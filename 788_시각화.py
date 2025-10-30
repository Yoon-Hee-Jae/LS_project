
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans 
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import HuberRegressor # Meta Learner 변경
from sklearn.linear_model import RidgeCV 

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
# 파일 경로를 사용자 환경에 맞게 조정하세요.
train = pd.read_csv("./data/train_.csv")
test = pd.read_csv("./data/test_.csv")

# -----------------------------
# 1) 전역 상수 정의
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")

MAX_PRICE = 1.0
MID_PRICE = 0.6
LIGHT_PRICE = 0.4

# -----------------------------
# 1.5) 시간 파생 및 TOU 인코딩
# -----------------------------

def adjust_hour(dt):
    if pd.isna(dt): return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour

def get_tou_relative_price(m, h, period_flag):
    # TOU 로직은 이전과 동일
    if period_flag == 1: 
        if m in [7, 8]:  # Summer
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        
        elif m in [12, 1, 2]:  # Winter
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE

        else:  # Spring/Fall
            if (9 <= h < 23): return MID_PRICE
            return LIGHT_PRICE

    else: # 2024-10-24 이전
        if m in [7, 8]:  # Summer
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        
        elif m in [12, 1, 2]:  # Winter
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE

        else:  # Spring/Fall
            if (9 <= h < 23): return MID_PRICE
            return LIGHT_PRICE


def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["날짜"] = df['측정일시'].dt.date 
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int) 
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    
    df["tou_relative_price"] = df.apply(lambda row: get_tou_relative_price(row["월"], row["시간"], row["period_flag"]), axis=1)
    
    df["tou_load_index"] = df.apply(lambda row: 3 if row["tou_relative_price"] == MAX_PRICE else (2 if row["tou_relative_price"] == MID_PRICE else 1), axis=1)
    df["tou_price_code"] = df["period_flag"].astype(str) + "_" + df["tou_load_index"].astype(str)
    
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    return df

train = enrich(train).sort_values("측정일시").reset_index(drop=True)
test = enrich(test).sort_values("측정일시").reset_index(drop=True)

# -----------------------------
# 2) 인코딩 (작업유형 및 TOU 코드)
# -----------------------------
le_job = LabelEncoder()
train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
def safe_transform(le, series, mode_val):
    series_mapped = series.astype(str).map(lambda s: '-1' if s not in le.classes_ else s)
    return le.transform(series_mapped.replace('-1', mode_val))

test["작업유형_encoded"] = safe_transform(le_job, test["작업유형"], train["작업유형"].mode()[0])

le_tou = LabelEncoder()
train["tou_price_code_encoded"] = le_tou.fit_transform(train["tou_price_code"].astype(str))
test["tou_price_code_encoded"] = safe_transform(le_tou, test["tou_price_code"], train["tou_price_code"].mode()[0])

train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
test["시간_작업유형_encoded"] = safe_transform(le_tj, test["시간_작업유형"], train["시간_작업유형"].mode()[0])


# -----------------------------
# 2.5) 요금적용전력 (Demand Charge) 실제값 계산
# -----------------------------

# 📌 실제 요금적용전력 (Demand Charge) Target 계산 함수 (Loop 기반)
def calculate_demand_charge_true(df):
    df["피상전력_sim"] = np.sqrt(df["전력사용량(kWh)"]**2 + df["지상무효전력량(kVarh)"]**2)
    df["요금적용전력_kW_true"] = 0.0
    demand_months = [12, 1, 2, 7, 8, 9] 
    
    for idx in df.index:
        current_date = df.loc[idx, "측정일시"]
        start_date = current_date - pd.DateOffset(months=12)
        history_df = df.loc[(df["측정일시"] >= start_date) & 
                            (df["측정일시"] < current_date) & 
                            (df["월"].isin(demand_months))]
        
        current_max_demand = 0.0
        if not history_df.empty:
            max_demand = history_df["피상전력_sim"].max()
            current_max_demand = max(current_max_demand, max_demand)

        if current_date.month in demand_months:
             current_max_demand = max(current_max_demand, df.loc[idx, "피상전력_sim"])

        df.loc[idx, "요금적용전력_kW_true"] = current_max_demand

    df.fillna(method='bfill', inplace=True)
    return df.fillna(0)

train = calculate_demand_charge_true(train)
# test에는 true 값을 알 수 없으므로, train에서만 계산하여 예측 Target으로 사용합니다.

# -----------------------------
# 3) Stage1: 전력특성 및 요금적용전력 예측 (Demand Charge 예측 모델 추가)
# -----------------------------
targets_s1 = ["전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)", 
              "지상역률(%)", "진상역률(%)", "요금적용전력_kW_true"] # 📌 Demand Charge Target 추가

feat_s1 = ["월","일","요일","시간","주말여부","겨울여부","period_flag",
           "sin_time","cos_time","sin_day", "cos_day", "sin_month", "cos_month",
           "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded"] 

stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(n_estimators=2500, learning_rate=0.012, num_leaves=128, random_state=42),
    "지상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "진상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "지상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
    "진상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
    # 📌 Demand Charge 예측 모델: Log 변환을 통해 안정성 확보
    "요금적용전력_kW_true": LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=64, random_state=42, 
                                          subsample=0.8, colsample_bytree=0.8),
}

tscv = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index)
stage1_test_pred = pd.DataFrame(index=test.index)
train_targets_true = train[targets_s1].copy()

for tgt in targets_s1:
    oof_pred = np.full(len(train), np.nan, dtype=float)
    model = stage1_models[tgt]
    
    # Demand Charge Target에만 Log 변환 적용
    current_target = train_targets_true[tgt].copy()
    is_demand_target = (tgt == "요금적용전력_kW_true")
    if is_demand_target:
        current_target = np.log1p(current_target)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train), start=1):
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(train.iloc[tr_idx][feat_s1], current_target.iloc[tr_idx])
        oof_pred[va_idx] = fold_model.predict(train.iloc[va_idx][feat_s1])

    missing = np.isnan(oof_pred)
    if missing.any():
        full_model = model.__class__(**model.get_params())
        full_model.fit(train[feat_s1], current_target)
        oof_pred[missing] = full_model.predict(train.loc[missing, feat_s1])
        
    if is_demand_target:
        oof_pred = np.expm1(oof_pred).clip(min=0) # 역변환 및 0 미만 클립

    stage1_oof[tgt] = oof_pred
    
    final_model = model.__class__(**model.get_params())
    final_model.fit(train[feat_s1], current_target)
    test_pred = final_model.predict(test[feat_s1])
    
    if is_demand_target:
        test_pred = np.expm1(test_pred).clip(min=0) # 역변환 및 0 미만 클립
        
    stage1_test_pred[tgt] = test_pred

for tgt in targets_s1:
    # Target 이름 정리: 요금적용전력_kW_true 예측값을 요금적용전력_kW로 사용
    new_col_name = "요금적용전력_kW" if tgt == "요금적용전력_kW_true" else tgt
    train[new_col_name] = stage1_oof[tgt]
    test[new_col_name] = stage1_test_pred[tgt]
    
# 기존 피상전력_sim 재계산: 예측된 전력, 무효전력으로 계산 
train["피상전력_sim"] = np.sqrt(train["전력사용량(kWh)"]**2 + train["지상무효전력량(kVarh)"]**2)
test["피상전력_sim"] = np.sqrt(test["전력사용량(kWh)"]**2 + test["지상무효전력량(kVarh)"]**2)


# -----------------------------
# 3.5) Stage1 예측값 후처리 및 물리적 특성 추가 (PF 재계산 및 PF_diff)
# -----------------------------
def post_process_stage1(df):
    P = df["전력사용량(kWh)"]
    Q = df["지상무효전력량(kVarh)"]
    
    safe_denominator = np.sqrt(P**2 + Q**2) + 1e-6
    df["PF_recalc"] = 100 * P / safe_denominator
    df["PF_recalc"] = df["PF_recalc"].clip(upper=100.0) 
    
    df["PF_diff"] = df["PF_recalc"] - df["지상역률(%)"]
    
    # 📌 노이즈 감소: 전력사용량이 0에 가까우면 PF 관련 특성을 안정화 (임계값 0.5)
    is_low_kwh = (df["전력사용량(kWh)"] < 0.5)
    df["PF_recalc"] = np.where(is_low_kwh, 95.0, df["PF_recalc"])
    df["PF_diff"] = np.where(is_low_kwh, 0.0, df["PF_diff"])
    
    return df

train = post_process_stage1(train)
test = post_process_stage1(test)

# -----------------------------
# 4) 역률 규정 피처
# -----------------------------
# (코드 동일)
def add_pf_features_regulated(df):
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0) * 0.01
    df["역률_보상율"] = (df["유효역률(%)"] - 90).clip(lower=0) * 0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    
    df["주간여부"] = df["시간"].isin(range(9, 23)).astype(int)
    
    df["지상역률_보정"] = df["PF_recalc"].clip(lower=60)
    df["지상역률_주간클립"] = np.where(df["주간여부"] == 1, 
                                    df["지상역률_보정"].clip(upper=95), 
                                    df["지상역률_보정"])
    
    df["역률부족폭_94"] = (94 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    df["역률부족폭_90"] = (90 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    df["역률부족폭_92"] = (92 - df["지상역률_주간클립"]).clip(lower=0) * df["주간여부"]
    
    df["역률우수"] = (df["지상역률_주간클립"] >= 95).astype(int) 
    
    df["야간여부"] = (1 - df["주간여부"]).astype(int)
    df["진상역률_페널티"] = (95 - df["진상역률(%)"]).clip(lower=0) * df["야간여부"]
    
    df["법적페널티"] = ((df["지상역률_주간클립"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["실질위험"] = ((df["지상역률_주간클립"] < 94) & (df["주간여부"] == 1)).astype(int)
    df["극저역률"] = ((df["지상역률_주간클립"] < 85) & (df["주간여부"] == 1)).astype(int)
    
    return df

train = add_pf_features_regulated(train)
test = add_pf_features_regulated(test)


# -----------------------------
# 5) Lag/Rolling (Demand Charge Loop 삭제)
# -----------------------------
# 📌 요금적용전력_kW는 이미 Stage 1에서 예측되었으므로, Lag/Rolling만 계산
def add_lag_roll(df, hist_data, is_train=True):
    df["kwh_lag1"] = df["전력사용량(kWh)"].shift(1)
    df["kwh_lag24"] = df["전력사용량(kWh)"].shift(24)
    df["kwh_roll24_mean"] = df["전력사용량(kWh)"].shift(1).rolling(24).mean()
    df["kwh_roll24_std"] = df["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
    
    if is_train:
        df.fillna(method='bfill', inplace=True)
        return df.fillna(0)
    
    else: 
        hist_data_kwh = list(hist_data["kwh"].values.astype(float))
        
        for i in range(len(df)):
            df.loc[df.index[i], "kwh_lag1"] = hist_data_kwh[-1] if len(hist_data_kwh) >= 1 else 0
            df.loc[df.index[i], "kwh_lag24"] = hist_data_kwh[-24] if len(hist_data_kwh) >= 24 else 0
            arr24 = np.array(hist_data_kwh[-24:])
            df.loc[df.index[i], "kwh_roll24_mean"] = arr24.mean() if arr24.size > 0 else 0
            df.loc[df.index[i], "kwh_roll24_std"] = arr24.std() if arr24.size > 1 else 0
            
            hist_data_kwh.append(df.loc[df.index[i], "전력사용량(kWh)"])
            
        return df

hist_data_train = {
    "kwh": train["전력사용량(kWh)"],
}
hist_data_test = {
    "kwh": train["전력사용량(kWh)"].copy(),
}

train = add_lag_roll(train, hist_data_train, is_train=True)
test = add_lag_roll(test, hist_data_test, is_train=False)

# -----------------------------
# 6) 고급 피처 추가 
# -----------------------------
kwh_mean_day_hour = train.groupby(["요일", "시간"])["전력사용량(kWh)"].mean().reset_index()
kwh_mean_day_hour.rename(columns={"전력사용량(kWh)": "kwh_요일_시간_평균"}, inplace=True)
train = pd.merge(train, kwh_mean_day_hour, on=["요일", "시간"], how="left")
test = pd.merge(test, kwh_mean_day_hour, on=["요일", "시간"], how="left")

def add_advanced_features_hybrid(df, train_means=None):
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"] 
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_주간클립"] + 1e-6) 
    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률_주간클립"] >= 94)).astype(int)

    if train_means: 
        df["역률_월평균"] = df["월"].map(train_means["역률_월평균"])
        df["역률_월평균"].fillna(train_means["역률_월평균"].mean(), inplace=True) 
    else: 
        df["역률_월평균"] = df.groupby("월")["지상역률_주간클립"].transform("mean")

    df["역률_월평균차이"] = df["지상역률_주간클립"] - df["역률_월평균"]
    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    df["kwh_변화율_24h"] = ((df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6))
    df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    
    df["kwh_시간대비_요일"] = df["전력사용량(kWh)"] / (df["kwh_요일_시간_평균"] + 1e-6)
    df.drop("kwh_요일_시간_평균", axis=1, inplace=True)
    
    df["총무효전력"] = df["지상무효전력량(kVarh)"] + df["진상무효전력량(kVarh)"]
    
    return df

train_means_for_test = {"역률_월평균": train.groupby("월")["지상역률_주간클립"].mean()}
train = add_advanced_features_hybrid(train)
test = add_advanced_features_hybrid(test, train_means=train_means_for_test)

# -----------------------------
# 6.5) 일일 전력 사용 패턴 피처 생성
# -----------------------------
print("\n🔄 일일 전력 사용 패턴 피처 생성 중...")
train_pattern_pivot = train.pivot_table(index='날짜', columns='시간', values='전력사용량(kWh)')
train_pattern_pivot = train_pattern_pivot.fillna(train_pattern_pivot.mean().mean())

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
train_pattern_pivot['일일패턴유형'] = kmeans.fit_predict(train_pattern_pivot)
pattern_map = train_pattern_pivot[['일일패턴유형']].reset_index()
train = pd.merge(train, pattern_map, on='날짜', how='left')

test_pattern_pivot = test.pivot_table(index='날짜', columns='시간', values='전력사용량(kWh)')
test_pattern_pivot = test_pattern_pivot.fillna(train_pattern_pivot.drop('일일패턴유형', axis=1).mean().mean())
train_cols_no_target = train_pattern_pivot.drop(columns='일일패턴유형').columns
test_pattern_pivot = test_pattern_pivot.reindex(columns=train_cols_no_target, fill_value=train_pattern_pivot.drop('일일패턴유형', axis=1).mean().mean())
test_pattern_pivot['일일패턴유형'] = kmeans.predict(test_pattern_pivot[train_cols_no_target])
test_pattern_map = test_pattern_pivot[['일일패턴유형']].reset_index()
test = pd.merge(test, test_pattern_map, on='날짜', how='left')

most_frequent_pattern = train['일일패턴유형'].mode()[0]
train['일일패턴유형'].fillna(most_frequent_pattern, inplace=True)
test['일일패턴유형'].fillna(most_frequent_pattern, inplace=True)
train['일일패턴유형'] = train['일일패턴유형'].astype(int)
test['일일패턴유형'] = test['일일패턴유형'].astype(int)
print(f"✅ 일일 전력 패턴 유형 생성 완료. 가장 흔한 유형: {most_frequent_pattern}")


# -----------------------------
# 7) Stage2 Feature Set
# -----------------------------
feat_s2 = [
    "월","일","요일","시간","주말여부","겨울여부","period_flag",
    "sin_day", "cos_day", "sin_month", "cos_month", 
    "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded",
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)",
    "진상역률(%)",
    "유효역률(%)","역률_조정요율",
    "지상역률_보정", "지상역률_주간클립", "주간여부", "야간여부", 
    "법적페널티","실질위험","극저역률","역률부족폭_94", "역률부족폭_92", 
    "진상역률_페널티", "총무효전력", 
    "PF_recalc", "PF_diff", 
    "무효유효비율","부하역률곱", "역률당전력","가을위험","동절기안정","역률_월평균",
    "역률_월평균차이","kwh_roll24_cv","kwh_변화율_24h","전력급등","kwh_lag1",
    "kwh_lag24","kwh_roll24_mean","kwh_roll24_std",
    "kwh_시간대비_요일", 
    "요금적용전력_kW", "피상전력_sim", 
    "일일패턴유형" 
]
print(f"\n💡 Stage 2 피처 개수: {len(feat_s2)}")

# -----------------------------
# 8) Stage2 학습 (Hyperparameter 및 Meta Learner 변경)
# -----------------------------
X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)
X_te = test[feat_s2].copy()

# 📌 Hyperparameter 조정 (n_estimators 증가, 정규화 강화)
LGB_PARAMS = dict(n_estimators=4000, learning_rate=0.012, num_leaves=75, subsample=0.8, colsample_bytree=0.8, reg_alpha=5, reg_lambda=6, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=4000, learning_rate=0.012, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=6, reg_alpha=3, random_state=42, n_jobs=-1)
CAT_PARAMS = dict(iterations=3000, learning_rate=0.015, depth=7, l2_leaf_reg=8, random_seed=42, verbose=0, thread_count=-1) # l2_leaf_reg 강화

base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS)
}

# 📌 Meta Learner 변경 (Outlier에 강한 HuberRegressor)
meta_learner = HuberRegressor(epsilon=1.35) 
tscv_s2 = TimeSeriesSplit(n_splits=5)

oof_preds_s2 = pd.DataFrame(index=X_all.index, columns=base_models.keys(), dtype=float)
test_preds_s2 = np.zeros((len(X_te), len(base_models)))

print("\n🚀 Stage 2 모델 학습 및 OOF 예측 생성 시작...")
for fold, (tr_idx, va_idx) in enumerate(tscv_s2.split(X_all), start=1):
    print(f"--- Fold {fold} ---")
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr_log = y_all_log.iloc[tr_idx]

    fold_test_preds = [] 

    for name, model in base_models.items():
        print(f"  Training {name}...")
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_tr, y_tr_log)

        oof_pred = fold_model.predict(X_va)
        oof_preds_s2.iloc[va_idx, list(base_models.keys()).index(name)] = oof_pred

        fold_test_preds.append(fold_model.predict(X_te))

    test_preds_s2 += np.mean(fold_test_preds, axis=0)[:, np.newaxis] / tscv_s2.n_splits

print("\n✅ OOF 예측 생성 완료.")

# Meta-Learner 학습 
oof_valid_idx = oof_preds_s2.dropna().index
print(f"\n🧠 Meta-Learner ({meta_learner.__class__.__name__}) 학습 시작 (데이터 {len(oof_valid_idx)}개)...")
# HuberRegressor는 Log 변환된 OOF 예측값을 입력으로 받습니다.
meta_learner.fit(oof_preds_s2.loc[oof_valid_idx], y_all_log.loc[oof_valid_idx])
print(f"✅ Meta-Learner 학습 완료.")

# 최종 Test 예측
print("\n🧪 최종 Test 예측 생성...")
meta_test_input = pd.DataFrame(test_preds_s2, columns=base_models.keys(), index=X_te.index)
pred_te_log = meta_learner.predict(meta_test_input)
pred_te = np.expm1(pred_te_log)

# OOF 검증 점수 계산
oof_pred_final_log = meta_learner.predict(oof_preds_s2.loc[oof_valid_idx])
oof_pred_final = np.expm1(oof_pred_final_log)
oof_mae = mean_absolute_error(y_all.loc[oof_valid_idx], oof_pred_final)
oof_r2 = r2_score(y_all.loc[oof_valid_idx], oof_pred_final)
print(f"\n📊 OOF 검증 (Stacking): MAE={oof_mae:.2f} | R²={oof_r2:.4f}")


# -----------------------------
# 9) 후처리 및 제출
# -----------------------------
# 📌 예측 범위 클리핑을 전체 범위(Max 42만)에 가깝게 완화하여 과소평가 문제 개선 시도
low, high = np.percentile(pred_te, [0.01, 99.9]) 
pred_te = np.clip(pred_te, low, high)
pred_te = np.clip(pred_te, a_min=500, a_max=450000) # 현실적인 최소/최대값으로 클립

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_demand_focused_v3.csv", index=False) 
print("\n💾 submission_demand_focused_v3.csv 저장 완료!")
print(f"예측 범위: {pred_te.min():.2f} ~ {pred_te.max():.2f}")
print(f"예측 평균: {pred_te.mean():.2f}")

# -----------------------------
# 11) 실전형 EDA 시각화 (콘솔 출력)
# -----------------------------
print("\n📊 실전형 EDA 시각화 실행 중...")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
eda_df = train.copy()
eda_df = eda_df.replace([np.inf, -np.inf], np.nan)
eda_df["PF_band"] = pd.cut(
    eda_df["PF_recalc"],
    bins=[-np.inf, 90, 94, np.inf],
    labels=["PF<90", "90~94", "≥95"],
)
eda_df["PF_band"] = eda_df["PF_band"].astype(str)
eda_df["주간라벨"] = eda_df["주간여부"].map({1: "주간", 0: "야간"}).fillna("미확인")
eda_df["PF_90_cut"] = np.where(eda_df["PF_recalc"] < 90, "PF<90", "PF≥90")
eda_df["TOU라벨"] = eda_df["tou_relative_price"].map({MAX_PRICE: "최대부하", MID_PRICE: "중간부하", LIGHT_PRICE: "경부하"}).fillna("미확인")
eda_df["일일패턴유형"] = eda_df["일일패턴유형"].astype("Int64")
# 1) 전기요금 vs DemandCharge / 역률 / (부하 or TOU) 박스플롯
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
eda_box = eda_df.dropna(subset=["전기요금(원)", "요금적용전력_kW", "PF_recalc"])
eda_box["요금적용전력_bin"] = pd.qcut(eda_box["요금적용전력_kW"], 5, duplicates="drop")
eda_box["PF_bin"] = pd.qcut(eda_box["PF_recalc"], 5, duplicates="drop")
sns.boxplot(data=eda_box, x="요금적용전력_bin", y="전기요금(원)", ax=axes[0])
axes[0].set_title("전기요금 vs 요금적용전력 (qcut)")
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=eda_box, x="PF_bin", y="전기요금(원)", ax=axes[1])
axes[1].set_title("전기요금 vs PF 재계산 (qcut)")
axes[1].tick_params(axis="x", rotation=30)
if "부하구분" in eda_df.columns:
    sns.boxplot(data=eda_df, x="부하구분", y="전기요금(원)", ax=axes[2])
    axes[2].set_title("전기요금 vs 부하구분")
else:
    sns.boxplot(data=eda_df, x="TOU라벨", y="전기요금(원)", ax=axes[2])
    axes[2].set_title("전기요금 vs TOU 단계")
axes[2].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()
# 2) 주요 그룹 비교 (주간/야간, TOU별, PF<90 vs ≥90, 패턴유형별)
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
sns.boxplot(data=eda_df, x="주간라벨", y="전기요금(원)", ax=axes[0, 0])
axes[0, 0].set_title("주간/야간별 전기요금")
sns.boxplot(data=eda_df, x="TOU라벨", y="전기요금(원)", ax=axes[0, 1])
axes[0, 1].set_title("TOU 단계별 전기요금")
axes[0, 1].tick_params(axis="x", rotation=15)
sns.boxplot(data=eda_df, x="PF_90_cut", y="전기요금(원)", ax=axes[1, 0])
axes[1, 0].set_title("PF 90 기준 전기요금")
sns.boxplot(data=eda_df, x="일일패턴유형", y="전기요금(원)", ax=axes[1, 1])
axes[1, 1].set_title("일일 패턴 유형별 전기요금")
plt.tight_layout()
plt.show()
# 3) Partial dependence 스타일 구간 비교
def plot_partial_dependence(df, feature, target, hue, ax, q=10):
    sub = df[[feature, target, hue]].dropna()
    if sub.empty:
        ax.set_visible(False)
        return
    try:
        sub["bin"] = pd.qcut(sub[feature], q=q, duplicates="drop")
    except ValueError:
        sub["bin"] = pd.cut(sub[feature], bins=q)
    stats = (
        sub.groupby([hue, "bin"])[target]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats["bin_center"] = stats["bin"].apply(lambda x: x.mid if hasattr(x, "mid") else np.nan)
    sns.lineplot(data=stats, x="bin_center", y="mean", hue=hue, marker="o", ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"{target} 평균")
    ax.set_title(f"{feature} 구간별 평균 {target}")
    ax.legend(title=hue, loc="best")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
partial_features = ["전력사용량(kWh)", "요금적용전력_kW", "무효유효비율"]
for ax, feat in zip(axes, partial_features):
    plot_partial_dependence(eda_df, feat, "전기요금(원)", "PF_band", ax, q=10)
plt.tight_layout()
plt.show()
# 4) 전력사용량 대비 요금의 PF 구간별 기울기 비교
fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
for ax, band in zip(axes, ["PF<90", "90~94", "≥95"]):
    band_df = eda_df[eda_df["PF_band"] == band][["전력사용량(kWh)", "전기요금(원)"]].dropna()
    if len(band_df) < 10:
        ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center")
        ax.set_title(f"{band}")
        ax.set_xlabel("전력사용량(kWh)")
        ax.set_ylabel("전기요금(원)")
        continue
    sns.regplot(
        data=band_df,
        x="전력사용량(kWh)",
        y="전기요금(원)",
        scatter_kws={"alpha": 0.3, "s": 20},
        line_kws={"color": "red"},
        ax=ax,
    )
    ax.set_title(f"{band} 구간")
    ax.set_xlabel("전력사용량(kWh)")
    ax.set_ylabel("전기요금(원)")
plt.tight_layout()
plt.show()
# 5) 요금 급등(상위 10%) 구간 집중 분석
spike_threshold = eda_df["전기요금(원)"].quantile(0.9)
spike_df = eda_df[eda_df["전기요금(원)"] >= spike_threshold]
compare_targets = {
    "PF_recalc": "PF 재계산",
    "무효유효비율": "Q/P (무효/유효)",
    "tou_relative_price": "TOU 상대요금",
    "kwh_변화율_24h": "24시간 전 대비 kWh 변화율",
}
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for ax, (col, label) in zip(axes.flatten(), compare_targets.items()):
    base_series = eda_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    spike_series = spike_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if base_series.empty or spike_series.empty:
        ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center")
        ax.set_title(label)
        continue
    sns.kdeplot(base_series, label="전체", ax=ax, fill=True, alpha=0.4)
    sns.kdeplot(spike_series, label="상위 10%", ax=ax, fill=True, alpha=0.4)
    ax.set_title(label)
    ax.legend()
plt.tight_layout()
plt.show()
print("\n✅ 실전형 EDA 시각화 출력 완료.")