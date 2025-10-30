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
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
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
    if period_flag == 1: 
        if m in [7, 8]:
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        elif m in [12, 1, 2]:
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
        else:
            if (9 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
    else:
        if m in [7, 8]:
            if (10 <= h < 12) or (13 <= h < 17): return MAX_PRICE
            if (9 <= h < 10) or (12 <= h < 13) or (17 <= h < 22): return MID_PRICE
            return LIGHT_PRICE
        elif m in [12, 1, 2]:
            if (9 <= h < 12) or (17 <= h < 22): return MAX_PRICE
            if (12 <= h < 17) or (22 <= h < 23): return MID_PRICE
            return LIGHT_PRICE
        else:
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
# 2) 인코딩
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
# 2.5) 🔥 단순화된 수요요금 계산 (벡터화)
# -----------------------------
def calculate_demand_charge_simple(df):
    """단순하고 빠른 수요요금 계산"""
    df = df.sort_values('측정일시').copy()
    df["피상전력_sim"] = np.sqrt(df["전력사용량(kWh)"]**2 + df["지상무효전력량(kVarh)"]**2)
    
    # 계절별 피크만 고려
    demand_months = [7, 8, 9, 12, 1, 2]
    df['is_demand_season'] = df['월'].isin(demand_months).astype(int)
    
    # 최근 365일 피크 (간단한 rolling)
    df['요금적용전력_kW_true'] = (
        df['피상전력_sim']
        .rolling(window=min(8760, len(df)), min_periods=1)
        .max()
    )
    
    return df

print("\n🔄 수요요금 계산 중...")
train = calculate_demand_charge_simple(train)

# -----------------------------
# 3) Stage1: 전력특성 예측
# -----------------------------
targets_s1 = ["전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)", 
              "지상역률(%)", "진상역률(%)", "요금적용전력_kW_true"]

feat_s1 = ["월","일","요일","시간","주말여부","겨울여부","period_flag",
           "sin_time","cos_time","sin_day", "cos_day", "sin_month", "cos_month",
           "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded"] 

# 🔥 하이퍼파라미터 원복 (과적합 방지)
stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(n_estimators=2000, learning_rate=0.015, num_leaves=96, random_state=42),
    "지상무효전력량(kVarh)": CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
    "진상무효전력량(kVarh)": CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
    "지상역률(%)": LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=64, random_state=42),
    "진상역률(%)": LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=64, random_state=42),
    "요금적용전력_kW_true": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=48, random_state=42),
}

tscv = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index)
stage1_test_pred = pd.DataFrame(index=test.index)
train_targets_true = train[targets_s1].copy()

print("\n🚀 Stage 1: 전력특성 예측 시작...")
for tgt in targets_s1:
    print(f"  학습 중: {tgt}")
    oof_pred = np.full(len(train), np.nan, dtype=float)
    model = stage1_models[tgt]
    
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
        oof_pred = np.expm1(oof_pred).clip(min=0)

    stage1_oof[tgt] = oof_pred
    
    final_model = model.__class__(**model.get_params())
    final_model.fit(train[feat_s1], current_target)
    test_pred = final_model.predict(test[feat_s1])
    
    if is_demand_target:
        test_pred = np.expm1(test_pred).clip(min=0)
        
    stage1_test_pred[tgt] = test_pred

for tgt in targets_s1:
    new_col_name = "요금적용전력_kW" if tgt == "요금적용전력_kW_true" else tgt
    train[new_col_name] = stage1_oof[tgt]
    test[new_col_name] = stage1_test_pred[tgt]
    
train["피상전력_sim"] = np.sqrt(train["전력사용량(kWh)"]**2 + train["지상무효전력량(kVarh)"]**2)
test["피상전력_sim"] = np.sqrt(test["전력사용량(kWh)"]**2 + test["지상무효전력량(kVarh)"]**2)

print("✅ Stage 1 완료")

# -----------------------------
# 3.5) Stage1 후처리
# -----------------------------
def post_process_stage1(df):
    P = df["전력사용량(kWh)"]
    Q = df["지상무효전력량(kVarh)"]
    
    safe_denominator = np.sqrt(P**2 + Q**2) + 1e-6
    df["PF_recalc"] = 100 * P / safe_denominator
    df["PF_recalc"] = df["PF_recalc"].clip(upper=100.0) 
    
    df["PF_diff"] = df["PF_recalc"] - df["지상역률(%)"]
    
    is_low_kwh = (df["전력사용량(kWh)"] < 0.5)
    df["PF_recalc"] = np.where(is_low_kwh, 95.0, df["PF_recalc"])
    df["PF_diff"] = np.where(is_low_kwh, 0.0, df["PF_diff"])
    
    return df

train = post_process_stage1(train)
test = post_process_stage1(test)

# -----------------------------
# 4) 역률 규정 피처
# -----------------------------
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
    df["역률우수"] = (df["지상역률_주간클립"] >= 95).astype(int) 
    
    df["법적페널티"] = ((df["지상역률_주간클립"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["실질위험"] = ((df["지상역률_주간클립"] < 94) & (df["주간여부"] == 1)).astype(int)
    
    return df

train = add_pf_features_regulated(train)
test = add_pf_features_regulated(test)

# -----------------------------
# 5) 🔥 핵심 교차항만 추가
# -----------------------------
def add_critical_interactions(df):
    # TOU × 역률부족
    df['tou_pf_risk'] = df['tou_relative_price'] * df['역률부족폭_94']
    
    # 위험 구간
    df['critical_zone'] = (
        (df['tou_relative_price'] == MAX_PRICE) & 
        (df['PF_recalc'] >= 90) & 
        (df['PF_recalc'] < 94)
    ).astype(int)
    
    # 전력 × 역률
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    
    return df

train = add_critical_interactions(train)
test = add_critical_interactions(test)

# -----------------------------
# 6) Lag/Rolling
# -----------------------------
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

hist_data_train = {"kwh": train["전력사용량(kWh)"]}
hist_data_test = {"kwh": train["전력사용량(kWh)"].copy()}

train = add_lag_roll(train, hist_data_train, is_train=True)
test = add_lag_roll(test, hist_data_test, is_train=False)

# -----------------------------
# 7) 고급 피처
# -----------------------------
kwh_mean_day_hour = train.groupby(["요일", "시간"])["전력사용량(kWh)"].mean().reset_index()
kwh_mean_day_hour.rename(columns={"전력사용량(kWh)": "kwh_요일_시간_평균"}, inplace=True)
train = pd.merge(train, kwh_mean_day_hour, on=["요일", "시간"], how="left")
test = pd.merge(test, kwh_mean_day_hour, on=["요일", "시간"], how="left")

def add_advanced_features(df, train_means=None):
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_주간클립"] + 1e-6) 

    if train_means: 
        df["역률_월평균"] = df["월"].map(train_means["역률_월평균"])
        df["역률_월평균"].fillna(train_means["역률_월평균"].mean(), inplace=True) 
    else: 
        df["역률_월평균"] = df.groupby("월")["지상역률_주간클립"].transform("mean")

    df["역률_월평균차이"] = df["지상역률_주간클립"] - df["역률_월평균"]
    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    df["kwh_변화율_24h"] = ((df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6))
    
    df["kwh_시간대비_요일"] = df["전력사용량(kWh)"] / (df["kwh_요일_시간_평균"] + 1e-6)
    df.drop("kwh_요일_시간_평균", axis=1, inplace=True)
    
    df["총무효전력"] = df["지상무효전력량(kVarh)"] + df["진상무효전력량(kVarh)"]
    
    return df

train_means_for_test = {"역률_월평균": train.groupby("월")["지상역률_주간클립"].mean()}
train = add_advanced_features(train)
test = add_advanced_features(test, train_means=train_means_for_test)

# -----------------------------
# 8) 일일 패턴
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

# -----------------------------
# 9) 🔥 Stage2 Feature Set (최소화)
# -----------------------------
feat_s2 = [
    "월","일","요일","시간","주말여부","겨울여부","period_flag",
    "sin_time","cos_time","sin_day", "cos_day", "sin_month", "cos_month",
    "작업유형_encoded", "tou_relative_price", "tou_price_code_encoded", "시간_작업유형_encoded",
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)",
    "지상역률(%)","진상역률(%)",
    "유효역률(%)","역률_조정요율",
    "지상역률_보정", "지상역률_주간클립", "주간여부", 
    "법적페널티","실질위험","역률부족폭_94", "역률부족폭_90", "역률우수",
    "총무효전력", 
    "PF_recalc", "PF_diff", 
    "무효유효비율","부하역률곱", "역률당전력",
    "역률_월평균","역률_월평균차이",
    "kwh_roll24_cv","kwh_변화율_24h",
    "kwh_lag1","kwh_lag24","kwh_roll24_mean","kwh_roll24_std",
    "kwh_시간대비_요일", 
    "요금적용전력_kW", "피상전력_sim", 
    "tou_pf_risk", "critical_zone",
    "일일패턴유형" 
]

print(f"\n💡 Stage 2 피처 개수: {len(feat_s2)}")

# -----------------------------
# 10) 🔥 Stage2 학습 (보수적 하이퍼파라미터)
# -----------------------------
X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)
X_te = test[feat_s2].copy()

# 🔥 하이퍼파라미터 보수적으로 조정 (과적합 방지)
LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.015, num_leaves=64, 
                  subsample=0.85, colsample_bytree=0.85, 
                  reg_alpha=3, reg_lambda=3, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.015, max_depth=6, 
                  subsample=0.85, colsample_bytree=0.85, 
                  reg_lambda=3, reg_alpha=2, random_state=42, n_jobs=-1)
CAT_PARAMS = dict(iterations=2500, learning_rate=0.02, depth=7, 
                  l2_leaf_reg=5, random_seed=42, verbose=0, thread_count=-1)

base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS)
}

# 🔥 Meta-Learner 단순화 (Ridge)
meta_learner = Ridge(alpha=10.0)
tscv_s2 = TimeSeriesSplit(n_splits=5)

oof_preds_s2 = pd.DataFrame(index=X_all.index, columns=base_models.keys(), dtype=float)
test_preds_s2 = np.zeros((len(X_te), len(base_models)))

print("\n🚀 Stage 2 모델 학습 시작...")
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

    test_preds_s2 += np.column_stack(fold_test_preds) / tscv_s2.n_splits

print("\n✅ OOF 예측 생성 완료.")

# Meta-Learner 학습 
oof_valid_idx = oof_preds_s2.dropna().index
print(f"\n🧠 Meta-Learner 학습 시작...")
meta_learner.fit(oof_preds_s2.loc[oof_valid_idx], y_all_log.loc[oof_valid_idx])

# 최종 Test 예측
meta_test_input = pd.DataFrame(test_preds_s2, columns=base_models.keys(), index=X_te.index)
pred_te_log = meta_learner.predict(meta_test_input)
pred_te = np.expm1(pred_te_log)

# OOF 검증
oof_pred_final_log = meta_learner.predict(oof_preds_s2.loc[oof_valid_idx])
oof_pred_final = np.expm1(oof_pred_final_log)
oof_mae = mean_absolute_error(y_all.loc[oof_valid_idx], oof_pred_final)
oof_r2 = r2_score(y_all.loc[oof_valid_idx], oof_pred_final)
print(f"\n📊 OOF 검증: MAE={oof_mae:.2f} | R²={oof_r2:.4f}")

# 월별 MAE
oof_valid_months = train.loc[oof_valid_idx, "월"]
monthly_mae = {}
for month in sorted(oof_valid_months.dropna().unique()):
    month_index = oof_valid_months.index[oof_valid_months == month]
    if len(month_index) == 0:
        continue
    month_mae = mean_absolute_error(y_all.loc[month_index], oof_pred_final[oof_valid_months == month])
    monthly_mae[int(month)] = month_mae

if monthly_mae:
    print("\n📆 월별 OOF MAE:")
    for month in sorted(monthly_mae):
        print(f"  {month}월 MAE={monthly_mae[month]:.2f}")

# -----------------------------
# 11) 🔥 보수적 후처리 (Train 분포 유지)
# -----------------------------
print("\n🔧 후처리 적용 중...")

# Train의 실제 분포 기반
train_p01 = np.percentile(y_all, 0.5)
train_p99 = np.percentile(y_all, 99.5)
train_mean = y_all.mean()
train_std = y_all.std()

print(f"Train 통계: 평균={train_mean:.2f}, 표준편차={train_std:.2f}")
print(f"Train 범위: {y_all.min():.2f} ~ {y_all.max():.2f}")

# 🔥 극단값만 부드럽게 조정
pred_te_adjusted = pred_te.copy()

# 하한선: 0.5 percentile 기준
lower_bound = max(0, train_p01 * 0.8)
pred_te_adjusted = np.where(
    pred_te < lower_bound,
    lower_bound + (pred_te - pred_te.min()) * 0.2,
    pred_te
)

# 상한선: 99.5 percentile 기준 (여유있게)
upper_bound = train_p99 * 1.05
pred_te_adjusted = np.where(
    pred_te_adjusted > upper_bound,
    upper_bound + (pred_te_adjusted - upper_bound) * 0.5,
    pred_te_adjusted
)

# 최종 물리적 제약
pred_te_final = np.clip(pred_te_adjusted, a_min=0, a_max=train_p99 * 1.1)

# -----------------------------
# 12) 제출 파일 생성
# -----------------------------
submission = pd.DataFrame({"id": test["id"], "target": pred_te_final})
submission.to_csv("submission_optimized_v5.csv", index=False) 
print("\n💾 submission_optimized_v5.csv 저장 완료!")
print(f"예측 범위: {pred_te_final.min():.2f} ~ {pred_te_final.max():.2f}")
print(f"예측 평균: {pred_te_final.mean():.2f} (Train: {train_mean:.2f})")
print(f"예측 표준편차: {pred_te_final.std():.2f} (Train: {train_std:.2f})")

# -----------------------------
# 13) Meta 계수 분석
# -----------------------------
print("\n📊 Meta-Learner 계수:")
w = pd.Series(meta_learner.coef_, index=list(base_models.keys()))
w_norm = (w / w.abs().sum()).sort_values(ascending=False)
print(w_norm.round(3))

# -----------------------------
# 14) 🔥 간단한 Feature Importance (LightGBM 기준)
# -----------------------------
print("\n🔍 Feature Importance (Top 30)...")
lgb_model = LGBMRegressor(**LGB_PARAMS)
lgb_model.fit(X_all, y_all_log)
feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feat_imp.head(30).to_string(index=False))
feat_imp.to_csv("feature_importance_v5.csv", index=False)
print("➡ feature_importance_v5.csv 저장")

# -----------------------------
# 15) 🎯 예측 분포 비교
# -----------------------------
print("\n📊 예측 분포 비교:")
print(f"Train 분위수 (10%, 25%, 50%, 75%, 90%):")
train_quantiles = np.percentile(y_all, [10, 25, 50, 75, 90])
print(f"  {train_quantiles}")

print(f"Test 예측 분위수 (10%, 25%, 50%, 75%, 90%):")
test_quantiles = np.percentile(pred_te_final, [10, 25, 50, 75, 90])
print(f"  {test_quantiles}")

# -----------------------------
# 16) 🔥 추가 앙상블: 단순 평균과 혼합
# -----------------------------
print("\n🔄 추가 앙상블 생성 중...")

# 베이스 모델들의 단순 평균
test_preds_avg = np.mean(test_preds_s2, axis=1)
pred_te_simple = np.expm1(test_preds_avg)

# 🔥 Meta 예측과 단순 평균 혼합 (60:40)
pred_te_blended = 0.6 * pred_te_final + 0.4 * pred_te_simple
pred_te_blended = np.clip(pred_te_blended, a_min=0, a_max=train_p99 * 1.1)

submission_blended = pd.DataFrame({"id": test["id"], "target": pred_te_blended})
submission_blended.to_csv("submission_blended_v5.csv", index=False)
print("💾 submission_blended_v5.csv 저장 완료! (Meta 60% + Simple 40%)")
print(f"Blended 예측 범위: {pred_te_blended.min():.2f} ~ {pred_te_blended.max():.2f}")
print(f"Blended 예측 평균: {pred_te_blended.mean():.2f}")

# -----------------------------
# 17) 🎯 고위험 구간 분석
# -----------------------------
print("\n🎯 고위험 구간 성능 분석...")
risk_samples = train.loc[oof_valid_idx].query(
    "tou_relative_price == 1.0 and PF_recalc < 94"
)

if len(risk_samples) > 100:
    risk_idx = risk_samples.index
    risk_mae = mean_absolute_error(
        y_all.loc[risk_idx], 
        oof_pred_final[oof_valid_idx.isin(risk_idx)]
    )
    print(f"고위험 구간 MAE: {risk_mae:.2f} (샘플 수: {len(risk_samples)})")
    
    normal_idx = oof_valid_idx[~oof_valid_idx.isin(risk_idx)]
    normal_mae = mean_absolute_error(
        y_all.loc[normal_idx], 
        oof_pred_final[oof_valid_idx.isin(normal_idx)]
    )
    print(f"일반 구간 MAE: {normal_mae:.2f}")
    print(f"위험도 비율: {risk_mae / normal_mae:.2f}x")

print("\n" + "="*60)
print("✅ 최적화 완료!")
print("="*60)
print("\n📌 제출 파일:")
print("  1. submission_optimized_v5.csv (메인)")
print("  2. submission_blended_v5.csv (앙상블 혼합)")
print("\n💡 권장 제출 순서:")
print("  1단계: submission_optimized_v5.csv")
print("  2단계: submission_blended_v5.csv")
print("="*60)