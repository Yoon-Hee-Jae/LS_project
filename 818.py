import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge # Stacking Meta-Learner

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
# 파일 경로를 사용자 환경에 맞게 조정하세요.
train = pd.read_csv("./data/train_.csv") 
test = pd.read_csv("./data/test_.csv")

# -----------------------------
# 1) 시간 및 인코딩 파생 (강화)
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")

def adjust_hour(dt):
    if pd.isna(dt):
        return np.nan
    # 00:00 -> 23:xx로 조정하는 대신, 일관성을 위해 00:xx는 0, 01:xx는 1 등으로 처리
    return dt.hour if dt.minute >= 15 else (dt.hour - 1) % 24 
    # 원래 코드의 논리를 유지: 00분이면 이전 시간대로.

def band_of_hour(h):
    if (22 <= h <= 23) or (0 <= h <= 7):
        return "경부하"
    if 16 <= h <= 21:
        return "최대부하"
    return "중간부하"

def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["시간"] = df["측정일시"].apply(lambda x: adjust_hour(x)) # apply 방식 변경
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["부하구분"] = df["시간"].apply(band_of_hour)
    
    # 추가 시간 피처
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    
    return df

train = enrich(train).sort_values("측정일시").reset_index(drop=True)
test = enrich(test).sort_values("측정일시").reset_index(drop=True)

# 인코딩
le_job = LabelEncoder()
train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
test["작업유형_encoded"] = le_job.transform(test["작업유형"].astype(str))

le_band = LabelEncoder()
train["부하구분_encoded"] = le_band.fit_transform(train["부하구분"].astype(str))
test["부하구분_encoded"] = le_band.transform(test["부하구분"].astype(str))

train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
test["시간_작업유형_encoded"] = le_tj.transform(test["시간_작업유형"])

# -----------------------------
# 2) Stage1: 전력특성 예측 (Stage1 모델의 n_estimators 증가)
# -----------------------------
targets_s1 = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]
feat_s1 = [
    "월", "일", "요일", "시간", "주말여부", "겨울여부", "period_flag",
    "sin_time", "cos_time", "sin_day", "cos_day", "sin_month", "cos_month",
    "작업유형_encoded", "부하구분_encoded", "시간_작업유형_encoded",
]

stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(
        n_estimators=4000, learning_rate=0.008, num_leaves=128,  # n_estimators 증가
        subsample=0.85, colsample_bytree=0.85, random_state=42
    ),
    "지상무효전력량(kVarh)": CatBoostRegressor(
        iterations=3500, learning_rate=0.02, depth=8, verbose=0, random_seed=42 # iterations 증가
    ),
    "진상무효전력량(kVarh)": CatBoostRegressor(
        iterations=3500, learning_rate=0.02, depth=8, verbose=0, random_seed=42 # iterations 증가
    ),
    "지상역률(%)": LGBMRegressor(
        n_estimators=3500, learning_rate=0.015, num_leaves=96, random_state=42 # n_estimators 증가
    ),
    "진상역률(%)": LGBMRegressor(
        n_estimators=3500, learning_rate=0.015, num_leaves=96, random_state=42 # n_estimators 증가
    ),
}

tscv = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index)
stage1_test_pred = pd.DataFrame(index=test.index)
train_targets_true = train[targets_s1].copy()

for tgt in targets_s1:
    oof_pred = np.full(len(train), np.nan, dtype=float)
    model = stage1_models[tgt]

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train), start=1):
        fold_model = stage1_models[tgt].__class__(**stage1_models[tgt].get_params())
        fold_model.fit(train.iloc[tr_idx][feat_s1], train.iloc[tr_idx][tgt])
        oof_pred[va_idx] = fold_model.predict(train.iloc[va_idx][feat_s1])

    missing = np.isnan(oof_pred)
    if missing.any():
        full_model = stage1_models[tgt].__class__(**stage1_models[tgt].get_params())
        full_model.fit(train[feat_s1], train[tgt])
        oof_pred[missing] = full_model.predict(train.loc[missing, feat_s1])

    stage1_oof[tgt] = oof_pred

    final_model = stage1_models[tgt].__class__(**stage1_models[tgt].get_params())
    final_model.fit(train[feat_s1], train_targets_true[tgt])
    stage1_test_pred[tgt] = final_model.predict(test[feat_s1])

for tgt in targets_s1:
    train[f"{tgt}_true"] = train_targets_true[tgt]
    train[tgt] = stage1_oof[tgt]
    test[tgt] = stage1_test_pred[tgt]

# -----------------------------
# 3) 역률 피처 (코드1 기반 + 추가)
# -----------------------------
def add_pf_features(df):
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0) * 0.01
    df["역률_보상율"] = (df["유효역률(%)"] - 90).clip(lower=0) * 0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    
    df["지상역률_보정"] = df["지상역률(%)"].clip(lower=60)
    df["주간여부"] = df["부하구분"].isin(["중간부하", "최대부하"]).astype(int)
    
    df["법적페널티"] = ((df["지상역률_보정"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["실질위험"] = ((df["지상역률_보정"] < 94) & (df["주간여부"] == 1)).astype(int)
    df["극저역률"] = ((df["지상역률_보정"] < 85) & (df["주간여부"] == 1)).astype(int)
    df["역률부족폭_94"] = (94 - df["지상역률_보정"]).clip(lower=0) * df["주간여부"]
    
    # 추가 역률 피처
    df["역률부족폭_90"] = (90 - df["지상역률_보정"]).clip(lower=0) * df["주간여부"]
    df["역률우수"] = (df["지상역률_보정"] >= 95).astype(int)
    df["역률_60_85"] = ((df["지상역률_보정"] >= 60) & (df["지상역률_보정"] < 85) & (df["주간여부"] == 1)).astype(int)
    df["역률_85_90"] = ((df["지상역률_보정"] >= 85) & (df["지상역률_보정"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["역률_90_94"] = ((df["지상역률_보정"] >= 90) & (df["지상역률_보정"] < 94) & (df["주간여부"] == 1)).astype(int)
    df["역률_94_이상"] = ((df["지상역률_보정"] >= 94) & (df["주간여부"] == 1)).astype(int)
    df["주간_부족률"] = df["주간여부"] * (90 - df["지상역률_보정"]).clip(lower=0)
    df["주간_추가요율"] = df["주간_부족률"] * 0.01
    df["부하역률곱_강화"] = df["전력사용량(kWh)"] * df["역률부족폭_94"] * df["주간여부"] * 10
    df["역률부족_경부하"] = (df["부하구분"] == "경부하").astype(int) * df["역률부족폭_94"]
    df["역률부족_중간부하"] = (df["부하구분"] == "중간부하").astype(int) * df["역률부족폭_94"]
    df["역률부족_최대부하"] = (df["부하구분"] == "최대부하").astype(int) * df["역률부족폭_94"]
    
    return df

train = add_pf_features(train)
test = add_pf_features(test)

# -----------------------------
# 4) Lag/Rolling (전력 및 무효전력에 대한 시계열 특징 강화)
# -----------------------------

# 전력 사용량 (kWh) Lag/Rolling (기존)
train["kwh_lag1"] = train["전력사용량(kWh)"].shift(1)
train["kwh_lag24"] = train["전력사용량(kWh)"].shift(24)
train["kwh_lag96"] = train["전력사용량(kWh)"].shift(96)
train["kwh_lag672"] = train["전력사용량(kWh)"].shift(672)
train["kwh_lag168"] = train["전력사용량(kWh)"].shift(168)

train["kwh_roll24_mean"] = train["전력사용량(kWh)"].shift(1).rolling(24).mean()
train["kwh_roll24_std"] = train["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
train["kwh_roll96_mean"] = train["전력사용량(kWh)"].shift(1).rolling(96).mean()
train["kwh_roll96_std"] = train["전력사용량(kWh)"].shift(1).rolling(96).std().fillna(0)
train["kwh_roll24_cv"] = train["kwh_roll24_std"] / (train["kwh_roll24_mean"] + 1e-6)
train["kwh_roll12_mean"] = train["전력사용량(kWh)"].shift(1).rolling(12).mean()
train["kwh_roll12_std"] = train["전력사용량(kWh)"].shift(1).rolling(12).std().fillna(0)
train["kwh_roll12_cv"] = train["kwh_roll12_std"] / (train["kwh_roll12_mean"] + 1e-6)
train["kwh_roll48_mean"] = train["전력사용량(kWh)"].shift(1).rolling(48).mean()
train["kwh_roll48_std"] = train["전력사용량(kWh)"].shift(1).rolling(48).std().fillna(0)
train["kwh_roll48_cv"] = train["kwh_roll48_std"] / (train["kwh_roll48_mean"] + 1e-6)
train["kwh_roll24_range"] = (
    train["전력사용량(kWh)"].shift(1).rolling(24).max()
    - train["전력사용량(kWh)"].shift(1).rolling(24).min()
)
train["kwh_lag24_ratio"] = train["전력사용량(kWh)"] / (train["kwh_lag24"] + 1e-6)
train["kwh_roll24_ratio"] = train["전력사용량(kWh)"] / (train["kwh_roll24_mean"] + 1e-6)
train["kwh_lag168_ratio"] = train["전력사용량(kWh)"] / (train["kwh_lag168"] + 1e-6)
train["kwh_vs_어제"] = (train["전력사용량(kWh)"] - train["kwh_lag24"]) / (train["kwh_lag24"] + 1e-6)
train["전력급등"] = (train["kwh_vs_어제"] > 0.5).astype(int)
train["위험_변동성"] = train["실질위험"] * train["kwh_roll24_cv"]

# 지상 무효 전력량 (kVarh) Lag/Rolling (추가)
train["kvarh_lag1"] = train["지상무효전력량(kVarh)"].shift(1)
train["kvarh_roll24_mean"] = train["지상무효전력량(kVarh)"].shift(1).rolling(24).mean()
train["kvarh_roll96_mean"] = train["지상무효전력량(kVarh)"].shift(1).rolling(96).mean()

# Test 데이터에 대한 시계열 특징 생성 (전력 사용량)
hist_kwh = list(train["전력사용량(kWh)"].tail(672).values.astype(float))
hist_kvarh = list(train["지상무효전력량(kVarh)"].tail(672).values.astype(float))

kwh_lag1_list, kwh_lag24_list, kwh_lag96_list, kwh_lag672_list, kwh_lag168_list = [], [], [], [], []
kwh_roll24_mean_list, kwh_roll24_std_list = [], []
kwh_roll96_mean_list, kwh_roll96_std_list = [], []
kwh_roll24_cv_list = []
kwh_roll12_mean_list, kwh_roll12_std_list, kwh_roll12_cv_list = [], [], []
kwh_roll48_mean_list, kwh_roll48_std_list, kwh_roll48_cv_list = [], [], []
kwh_roll24_range_list = []
kwh_lag24_ratio_list, kwh_roll24_ratio_list, kwh_lag168_ratio_list = [], [], []
kwh_vs_yesterday_list = []
kvarh_lag1_list, kvarh_roll24_mean_list, kvarh_roll96_mean_list = [], [], []

for _, row in test.iterrows():
    kwh_lag1 = hist_kwh[-1] if len(hist_kwh) >= 1 else np.nan
    kwh_lag24 = hist_kwh[-24] if len(hist_kwh) >= 24 else np.nan
    kwh_lag96 = hist_kwh[-96] if len(hist_kwh) >= 96 else np.nan
    kwh_lag672 = hist_kwh[-672] if len(hist_kwh) >= 672 else np.nan
    kwh_lag168 = hist_kwh[-168] if len(hist_kwh) >= 168 else np.nan

    arr24 = np.array(hist_kwh[-24:])
    arr96 = np.array(hist_kwh[-96:])
    arr12 = np.array(hist_kwh[-12:])
    arr48 = np.array(hist_kwh[-48:])

    roll24_mean = arr24.mean() if arr24.size > 0 else np.nan
    roll24_std = arr24.std() if arr24.size > 1 else 0
    roll96_mean = arr96.mean() if arr96.size > 0 else np.nan
    roll96_std = arr96.std() if arr96.size > 1 else 0
    roll12_mean = arr12.mean() if arr12.size > 0 else np.nan
    roll12_std = arr12.std() if arr12.size > 1 else 0
    roll48_mean = arr48.mean() if arr48.size > 0 else np.nan
    roll48_std = arr48.std() if arr48.size > 1 else 0
    roll24_range = (arr24.max() - arr24.min()) if arr24.size > 0 else np.nan

    kwh_lag1_list.append(kwh_lag1)
    kwh_lag24_list.append(kwh_lag24)
    kwh_lag96_list.append(kwh_lag96)
    kwh_lag672_list.append(kwh_lag672)
    kwh_lag168_list.append(kwh_lag168)
    kwh_roll24_mean_list.append(roll24_mean)
    kwh_roll24_std_list.append(roll24_std)
    kwh_roll96_mean_list.append(roll96_mean)
    kwh_roll96_std_list.append(roll96_std)
    kwh_roll12_mean_list.append(roll12_mean)
    kwh_roll12_std_list.append(roll12_std)
    kwh_roll48_mean_list.append(roll48_mean)
    kwh_roll48_std_list.append(roll48_std)
    kwh_roll24_range_list.append(roll24_range)
    kwh_roll24_cv_list.append(roll24_std / (roll24_mean + 1e-6) if not np.isnan(roll24_mean) else np.nan)
    kwh_roll12_cv_list.append(roll12_std / (roll12_mean + 1e-6) if not np.isnan(roll12_mean) else np.nan)
    kwh_roll48_cv_list.append(roll48_std / (roll48_mean + 1e-6) if not np.isnan(roll48_mean) else np.nan)

    kwh_lag24_ratio_list.append(row["전력사용량(kWh)"] / (kwh_lag24 + 1e-6) if not np.isnan(kwh_lag24) else np.nan)
    kwh_roll24_ratio_list.append(row["전력사용량(kWh)"] / (roll24_mean + 1e-6) if not np.isnan(roll24_mean) else np.nan)
    kwh_lag168_ratio_list.append(row["전력사용량(kWh)"] / (kwh_lag168 + 1e-6) if not np.isnan(kwh_lag168) else np.nan)
    kwh_vs_yesterday_list.append((row["전력사용량(kWh)"] - kwh_lag24) / (kwh_lag24 + 1e-6) if not np.isnan(kwh_lag24) else np.nan)

    # kVarh
    kvarh_lag1 = hist_kvarh[-1] if len(hist_kvarh) >= 1 else np.nan
    arr24_kvarh = np.array(hist_kvarh[-24:])
    arr96_kvarh = np.array(hist_kvarh[-96:])
    kvarh_roll24_mean = arr24_kvarh.mean() if arr24_kvarh.size > 0 else np.nan
    kvarh_roll96_mean = arr96_kvarh.mean() if arr96_kvarh.size > 0 else np.nan

    kvarh_lag1_list.append(kvarh_lag1)
    kvarh_roll24_mean_list.append(kvarh_roll24_mean)
    kvarh_roll96_mean_list.append(kvarh_roll96_mean)

    hist_kwh.append(row["전력사용량(kWh)"])
    hist_kvarh.append(row["지상무효전력량(kVarh)"])

test["kwh_lag1"] = kwh_lag1_list
test["kwh_lag24"] = kwh_lag24_list
test["kwh_lag96"] = kwh_lag96_list
test["kwh_lag672"] = kwh_lag672_list
test["kwh_lag168"] = kwh_lag168_list
test["kwh_roll24_mean"] = kwh_roll24_mean_list
test["kwh_roll24_std"] = kwh_roll24_std_list
test["kwh_roll96_mean"] = kwh_roll96_mean_list
test["kwh_roll96_std"] = kwh_roll96_std_list
test["kwh_roll24_cv"] = kwh_roll24_cv_list
test["kwh_roll12_mean"] = kwh_roll12_mean_list
test["kwh_roll12_std"] = kwh_roll12_std_list
test["kwh_roll12_cv"] = kwh_roll12_cv_list
test["kwh_roll48_mean"] = kwh_roll48_mean_list
test["kwh_roll48_std"] = kwh_roll48_std_list
test["kwh_roll48_cv"] = kwh_roll48_cv_list
test["kwh_roll24_range"] = kwh_roll24_range_list
test["kwh_lag24_ratio"] = kwh_lag24_ratio_list
test["kwh_roll24_ratio"] = kwh_roll24_ratio_list
test["kwh_lag168_ratio"] = kwh_lag168_ratio_list
test["kwh_vs_어제"] = kwh_vs_yesterday_list
test["전력급등"] = (np.array(kwh_vs_yesterday_list) > 0.5).astype(int)
test["kvarh_lag1"] = kvarh_lag1_list
test["kvarh_roll24_mean"] = kvarh_roll24_mean_list
test["kvarh_roll96_mean"] = kvarh_roll96_mean_list
test["위험_변동성"] = test["실질위험"] * test["kwh_roll24_cv"]


# -----------------------------
# 5) 고급 피처 (시계열 및 그룹화 피처 강화)
# -----------------------------
# 요일-시간대별 평균 전력 사용량 (Train)
kwh_mean_day_hour = train.groupby(["요일", "시간"])["전력사용량(kWh)"].mean().reset_index()
kwh_mean_day_hour.rename(columns={"전력사용량(kWh)": "kwh_요일_시간_평균"}, inplace=True)
train = pd.merge(train, kwh_mean_day_hour, on=["요일", "시간"], how="left")
test = pd.merge(test, kwh_mean_day_hour, on=["요일", "시간"], how="left")

def add_advanced_features(df, is_train=True):
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_보정"] + 1e-6)
    
    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률_보정"] >= 94)).astype(int)
    
    if is_train:
        df["역률_월평균"] = df.groupby("월")["지상역률_보정"].transform("mean")
        df["전력사용_시간평균"] = df.groupby("시간")["전력사용량(kWh)"].transform("mean")
    else: # Test 데이터는 Train 데이터의 그룹 평균을 사용해야 합니다.
        df["역률_월평균"] = df["월"].map(train.groupby("월")["지상역률_보정"].mean())
        df["전력사용_시간평균"] = df["시간"].map(train.groupby("시간")["전력사용량(kWh)"].mean())
        # 만약 test 월이 train에 없다면 (여기선 12월), 가장 유사한 월의 평균을 사용하거나 전체 평균을 사용합니다.
        # 여기서는 간단히 NaN이 되지 않도록 처리만 합니다.
        df["역률_월평균"].fillna(df["역률_월평균"].mean(), inplace=True)
        df["전력사용_시간평균"].fillna(df["전력사용_시간평균"].mean(), inplace=True)
    
    df["역률_월평균차이"] = df["지상역률_보정"] - df["역률_월평균"]
    df["전력사용_시간대비"] = df["전력사용량(kWh)"] / (df["전력사용_시간평균"] + 1e-6)

    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    df["kwh_변화율_24h"] = (df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6)
    df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    
    # 추가 고급 피처
    df["kwh_roll96_cv"] = df["kwh_roll96_std"] / (df["kwh_roll96_mean"] + 1e-6)
    df["총무효전력"] = df["지상무효전력량(kVarh)"] + df["진상무효전력량(kVarh)"]
    df["무효전력비중"] = df["총무효전력"] / (df["전력사용량(kWh)"] + df["총무효전력"] + 1e-6)
    
    # 🆕 요일-시간대별 평균 전력 사용량 대비 피처
    df["kwh_시간대비_요일"] = df["전력사용량(kWh)"] / (df["kwh_요일_시간_평균"] + 1e-6)
    df.drop("kwh_요일_시간_평균", axis=1, inplace=True)
    
    return df

train = add_advanced_features(train, is_train=True)
test = add_advanced_features(test, is_train=False)

# -----------------------------
# 6) Stage2: 요금 예측 (TimeSeriesSplit 기반 Stacking 앙상블)
# -----------------------------
feat_s2 = [
    "월", "일", "요일", "시간", "주말여부", "겨울여부", "period_flag",
    "sin_time", "cos_time", "sin_day", "cos_day", "sin_month", "cos_month",
    "작업유형_encoded", "부하구분_encoded", "시간_작업유형_encoded",
    # Stage 1 예측 값
    "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
    "지상역률(%)", "진상역률(%)", 
    # Lag/Rolling Features (강화)
    "kwh_lag1", "kwh_lag24", "kwh_lag96", "kwh_lag672", "kwh_lag168",
    "kwh_roll24_mean", "kwh_roll24_std", "kwh_roll96_mean", "kwh_roll96_std",
    "kvarh_lag1", "kvarh_roll24_mean", "kvarh_roll96_mean", # kVarh Lag/Rolling
    # Advanced Features
    "유효역률(%)", "역률_조정요율", "지상역률_보정", "주간여부", "법적페널티", "실질위험", "극저역률",
    "역률부족폭_94", "역률부족폭_90", "역률우수",
    "역률_60_85", "역률_85_90", "역률_90_94", "역률_94_이상",
    "주간_부족률", "주간_추가요율", "부하역률곱_강화",
    "역률부족_경부하", "역률부족_중간부하", "역률부족_최대부하",
    "무효유효비율", "부하역률곱", "역률당전력",
    "가을위험", "동절기안정", "역률_월평균", "역률_월평균차이",
    "kwh_roll24_cv", "위험_변동성", "kwh_변화율_24h", "전력급등",
    "kwh_roll12_mean", "kwh_roll12_std", "kwh_roll12_cv",
    "kwh_roll48_mean", "kwh_roll48_std", "kwh_roll48_cv",
    "kwh_roll24_range", "kwh_lag24_ratio", "kwh_roll24_ratio",
    "kwh_lag168_ratio", "kwh_vs_어제",
    "kwh_roll96_cv", "전력사용_시간평균", "전력사용_시간대비",
    "총무효전력", "무효전력비중", "kwh_시간대비_요일", # 강화된 피처
]

X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)
sample_weights_all = np.ones(len(y_all), dtype=float)
sample_weights_all[y_all > 3000] = 2.0
sample_weights_all[y_all > 5000] = 3.0
sample_weights_all[y_all > 10000] = 5.0
sample_weights_all[X_all["실질위험"] == 1] *= 2.0

# Target Encoded Features는 여기서 제외했습니다. Stage 2 모델은 LGBM, XGB, CatBoost가 잘 처리합니다.

# 🆕 Stage2 모델 (하이퍼파라미터 조정)
LGB_PARAMS = dict(
    n_estimators=3500, learning_rate=0.012, num_leaves=128, subsample=0.85, 
    colsample_bytree=0.85, reg_alpha=3, reg_lambda=4, min_child_samples=18, 
    random_state=42, n_jobs=-1
)
XGB_PARAMS = dict(
    n_estimators=3500, learning_rate=0.012, max_depth=9, subsample=0.8, 
    colsample_bytree=0.8, reg_lambda=4, reg_alpha=2, min_child_weight=3, 
    random_state=42, n_jobs=-1
)
CAT_PARAMS = dict(
    iterations=3000, learning_rate=0.015, depth=8, l2_leaf_reg=4, 
    random_seed=42, verbose=0, thread_count=-1
)

# -----------------------------
# Stacking (OOF 기반)
# -----------------------------
tscv_s2 = TimeSeriesSplit(n_splits=5)
# OOF 예측 결과를 저장할 데이터프레임
oof_preds_s2 = pd.DataFrame(index=X_all.index, columns=["lgb", "xgb", "cat"])

base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS)
}

# OOF 예측 생성
for fold, (tr_idx, va_idx) in enumerate(tscv_s2.split(X_all), start=1):
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr_log, y_va_log = y_all_log.iloc[tr_idx], y_all_log.iloc[va_idx]
    w_tr = sample_weights_all[tr_idx]
    
    for name, model in base_models.items():
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_tr, y_tr_log, sample_weight=w_tr)
        oof_preds_s2.loc[va_idx, name] = fold_model.predict(X_va)

# Meta-Learner 학습 (Log Scale)
meta_learner = Ridge(alpha=100) # MAE에 강건한 Ridge 사용
# OOF 예측이 없는 부분(초기 폴드)은 제외하고 학습합니다.
oof_valid_idx = oof_preds_s2.dropna().index
meta_learner.fit(oof_preds_s2.loc[oof_valid_idx], y_all_log.loc[oof_valid_idx])

# 11월 검증 (단일 분할)
idx_va = train["월"] == 11
X_va = X_all[idx_va]
y_va = y_all[idx_va]

# 각 모델의 11월 예측 (전체 데이터 학습 + 가중치 적용)
lgb_full = LGBMRegressor(**LGB_PARAMS)
lgb_full.fit(X_all, y_all_log, sample_weight=sample_weights_all)
pred_lgb_va = np.expm1(lgb_full.predict(X_va))

xgb_full = XGBRegressor(**XGB_PARAMS)
xgb_full.fit(X_all, y_all_log, sample_weight=sample_weights_all)
pred_xgb_va = np.expm1(xgb_full.predict(X_va))

cat_full = CatBoostRegressor(**CAT_PARAMS)
cat_full.fit(X_all, y_all_log, sample_weight=sample_weights_all)
pred_cat_va = np.expm1(cat_full.predict(X_va))

# Meta-Learner 예측 (OOF 예측이 아니므로, 전체 데이터로 학습한 Base Model의 예측을 Meta-Learner에 입력)
X_meta_va = pd.DataFrame({
    "lgb": np.log1p(pred_lgb_va),
    "xgb": np.log1p(pred_xgb_va),
    "cat": np.log1p(pred_cat_va),
}, index=X_va.index)
pred_va = np.expm1(meta_learner.predict(X_meta_va))

mae = mean_absolute_error(y_va, pred_va)
r2 = r2_score(y_va, pred_va)
print(f"\n📊 11월 검증 (Stacking): MAE={mae:.2f} | R²={r2:.4f}")

# -----------------------------
# 7) Test(12월) 예측
# -----------------------------
X_te = test[feat_s2].copy()

# Base Model Test 예측 (Log Scale)
pred_lgb_te = lgb_full.predict(X_te)
pred_xgb_te = xgb_full.predict(X_te)
pred_cat_te = cat_full.predict(X_te)

# Meta-Learner 입력 데이터
X_meta_te = pd.DataFrame({
    "lgb": pred_lgb_te,
    "xgb": pred_xgb_te,
    "cat": pred_cat_te
}, index=X_te.index)

# 최종 Test 예측 (Meta-Learner 적용 후 expm1 변환)
pred_te = np.expm1(meta_learner.predict(X_meta_te))

# 더 보수적인 클리핑 (상위/하위 0.1% 제거)
low, high = np.percentile(pred_te, [0.1, 99.9])
pred_te = np.clip(pred_te, low, high)

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_stacking_mae600.csv", index=False)
print("\n💾 submission_stacking_mae600.csv 저장 완료!")
print(f"예측 범위: {pred_te.min():.2f} ~ {pred_te.max():.2f}")
print(f"예측 평균: {pred_te.mean():.2f}")

# Feature Importance (LGBM 기준)
feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(20).to_string(index=False))
