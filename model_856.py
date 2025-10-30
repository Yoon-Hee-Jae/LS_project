#!/usr/bin/env python
# ============================================================
# EDA 기반 역률 피처 강화판 + 고급 피처 (1,2,3)
#  - 전력×역률 교호작용
#  - 계절×역률 교호작용
#  - 부하 변동성
# ============================================================
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

warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
train = pd.read_csv("./data/train_.csv")
test = pd.read_csv("./data/test_.csv")

# -----------------------------
# 1) 시간 파생 (베이스와 동일)
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")


def adjust_hour(dt):
    if pd.isna(dt):
        return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour


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
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["부하구분"] = df["시간"].apply(band_of_hour)
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

train["시간_작업유형"] = (
    train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
)
test["시간_작업유형"] = (
    test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
test["시간_작업유형_encoded"] = le_tj.transform(test["시간_작업유형"])

# -----------------------------
# 2) Stage1: 전력특성 예측
# -----------------------------
targets_s1 = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]
feat_s1 = [
    "월",
    "일",
    "요일",
    "시간",
    "주말여부",
    "겨울여부",
    "period_flag",
    "sin_time",
    "cos_time",
    "작업유형_encoded",
    "부하구분_encoded",
    "시간_작업유형_encoded",
]

stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(
        n_estimators=2500, learning_rate=0.012, num_leaves=128, random_state=42
    ),
    "지상무효전력량(kVarh)": CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42
    ),
    "진상무효전력량(kVarh)": CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42
    ),
    "지상역률(%)": LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42
    ),
    "진상역률(%)": LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42
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

    # TimeSeriesSplit keeps early segment (first fold) as validation; ensure no NaNs
    missing = np.isnan(oof_pred)
    if missing.any():
        full_model = stage1_models[tgt].__class__(**stage1_models[tgt].get_params())
        full_model.fit(train[feat_s1], train[tgt])
        oof_pred[missing] = full_model.predict(train.loc[missing, feat_s1])

    stage1_oof[tgt] = oof_pred

    # 최종 12월 예측용 모델 (1~11월 전체로 학습)
    final_model = stage1_models[tgt].__class__(**stage1_models[tgt].get_params())
    final_model.fit(train[feat_s1], train_targets_true[tgt])
    stage1_test_pred[tgt] = final_model.predict(test[feat_s1])

# Stage1 예측치로 train/test 갱신
for tgt in targets_s1:
    train[f"{tgt}_true"] = train_targets_true[tgt]
    train[tgt] = stage1_oof[tgt]
    test[tgt] = stage1_test_pred[tgt]

# Stage1 예측치 분포 확인
print("\n📊 Stage1 지상역률(%) 예측 분포:")
print(train["지상역률(%)"].describe())
print(f"95% 초과 건수: {(train['지상역률(%)'] > 95).sum()}건")
print(f"94% 미만 건수: {(train['지상역률(%)'] < 94).sum()}건")

# -----------------------------
# 3) EDA 기반 역률 피처 생성
# -----------------------------
def add_pf_features(df: pd.DataFrame) -> pd.DataFrame:
    """EDA 인사이트 기반 역률 피처 생성"""
    # 기본 역률 피처
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0) * 0.01
    df["역률_보상율"] = (df["유효역률(%)"] - 90).clip(lower=0) * 0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    
    # EDA 기반 새 피처
    df["지상역률_보정"] = df["지상역률(%)"].clip(lower=60)  # upper 제거 (95% 이상 정보 보존)
    df["주간여부"] = df["부하구분"].isin(["중간부하", "최대부하"]).astype(int)
    
    # 법적 페널티 (90% 미만)
    df["법적페널티"] = ((df["지상역률_보정"] < 90) & (df["주간여부"] == 1)).astype(int)
    
    # 실질 위험 (94% 미만) ← EDA에서 발견한 핵심 임계점!
    df["실질위험"] = ((df["지상역률_보정"] < 94) & (df["주간여부"] == 1)).astype(int)
    
    # 극저 역률 (85% 미만) ← 법적페널티와 차별화
    df["극저역률"] = ((df["지상역률_보정"] < 85) & (df["주간여부"] == 1)).astype(int)
    
    # 역률 부족폭 (94% 기준, 야간 노이즈 제거)
    df["역률부족폭_94"] = (94 - df["지상역률_보정"]).clip(lower=0) * df["주간여부"]
    
    return df


train = add_pf_features(train)
test = add_pf_features(test)

# -----------------------------
# 4) Lag/Rolling 생성 (전력사용량)
# -----------------------------
# lag / rolling for train (shifted to avoid leakage)
train["kwh_lag1"] = train["전력사용량(kWh)"].shift(1)
train["kwh_lag24"] = train["전력사용량(kWh)"].shift(24)
train["kwh_lag96"] = train["전력사용량(kWh)"].shift(96)
train["kwh_lag672"] = train["전력사용량(kWh)"].shift(672)

train["kwh_roll24_mean"] = train["전력사용량(kWh)"].shift(1).rolling(24).mean()
train["kwh_roll24_std"] = (
    train["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
)
train["kwh_roll96_mean"] = train["전력사용량(kWh)"].shift(1).rolling(96).mean()
train["kwh_roll96_std"] = (
    train["전력사용량(kWh)"].shift(1).rolling(96).std().fillna(0)
)
train["kwh_roll672_mean"] = train["전력사용량(kWh)"].shift(1).rolling(672).mean()
train["kwh_roll672_std"] = (
    train["전력사용량(kWh)"].shift(1).rolling(672).std().fillna(0)
)

# lag/rolling for test using recursive approach
hist = list(train["전력사용량(kWh)"].tail(672).values.astype(float))
lag1_list, lag24_list, lag96_list, lag672_list = [], [], [], []
r24m, r24s, r96m, r96s, r672m, r672s = [], [], [], [], [], []

for y in test["전력사용량(kWh)"].values.astype(float):
    lag1_list.append(hist[-1] if len(hist) >= 1 else np.nan)
    lag24_list.append(hist[-24] if len(hist) >= 24 else np.nan)
    lag96_list.append(hist[-96] if len(hist) >= 96 else np.nan)
    lag672_list.append(hist[-672] if len(hist) >= 672 else np.nan)

    arr24 = np.array(hist[-24:]) if len(hist) >= 24 else np.array(hist)
    arr96 = np.array(hist[-96:]) if len(hist) >= 96 else np.array(hist)
    arr672 = np.array(hist[-672:]) if len(hist) >= 672 else np.array(hist)

    r24m.append(arr24.mean() if arr24.size > 0 else np.nan)
    r24s.append(arr24.std() if arr24.size > 1 else 0)
    r96m.append(arr96.mean() if arr96.size > 0 else np.nan)
    r96s.append(arr96.std() if arr96.size > 1 else 0)
    r672m.append(arr672.mean() if arr672.size > 0 else np.nan)
    r672s.append(arr672.std() if arr672.size > 1 else 0)

    hist.append(y)

test["kwh_lag1"] = lag1_list
test["kwh_lag24"] = lag24_list
test["kwh_lag96"] = lag96_list
test["kwh_lag672"] = lag672_list
test["kwh_roll24_mean"] = r24m
test["kwh_roll24_std"] = r24s
test["kwh_roll96_mean"] = r96m
test["kwh_roll96_std"] = r96s
test["kwh_roll672_mean"] = r672m
test["kwh_roll672_std"] = r672s

# -----------------------------
# 5) 고급 피처 추가 (1,2,3)
# -----------------------------
def add_advanced_features(df, is_train=True):
    """
    고급 피처 추가
    1. 전력×역률 교호작용
    2. 계절×역률 교호작용
    3. 부하 변동성
    """
    # === 1. 전력×역률 교호작용 ===
    # 무효전력 / 유효전력 비율 (역률이 나쁠수록 커짐)
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    
    # 전력사용량 × 역률부족폭 (큰 부하 + 나쁜 역률 = 큰 패널티)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    
    # 역률 대비 전력사용량
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_보정"] + 1e-6)
    
    # === 2. 계절×역률 교호작용 ===
    # 9-10월 위험구간 (EDA에서 발견한 최악의 조합)
    df["가을위험"] = (
        (df["월"].isin([9, 10])) & 
        (df["실질위험"] == 1)
    ).astype(int)
    
    # 1-2월 고부하 + 역률양호 (난방 시즌)
    df["동절기안정"] = (
        (df["겨울여부"] == 1) & 
        (df["지상역률_보정"] >= 94)
    ).astype(int)
    
    # 월별 평균 역률 대비 편차
    df["역률_월평균"] = df.groupby("월")["지상역률_보정"].transform("mean")
    df["역률_월평균차이"] = df["지상역률_보정"] - df["역률_월평균"]
    
    # === 3. 부하 변동성 ===
    # 최근 24시간 변동계수 (CV = std/mean)
    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    
    if is_train:
        # lag 대비 변화율
        df["kwh_변화율_24h"] = (
            (df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6)
        )
        
        # 급등 플래그 (전날 대비 50% 이상 증가)
        df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    else:
        # test는 lag24가 있으므로 동일하게 계산 가능
        df["kwh_변화율_24h"] = (
            (df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6)
        )
        df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    
    return df


train = add_advanced_features(train, is_train=True)
test = add_advanced_features(test, is_train=False)

# 새 피처 분포 확인
print("\n📊 새 역률 피처 분포 (train):")
print(f"법적페널티 발생: {train['법적페널티'].sum()}건 ({train['법적페널티'].mean()*100:.1f}%)")
print(f"실질위험 발생: {train['실질위험'].sum()}건 ({train['실질위험'].mean()*100:.1f}%)")
print(f"극저역률 발생: {train['극저역률'].sum()}건 ({train['극저역률'].mean()*100:.1f}%)")
print(f"가을위험 발생: {train['가을위험'].sum()}건 ({train['가을위험'].mean()*100:.1f}%)")
print(f"역률부족폭_94 평균: {train['역률부족폭_94'].mean():.3f}")
print(f"부하역률곱 평균: {train['부하역률곱'].mean():.3f}")

# -----------------------------
# 6) Stage2: 요금 예측 (EDA + 고급 피처)
# -----------------------------
feat_s2 = [
    "월",
    "일",
    "요일",
    "시간",
    "주말여부",
    "겨울여부",
    "period_flag",
    "sin_time",
    "cos_time",
    "작업유형_encoded",
    "부하구분_encoded",
    "시간_작업유형_encoded",
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
    "유효역률(%)",
    "역률_조정요율",
    # EDA 기반 역률 피처
    "지상역률_보정",
    "주간여부",
    "법적페널티",
    "실질위험",
    "극저역률",
    "역률부족폭_94",
    # 고급 피처 (1,2,3)
    "무효유효비율",
    "부하역률곱",
    "역률당전력",
    "가을위험",
    "동절기안정",
    "역률_월평균",
    "역률_월평균차이",
    "kwh_roll24_cv",
    "kwh_변화율_24h",
    "전력급등",
    # lag/rolling
    "kwh_lag1",
    "kwh_lag24",
    "kwh_lag96",
    "kwh_lag672",
    "kwh_roll24_mean",
    "kwh_roll24_std",
    "kwh_roll96_mean",
    "kwh_roll96_std",
    "kwh_roll672_mean",
    "kwh_roll672_std",
]

X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()

idx_tr = train["월"] < 11
idx_va = train["월"] == 11

X_tr = X_all[idx_tr]
X_va = X_all[idx_va]
y_tr = y_all[idx_tr]
y_va = y_all[idx_va]

y_tr_log = np.log1p(y_tr)
y_all_log = np.log1p(y_all)

LGB_PARAMS = dict(
    n_estimators=2300,
    learning_rate=0.02,
    num_leaves=96,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=3,
    reg_lambda=4,
    random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=2300,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=4,
    reg_alpha=1,
    random_state=42,
)
CAT_PARAMS = dict(
    iterations=2000,
    learning_rate=0.02,
    depth=7,
    l2_leaf_reg=4,
    random_seed=42,
    verbose=0,
)

lgb = LGBMRegressor(**LGB_PARAMS)
xgb = XGBRegressor(**XGB_PARAMS)
cat = CatBoostRegressor(**CAT_PARAMS)

lgb.fit(X_tr, y_tr_log)
xgb.fit(X_tr, y_tr_log)
cat.fit(X_tr, y_tr_log)

pred_va = (
    0.5 * np.expm1(lgb.predict(X_va))
    + 0.3 * np.expm1(xgb.predict(X_va))
    + 0.2 * np.expm1(cat.predict(X_va))
)
mae = mean_absolute_error(y_va, pred_va)
r2 = r2_score(y_va, pred_va)
print(f"\n📊 11월 검증: MAE={mae:.2f} | R²={r2:.4f}")

# Feature Importance 확인 (LightGBM 기준)
feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(20).to_string(index=False))

plt.figure(figsize=(8, 4.8))
plt.hist(y_va, bins=60, alpha=0.5, density=True, label="Actual (11월)", color="#6BA3D6")
plt.hist(pred_va, bins=60, alpha=0.5, density=True, label="Pred (11월)", color="#F3C969")
plt.title("📈 11월 전기요금 분포 (Actual vs Pred)")
plt.xlabel("전기요금(원)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Test(12월) 예측
# -----------------------------
lgb_full = LGBMRegressor(**LGB_PARAMS)
xgb_full = XGBRegressor(**XGB_PARAMS)
cat_full = CatBoostRegressor(**CAT_PARAMS)

lgb_full.fit(X_all, y_all_log)
xgb_full.fit(X_all, y_all_log)
cat_full.fit(X_all, y_all_log)

X_te = test[feat_s2].copy()
pred_te = (
    0.5 * np.expm1(lgb_full.predict(X_te))
    + 0.3 * np.expm1(xgb_full.predict(X_te))
    + 0.2 * np.expm1(cat_full.predict(X_te))
)

low, high = np.percentile(pred_te, [0.2, 99.8])
pred_te = np.clip(pred_te, low, high)

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_advanced_v1.csv", index=False)
print("\n💾 submission_advanced_v1.csv 저장 완료!")

# Test 세트 역률 위험 분포 확인
print("\n📊 Test(12월) 역률 위험 분포:")
print(f"법적페널티 예상: {test['법적페널티'].sum()}건 ({test['법적페널티'].mean()*100:.1f}%)")
print(f"실질위험 예상: {test['실질위험'].sum()}건 ({test['실질위험'].mean()*100:.1f}%)")
print(f"극저역률 예상: {test['극저역률'].sum()}건 ({test['극저역률'].mean()*100:.1f}%)")
print(f"가을위험 예상: {test['가을위험'].sum()}건 ({test['가을위험'].mean()*100:.1f}%)")
print(f"전력급등 예상: {test['전력급등'].sum()}건 ({test['전력급등'].mean()*100:.1f}%)")



#####################################
# -----------------------------
# 8) 상세 분석: Feature Importance & 11월 성능
# -----------------------------

# === Feature Importance 상세 분석 ===
print("\n" + "="*70)
print("🔍 FEATURE IMPORTANCE 분석 (LightGBM 기준)")
print("="*70)

feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(20).to_string(index=False))

# 카테고리별 중요도 합계
print("\n📊 카테고리별 중요도:")

# EDA 기반 역률 피처
eda_features = ['지상역률_보정', '주간여부', '법적페널티', '실질위험', '극저역률', '역률부족폭_94']
eda_imp = feat_imp[feat_imp['feature'].isin(eda_features)]['importance'].sum()
print(f"EDA 역률 피처: {eda_imp:.1f}")

# 고급 피처 (1,2,3)
advanced_features = [
    '무효유효비율', '부하역률곱', '역률당전력',  # 1. 전력×역률
    '가을위험', '동절기안정', '역률_월평균', '역률_월평균차이',  # 2. 계절×역률
    'kwh_roll24_cv', 'kwh_변화율_24h', '전력급등'  # 3. 부하 변동성
]
advanced_imp = feat_imp[feat_imp['feature'].isin(advanced_features)]['importance'].sum()
print(f"고급 피처 (1,2,3): {advanced_imp:.1f}")

# Lag/Rolling
lag_features = [f for f in feat_s2 if 'lag' in f or 'roll' in f]
lag_imp = feat_imp[feat_imp['feature'].isin(lag_features)]['importance'].sum()
print(f"Lag/Rolling 피처: {lag_imp:.1f}")

# 기본 시간 피처
time_features = ['월', '일', '요일', '시간', '주말여부', '겨울여부', 'sin_time', 'cos_time']
time_imp = feat_imp[feat_imp['feature'].isin(time_features)]['importance'].sum()
print(f"시간 피처: {time_imp:.1f}")

# Stage1 예측치
stage1_features = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)', 
                   '지상역률(%)', '진상역률(%)', '유효역률(%)']
stage1_imp = feat_imp[feat_imp['feature'].isin(stage1_features)]['importance'].sum()
print(f"Stage1 예측치: {stage1_imp:.1f}")

# === 고급 피처별 상세 분석 ===
print("\n" + "="*70)
print("🎯 고급 피처 (1,2,3) 상세 순위")
print("="*70)

advanced_imp_detail = feat_imp[feat_imp['feature'].isin(advanced_features)].copy()
advanced_imp_detail['category'] = advanced_imp_detail['feature'].apply(
    lambda x: '1.전력×역률' if x in ['무효유효비율', '부하역률곱', '역률당전력']
    else '2.계절×역률' if x in ['가을위험', '동절기안정', '역률_월평균', '역률_월평균차이']
    else '3.부하변동성'
)
print(advanced_imp_detail.to_string(index=False))

# === 11월 성능 상세 분석 ===
print("\n" + "="*70)
print("📊 11월 검증 성능 상세 분석")
print("="*70)

mae = mean_absolute_error(y_va, pred_va)
r2 = r2_score(y_va, pred_va)
mape = np.mean(np.abs((y_va - pred_va) / y_va)) * 100
rmse = np.sqrt(np.mean((y_va - pred_va) ** 2))

print(f"\n전체 성능:")
print(f"  MAE  : {mae:.2f}원")
print(f"  RMSE : {rmse:.2f}원")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f}%")

# 요금 구간별 성능
print(f"\n요금 구간별 MAE:")
y_va_series = pd.Series(y_va.values, index=y_va.index)
pred_va_series = pd.Series(pred_va, index=y_va.index)

bins = [0, 1000, 3000, 5000, 10000, np.inf]
labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k+']
y_va_binned = pd.cut(y_va_series, bins=bins, labels=labels)

for label in labels:
    mask = (y_va_binned == label)
    if mask.sum() > 0:
        mae_bin = mean_absolute_error(y_va_series[mask], pred_va_series[mask])
        count = mask.sum()
        print(f"  {label:8s}: MAE={mae_bin:7.2f}원 (n={count:4d})")

# 역률 위험 구간별 성능 (EDA 인사이트 검증!)
print(f"\n역률 위험 구간별 MAE:")
va_data = train[train["월"] == 11].copy()
va_data["pred"] = pred_va_series.values

# 실질위험 여부
for risk_val in [0, 1]:
    mask = (va_data["실질위험"] == risk_val)
    if mask.sum() > 0:
        mae_risk = mean_absolute_error(
            va_data.loc[mask, "전기요금(원)"], 
            va_data.loc[mask, "pred"]
        )
        count = mask.sum()
        risk_label = "94% 미만" if risk_val == 1 else "94% 이상"
        print(f"  {risk_label:10s}: MAE={mae_risk:7.2f}원 (n={count:4d})")

# 주간/야간별 성능
print(f"\n주간/야간별 MAE:")
for period_val in [0, 1]:
    mask = (va_data["주간여부"] == period_val)
    if mask.sum() > 0:
        mae_period = mean_absolute_error(
            va_data.loc[mask, "전기요금(원)"], 
            va_data.loc[mask, "pred"]
        )
        count = mask.sum()
        period_label = "주간" if period_val == 1 else "야간"
        print(f"  {period_label:6s}: MAE={mae_period:7.2f}원 (n={count:4d})")

# 잔차 분석
residuals = y_va - pred_va
print(f"\n잔차 분석:")
print(f"  평균 잔차    : {residuals.mean():7.2f}원")
print(f"  잔차 표준편차: {residuals.std():7.2f}원")
print(f"  최대 과대예측: {residuals.min():7.2f}원")
print(f"  최대 과소예측: {residuals.max():7.2f}원")

# 과대/과소 예측 비율
over_predict = (residuals < 0).sum()
under_predict = (residuals > 0).sum()
print(f"  과대예측 비율: {over_predict/len(residuals)*100:.1f}% ({over_predict}건)")
print(f"  과소예측 비율: {under_predict/len(residuals)*100:.1f}% ({under_predict}건)")

# === 시각화: 잔차 분포 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 실제 vs 예측 산점도
axes[0].scatter(y_va, pred_va, alpha=0.3, s=10)
axes[0].plot([y_va.min(), y_va.max()], [y_va.min(), y_va.max()], 'r--', lw=2)
axes[0].set_xlabel('실제 요금 (원)')
axes[0].set_ylabel('예측 요금 (원)')
axes[0].set_title(f'실제 vs 예측\n(R²={r2:.4f})')
axes[0].grid(alpha=0.3)

# 2. 잔차 분포
axes[1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('잔차 (실제 - 예측)')
axes[1].set_ylabel('빈도')
axes[1].set_title(f'잔차 분포\n(평균={residuals.mean():.1f}원)')
axes[1].grid(alpha=0.3)

# 3. 잔차 vs 실제 요금
axes[2].scatter(y_va, residuals, alpha=0.3, s=10, color='coral')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('실제 요금 (원)')
axes[2].set_ylabel('잔차 (원)')
axes[2].set_title('잔차 패턴 분석')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('validation_analysis.png', dpi=150, bbox_inches='tight')
print("\n💾 validation_analysis.png 저장 완료!")
plt.show()

# === Feature Importance 시각화 ===
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 25
feat_imp_top = feat_imp.head(top_n)

colors = []
for feat in feat_imp_top['feature']:
    if feat in advanced_features:
        if feat in ['무효유효비율', '부하역률곱', '역률당전력']:
            colors.append('#FF6B6B')  # 빨강: 전력×역률
        elif feat in ['가을위험', '동절기안정', '역률_월평균', '역률_월평균차이']:
            colors.append('#4ECDC4')  # 청록: 계절×역률
        else:
            colors.append('#FFE66D')  # 노랑: 부하변동성
    elif feat in eda_features:
        colors.append('#95E1D3')  # 연두: EDA 역률
    else:
        colors.append('#C7CEEA')  # 회색: 기타

ax.barh(range(top_n), feat_imp_top['importance'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(feat_imp_top['feature'], fontsize=9)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title(f'Top {top_n} Feature Importance (LightGBM)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='1.전력×역률'),
    Patch(facecolor='#4ECDC4', label='2.계절×역률'),
    Patch(facecolor='#FFE66D', label='3.부하변동성'),
    Patch(facecolor='#95E1D3', label='EDA 역률'),
    Patch(facecolor='#C7CEEA', label='기타')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=150, bbox_inches='tight')
print("💾 feature_importance_analysis.png 저장 완료!")
plt.show()

print("\n" + "="*70)
print("✅ 분석 완료!")
print("="*70)