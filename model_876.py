# ============================================================
# 안정 복구판: 베이스모델 + 재귀 lag 생성 (요금표/보정 제거)
#  - 목표: 캐글 점수 900 전후 회복, 이어서 700대 재도전
# ============================================================
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings; warnings.filterwarnings("ignore")

# -----------------------------
# 0) Load
# -----------------------------
train = pd.read_csv("./data/train_.csv")
test  = pd.read_csv("./data/test_.csv")

# -----------------------------
# 1) 시간 파생 (베이스와 동일)
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")
def adjust_hour(dt):
    if pd.isna(dt): return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour
def band_of_hour(h):
    if (22 <= h <= 23) or (0 <= h <= 7): return "경부하"
    elif 16 <= h <= 21: return "최대부하"
    else: return "중간부하"

def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"]>=5).astype(int)
    df["겨울여부"] = df["월"].isin([11,12,1,2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2*np.pi*df["시간"]/24)
    df["cos_time"] = np.cos(2*np.pi*df["시간"]/24)
    df["부하구분"] = df["시간"].apply(band_of_hour)
    return df

train = enrich(train).sort_values("측정일시").reset_index(drop=True)
test  = enrich(test ).sort_values("측정일시").reset_index(drop=True)

# 인코딩
le_job = LabelEncoder()
train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
test["작업유형_encoded"]  = le_job.transform(test["작업유형"].astype(str))

le_band = LabelEncoder()
train["부하구분_encoded"] = le_band.fit_transform(train["부하구분"].astype(str))
test["부하구분_encoded"]  = le_band.transform(test["부하구분"].astype(str))

train["시간_작업유형"] = train["시간"].astype(str)+"_"+train["작업유형_encoded"].astype(str)
test ["시간_작업유형"] = test ["시간"].astype(str)+"_"+test ["작업유형_encoded"].astype(str)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
test ["시간_작업유형_encoded"]  = le_tj.transform(test["시간_작업유형"])

# -----------------------------
# 2) Stage1: 전력특성 예측 (베이스 그대로)
# -----------------------------
targets_s1 = [
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)",
    "지상역률(%)","진상역률(%)"
]
feat_s1 = [
    "월","일","요일","시간","주말여부","겨울여부","period_flag",
    "sin_time","cos_time","작업유형_encoded","부하구분_encoded","시간_작업유형_encoded"
]

model_map = {
    "전력사용량(kWh)"      : LGBMRegressor(n_estimators=2500, learning_rate=0.012, num_leaves=128, random_state=42),
    "지상무효전력량(kVarh)" : CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "진상무효전력량(kVarh)" : CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "지상역률(%)"         : LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
    "진상역률(%)"         : LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
}

# ← 위 오타 수정:
model_map["진상역률(%)"] = LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42)

pred_test = pd.DataFrame({"id": test["id"]})
for tgt in targets_s1:
    m = model_map[tgt]
    m.fit(train[feat_s1], train[tgt])
    pred_test[tgt] = m.predict(test[feat_s1])
test = test.merge(pred_test, on="id", how="left")

# 유효역률 파생
def add_pf(df):
    df["유효역률(%)"] = df[["지상역률(%)","진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0)*0.01
    df["역률_보상율"]   = (df["유효역률(%)"] - 90).clip(lower=0)*0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    return df
train = add_pf(train)
test  = add_pf(test)

# -----------------------------
# 3) 재귀 lag/rolling 생성 (test에 진짜 값 채우기)
# -----------------------------
# train에서 kWh로 lag24/roll 생성
train["kwh_lag24"] = train["전력사용량(kWh)"].shift(24)
train["kwh_roll24_mean"] = train["전력사용량(kWh)"].shift(1).rolling(24, min_periods=1).mean()
train["kwh_roll24_std"]  = train["전력사용량(kWh)"].shift(1).rolling(24, min_periods=1).std().fillna(0)

# test는 12월이므로, 11월 마지막 24시간을 시드로 재귀 생성
last24 = train[["측정일시","전력사용량(kWh)"]].tail(24).copy()
hist = list(last24["전력사용량(kWh)"].values.astype(float))

kwh_pred = test["전력사용량(kWh)"].values.astype(float).copy()
lag24_list, roll24m_list, roll24s_list = [], [], []
for i in range(len(kwh_pred)):
    # 현재 시점의 lag24/rolling은 '직전 24시간' 히스토리로 계산
    lag24_list.append(hist[-24] if len(hist)>=24 else np.nan)
    arr = np.array(hist[-24:]) if len(hist)>=24 else np.array(hist)
    roll24m_list.append(arr.mean() if arr.size>0 else np.nan)
    roll24s_list.append(arr.std()  if arr.size>1 else 0.0)
    # 다음 시점을 위해 현재 예측 사용량을 히스토리에 push
    hist.append(kwh_pred[i])

test["kwh_lag24"] = lag24_list
test["kwh_roll24_mean"] = roll24m_list
test["kwh_roll24_std"]  = roll24s_list

# -----------------------------
# 4) Stage2: 요금 예측 (Log1p, 앙상블, 보정 제거)
# -----------------------------
feat_s2 = [
    "월","일","요일","시간","주말여부","겨울여부","period_flag",
    "sin_time","cos_time","작업유형_encoded","부하구분_encoded","시간_작업유형_encoded",
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)",
    "지상역률(%)","진상역률(%)","유효역률(%)","역률_조정요율",
    "kwh_lag24","kwh_roll24_mean","kwh_roll24_std"
]

X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
idx_tr = train["월"]<11; idx_va = train["월"]==11
X_tr, y_tr = X_all[idx_tr], np.log1p(y_all[idx_tr])
X_va, y_va = X_all[idx_va], y_all[idx_va]

lgb = LGBMRegressor(n_estimators=2300, learning_rate=0.02, num_leaves=96, subsample=0.9, colsample_bytree=0.9, reg_alpha=3, reg_lambda=4, random_state=42)
xgb = XGBRegressor(n_estimators=2300, learning_rate=0.02, max_depth=8, subsample=0.9, colsample_bytree=0.9, reg_lambda=4, reg_alpha=1, random_state=42)
cat = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=7, l2_leaf_reg=4, random_seed=42, verbose=0)

lgb.fit(X_tr, y_tr); xgb.fit(X_tr, y_tr); cat.fit(X_tr, y_tr)
pred_va = 0.5*np.expm1(lgb.predict(X_va)) + 0.3*np.expm1(xgb.predict(X_va)) + 0.2*np.expm1(cat.predict(X_va))

mae = mean_absolute_error(y_va, pred_va); r2 = r2_score(y_va, pred_va)
print(f"📊 11월 검증: MAE={mae:.2f} | R²={r2:.4f}")

# 히스토그램
plt.figure(figsize=(8,4.8))
plt.hist(y_va,   bins=60, alpha=0.5, density=True, label="Actual (11월)", color="#6BA3D6")
plt.hist(pred_va, bins=60, alpha=0.5, density=True, label="Pred (11월)",   color="#F3C969")
plt.title("📈 11월 전기요금 분포 (Actual vs Pred)"); plt.xlabel("전기요금(원)"); plt.ylabel("Density")
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# 5) Test(12월) 예측 (보정/계수/바이어스 없음)
# -----------------------------
X_te = test[feat_s2].copy()
pred_te = 0.5*np.expm1(lgb.predict(X_te)) + 0.3*np.expm1(xgb.predict(X_te)) + 0.2*np.expm1(cat.predict(X_te))

# 이상치 안정화(극단값 클리핑)
low, high = np.percentile(pred_te, [0.2, 99.8])
pred_te = np.clip(pred_te, low, high)

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_recover.csv", index=False)
print("💾 submission_recover.csv 저장 완료!")

