# ============================================================
# LS 전력요금 예측 (개선된 Stacking 적용 버전)
# ============================================================
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans # KMeans 추가
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV # RidgeCV 사용 (자동 Alpha 찾기)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 0) Load
# -----------------------------
# 파일 경로는 동일하다고 가정
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# -----------------------------
# 1) 시간 파생 (기존 코드 유지)
# -----------------------------
REF_DATE = pd.Timestamp("2024-10-24")
def adjust_hour(dt):
    if pd.isna(dt): return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour
def band_of_hour(h):
    if (22 <= h <= 23) or (0 <= h <= 7): return "경부하"
    if 16 <= h <= 21: return "최대부하"
    return "중간부하"
def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["날짜"] = df['측정일시'].dt.date # <-- '날짜' 컬럼 추가 (KMeans용)
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

# -----------------------------
# 2) 인코딩 (Test 데이터 에러 처리 추가)
# -----------------------------
le_job = LabelEncoder()
train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
# Test 데이터 변환 시 unseen 라벨 처리 (가장 흔한 값으로 대체)
test["작업유형_encoded"] = test["작업유형"].astype(str).map(lambda s: '-1' if s not in le_job.classes_ else s)
test["작업유형_encoded"] = le_job.transform(test["작업유형_encoded"].replace('-1', train["작업유형"].mode()[0])) # mode()로 대체

le_band = LabelEncoder()
train["부하구분_encoded"] = le_band.fit_transform(train["부하구분"].astype(str))
# Test 데이터 변환 시 unseen 라벨 처리
test["부하구분_encoded"] = test["부하구분"].astype(str).map(lambda s: '-1' if s not in le_band.classes_ else s)
test["부하구분_encoded"] = le_band.transform(test["부하구분_encoded"].replace('-1', train["부하구분"].mode()[0]))

train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
le_tj = LabelEncoder()
train["시간_작업유형_encoded"] = le_tj.fit_transform(train["시간_작업유형"])
# Test 데이터 변환 시 unseen 라벨 처리
test["시간_작업유형_encoded"] = test["시간_작업유형"].map(lambda s: '-1' if s not in le_tj.classes_ else s)
test["시간_작업유형_encoded"] = le_tj.transform(test["시간_작업유형_encoded"].replace('-1', train["시간_작업유형"].mode()[0]))

# -----------------------------
# 3) Stage1: 전력특성 예측 (기존 코드 유지)
# -----------------------------
targets_s1 = ["전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)", "지상역률(%)", "진상역률(%)"]
feat_s1 = ["월","일","요일","시간","주말여부","겨울여부","period_flag",
           "sin_time","cos_time","작업유형_encoded","부하구분_encoded","시간_작업유형_encoded"]
stage1_models = { # 하이퍼파라미터는 원본 유지
    "전력사용량(kWh)": LGBMRegressor(n_estimators=2500, learning_rate=0.012, num_leaves=128, random_state=42),
    "지상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "진상무효전력량(kVarh)": CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, verbose=0, random_seed=42),
    "지상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
    "진상역률(%)": LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=96, random_state=42),
}
tscv = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index)
stage1_test_pred = pd.DataFrame(index=test.index)
train_targets_true = train[targets_s1].copy() # 원본 타겟값 저장

for tgt in targets_s1:
    oof_pred = np.full(len(train), np.nan, dtype=float)
    model = stage1_models[tgt]
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train), start=1):
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(train.iloc[tr_idx][feat_s1], train_targets_true.iloc[tr_idx][tgt]) # 원본 타겟으로 학습
        oof_pred[va_idx] = fold_model.predict(train.iloc[va_idx][feat_s1])
    missing = np.isnan(oof_pred)
    if missing.any():
        full_model = model.__class__(**model.get_params())
        full_model.fit(train[feat_s1], train_targets_true[tgt]) # 원본 타겟으로 학습
        oof_pred[missing] = full_model.predict(train.loc[missing, feat_s1])
    stage1_oof[tgt] = oof_pred
    final_model = model.__class__(**model.get_params())
    final_model.fit(train[feat_s1], train_targets_true[tgt]) # 원본 타겟으로 학습
    stage1_test_pred[tgt] = final_model.predict(test[feat_s1])

# Stage1 예측 결과를 train, test 데이터프레임에 업데이트
for tgt in targets_s1:
    train[tgt] = stage1_oof[tgt]
    test[tgt] = stage1_test_pred[tgt]

# -----------------------------
# 4) EDA 기반 역률 피처 생성 (기존 코드 유지)
# -----------------------------
def add_pf_features(df: pd.DataFrame) -> pd.DataFrame:
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
    # 역률 구간 더미
    df["역률_60_85"] = (
        (df["지상역률_보정"] >= 60)
        & (df["지상역률_보정"] < 85)
        & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_85_90"] = (
        (df["지상역률_보정"] >= 85)
        & (df["지상역률_보정"] < 90)
        & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_90_94"] = (
        (df["지상역률_보정"] >= 90)
        & (df["지상역률_보정"] < 94)
        & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_94_이상"] = (
        (df["지상역률_보정"] >= 94) & (df["주간여부"] == 1)
    ).astype(int)
    # 강화된 역률 기반 피처
    df["부하역률곱_강화"] = (
        df["전력사용량(kWh)"] * df["역률부족폭_94"] * df["주간여부"] * 10
    )
    df["주간_부족률"] = df["주간여부"] * (90 - df["지상역률_보정"]).clip(lower=0)
    df["주간_추가요율"] = df["주간_부족률"] * 0.01
    return df
train = add_pf_features(train)
test = add_pf_features(test)

# -----------------------------
# 5) Lag/Rolling 생성 (Test 부분 정리)
# -----------------------------
# Train 데이터 Lag/Rolling (기존과 동일)
train["kwh_lag1"] = train["전력사용량(kWh)"].shift(1)
train["kwh_lag24"] = train["전력사용량(kWh)"].shift(24)
train["kwh_lag336"] = train["전력사용량(kWh)"].shift(336)
train["kwh_lag336_ratio"] = train["전력사용량(kWh)"] / (train["kwh_lag336"] + 1e-6)
train["kwh_roll12_mean"] = train["전력사용량(kWh)"].shift(1).rolling(12).mean()
train["kwh_roll12_std"] = (
    train["전력사용량(kWh)"].shift(1).rolling(12).std().fillna(0)
)
train["kwh_roll24_mean"] = train["전력사용량(kWh)"].shift(1).rolling(24).mean()
train["kwh_roll24_std"] = (
    train["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
)

# 무효전력 Lag/Rolling
train["kvarh_lag1"] = train["지상무효전력량(kVarh)"].shift(1)
train["kvarh_lag24"] = train["지상무효전력량(kVarh)"].shift(24)
train["kvarh_lag96"] = train["지상무효전력량(kVarh)"].shift(96)
train["kvarh_roll24_mean"] = (
    train["지상무효전력량(kVarh)"].shift(1).rolling(24).mean()
)
train["kvarh_roll24_std"] = (
    train["지상무효전력량(kVarh)"].shift(1).rolling(24).std().fillna(0)
)
train["kvarh_roll96_mean"] = (
    train["지상무효전력량(kVarh)"].shift(1).rolling(96).mean()
)
train["kvarh_roll96_std"] = (
    train["지상무효전력량(kVarh)"].shift(1).rolling(96).std().fillna(0)
)
train["kvarh_변화율_24h"] = (
    (train["지상무효전력량(kVarh)"] - train["kvarh_lag1"])
    / (train["kvarh_lag1"] + 1e-6)
)
train["무효전력_급등"] = (train["kvarh_변화율_24h"] > 0.5).astype(int)
train["전력품질지수"] = (
    train["kvarh_roll24_mean"] / (train["kwh_roll24_mean"] + 1e-6)
)

# Test 데이터 Lag/Rolling (확장)
hist_kwh = list(train["전력사용량(kWh)"].tail(672).values.astype(float))
hist_kvarh = list(train["지상무효전력량(kVarh)"].tail(672).values.astype(float))

kwh_lag1_list, kwh_lag24_list, kwh_lag336_list = [], [], []
kwh_roll12_mean_list, kwh_roll12_std_list = [], []
kwh_roll24_mean_list, kwh_roll24_std_list = [], []
kwh_lag336_ratio_list = []

kvarh_lag1_list, kvarh_lag24_list, kvarh_lag96_list = [], [], []
kvarh_roll24_mean_list, kvarh_roll24_std_list = [], []
kvarh_roll96_mean_list, kvarh_roll96_std_list = [], []
kvarh_change24_list, kvarh_spike_list = [], []

for i in range(len(test)):
    kwh_lag1 = hist_kwh[-1] if len(hist_kwh) >= 1 else np.nan
    kwh_lag24 = hist_kwh[-24] if len(hist_kwh) >= 24 else np.nan
    kwh_lag336 = hist_kwh[-336] if len(hist_kwh) >= 336 else np.nan

    kwh_lag1_list.append(kwh_lag1)
    kwh_lag24_list.append(kwh_lag24)
    kwh_lag336_list.append(kwh_lag336)

    arr12 = np.array(hist_kwh[-12:])
    arr24 = np.array(hist_kwh[-24:])
    mean12 = arr12.mean() if arr12.size > 0 else np.nan
    std12 = arr12.std() if arr12.size > 1 else 0
    mean24 = arr24.mean() if arr24.size > 0 else np.nan
    std24 = arr24.std() if arr24.size > 1 else 0

    kwh_roll12_mean_list.append(mean12)
    kwh_roll12_std_list.append(std12)
    kwh_roll24_mean_list.append(mean24)
    kwh_roll24_std_list.append(std24)

    y_kwh = test.loc[i, "전력사용량(kWh)"]
    kwh_lag336_ratio_list.append(
        y_kwh / (kwh_lag336 + 1e-6) if not np.isnan(kwh_lag336) else np.nan
    )

    kvarh_lag1 = hist_kvarh[-1] if len(hist_kvarh) >= 1 else np.nan
    kvarh_lag24 = hist_kvarh[-24] if len(hist_kvarh) >= 24 else np.nan
    kvarh_lag96 = hist_kvarh[-96] if len(hist_kvarh) >= 96 else np.nan

    kvarh_lag1_list.append(kvarh_lag1)
    kvarh_lag24_list.append(kvarh_lag24)
    kvarh_lag96_list.append(kvarh_lag96)

    arr24_k = np.array(hist_kvarh[-24:])
    arr96_k = np.array(hist_kvarh[-96:])
    mean24_k = arr24_k.mean() if arr24_k.size > 0 else np.nan
    std24_k = arr24_k.std() if arr24_k.size > 1 else 0
    mean96_k = arr96_k.mean() if arr96_k.size > 0 else np.nan
    std96_k = arr96_k.std() if arr96_k.size > 1 else 0

    kvarh_roll24_mean_list.append(mean24_k)
    kvarh_roll24_std_list.append(std24_k)
    kvarh_roll96_mean_list.append(mean96_k)
    kvarh_roll96_std_list.append(std96_k)

    y_kvarh = test.loc[i, "지상무효전력량(kVarh)"]
    change24 = (
        (y_kvarh - kvarh_lag1) / (kvarh_lag1 + 1e-6)
        if not np.isnan(kvarh_lag1)
        else np.nan
    )
    kvarh_change24_list.append(change24)
    kvarh_spike_list.append(int(change24 > 0.5) if not np.isnan(change24) else 0)

    hist_kwh.append(y_kwh)
    hist_kvarh.append(y_kvarh)

test["kwh_lag1"] = kwh_lag1_list
test["kwh_lag24"] = kwh_lag24_list
test["kwh_lag336"] = kwh_lag336_list
test["kwh_lag336_ratio"] = kwh_lag336_ratio_list
test["kwh_roll12_mean"] = kwh_roll12_mean_list
test["kwh_roll12_std"] = kwh_roll12_std_list
test["kwh_roll24_mean"] = kwh_roll24_mean_list
test["kwh_roll24_std"] = kwh_roll24_std_list

test["kvarh_lag1"] = kvarh_lag1_list
test["kvarh_lag24"] = kvarh_lag24_list
test["kvarh_lag96"] = kvarh_lag96_list
test["kvarh_roll24_mean"] = kvarh_roll24_mean_list
test["kvarh_roll24_std"] = kvarh_roll24_std_list
test["kvarh_roll96_mean"] = kvarh_roll96_mean_list
test["kvarh_roll96_std"] = kvarh_roll96_std_list
test["kvarh_변화율_24h"] = kvarh_change24_list
test["무효전력_급등"] = kvarh_spike_list
test["전력품질지수"] = (
    test["kvarh_roll24_mean"] / (test["kwh_roll24_mean"] + 1e-6)
)

# -----------------------------
# 6) 고급 피처 추가 (코드 중복 제거)
# -----------------------------
def add_advanced_features(df, train_means=None): # is_train 대신 train_means 전달
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_보정"] + 1e-6)
    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률_보정"] >= 94)).astype(int)

    if train_means: # Test 데이터 처리
        df["역률_월평균"] = df["월"].map(train_means["역률_월평균"])
        df["역률_월평균"].fillna(train_means["역률_월평균"].mean(), inplace=True) # 혹시 모를 NaN 처리
    else: # Train 데이터 처리
        df["역률_월평균"] = df.groupby("월")["지상역률_보정"].transform("mean")

    df["역률_월평균차이"] = df["지상역률_보정"] - df["역률_월평균"]
    df["kwh_roll24_cv"] = df["kwh_roll24_std"] / (df["kwh_roll24_mean"] + 1e-6)
    df["kwh_roll12_cv"] = df["kwh_roll12_std"] / (df["kwh_roll12_mean"] + 1e-6)
    df["kvarh_roll24_cv"] = df["kvarh_roll24_std"] / (df["kvarh_roll24_mean"] + 1e-6)
    df["kvarh_roll96_cv"] = df["kvarh_roll96_std"] / (df["kvarh_roll96_mean"] + 1e-6)
    # 변화율/급등 피처 (if/else 밖으로 이동)
    df["kwh_변화율_24h"] = ((df["전력사용량(kWh)"] - df["kwh_lag24"]) / (df["kwh_lag24"] + 1e-6))
    df["전력급등"] = (df["kwh_변화율_24h"] > 0.5).astype(int)
    # 역률 × 변동성/무효전력 교호작용
    df["역률부족_변동곱"] = df["역률부족폭_94"] * df["kwh_roll24_cv"] * 100
    df["역률부족_무효전력"] = df["역률부족폭_94"] * df["지상무효전력량(kVarh)"]
    df["주간위험_급변"] = df["실질위험"] * df["kwh_roll12_cv"] * 100
    df["무효전력비율_24h"] = df["kvarh_roll24_mean"] / (df["kwh_roll24_mean"] + 1e-6)
    df["무효전력비율_변화"] = (
        (df["지상무효전력량(kVarh)"] - df["kvarh_lag1"]) / (df["kvarh_lag1"] + 1e-6)
    )
    df["전력품질지수"] = df["무효전력비율_24h"]
    if "역률_90_94" in df.columns:
        df["역률_90_94_강화"] = df["역률_90_94"] * df["전력사용량(kWh)"] * 0.1
    return df

# Train 평균 계산
train_means_for_test = {"역률_월평균": train.groupby("월")["지상역률_보정"].mean()}
train = add_advanced_features(train)
test = add_advanced_features(test, train_means=train_means_for_test)

# -----------------------------
# 6.5) 일일 작업 유형 패턴 피처 생성 (기존 코드 유지)
# -----------------------------
print("\n🔄 일일 작업 유형 패턴 피처 생성 중...")
# 날짜별, 시간대별 작업 유형_encoded를 피벗 테이블로 변환
# 주의: enrich 함수에 '날짜' 컬럼 생성이 추가되었는지 확인 필요 (이미 위에서 추가함)
train_pattern_pivot = train.pivot_table(index='날짜', columns='시간', values='작업유형_encoded')
train_pattern_pivot = train_pattern_pivot.fillna(-1) # 결측치를 -1로 채움
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
train_pattern_pivot['일일패턴유형'] = kmeans.fit_predict(train_pattern_pivot)
pattern_map = train_pattern_pivot[['일일패턴유형']].reset_index()
train = pd.merge(train, pattern_map, on='날짜', how='left')

# Test 데이터 적용
test_pattern_pivot = test.pivot_table(index='날짜', columns='시간', values='작업유형_encoded')
test_pattern_pivot = test_pattern_pivot.fillna(-1)
# Train 컬럼 기준으로 맞추기 (순서 중요)
train_cols_no_target = train_pattern_pivot.drop(columns='일일패턴유형').columns
test_pattern_pivot = test_pattern_pivot.reindex(columns=train_cols_no_target, fill_value=-1)
test_pattern_pivot['일일패턴유형'] = kmeans.predict(test_pattern_pivot[train_cols_no_target]) # 순서 맞춘 데이터로 예측
test_pattern_map = test_pattern_pivot[['일일패턴유형']].reset_index()
test = pd.merge(test, test_pattern_map, on='날짜', how='left')

# NaN 처리 (가장 흔한 값으로)
most_frequent_pattern = train['일일패턴유형'].mode()[0]
train['일일패턴유형'].fillna(most_frequent_pattern, inplace=True)
test['일일패턴유형'].fillna(most_frequent_pattern, inplace=True)
train['일일패턴유형'] = train['일일패턴유형'].astype(int)
test['일일패턴유형'] = test['일일패턴유형'].astype(int)
print(f"✅ 일일 패턴 유형 생성 완료. 가장 흔한 유형: {most_frequent_pattern}")

# -----------------------------
# 7) Stage2 Feature Set (패턴 피처 추가 확인)
# -----------------------------
feat_s2 = [ # 기존 피처 리스트 사용 + 패턴 / 강화 피처 추가
    "월","일","요일","시간","주말여부","겨울여부","period_flag","sin_time","cos_time",
    "작업유형_encoded","부하구분_encoded","시간_작업유형_encoded",
    "전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)",
    "지상역률(%)","진상역률(%)","유효역률(%)","역률_조정요율","지상역률_보정",
    "주간여부","법적페널티","실질위험","극저역률","역률부족폭_94",
    "역률_60_85","역률_85_90","역률_90_94","역률_94_이상","역률_90_94_강화",
    "부하역률곱","부하역률곱_강화","주간_부족률","주간_추가요율",
    "무효유효비율","역률당전력","가을위험","동절기안정",
    "역률_월평균","역률_월평균차이",
    "역률부족_변동곱","역률부족_무효전력","주간위험_급변",
    "무효전력비율_24h","무효전력비율_변화","전력품질지수",
    "kwh_lag1","kwh_lag24","kwh_lag336","kwh_lag336_ratio",
    "kwh_roll12_mean","kwh_roll12_std","kwh_roll12_cv",
    "kwh_roll24_mean","kwh_roll24_std","kwh_roll24_cv",
    "kwh_변화율_24h","전력급등",
    "kvarh_lag1","kvarh_lag24","kvarh_lag96",
    "kvarh_roll24_mean","kvarh_roll24_std","kvarh_roll24_cv",
    "kvarh_roll96_mean","kvarh_roll96_std","kvarh_roll96_cv",
    "kvarh_변화율_24h","무효전력_급등",
    "일일패턴유형"
]
print(f"\n💡 Stage 2 피처 개수: {len(feat_s2)}")

# -----------------------------
# 8) Stage2 학습 (TimeSeriesSplit 기반 Stacking으로 변경)
# -----------------------------
X_all = train[feat_s2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)
X_te = test[feat_s2].copy()

# Base 모델 정의 (하이퍼파라미터는 원본 유지)
LGB_PARAMS = dict(n_estimators=2300, learning_rate=0.02, num_leaves=96, subsample=0.9, colsample_bytree=0.9, reg_alpha=3, reg_lambda=4, random_state=42)
XGB_PARAMS = dict(n_estimators=2300, learning_rate=0.02, max_depth=8, subsample=0.9, colsample_bytree=0.9, reg_lambda=4, reg_alpha=1, random_state=42)
CAT_PARAMS = dict(iterations=2000, learning_rate=0.02, depth=7, l2_leaf_reg=4, random_seed=42, verbose=0)
base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS)
}

# Meta 모델 정의
meta_learner = RidgeCV(alphas=np.logspace(-2, 2, 10), cv=None) # CV는 직접 하므로 None, Alpha 범위 지정

# TimeSeriesSplit 설정 (Stage 1과 동일하게 5-Fold)
tscv_s2 = TimeSeriesSplit(n_splits=5)

# OOF 예측값 및 Test 예측값 저장 배열 초기화
oof_preds_s2 = pd.DataFrame(index=X_all.index, columns=base_models.keys(), dtype=float)
test_preds_s2 = np.zeros((len(X_te), len(base_models)))

print("\n🚀 Stage 2 모델 학습 및 OOF 예측 생성 시작...")
for fold, (tr_idx, va_idx) in enumerate(tscv_s2.split(X_all), start=1):
    print(f"--- Fold {fold} ---")
    X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_tr_log, y_va_log = y_all_log.iloc[tr_idx], y_all_log.iloc[va_idx]

    fold_test_preds = [] # 현재 Fold에서의 Test 예측값 저장용

    for name, model in base_models.items():
        print(f"  Training {name}...")
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_tr, y_tr_log)

        # OOF 예측값 저장
        oof_pred = fold_model.predict(X_va)
        oof_preds_s2.iloc[va_idx, list(base_models.keys()).index(name)] = oof_pred

        # Test 예측값 누적 (각 Fold 모델의 예측을 평균내기 위함)
        fold_test_preds.append(fold_model.predict(X_te))

    # 현재 Fold의 Test 예측값들을 평균하여 누적 배열에 더함
    test_preds_s2 += np.mean(fold_test_preds, axis=0)[:, np.newaxis] / tscv_s2.n_splits

print("\n✅ OOF 예측 생성 완료.")

# Meta-Learner 학습 (OOF 예측값이 있는 부분만 사용)
oof_valid_idx = oof_preds_s2.dropna().index
print(f"\n🧠 Meta-Learner ({meta_learner.__class__.__name__}) 학습 시작 (데이터 {len(oof_valid_idx)}개)...")
meta_learner.fit(oof_preds_s2.loc[oof_valid_idx], y_all_log.loc[oof_valid_idx])
print(f"✅ Meta-Learner 학습 완료. 최적 Alpha: {meta_learner.alpha_:.4f}")
# 최종 가중치 확인 (RidgeCV는 coef_가 직접 가중치 역할)
final_weights = meta_learner.coef_ / meta_learner.coef_.sum()
print("⚙️ 최종 가중치:", {name: f"{w:.3f}" for name, w in zip(base_models.keys(), final_weights)})

# 최종 Test 예측 (평균낸 Base 모델 예측값에 Meta Learner 적용)
print("\n🧪 최종 Test 예측 생성...")
meta_test_input = pd.DataFrame(test_preds_s2, columns=base_models.keys(), index=X_te.index)
pred_te_log = meta_learner.predict(meta_test_input)
pred_te = np.expm1(pred_te_log)

# OOF 검증 점수 계산 (Optional, 모델 성능 평가용)
oof_pred_final_log = meta_learner.predict(oof_preds_s2.loc[oof_valid_idx])
oof_pred_final = np.expm1(oof_pred_final_log)
oof_mae = mean_absolute_error(y_all.loc[oof_valid_idx], oof_pred_final)
oof_r2 = r2_score(y_all.loc[oof_valid_idx], oof_pred_final)
print(f"\n📊 OOF 검증 (Stacking): MAE={oof_mae:.2f} | R²={oof_r2:.4f}")


# -----------------------------
# 9) 후처리 및 제출
# -----------------------------
low, high = np.percentile(pred_te, [0.2, 99.8]) # 클리핑 범위는 유지
pred_te = np.clip(pred_te, low, high)

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv("submission_ridge_stacking_cv.csv", index=False) # 파일명 변경
print("\n💾 submission_ridge_stacking_cv.csv 저장 완료!")
print(f"예측 범위: {pred_te.min():.2f} ~ {pred_te.max():.2f}")
print(f"예측 평균: {pred_te.mean():.2f}")

# Feature Importance (전체 데이터로 학습한 LGBM 모델 기준)
print("\n🚀 전체 데이터로 LGBM 모델 재학습 (Feature Importance 용)...")
lgb_full = LGBMRegressor(**LGB_PARAMS).fit(X_all, y_all_log)
feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb_full.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(70).to_string(index=False))


feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb_full.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(70).to_string(index=False))


print(f"예측 평균: {pred_te.mean():.2f}")
# Feature Importance (전체 데이터로 학습한 LGBM 모델 기준)␊
print("\n🚀 전체 데이터로 LGBM 모델 재학습 (Feature Importance 용)...")
lgb_full = LGBMRegressor(**LGB_PARAMS).fit(X_all, y_all_log)
feat_imp = pd.DataFrame({
    'feature': feat_s2,
    'importance': lgb_full.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔝 Top 20 중요 피처:")
print(feat_imp.head(20).to_string(index=False))
# -----------------------------
# 10) 월별 검증 (1~11월)
# -----------------------------
xgb_full = XGBRegressor(**XGB_PARAMS)
xgb_full.fit(X_all, y_all_log)
cat_full = CatBoostRegressor(**CAT_PARAMS)
cat_full.fit(X_all, y_all_log)
stack_inputs_full = pd.DataFrame({
    "lgb": lgb_full.predict(X_all),
    "xgb": xgb_full.predict(X_all),
    "cat": cat_full.predict(X_all),
}, index=X_all.index)
train_preds_full = np.expm1(meta_learner.predict(stack_inputs_full))
monthly_scores = []
for month in sorted(train["월"].unique()):
    mask = train["월"] == month
    mae_month = mean_absolute_error(y_all[mask], train_preds_full[mask])
    r2_month = r2_score(y_all[mask], train_preds_full[mask])
    monthly_scores.append((month, mae_month, r2_month))
    print(f"📅 {month}월 MAE={mae_month:.2f} | R²={r2_month:.4f}")
if monthly_scores:
    best_month, best_mae, best_r2 = min(monthly_scores, key=lambda x: x[1])
    print(f"\n✅ MAE가 가장 낮은 달: {best_month}월 (MAE={best_mae:.2f}, R²={best_r2:.4f})")
