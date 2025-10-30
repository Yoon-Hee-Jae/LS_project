import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0) Load
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

train = pd.read_csv(DATA_DIR / "train_.csv")
test = pd.read_csv(DATA_DIR / "test_.csv")


# ---------------------------------------------------------------------------
# 1) Time Features & Encoding
# ---------------------------------------------------------------------------
REF_DATE = pd.Timestamp("2024-10-24")


def adjust_hour(dt: pd.Timestamp) -> float:
    if pd.isna(dt):
        return np.nan
    return dt.hour if dt.minute >= 15 else (dt.hour - 1) % 24


def band_of_hour(hour: float) -> str:
    if pd.isna(hour):
        return "unknown"
    if (22 <= hour <= 23) or (0 <= hour <= 7):
        return "경부하"
    if 16 <= hour <= 21:
        return "최대부하"
    return "중간부하"


def enrich_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    df["부하구분"] = df["시간"].apply(band_of_hour)
    return df


train = enrich_time(train).sort_values("측정일시").reset_index(drop=True)
test = enrich_time(test).sort_values("측정일시").reset_index(drop=True)

job_encoder = LabelEncoder()
train["작업유형_encoded"] = job_encoder.fit_transform(train["작업유형"].astype(str))
test["작업유형_encoded"] = job_encoder.transform(test["작업유형"].astype(str))

band_encoder = LabelEncoder()
train["부하구분_encoded"] = band_encoder.fit_transform(train["부하구분"].astype(str))
test["부하구분_encoded"] = band_encoder.transform(test["부하구분"].astype(str))

train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)

combo_encoder = LabelEncoder()
train["시간_작업유형_encoded"] = combo_encoder.fit_transform(train["시간_작업유형"])
test["시간_작업유형_encoded"] = combo_encoder.transform(test["시간_작업유형"])


# ---------------------------------------------------------------------------
# 2) Stage1 Models (OOF)
# ---------------------------------------------------------------------------
targets_stage1 = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]

features_stage1 = [
    "월",
    "일",
    "요일",
    "시간",
    "주말여부",
    "겨울여부",
    "period_flag",
    "sin_time",
    "cos_time",
    "sin_day",
    "cos_day",
    "sin_month",
    "cos_month",
    "작업유형_encoded",
    "부하구분_encoded",
    "시간_작업유형_encoded",
]

stage1_models = {
    "전력사용량(kWh)": LGBMRegressor(
        n_estimators=3500,
        learning_rate=0.01,
        num_leaves=128,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    ),
    "지상무효전력량(kVarh)": CatBoostRegressor(
        iterations=3200,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_seed=42,
        thread_count=-1,
    ),
    "진상무효전력량(kVarh)": CatBoostRegressor(
        iterations=3200,
        learning_rate=0.02,
        depth=8,
        verbose=0,
        random_seed=42,
        thread_count=-1,
    ),
    "지상역률(%)": LGBMRegressor(
        n_estimators=3200,
        learning_rate=0.012,
        num_leaves=96,
        random_state=42,
        n_jobs=-1,
    ),
    "진상역률(%)": LGBMRegressor(
        n_estimators=3200,
        learning_rate=0.012,
        num_leaves=96,
        random_state=42,
        n_jobs=-1,
    ),
}

tscv_stage1 = TimeSeriesSplit(n_splits=5)
stage1_oof = pd.DataFrame(index=train.index, columns=targets_stage1)
stage1_test = pd.DataFrame(index=test.index, columns=targets_stage1)

for target in targets_stage1:
    model = stage1_models[target]
    oof_pred = np.full(len(train), np.nan)
    test_fold_sum = np.zeros(len(test), dtype=float)

    for fold, (idx_tr, idx_va) in enumerate(tscv_stage1.split(train), start=1):
        X_tr = train.iloc[idx_tr][features_stage1]
        y_tr = train.iloc[idx_tr][target]
        X_va = train.iloc[idx_va][features_stage1]

        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_tr, y_tr)
        oof_pred[idx_va] = model_fold.predict(X_va)
        test_fold_sum += model_fold.predict(test[features_stage1])

    stage1_oof[target] = oof_pred
    stage1_test[target] = test_fold_sum / tscv_stage1.get_n_splits()

    # 초기 구간 보정 (shift로 인한 NaN)
    nan_mask = stage1_oof[target].isna()
    if nan_mask.any():
        filler = model.__class__(**model.get_params())
        filler.fit(train[features_stage1], train[target])
        stage1_oof.loc[nan_mask, target] = filler.predict(train.loc[nan_mask, features_stage1])

# Stage1 예측값으로 대체 (true 값은 보존)
for tgt in targets_stage1:
    train[f"{tgt}_true"] = train[tgt]
    train[tgt] = stage1_oof[tgt]
    test[tgt] = stage1_test[tgt]


# ---------------------------------------------------------------------------
# 3) Stage1 Derived Features
# ---------------------------------------------------------------------------
def add_pf_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["역률부족폭_90"] = (90 - df["지상역률(%)"]).clip(lower=0)
    df["역률부족폭_94"] = (94 - df["지상역률(%)"]).clip(lower=0)
    df["역률우수"] = (df["지상역률(%)"] >= 95).astype(int)

    df["주간여부"] = df["시간"].between(9, 22).astype(int)
    df["법적페널티"] = ((df["주간여부"] == 1) & (df["지상역률(%)"] < 90)).astype(int)
    df["실질위험"] = ((df["주간여부"] == 1) & (df["지상역률(%)"] < 94)).astype(int)
    df["극저역률"] = ((df["주간여부"] == 1) & (df["지상역률(%)"] < 85)).astype(int)

    df["주간_부족률"] = df["주간여부"] * (90 - df["지상역률(%)"]).clip(lower=0)
    df["주간_추가요율"] = df["주간_부족률"] * 0.01

    df["부하역률곱_강화"] = (
        df["전력사용량(kWh)"] * df["역률부족폭_94"] * df["주간여부"] * 10
    )
    df["역률부족_경부하"] = (df["부하구분"] == "경부하").astype(int) * df["역률부족폭_94"]
    df["역률부족_중간부하"] = (df["부하구분"] == "중간부하").astype(int) * df["역률부족폭_94"]
    df["역률부족_최대부하"] = (df["부하구분"] == "최대부하").astype(int) * df["역률부족폭_94"]

    df["역률_60_85"] = (
        (df["지상역률(%)"].between(60, 85, inclusive="left")) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_85_90"] = (
        (df["지상역률(%)"].between(85, 90, inclusive="left")) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_90_94"] = (
        (df["지상역률(%)"].between(90, 94, inclusive="left")) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_94_이상"] = ((df["지상역률(%)"] >= 94) & (df["주간여부"] == 1)).astype(int)
    return df


train = add_pf_features(train)
test = add_pf_features(test)


# ---------------------------------------------------------------------------
# 4) Lag / Rolling Features
# ---------------------------------------------------------------------------
def add_lag_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_aug = df_train.copy()
    test_aug = df_test.copy()

    # Train lag/rolling
    train_aug["kwh_lag1"] = train_aug["전력사용량(kWh)"].shift(1)
    train_aug["kwh_lag24"] = train_aug["전력사용량(kWh)"].shift(24)
    train_aug["kwh_lag96"] = train_aug["전력사용량(kWh)"].shift(96)
    train_aug["kwh_lag168"] = train_aug["전력사용량(kWh)"].shift(168)
    train_aug["kwh_lag672"] = train_aug["전력사용량(kWh)"].shift(672)

    for window in [12, 24, 48, 96]:
        roll_mean = train_aug["전력사용량(kWh)"].shift(1).rolling(window).mean()
        roll_std = train_aug["전력사용량(kWh)"].shift(1).rolling(window).std().fillna(0)
        train_aug[f"kwh_roll{window}_mean"] = roll_mean
        train_aug[f"kwh_roll{window}_std"] = roll_std
        train_aug[f"kwh_roll{window}_cv"] = roll_std / (roll_mean + 1e-6)

    train_aug["kwh_roll24_range"] = (
        train_aug["전력사용량(kWh)"].shift(1).rolling(24).max()
        - train_aug["전력사용량(kWh)"].shift(1).rolling(24).min()
    )
    train_aug["kwh_lag24_ratio"] = train_aug["전력사용량(kWh)"] / (train_aug["kwh_lag24"] + 1e-6)
    train_aug["kwh_roll24_ratio"] = train_aug["전력사용량(kWh)"] / (train_aug["kwh_roll24_mean"] + 1e-6)
    train_aug["kwh_lag168_ratio"] = train_aug["전력사용량(kWh)"] / (train_aug["kwh_lag168"] + 1e-6)
    train_aug["kwh_vs_어제"] = (train_aug["전력사용량(kWh)"] - train_aug["kwh_lag24"]) / (
        train_aug["kwh_lag24"] + 1e-6
    )
    train_aug["전력급등"] = (train_aug["kwh_vs_어제"] > 0.5).astype(int)
    train_aug["위험_변동성"] = train_aug["실질위험"] * train_aug["kwh_roll24_cv"]

    # kVarh lags/rolling
    train_aug["kvarh_lag1"] = train_aug["지상무효전력량(kVarh)"].shift(1)
    train_aug["kvarh_roll24_mean"] = train_aug["지상무효전력량(kVarh)"].shift(1).rolling(24).mean()
    train_aug["kvarh_roll96_mean"] = train_aug["지상무효전력량(kVarh)"].shift(1).rolling(96).mean()

    # Recursive generation for test
    hist_kwh = list(train_aug["전력사용량(kWh)"].tail(672).values.astype(float))
    hist_kvarh = list(train_aug["지상무효전력량(kVarh)"].tail(672).values.astype(float))

    kwh_features = {name: [] for name in [
        "kwh_lag1", "kwh_lag24", "kwh_lag96", "kwh_lag168", "kwh_lag672",
        "kwh_roll12_mean", "kwh_roll12_std", "kwh_roll12_cv",
        "kwh_roll24_mean", "kwh_roll24_std", "kwh_roll24_cv",
        "kwh_roll48_mean", "kwh_roll48_std", "kwh_roll48_cv",
        "kwh_roll96_mean", "kwh_roll96_std", "kwh_roll96_cv",
        "kwh_roll24_range", "kwh_lag24_ratio", "kwh_roll24_ratio",
        "kwh_lag168_ratio", "kwh_vs_어제"
    ]}
    kvarh_features = {name: [] for name in ["kvarh_lag1", "kvarh_roll24_mean", "kvarh_roll96_mean"]}

    for _, row in test_aug.iterrows():
        kwh_vals = np.array(hist_kwh)
        kwh_lag1 = kwh_vals[-1] if kwh_vals.size >= 1 else np.nan
        kwh_lag24 = kwh_vals[-24] if kwh_vals.size >= 24 else np.nan
        kwh_lag96 = kwh_vals[-96] if kwh_vals.size >= 96 else np.nan
        kwh_lag168 = kwh_vals[-168] if kwh_vals.size >= 168 else np.nan
        kwh_lag672 = kwh_vals[-672] if kwh_vals.size >= 672 else np.nan

        def recent(arr, window):
            return arr[-window:] if arr.size >= window else arr

        arr12 = recent(kwh_vals, 12)
        arr24 = recent(kwh_vals, 24)
        arr48 = recent(kwh_vals, 48)
        arr96 = recent(kwh_vals, 96)

        roll_means = {
            12: arr12.mean() if arr12.size else np.nan,
            24: arr24.mean() if arr24.size else np.nan,
            48: arr48.mean() if arr48.size else np.nan,
            96: arr96.mean() if arr96.size else np.nan,
        }
        roll_stds = {
            12: arr12.std() if arr12.size > 1 else 0,
            24: arr24.std() if arr24.size > 1 else 0,
            48: arr48.std() if arr48.size > 1 else 0,
            96: arr96.std() if arr96.size > 1 else 0,
        }

        kwh_features["kwh_lag1"].append(kwh_lag1)
        kwh_features["kwh_lag24"].append(kwh_lag24)
        kwh_features["kwh_lag96"].append(kwh_lag96)
        kwh_features["kwh_lag168"].append(kwh_lag168)
        kwh_features["kwh_lag672"].append(kwh_lag672)
        kwh_features["kwh_roll12_mean"].append(roll_means[12])
        kwh_features["kwh_roll24_mean"].append(roll_means[24])
        kwh_features["kwh_roll48_mean"].append(roll_means[48])
        kwh_features["kwh_roll96_mean"].append(roll_means[96])
        kwh_features["kwh_roll12_std"].append(roll_stds[12])
        kwh_features["kwh_roll24_std"].append(roll_stds[24])
        kwh_features["kwh_roll48_std"].append(roll_stds[48])
        kwh_features["kwh_roll96_std"].append(roll_stds[96])
        kwh_features["kwh_roll12_cv"].append(roll_stds[12] / (roll_means[12] + 1e-6) if not np.isnan(roll_means[12]) else np.nan)
        kwh_features["kwh_roll24_cv"].append(roll_stds[24] / (roll_means[24] + 1e-6) if not np.isnan(roll_means[24]) else np.nan)
        kwh_features["kwh_roll48_cv"].append(roll_stds[48] / (roll_means[48] + 1e-6) if not np.isnan(roll_means[48]) else np.nan)
        kwh_features["kwh_roll96_cv"].append(roll_stds[96] / (roll_means[96] + 1e-6) if not np.isnan(roll_means[96]) else np.nan)
        kwh_features["kwh_roll24_range"].append(
            arr24.max() - arr24.min() if arr24.size else np.nan
        )
        kwh_features["kwh_lag24_ratio"].append(
            row["전력사용량(kWh)"] / (kwh_lag24 + 1e-6) if not np.isnan(kwh_lag24) else np.nan
        )
        kwh_features["kwh_roll24_ratio"].append(
            row["전력사용량(kWh)"] / (roll_means[24] + 1e-6) if not np.isnan(roll_means[24]) else np.nan
        )
        kwh_features["kwh_lag168_ratio"].append(
            row["전력사용량(kWh)"] / (kwh_lag168 + 1e-6) if not np.isnan(kwh_lag168) else np.nan
        )
        kwh_features["kwh_vs_어제"].append(
            (row["전력사용량(kWh)"] - kwh_lag24) / (kwh_lag24 + 1e-6) if not np.isnan(kwh_lag24) else np.nan
        )

        kvarh_vals = np.array(hist_kvarh)
        kvarh_lag1 = kvarh_vals[-1] if kvarh_vals.size >= 1 else np.nan
        arr24_kvarh = kvarh_vals[-24:] if kvarh_vals.size >= 24 else kvarh_vals
        arr96_kvarh = kvarh_vals[-96:] if kvarh_vals.size >= 96 else kvarh_vals

        kvarh_features["kvarh_lag1"].append(kvarh_lag1)
        kvarh_features["kvarh_roll24_mean"].append(arr24_kvarh.mean() if arr24_kvarh.size else np.nan)
        kvarh_features["kvarh_roll96_mean"].append(arr96_kvarh.mean() if arr96_kvarh.size else np.nan)

        hist_kwh.append(row["전력사용량(kWh)"])
        hist_kvarh.append(row["지상무효전력량(kVarh)"])

    for key, values in kwh_features.items():
        test_aug[key] = values
    for key, values in kvarh_features.items():
        test_aug[key] = values

    test_aug["전력급등"] = (np.array(kwh_features["kwh_vs_어제"]) > 0.5).astype(int)
    test_aug["위험_변동성"] = test_aug["실질위험"] * test_aug["kwh_roll24_cv"]

    return train_aug, test_aug


train, test = add_lag_features(train, test)


# ---------------------------------------------------------------------------
# 5) Advanced Aggregations
# ---------------------------------------------------------------------------
mean_day_hour = (
    train.groupby(["요일", "시간"])["전력사용량(kWh)"].mean().rename("kwh_요일_시간_평균").reset_index()
)
train = train.merge(mean_day_hour, on=["요일", "시간"], how="left")
test = test.merge(mean_day_hour, on=["요일", "시간"], how="left")


def add_advanced_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률(%)"] + 1e-6)
    df["총무효전력"] = df["지상무효전력량(kVarh)"] + df["진상무효전력량(kVarh)"]
    df["무효전력비중"] = df["총무효전력"] / (df["전력사용량(kWh)"] + df["총무효전력"] + 1e-6)

    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률(%)"] >= 94)).astype(int)

    if is_train:
        df["역률_월평균"] = df.groupby("월")["지상역률(%)"].transform("mean")
        df["전력사용_시간평균"] = df.groupby("시간")["전력사용량(kWh)"].transform("mean")
    else:
        monthly_mean = train.groupby("월")["지상역률(%)"].mean()
        hourly_mean = train.groupby("시간")["전력사용량(kWh)"].mean()
        df["역률_월평균"] = df["월"].map(monthly_mean)
        df["전력사용_시간평균"] = df["시간"].map(hourly_mean)
        df["역률_월평균"].fillna(monthly_mean.mean(), inplace=True)
        df["전력사용_시간평균"].fillna(hourly_mean.mean(), inplace=True)

    df["역률_월평균차이"] = df["지상역률(%)"] - df["역률_월평균"]
    df["전력사용_시간대비"] = df["전력사용량(kWh)"] / (df["전력사용_시간평균"] + 1e-6)
    df["kwh_시간대비_요일"] = df["전력사용량(kWh)"] / (df["kwh_요일_시간_평균"] + 1e-6)
    df.drop(columns="kwh_요일_시간_평균", inplace=True)
    return df


train = add_advanced_features(train, is_train=True)
test = add_advanced_features(test, is_train=False)


# ---------------------------------------------------------------------------
# 6) Stage2 Training (Stacking + Bias Correction)
# ---------------------------------------------------------------------------
features_stage2 = [
    # Time / categorical
    "월", "일", "요일", "시간", "주말여부", "겨울여부", "period_flag",
    "sin_time", "cos_time", "sin_day", "cos_day", "sin_month", "cos_month",
    "작업유형_encoded", "부하구분_encoded", "시간_작업유형_encoded",
    # Stage1 outputs
    "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
    "지상역률(%)", "진상역률(%)", "유효역률(%)",
    # Lag / rolling
    "kwh_lag1", "kwh_lag24", "kwh_lag96", "kwh_lag168", "kwh_lag672",
    "kwh_roll12_mean", "kwh_roll12_std", "kwh_roll12_cv",
    "kwh_roll24_mean", "kwh_roll24_std", "kwh_roll24_cv",
    "kwh_roll48_mean", "kwh_roll48_std", "kwh_roll48_cv",
    "kwh_roll96_mean", "kwh_roll96_std", "kwh_roll96_cv",
    "kwh_roll24_range",
    "kwh_lag24_ratio", "kwh_roll24_ratio", "kwh_lag168_ratio", "kwh_vs_어제",
    "전력급등", "위험_변동성",
    "kvarh_lag1", "kvarh_roll24_mean", "kvarh_roll96_mean",
    # PF features
    "역률부족폭_90", "역률부족폭_94", "역률우수",
    "주간여부", "법적페널티", "실질위험", "극저역률",
    "주간_부족률", "주간_추가요율", "부하역률곱", "부하역률곱_강화",
    "역률부족_경부하", "역률부족_중간부하", "역률부족_최대부하",
    "역률_60_85", "역률_85_90", "역률_90_94", "역률_94_이상",
    "무효유효비율", "역률당전력", "총무효전력", "무효전력비중",
    "가을위험", "동절기안정",
    "역률_월평균", "역률_월평균차이",
    "전력사용_시간평균", "전력사용_시간대비", "kwh_시간대비_요일",
]

X_all = train[features_stage2].copy()
y_all = train["전기요금(원)"].copy()
y_all_log = np.log1p(y_all)

sample_weights = np.ones(len(y_all), dtype=float)
sample_weights[y_all > 3000] = 2.0
sample_weights[y_all > 5000] = 3.0
sample_weights[y_all > 10000] = 5.0
sample_weights[X_all["실질위험"] == 1] *= 2.0

LGB_PARAMS = dict(
    n_estimators=3500,
    learning_rate=0.012,
    num_leaves=128,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=3,
    reg_lambda=4,
    min_child_samples=18,
    random_state=42,
    n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=3500,
    learning_rate=0.012,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=4,
    reg_alpha=2,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1,
)
CAT_PARAMS = dict(
    iterations=3000,
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=4,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

base_models = {
    "lgb": LGBMRegressor(**LGB_PARAMS),
    "xgb": XGBRegressor(**XGB_PARAMS),
    "cat": CatBoostRegressor(**CAT_PARAMS),
}

tscv_stage2 = TimeSeriesSplit(n_splits=5)
oof_pred = pd.DataFrame(index=X_all.index, columns=base_models.keys(), dtype=float)

for fold, (idx_tr, idx_va) in enumerate(tscv_stage2.split(X_all), start=1):
    X_tr, X_va = X_all.iloc[idx_tr], X_all.iloc[idx_va]
    y_tr, y_va = y_all_log.iloc[idx_tr], y_all_log.iloc[idx_va]
    w_tr = sample_weights[idx_tr]

    for name, model in base_models.items():
        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_pred.loc[idx_va, name] = model_fold.predict(X_va)

meta_learner = Ridge(alpha=100)
valid_idx = oof_pred.dropna().index
meta_learner.fit(oof_pred.loc[valid_idx], y_all_log.loc[valid_idx])

# Fit full models
model_lgb = LGBMRegressor(**LGB_PARAMS)
model_xgb = XGBRegressor(**XGB_PARAMS)
model_cat = CatBoostRegressor(**CAT_PARAMS)

model_lgb.fit(X_all, y_all_log, sample_weight=sample_weights)
model_xgb.fit(X_all, y_all_log, sample_weight=sample_weights)
model_cat.fit(X_all, y_all_log, sample_weight=sample_weights)


# ---------------------------------------------------------------------------
# 7) Validation (November) with Bias Buckets
# ---------------------------------------------------------------------------
va_mask = train["월"] == 11
X_va = X_all[va_mask]
y_va = y_all[va_mask]

va_preds = np.expm1(meta_learner.predict(pd.DataFrame({
    "lgb": model_lgb.predict(X_va),
    "xgb": model_xgb.predict(X_va),
    "cat": model_cat.predict(X_va),
}, index=X_va.index)))

mae = mean_absolute_error(y_va, va_preds)
r2 = r2_score(y_va, va_preds)
print(f"\n📊 11월 검증 (Stacking): MAE={mae:.2f} | R²={r2:.4f}")


# ---------------------------------------------------------------------------
# 8) Test Prediction (December) with Bias Buckets
# ---------------------------------------------------------------------------
X_te = test[features_stage2].copy()

pred_meta = meta_learner.predict(pd.DataFrame({
    "lgb": model_lgb.predict(X_te),
    "xgb": model_xgb.predict(X_te),
    "cat": model_cat.predict(X_te),
}, index=X_te.index))

pred_te = np.expm1(pred_meta)

low, high = np.percentile(pred_te, [0.1, 99.9])
pred_te = np.clip(pred_te, low, high)

submission = pd.DataFrame({"id": test["id"], "target": pred_te})
submission.to_csv(BASE_DIR / "submission_783_biasweighted.csv", index=False)

print("\n💾 submission_783_biasweighted.csv 저장 완료!")
print(f"예측 범위: {pred_te.min():.2f} ~ {pred_te.max():.2f}")
print(f"예측 평균: {pred_te.mean():.2f}")


# ---------------------------------------------------------------------------
# 9) Feature Importance (LightGBM)
# ---------------------------------------------------------------------------
feat_importance = pd.DataFrame({
    "feature": features_stage2,
    "importance": model_lgb.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n🔝 Top 20 Feature Importance (LGBM):")
print(feat_importance.head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# 10) Full Train Evaluation (January~November)
# ---------------------------------------------------------------------------
stack_inputs_all = pd.DataFrame({
    "lgb": model_lgb.predict(X_all),
    "xgb": model_xgb.predict(X_all),
    "cat": model_cat.predict(X_all),
}, index=X_all.index)
train_preds_all = np.expm1(meta_learner.predict(stack_inputs_all))

monthly_metrics = []
for month in sorted(train["월"].unique()):
    month_mask = train["월"] == month
    y_month = y_all[month_mask]
    pred_month = train_preds_all[month_mask]

    mae_month = mean_absolute_error(y_month, pred_month)
    r2_month = r2_score(y_month, pred_month)
    monthly_metrics.append((month, mae_month, r2_month))
    print(f"📅 {month}월 검증: MAE={mae_month:.2f} | R²={r2_month:.4f}")

best_month, best_mae, best_r2 = min(monthly_metrics, key=lambda x: x[1])
print(f"\n✅ 월별 MAE 최저: {best_month}월 (MAE={best_mae:.2f}, R²={best_r2:.4f})")
