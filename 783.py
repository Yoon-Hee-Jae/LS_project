import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_DIR = Path("./data")


# ---------------------------------------------------------------------------
# 1. 기본 전처리 & Stage1
# ---------------------------------------------------------------------------
def enrich_time_features(df: pd.DataFrame) -> pd.DataFrame:
    def band_of_hour(hour: int) -> str:
        if (22 <= hour <= 23) or (0 <= hour <= 7):
            return "경부하"
        if 16 <= hour <= 21:
            return "최대부하"
        return "중간부하"

    ref_date = pd.Timestamp("2024-10-24")
    df = df.copy()
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df = df.sort_values("측정일시").reset_index(drop=True)
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["시간"] = df["측정일시"].dt.hour
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["겨울여부"] = df["월"].isin([11, 12, 1, 2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= ref_date).astype(int)
    df["sin_time"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_time"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["부하구분"] = df["시간"].apply(band_of_hour)
    return df


def encode_categorical(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    le_job = LabelEncoder()
    train["작업유형_encoded"] = le_job.fit_transform(train["작업유형"].astype(str))
    test["작업유형_encoded"] = test["작업유형"].astype(str)
    test["작업유형_encoded"] = test["작업유형_encoded"].apply(
        lambda x: x if x in le_job.classes_ else le_job.classes_[0]
    )
    test["작업유형_encoded"] = le_job.transform(test["작업유형_encoded"])

    le_band = LabelEncoder()
    train["부하구분_encoded"] = le_band.fit_transform(train["부하구분"].astype(str))
    test["부하구분_encoded"] = test["부하구분"].astype(str)
    test["부하구분_encoded"] = test["부하구분_encoded"].apply(
        lambda x: x if x in le_band.classes_ else le_band.classes_[0]
    )
    test["부하구분_encoded"] = le_band.transform(test["부하구분_encoded"])

    train["시간_작업유형"] = train["시간"].astype(str) + "_" + train["작업유형_encoded"].astype(str)
    test["시간_작업유형"] = test["시간"].astype(str) + "_" + test["작업유형_encoded"].astype(str)
    le_combo = LabelEncoder()
    train["시간_작업유형_encoded"] = le_combo.fit_transform(train["시간_작업유형"])
    test["시간_작업유형_encoded"] = test["시간_작업유형"].apply(
        lambda x: x if x in le_combo.classes_ else le_combo.classes_[0]
    )
    test["시간_작업유형_encoded"] = le_combo.transform(test["시간_작업유형_encoded"])
    return train, test


def run_stage1(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [
        "전력사용량(kWh)",
        "지상무효전력량(kVarh)",
        "진상무효전력량(kVarh)",
        "지상역률(%)",
        "진상역률(%)",
    ]
    features = [
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
    model_factory = {
        "전력사용량(kWh)": lambda: LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, num_leaves=128, random_state=RANDOM_STATE
        ),
        "지상무효전력량(kVarh)": lambda: LGBMRegressor(
            n_estimators=1800, learning_rate=0.02, num_leaves=96, random_state=RANDOM_STATE
        ),
        "진상무효전력량(kVarh)": lambda: LGBMRegressor(
            n_estimators=1800, learning_rate=0.02, num_leaves=96, random_state=RANDOM_STATE
        ),
        "지상역률(%)": lambda: LGBMRegressor(
            n_estimators=1500, learning_rate=0.02, num_leaves=64, random_state=RANDOM_STATE
        ),
        "진상역률(%)": lambda: LGBMRegressor(
            n_estimators=1500, learning_rate=0.02, num_leaves=64, random_state=RANDOM_STATE
        ),
    }
    stage1_oof = pd.DataFrame(index=train.index)
    stage1_test = pd.DataFrame(index=test.index)

    for target in targets:
        model = model_factory[target]()
        oof_pred = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        fold_sizes = []

        months_sorted = sorted(train["월"].unique())
        for month in months_sorted:
            train_mask = train["월"] < month
            val_mask = train["월"] == month
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
            X_tr = train.loc[train_mask, features]
            y_tr = train.loc[train_mask, target]
            X_va = train.loc[val_mask, features]
            model.fit(X_tr, y_tr)
            oof_pred[val_mask] = model.predict(X_va)
            fold_sizes.append(val_mask.sum())
        zero_mask = oof_pred == 0
        model.fit(train[features], train[target])
        if zero_mask.any():
            oof_pred[zero_mask] = model.predict(train.loc[zero_mask, features])
        test_pred = model.predict(test[features])
        stage1_oof[target] = oof_pred
        stage1_test[target] = test_pred
    return stage1_oof, stage1_test


# ---------------------------------------------------------------------------
# 2. Stage2 Feature Engineering
# ---------------------------------------------------------------------------
def add_power_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["지상역률_보정"] = df["지상역률(%)"].clip(lower=60, upper=100)
    df["주간여부"] = df["부하구분"].isin(["중간부하", "최대부하"]).astype(int)
    df["법적페널티"] = ((df["지상역률_보정"] < 90) & (df["주간여부"] == 1)).astype(int)
    df["실질위험"] = ((df["지상역률_보정"] < 94) & (df["주간여부"] == 1)).astype(int)
    df["극저역률"] = ((df["지상역률_보정"] < 85) & (df["주간여부"] == 1)).astype(int)
    df["역률부족폭_94"] = (94 - df["지상역률_보정"]).clip(lower=0)
    df["유효역률(%)"] = df[["지상역률(%)", "진상역률(%)"]].max(axis=1)
    df["무효유효비율"] = df["지상무효전력량(kVarh)"] / (df["전력사용량(kWh)"] + 1e-6)
    df["부하역률곱"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    df["역률당전력"] = df["전력사용량(kWh)"] / (df["지상역률_보정"] + 1e-6)
    df["가을위험"] = ((df["월"].isin([9, 10])) & (df["실질위험"] == 1)).astype(int)
    df["동절기안정"] = ((df["겨울여부"] == 1) & (df["지상역률_보정"] >= 94)).astype(int)
    df["역률_월평균"] = df.groupby("월")["지상역률_보정"].transform("mean")
    df["역률_월평균차이"] = df["지상역률_보정"] - df["역률_월평균"]

    df["역률_60_85"] = (
        (df["지상역률_보정"] >= 60) & (df["지상역률_보정"] < 85) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_85_90"] = (
        (df["지상역률_보정"] >= 85) & (df["지상역률_보정"] < 90) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_90_94"] = (
        (df["지상역률_보정"] >= 90) & (df["지상역률_보정"] < 94) & (df["주간여부"] == 1)
    ).astype(int)
    df["역률_94_이상"] = ((df["지상역률_보정"] >= 94) & (df["주간여부"] == 1)).astype(int)

    df["역률_94cut_gap"] = df["역률부족폭_94"].copy()
    df["전력x94cut"] = df["전력사용량(kWh)"] * df["역률_94cut_gap"]
    df["kvarh_x94cut"] = df["지상무효전력량(kVarh)"] * df["역률_94cut_gap"]
    df["전력x주간위험"] = df["전력사용량(kWh)"] * df["실질위험"]
    df["전력x역률부족"] = df["전력사용량(kWh)"] * df["역률부족폭_94"]
    df["kvarh_x역률부족"] = df["지상무효전력량(kVarh)"] * df["역률부족폭_94"]
    return df


def add_sequence_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seq_cols = [
        "전력사용량(kWh)",
        "지상무효전력량(kVarh)",
    ]
    combined = (
        pd.concat(
            [
                train.assign(_dataset="train"),
                test.assign(_dataset="test"),
            ],
            ignore_index=True,
        )
        .sort_values("측정일시")
        .reset_index(drop=True)
    )
    combined["kwh_lag24"] = combined["전력사용량(kWh)"].shift(24)
    combined["kwh_roll24_mean"] = combined["전력사용량(kWh)"].shift(1).rolling(24).mean()
    combined["kwh_roll24_std"] = (
        combined["전력사용량(kWh)"].shift(1).rolling(24).std().fillna(0)
    )
    combined["kwh_변화율_24h"] = (
        (combined["전력사용량(kWh)"] - combined["kwh_lag24"]) / (combined["kwh_lag24"] + 1e-6)
    )

    combined["kvarh_lag24"] = combined["지상무효전력량(kVarh)"].shift(24)
    combined["kvarh_roll24_mean"] = (
        combined["지상무효전력량(kVarh)"].shift(1).rolling(24).mean()
    )
    combined["kvarh_roll24_std"] = (
        combined["지상무효전력량(kVarh)"].shift(1).rolling(24).std().fillna(0)
    )
    combined["전력품질지수"] = (
        combined["kvarh_roll24_mean"] / (combined["kwh_roll24_mean"] + 1e-6)
    )

    train_enhanced = combined.loc[combined["_dataset"] == "train"].drop(columns="_dataset")
    test_enhanced = combined.loc[combined["_dataset"] == "test"].drop(columns="_dataset")
    return train_enhanced.reset_index(drop=True), test_enhanced.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Custom CV & Sample Weights
# ---------------------------------------------------------------------------
def create_custom_folds(train: pd.DataFrame) -> list[dict]:
    fold_specs = [
        (list(range(1, 7)), [7]),
        (list(range(2, 8)), [8]),
        (list(range(3, 9)), [9]),
        (list(range(4, 10)), [10]),
        (list(range(5, 11)), [11]),
    ]
    folds = []
    for train_months, val_months in fold_specs:
        train_mask = train["월"].isin(train_months).values
        val_mask = train["월"].isin(val_months).values
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append(
            {
                "train_months": train_months,
                "val_months": val_months,
                "train_mask": train_mask,
                "val_mask": val_mask,
            }
        )
    return folds


def build_sample_weights(train: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(train))
    weights[train["월"] == 7] *= 2.0
    weights[train["실질위험"] == 1] *= 1.5
    return weights


# ---------------------------------------------------------------------------
# 4. 하이퍼파라미터 튜닝 (7월 기준)
# ---------------------------------------------------------------------------
def tune_model(
    model_name: str,
    model_cls,
    param_grid: list[dict],
    X: np.ndarray,
    y_log: np.ndarray,
    y_true: np.ndarray,
    sample_weights: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[dict, float]:
    best_score = np.inf
    best_params = None
    for params in param_grid:
        model = model_cls(**params)
        if model_name == "xgb":
            model.set_params(eval_metric="mae")
        model.fit(
            X[train_mask],
            y_log[train_mask],
            sample_weight=sample_weights[train_mask],
        )
        pred = np.expm1(model.predict(X[val_mask]))
        score = mean_absolute_error(y_true[val_mask], pred)
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


# ---------------------------------------------------------------------------
# 5. 메인 파이프라인
# ---------------------------------------------------------------------------
def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train = enrich_time_features(train)
    test = enrich_time_features(test)
    train, test = encode_categorical(train, test)

    stage1_oof, stage1_test = run_stage1(train, test)
    for col in stage1_oof.columns:
        train[col] = stage1_oof[col]
        test[col] = stage1_test[col]

    train = add_power_factor_features(train)
    test = add_power_factor_features(test)
    train, test = add_sequence_features(train, test)

    stage2_features = [
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
        "지상역률_보정",
        "주간여부",
        "실질위험",
        "역률부족폭_94",
        "무효유효비율",
        "부하역률곱",
        "역률당전력",
        "가을위험",
        "동절기안정",
        "역률_월평균",
        "역률_월평균차이",
        "역률_60_85",
        "역률_85_90",
        "역률_90_94",
        "역률_94_이상",
        "역률_94cut_gap",
        "전력x94cut",
        "kvarh_x94cut",
        "전력x주간위험",
        "전력x역률부족",
        "kvarh_x역률부족",
        "kwh_lag24",
        "kwh_roll24_mean",
        "kwh_roll24_std",
        "kwh_변화율_24h",
        "kvarh_lag24",
        "kvarh_roll24_mean",
        "kvarh_roll24_std",
        "전력품질지수",
    ]

    X_all = train[stage2_features].values
    X_te = test[stage2_features].values
    y_all = train["전기요금(원)"].values
    y_all_log = np.log1p(y_all)
    sample_weights = build_sample_weights(train)

    folds = create_custom_folds(train)
    print("📅 Custom CV 구조:")
    for idx, fold in enumerate(folds, 1):
        print(f"  Fold{idx}: train_months={fold['train_months']} | val_months={fold['val_months']}")

    first_fold = folds[0]
    param_grid_lgb = [
        {"learning_rate": lr, "num_leaves": nl, "n_estimators": ne, "subsample": ss, "colsample_bytree": cb, "random_state": RANDOM_STATE}
        for lr in [0.015, 0.02, 0.03]
        for nl in [48, 64, 96]
        for ne in [1200, 1600]
        for ss, cb in [(0.8, 0.8), (0.9, 0.9)]
    ][:25]
    param_grid_xgb = [
        {"learning_rate": lr, "max_depth": md, "n_estimators": ne, "subsample": ss, "colsample_bytree": cb, "reg_lambda": rl, "reg_alpha": ra, "random_state": RANDOM_STATE}
        for lr in [0.015, 0.02, 0.03]
        for md in [5, 7, 9]
        for ne in [1200, 1600]
        for ss, cb in [(0.8, 0.8), (0.9, 0.9)]
        for rl, ra in [(3, 0), (5, 1)]
    ][:20]
    param_grid_cat = [
        {"depth": dp, "learning_rate": lr, "iterations": it, "l2_leaf_reg": l2, "random_state": RANDOM_STATE, "verbose": 0}
        for dp in [6, 8]
        for lr in [0.02, 0.03]
        for it in [1500, 2000]
        for l2 in [3, 5, 7]
    ][:15]

    best_lgb_params, best_lgb_mae = tune_model(
        "lgb",
        LGBMRegressor,
        param_grid_lgb,
        X_all,
        y_all_log,
        y_all,
        sample_weights,
        first_fold["train_mask"],
        first_fold["val_mask"],
    )
    best_xgb_params, best_xgb_mae = tune_model(
        "xgb",
        XGBRegressor,
        param_grid_xgb,
        X_all,
        y_all_log,
        y_all,
        sample_weights,
        first_fold["train_mask"],
        first_fold["val_mask"],
    )
    if best_xgb_params is not None:
        best_xgb_params = {**best_xgb_params, "eval_metric": "mae"}
    best_cat_params, best_cat_mae = tune_model(
        "cat",
        CatBoostRegressor,
        param_grid_cat,
        X_all,
        y_all_log,
        y_all,
        sample_weights,
        first_fold["train_mask"],
        first_fold["val_mask"],
    )

    print("\n🎯 튜닝 결과 (7월 MAE 기준):")
    print(f"  LightGBM best MAE={best_lgb_mae:.2f} | params={best_lgb_params}")
    print(f"  XGBoost  best MAE={best_xgb_mae:.2f} | params={best_xgb_params}")
    print(f"  CatBoost best MAE={best_cat_mae:.2f} | params={best_cat_params}")

    oof_preds = pd.DataFrame(index=train.index, columns=["lgb", "xgb", "cat"], dtype=float)
    test_preds = {"lgb": np.zeros(len(test)), "xgb": np.zeros(len(test)), "cat": np.zeros(len(test))}

    for fold in folds:
        tr_mask = fold["train_mask"]
        va_mask = fold["val_mask"]

        lgb_model = LGBMRegressor(**best_lgb_params)
        lgb_model.fit(X_all[tr_mask], y_all_log[tr_mask], sample_weight=sample_weights[tr_mask])
        oof_preds.loc[va_mask, "lgb"] = lgb_model.predict(X_all[va_mask])
        test_preds["lgb"] += lgb_model.predict(X_te) / len(folds)

        xgb_model = XGBRegressor(**best_xgb_params)
        xgb_model.fit(
            X_all[tr_mask],
            y_all_log[tr_mask],
            sample_weight=sample_weights[tr_mask],
        )
        oof_preds.loc[va_mask, "xgb"] = xgb_model.predict(X_all[va_mask])
        test_preds["xgb"] += xgb_model.predict(X_te) / len(folds)

        cat_model = CatBoostRegressor(**best_cat_params)
        cat_model.fit(
            X_all[tr_mask],
            y_all_log[tr_mask],
            sample_weight=sample_weights[tr_mask],
        )
        oof_preds.loc[va_mask, "cat"] = cat_model.predict(X_all[va_mask])
        test_preds["cat"] += cat_model.predict(X_te) / len(folds)

    meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=True)
    oof_valid = oof_preds.dropna()
    meta_learner.fit(oof_valid.values, y_all_log[oof_valid.index])

    oof_stacked = np.expm1(meta_learner.predict(oof_valid.values))
    oof_pred_series = pd.Series(oof_stacked, index=oof_valid.index, name="pred")
    stacking_mae = mean_absolute_error(y_all[oof_valid.index], oof_stacked)
    stacking_r2 = r2_score(y_all[oof_valid.index], oof_stacked)
    print(f"\n📊 최종 Stacking OOF: MAE={stacking_mae:.2f} | R²={stacking_r2:.4f}")

    oof_valid_months = train.loc[oof_pred_series.index, "월"]
    monthly_mae = {}
    for month in sorted(oof_valid_months.dropna().unique()):
        month_idx = oof_valid_months[oof_valid_months == month].index
        if month_idx.empty:
            continue
        month_mae = mean_absolute_error(y_all.loc[month_idx], oof_pred_series.loc[month_idx])
        monthly_mae[int(month)] = float(month_mae)
    if monthly_mae:
        print("\n📆 월별 OOF MAE:")
        for month in sorted(monthly_mae):
            print(f"  {month}월 MAE={monthly_mae[month]:.2f}")
    else:
        print("\n⚠️ 월별 OOF MAE를 계산할 데이터가 없습니다.")

    test_stack_input = pd.DataFrame(test_preds)
    pred_test = np.expm1(meta_learner.predict(test_stack_input.values))
    submission = pd.DataFrame({"id": test["id"], "target": pred_test})
    submission.to_csv("submission_final_custom_cv.csv", index=False)

    lgb_full = LGBMRegressor(**best_lgb_params)
    lgb_full.fit(X_all, y_all_log, sample_weight=sample_weights)
    feature_importance = pd.DataFrame(
        {
            "feature": stage2_features,
            "importance": lgb_full.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    top30 = feature_importance.head(30)
    print("\n🔝 Top 30 Feature Importance (LightGBM):")
    print(top30.to_string(index=False))
    new_feature_set = {
        "역률_94cut_gap",
        "전력x94cut",
        "kvarh_x94cut",
        "전력x주간위험",
        "전력x역률부족",
        "kvarh_x역률부족",
    }
    present_new = sorted(f for f in new_feature_set if f in top30["feature"].values)
    print(f"\n🆕 신규 피처 Top30 포함: {', '.join(present_new) if present_new else 'None'}")


if __name__ == "__main__":
    main()
