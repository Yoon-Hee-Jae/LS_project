import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"   # 1~11월
TEST_PATH = DATA_DIR / "test.csv"     # 12월

LAG_COL = "지상역률(%)"
LEAD_COL = "진상역률(%)"
DAY_LABEL = "주간(09~23시)"
NIGHT_LABEL = "야간(23~09시)"

def load_and_resample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df = df.sort_values("측정일시").set_index("측정일시")
    df = df[[col for col in [LAG_COL, LEAD_COL] if col in df.columns]].copy()
    df["월"] = df.index.month
    df["slot_start"] = df.index.floor("30T")
    agg_cols = [col for col in [LAG_COL, LEAD_COL] if col in df.columns]
    df_30 = df.groupby("slot_start")[agg_cols].mean().reset_index()
    df_30["월"] = df_30["slot_start"].dt.month
    hours = df_30["slot_start"].dt.hour
    df_30["시간대"] = np.where((hours >= 9) & (hours < 23), DAY_LABEL, NIGHT_LABEL)
    return df_30

def apply_regulation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LAG_COL in df.columns:
        df[f"{LAG_COL}_보정"] = df[LAG_COL].clip(lower=60, upper=95)
    if LEAD_COL in df.columns:
        df[f"{LEAD_COL}_보정"] = df[LEAD_COL].clip(lower=60, upper=100)
    return df

def monthly_penalty(df_30: pd.DataFrame, label: str) -> pd.DataFrame:
    df_30 = apply_regulation(df_30)
    agg_dict = {}
    if LAG_COL in df_30.columns:
        agg_dict[LAG_COL] = "mean"
    if LEAD_COL in df_30.columns:
        agg_dict[LEAD_COL] = "mean"
    if f"{LAG_COL}_보정" in df_30.columns:
        agg_dict[f"{LAG_COL}_보정"] = "mean"
    if f"{LEAD_COL}_보정" in df_30.columns:
        agg_dict[f"{LEAD_COL}_보정"] = "mean"

    if not agg_dict:
        return pd.DataFrame()

    stats = df_30.groupby(["월", "시간대"]).agg(agg_dict).reset_index()
    if f"{LAG_COL}_보정" in stats.columns:
        stats["주간_기준역률"] = 90
        stats["주간_평균지상역률"] = np.where(
            stats["시간대"] == DAY_LABEL,
            stats[f"{LAG_COL}_보정"],
            np.nan,
        )
        stats["주간_부족률(%)"] = np.where(
            stats["시간대"] == DAY_LABEL,
            np.clip(stats["주간_기준역률"] - stats["주간_평균지상역률"], a_min=0, a_max=None),
            np.nan,
        )
        stats["주간_추가요율(%)"] = stats["주간_부족률(%)"] * 0.2
    if f"{LEAD_COL}_보정" in stats.columns:
        stats["야간_기준역률"] = 95
        stats["야간_평균진상역률"] = np.where(
            stats["시간대"] == NIGHT_LABEL, stats[f"{LEAD_COL}_보정"], np.nan
        )
        stats["야간_부족률(%)"] = np.where(
            stats["시간대"] == NIGHT_LABEL,
            np.clip(stats["야간_기준역률"] - stats["야간_평균진상역률"], a_min=0, a_max=None),
            np.nan,
        )
        stats["야간_추가요율(%)"] = stats["야간_부족률(%)"] * 0.2
    stats.insert(0, "데이터구분", label)
    return stats

train_30 = load_and_resample(TRAIN_PATH)
test_30 = load_and_resample(TEST_PATH)

train_summary = monthly_penalty(train_30, "train(1~11월)")
test_summary = monthly_penalty(test_30, "test(12월)") if set([LAG_COL, LEAD_COL]).issubset(test_30.columns) else pd.DataFrame()

result_df = pd.concat([train_summary, test_summary], ignore_index=True, sort=False)
print(result_df.round(3).fillna(""))

if not train_summary.empty:
    plot_df = train_summary.copy()
    plot_df = plot_df.melt(
        id_vars=["월", "시간대"],
        value_vars=["주간_평균지상역률", "야간_평균진상역률"],
        var_name="종류",
        value_name="역률",
    ).dropna()
    plt.figure(figsize=(10, 4))
    for label, grp in plot_df.groupby("종류"):
        plt.plot(grp["월"], grp["역률"], marker="o", label=label)
    plt.axhline(90, color="tab:blue", linestyle=":", label="지상 기준 90%")
    plt.axhline(95, color="tab:orange", linestyle=":", label="진상 기준 95%")
    plt.title("월별 평균 역률 (train)")
    plt.xlabel("월")
    plt.ylabel("역률(%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

############################################################

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
train = pd.read_csv(DATA_DIR / "train.csv")

train["측정일시"] = pd.to_datetime(train["측정일시"], errors="coerce")
train = train.sort_values("측정일시").set_index("측정일시")

# 30분 단위 역률·요금 평균
slot = train.resample("30T").agg({
    "전기요금(원)": "mean",
    "지상역률(%)": "mean",
    "진상역률(%)": "mean"
}).dropna()
slot["월"] = slot.index.month
slot["시간대"] = np.where(
    (slot.index.hour >= 9) & (slot.index.hour < 23),
    "주간(09~23시)",
    "야간(23~09시)",
)

# 역률 클리핑 규정 반영 (60~95/100)
slot["지상역률_보정"] = slot["지상역률(%)"].clip(lower=60, upper=95)
slot["진상역률_보정"] = slot["진상역률(%)"].clip(lower=60, upper=100)

# 주간·야간 나누기
day_df = slot[slot["시간대"] == "주간(09~23시)"].copy()
night_df = slot[slot["시간대"] == "야간(23~09시)"].copy()

# 기준 대비 부족률과 추가요율(%) 계산
day_df["부족률(%)"] = np.clip(90 - day_df["지상역률_보정"], 0, None)
day_df["추가요율(%)"] = day_df["부족률(%)"] * 0.2   # 기본요금 0.2%씩 추가

night_df["부족률(%)"] = np.clip(95 - night_df["진상역률_보정"], 0, None)
night_df["추가요율(%)"] = night_df["부족률(%)"] * 0.2

# 1) 시간대별 상관계수
corr_day = day_df["전기요금(원)"].corr(day_df["지상역률_보정"], method="spearman")
corr_night = night_df["전기요금(원)"].corr(night_df["진상역률_보정"], method="spearman")
print(f"[주간] Spearman corr(요금, 지상역률) = {corr_day:.4f}")
print(f"[야간] Spearman corr(요금, 진상역률) = {corr_night:.4f}")

# 2) 역률 구간별 평균 요금 테이블
day_bins = pd.cut(day_df["지상역률_보정"], bins=[60, 90, 92, 94, 95])
night_bins = pd.cut(night_df["진상역률_보정"], bins=[60, 95, 97, 99, 100])

day_summary = (
    day_df.groupby(day_bins)[["전기요금(원)", "부족률(%)", "추가요율(%)"]]
          .agg(["count", "mean"])
)

night_summary = (
    night_df.groupby(night_bins)[["전기요금(원)", "부족률(%)", "추가요율(%)"]]
            .agg(["count", "mean"])
)


print("\n[주간 지상역률 구간별 요금/부족률]")
print(day_summary.round(3))
print("\n[야간 진상역률 구간별 요금/부족률]")
print(night_summary.round(3))

# (선택) 월별 부족률 vs 요금 평균 테이블
monthly_day = (
    day_df.groupby("월")[["전기요금(원)", "지상역률_보정", "부족률(%)", "추가요율(%)"]]
    .mean()
    .round(3)
)
monthly_night = (
    night_df.groupby("월")[["전기요금(원)", "진상역률_보정", "부족률(%)", "추가요율(%)"]]
    .mean()
    .round(3)
)

print("\n[월별 주간 평균 요금과 지상역률]")
print(monthly_day)
print("\n[월별 야간 평균 요금과 진상역률]")
print(monthly_night)
