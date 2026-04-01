from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SESSION_WINDOWS = (
    ("asia_size", 18 * 60, 22 * 60),
    ("london_size", 2 * 60, 5 * 60),
    ("nyam_size", 9 * 60 + 30, 11 * 60),
)
DAILY_COLUMNS = ["trading_day", "asia_size", "london_size", "nyam_size"]


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    output_dir: Path
    rolling_window: int = 40


def assign_trading_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.hour >= 18:
        return (timestamp + pd.Timedelta(days=1)).normalize()
    return timestamp.normalize()


def _session_metric_for_minutes(minutes: pd.Series) -> pd.Series:
    metric = pd.Series(pd.NA, index=minutes.index, dtype="object")
    for session_name, start_minute, end_minute in SESSION_WINDOWS:
        in_window = (minutes >= start_minute) & (minutes < end_minute)
        metric.loc[in_window] = session_name
    return metric


def build_daily_session_sizes(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["DateTime_ET"] = pd.to_datetime(data["DateTime_ET"])
    minutes = data["DateTime_ET"].dt.hour * 60 + data["DateTime_ET"].dt.minute
    data["metric"] = _session_metric_for_minutes(minutes)
    data = data[data["metric"].notna()].copy()
    data["trading_day"] = data["DateTime_ET"].map(assign_trading_day)

    rows = []
    for (trading_day, metric), group in data.groupby(["trading_day", "metric"], sort=True):
        first_bar = group.sort_values("DateTime_ET").iloc[0]
        rows.append(
            {
                "trading_day": trading_day,
                "metric": metric,
                "session_size": (group["High"].max() - group["Low"].min()) / first_bar["Open"],
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    return (
        daily.pivot(index="trading_day", columns="metric", values="session_size")
        .reindex(columns=DAILY_COLUMNS[1:])
        .reset_index()
        .sort_values("trading_day")
        .reset_index(drop=True)
    )


def _rolling_spearman(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    values = []
    for idx in range(len(series_a)):
        left = max(0, idx - window + 1)
        pair = pd.DataFrame({"a": series_a.iloc[left : idx + 1], "b": series_b.iloc[left : idx + 1]}).dropna()
        if len(pair) < window:
            values.append(float("nan"))
        else:
            values.append(pair["a"].corr(pair["b"], method="spearman"))
    return pd.Series(values, index=series_a.index)


def _build_pairwise_rolling_correlations(
    daily: pd.DataFrame, left_col: str, right_col: str, output_col: str, window: int
) -> pd.DataFrame:
    pair = daily[["trading_day", left_col, right_col]].dropna().reset_index(drop=True)
    if pair.empty:
        return pd.DataFrame(columns=["trading_day", output_col])

    pair[output_col] = _rolling_spearman(pair[left_col], pair[right_col], window)
    return pair[["trading_day", output_col]]


def build_rolling_correlations(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    result = daily[["trading_day"]].copy()
    london_col = f"corr_asia_london_{window}"
    nyam_col = f"corr_asia_nyam_{window}"

    result = result.merge(
        _build_pairwise_rolling_correlations(daily, "asia_size", "london_size", london_col, window),
        on="trading_day",
        how="left",
    )
    result = result.merge(
        _build_pairwise_rolling_correlations(daily, "asia_size", "nyam_size", nyam_col, window),
        on="trading_day",
        how="left",
    )
    return result


def run_analysis(config: AnalysisConfig) -> None:
    source = pd.read_parquet(config.input_path)
    daily = build_daily_session_sizes(source)
    rolling = build_rolling_correlations(daily, window=config.rolling_window)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(config.output_dir / "daily_session_sizes.csv", index=False)
    rolling.to_csv(config.output_dir / "rolling_correlations.csv", index=False)


def main() -> None:
    config = AnalysisConfig(
        input_path=Path("data/nq_1m.parquet"),
        output_dir=Path("results"),
        rolling_window=40,
    )
    run_analysis(config)


if __name__ == "__main__":
    main()
