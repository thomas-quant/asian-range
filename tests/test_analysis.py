from pathlib import Path

import pandas as pd
import pytest

from asia_range.analysis import AnalysisConfig, run_analysis


def test_run_analysis_writes_daily_and_rolling_outputs(tmp_path: Path):
    source = pd.DataFrame(
        [
            {"DateTime_ET": "2020-08-31 18:00:00", "session": "ASIA", "Open": 100.0, "High": 103.0, "Low": 99.0},
            {"DateTime_ET": "2020-09-01 02:00:00", "session": "LONDON", "Open": 110.0, "High": 112.0, "Low": 109.0},
            {"DateTime_ET": "2020-09-01 09:30:00", "session": "NYAM", "Open": 120.0, "High": 123.0, "Low": 118.0},
            {"DateTime_ET": "2020-09-01 18:00:00", "session": "ASIA", "Open": 130.0, "High": 136.5, "Low": 129.0},
            {"DateTime_ET": "2020-09-02 02:00:00", "session": "LONDON", "Open": 140.0, "High": 144.2, "Low": 139.0},
            {"DateTime_ET": "2020-09-02 09:30:00", "session": "NYAM", "Open": 150.0, "High": 156.0, "Low": 149.0},
        ]
    )
    source["DateTime_ET"] = pd.to_datetime(source["DateTime_ET"])
    input_path = tmp_path / "input.parquet"
    source.to_parquet(input_path)

    config = AnalysisConfig(
        input_path=input_path,
        output_dir=tmp_path / "results",
        rolling_window=2,
    )
    run_analysis(config)

    daily = pd.read_csv(config.output_dir / "daily_session_sizes.csv", parse_dates=["trading_day"])
    rolling = pd.read_csv(config.output_dir / "rolling_correlations.csv", parse_dates=["trading_day"])

    assert list(daily.columns) == ["trading_day", "asia_size", "london_size", "nyam_size"]
    assert list(rolling.columns) == ["trading_day", "corr_asia_london_2", "corr_asia_nyam_2"]
    assert len(daily) == 2
    assert rolling.iloc[-1]["corr_asia_london_2"] == pytest.approx(1.0)
    assert rolling.iloc[-1]["corr_asia_nyam_2"] == pytest.approx(1.0)


def test_build_daily_session_sizes_maps_asia_to_following_trading_day():
    from asia_range.analysis import build_daily_session_sizes

    df = pd.DataFrame(
        [
            {"DateTime_ET": "2020-08-31 18:00:00", "session": "ASIA", "Open": 100.0, "High": 102.0, "Low": 99.0},
            {"DateTime_ET": "2020-08-31 18:01:00", "session": "ASIA", "Open": 101.0, "High": 103.0, "Low": 98.0},
            {"DateTime_ET": "2020-09-01 02:00:00", "session": "LONDON", "Open": 110.0, "High": 112.0, "Low": 109.0},
            {"DateTime_ET": "2020-09-01 09:30:00", "session": "NYAM", "Open": 120.0, "High": 121.0, "Low": 118.0},
        ]
    )
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])

    result = build_daily_session_sizes(df)

    row = result.loc[result["trading_day"] == pd.Timestamp("2020-09-01")].iloc[0]
    assert row["asia_size"] == 0.05
    assert row["london_size"] == (112.0 - 109.0) / 110.0
    assert row["nyam_size"] == (121.0 - 118.0) / 120.0


def test_build_rolling_correlations_uses_complete_pairs():
    from asia_range.analysis import build_rolling_correlations

    daily = pd.DataFrame(
        {
            "trading_day": pd.to_datetime(["2020-09-01", "2020-09-02", "2020-09-03"]),
            "asia_size": [0.01, 0.02, 0.03],
            "london_size": [0.02, 0.04, None],
            "nyam_size": [0.03, None, 0.09],
        }
    )

    result = build_rolling_correlations(daily, window=2)

    last = result.iloc[-1]
    assert pd.isna(last["corr_asia_london_2"])
    assert last["corr_asia_nyam_2"] == pytest.approx(1.0)


def test_build_daily_session_sizes_ignores_non_target_sessions_and_keeps_schema():
    from asia_range.analysis import build_daily_session_sizes

    df = pd.DataFrame(
        [
            {"DateTime_ET": "2020-09-01 12:00:00", "session": "LUNCH", "Open": 100.0, "High": 150.0, "Low": 50.0},
            {"DateTime_ET": "2020-09-01 18:00:00", "session": "ASIA", "Open": 100.0, "High": 101.0, "Low": 99.0},
        ]
    )
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])

    result = build_daily_session_sizes(df)

    assert list(result.columns) == ["trading_day", "asia_size", "london_size", "nyam_size"]
    assert result.iloc[0]["asia_size"] == pytest.approx(0.02)
    assert pd.isna(result.iloc[0]["london_size"])
    assert pd.isna(result.iloc[0]["nyam_size"])
