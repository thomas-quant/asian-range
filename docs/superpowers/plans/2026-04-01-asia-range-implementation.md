# Asia Range Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research script that computes normalized Asia, London, and NYAM session sizes by trading day and outputs 40-day rolling Spearman correlations for Asia vs London and Asia vs NYAM.

**Architecture:** Add a small Python analysis module that loads the parquet file, assigns a trading-day key on the `18:00 ET` boundary, aggregates session sizes from the dataset session labels, and writes daily and rolling-correlation outputs. Cover the session grouping and rolling-correlation behavior with focused pytest tests against synthetic in-memory data.

**Tech Stack:** Python 3, pandas, pytest, parquet input

---

### Task 1: Scaffold The Analysis Module And Public API

**Files:**
- Create: `src/asia_range/__init__.py`
- Create: `src/asia_range/analysis.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing API test**

```python
from pathlib import Path

from asia_range.analysis import AnalysisConfig, run_analysis


def test_run_analysis_writes_daily_and_rolling_outputs(tmp_path: Path):
    config = AnalysisConfig(
        input_path=tmp_path / "input.parquet",
        output_dir=tmp_path / "results",
        rolling_window=2,
    )
    run_analysis(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis.py::test_run_analysis_writes_daily_and_rolling_outputs -v`
Expected: FAIL with `ModuleNotFoundError` or missing `AnalysisConfig`

- [ ] **Step 3: Write minimal implementation**

```python
# src/asia_range/__init__.py
"""Asia range research package."""

from .analysis import AnalysisConfig, run_analysis

__all__ = ["AnalysisConfig", "run_analysis"]
```

```python
# src/asia_range/analysis.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    output_dir: Path
    rolling_window: int = 40


def run_analysis(config: AnalysisConfig) -> None:
    raise NotImplementedError("Implementation added in later tasks")
```

- [ ] **Step 4: Run test to verify it fails for the right reason**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_run_analysis_writes_daily_and_rolling_outputs -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 5: Commit**

```bash
git add src/asia_range/__init__.py src/asia_range/analysis.py tests/test_analysis.py
git commit -m "chore: scaffold asia range analysis module"
```

### Task 2: Add Trading-Day Mapping And Session Aggregation

**Files:**
- Modify: `src/asia_range/analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing aggregation test**

```python
import pandas as pd

from asia_range.analysis import build_daily_session_sizes


def test_build_daily_session_sizes_maps_asia_to_following_trading_day():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_build_daily_session_sizes_maps_asia_to_following_trading_day -v`
Expected: FAIL with missing `build_daily_session_sizes`

- [ ] **Step 3: Write minimal implementation**

```python
SESSION_MAP = {
    "ASIA": "asia_size",
    "LONDON": "london_size",
    "NYAM": "nyam_size",
}


def assign_trading_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.hour >= 18:
        return (timestamp + pd.Timedelta(days=1)).normalize()
    return timestamp.normalize()


def build_daily_session_sizes(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["DateTime_ET"] = pd.to_datetime(data["DateTime_ET"])
    data = data[data["session"].isin(SESSION_MAP)].copy()
    data["trading_day"] = data["DateTime_ET"].map(assign_trading_day)

    rows = []
    for (trading_day, session_name), group in data.groupby(["trading_day", "session"], sort=True):
        first_bar = group.sort_values("DateTime_ET").iloc[0]
        rows.append(
            {
                "trading_day": trading_day,
                "metric": SESSION_MAP[session_name],
                "session_size": (group["High"].max() - group["Low"].min()) / first_bar["Open"],
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=["trading_day", "asia_size", "london_size", "nyam_size"])

    return (
        daily.pivot(index="trading_day", columns="metric", values="session_size")
        .reset_index()
        .sort_values("trading_day")
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_build_daily_session_sizes_maps_asia_to_following_trading_day -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/asia_range/analysis.py tests/test_analysis.py
git commit -m "feat: add trading day session aggregation"
```

### Task 3: Add Rolling Spearman Correlations

**Files:**
- Modify: `src/asia_range/analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing rolling-correlation test**

```python
import pandas as pd

from asia_range.analysis import build_rolling_correlations


def test_build_rolling_correlations_uses_complete_pairs():
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
    assert last["corr_asia_nyam_2"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_build_rolling_correlations_uses_complete_pairs -v`
Expected: FAIL with missing `build_rolling_correlations`

- [ ] **Step 3: Write minimal implementation**

```python
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


def build_rolling_correlations(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    result = daily[["trading_day"]].copy()
    result[f"corr_asia_london_{window}"] = _rolling_spearman(daily["asia_size"], daily["london_size"], window)
    result[f"corr_asia_nyam_{window}"] = _rolling_spearman(daily["asia_size"], daily["nyam_size"], window)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_build_rolling_correlations_uses_complete_pairs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/asia_range/analysis.py tests/test_analysis.py
git commit -m "feat: add rolling spearman correlations"
```

### Task 4: Wire File I/O And CLI Entry Point

**Files:**
- Modify: `src/asia_range/analysis.py`
- Create: `scripts/run_analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing output test**

```python
from pathlib import Path

import pandas as pd

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

    config = AnalysisConfig(input_path=input_path, output_dir=tmp_path / "results", rolling_window=2)
    run_analysis(config)

    daily = pd.read_csv(config.output_dir / "daily_session_sizes.csv", parse_dates=["trading_day"])
    rolling = pd.read_csv(config.output_dir / "rolling_correlations.csv", parse_dates=["trading_day"])

    assert list(daily.columns) == ["trading_day", "asia_size", "london_size", "nyam_size"]
    assert list(rolling.columns) == ["trading_day", "corr_asia_london_2", "corr_asia_nyam_2"]
    assert len(daily) == 2
    assert rolling.iloc[-1]["corr_asia_london_2"] == 1.0
    assert rolling.iloc[-1]["corr_asia_nyam_2"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_run_analysis_writes_daily_and_rolling_outputs -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

```python
# scripts/run_analysis.py
from asia_range.analysis import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_run_analysis_writes_daily_and_rolling_outputs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/asia_range/analysis.py scripts/run_analysis.py tests/test_analysis.py
git commit -m "feat: add analysis output pipeline"
```

### Task 5: Run Full Test Suite And Produce Research Outputs

**Files:**
- Modify: `src/asia_range/analysis.py`
- Modify: `tests/test_analysis.py`
- Create: `results/daily_session_sizes.csv`
- Create: `results/rolling_correlations.csv`

- [ ] **Step 1: Add a session-filtering regression test**

```python
import pandas as pd

from asia_range.analysis import build_daily_session_sizes


def test_build_daily_session_sizes_ignores_non_target_sessions():
    df = pd.DataFrame(
        [
            {"DateTime_ET": "2020-09-01 12:00:00", "session": "LUNCH", "Open": 100.0, "High": 150.0, "Low": 50.0},
            {"DateTime_ET": "2020-09-01 18:00:00", "session": "ASIA", "Open": 100.0, "High": 101.0, "Low": 99.0},
        ]
    )
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])

    result = build_daily_session_sizes(df)

    assert list(result.columns) == ["trading_day", "asia_size"]
    assert result.iloc[0]["asia_size"] == 0.02
```

- [ ] **Step 2: Run test to verify it fails if needed**

Run: `PYTHONPATH=src pytest tests/test_analysis.py::test_build_daily_session_sizes_ignores_non_target_sessions -v`
Expected: FAIL only if session filtering is not already correct; otherwise PASS and no code change needed

- [ ] **Step 3: Refine implementation only if required by the test**

```python
target_sessions = data["session"].isin(SESSION_MAP)
data = data[target_sessions].copy()
```

- [ ] **Step 4: Run full verification and generate outputs**

Run: `PYTHONPATH=src pytest -v`
Expected: PASS

Run: `PYTHONPATH=src python3 scripts/run_analysis.py`
Expected: writes `results/daily_session_sizes.csv` and `results/rolling_correlations.csv`

- [ ] **Step 5: Commit**

```bash
git add src/asia_range/analysis.py tests/test_analysis.py results/daily_session_sizes.csv results/rolling_correlations.csv
git commit -m "feat: generate asia range research outputs"
```
