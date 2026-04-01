# Asia Session Range Predictiveness Design

## Goal

Measure whether the normalized size of the Asia session range is predictive of the normalized size of the later London and New York morning sessions.

This is a research-first study. It is not trying to build an ex-ante model yet. The first pass is descriptive: compute normalized session sizes, align them by trading day, and evaluate rolling correlations.

## Dataset

- Input file: `data/nq_1m.parquet`
- Relevant columns observed in the dataset:
  - `DateTime_ET`
  - `session`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`

The dataset already contains Eastern Time timestamps, so session slicing should be done directly from `DateTime_ET`.

## Trading Day Definition

Trading day runs from `18:00 ET` to `17:00 ET` the following calendar day.

Implication:

- Asia `2020-08-31 18:00-22:00 ET` belongs to trading day `2020-09-01`
- London and New York sessions that occur later in that same overnight/daytime cycle also belong to trading day `2020-09-01`

This trading-day key is required so one Asia session is paired with the later London and New York sessions from the same futures trading day.

## Session Definitions

Use these Eastern Time windows:

- Asia: `18:00-22:00 ET`
- London: `02:00-05:00 ET`
- New York morning: `09:30-11:00 ET`

These windows should be interpreted as start-inclusive and end-exclusive when working with 1-minute bars:

- Asia: `18:00 <= t < 22:00`
- London: `02:00 <= t < 05:00`
- New York morning: `09:30 <= t < 11:00`

Verification against the parquet dataset showed that the vendor `session` labels do not match these windows, so implementation must derive sessions directly from `DateTime_ET` rather than reusing the dataset `session` column for aggregation.

## Session Size Metric

For each trading day and each target session, compute normalized session size as:

`session_size = (session_high - session_low) / session_open`

Where:

- `session_high` is the maximum `High` across bars in the session
- `session_low` is the minimum `Low` across bars in the session
- `session_open` is the `Open` of the first bar in the session

This version intentionally avoids:

- ATR normalization
- logarithmic normalization
- slope or angle transforms

The purpose of this normalization is to make session sizes comparable across large changes in NQ price level over time.

## Research Table

Build one row per trading day with these columns:

- `trading_day`
- `asia_size`
- `london_size`
- `nyam_size`

Each value is the normalized session size for that trading day and session.

## Missing Data Handling

Session pairs should be evaluated only on complete observations.

Rules:

- If a trading day is missing `ASIA`, `LONDON`, or `NYAM`, keep the daily research table as-is with missing values
- When computing `ASIA` vs `LONDON`, use only rows where both `asia_size` and `london_size` are present
- When computing `ASIA` vs `NYAM`, use only rows where both `asia_size` and `nyam_size` are present

This avoids silently mixing incomplete days into the rolling statistics.

## Rolling Correlation

Use a rolling lookback window of `40` trading days.

Compute two rolling Spearman correlations:

- `corr_asia_london_40`
- `corr_asia_nyam_40`

Spearman is preferred over Pearson for the first pass because the research question is whether larger Asia ranges tend to coincide with larger later-session ranges, without requiring that the relationship be strictly linear.

## Outputs

Produce two research outputs:

1. Daily normalized session-size table with:
   - `trading_day`
   - `asia_size`
   - `london_size`
   - `nyam_size`
2. Rolling correlation table with:
   - `trading_day`
   - `corr_asia_london_40`
   - `corr_asia_nyam_40`

If useful during implementation, both outputs can be written to CSV or parquet in a `results/` directory.

## Verification

Implementation should verify the following before treating results as valid:

- Session aggregation matches expected time windows in Eastern Time
- Asia bars are assigned to the following trading day based on the `18:00 ET` boundary
- Session bar counts are roughly consistent with the session templates
- The first few trading days align correctly when manually spot-checked
- A direct hand-check on one 40-trading-day sample matches the rolling correlation output

## Scope Limits

This design does not yet include:

- predictive modeling beyond rolling correlation
- binary expansion/consolidation labels
- optimization of thresholds
- volatility adjustments
- path-based or slope-based geometry metrics

Those can be revisited only after this first-pass relationship study is validated.
