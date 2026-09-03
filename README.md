# tsanomaly

A dependency-free (Python stdlib only) time series anomaly detection toolkit.
It implements classical additive seasonal decomposition (centered
moving-average trend + period-averaged seasonal component) and scores the
leftover residuals with a choice of z-score, median/MAD "modified z-score",
or IQR ("Tukey's fences") detectors — plus a causal rolling z-score detector
for series with no fixed seasonal period. Includes a `demo` command that
generates a synthetic series with a *known* set of injected anomalies and
reports precision/recall/F1 against that ground truth, so the tool is
verifiable without needing a real dataset on hand.

## Why decomposition first?

Running a plain z-score over a raw series with a daily or weekly cycle
doesn't work well: the seasonal swing itself looks like "spread", so the
threshold either misses real anomalies or flags half the peaks and troughs
as outliers. Decomposing first — subtracting off the trend and the typical
shape of the cycle — leaves a residual that should look like noise around
zero when nothing unusual happened, which is what the point/threshold-based
detectors are actually built to catch.

## Tech stack

Python 3.9+, standard library only (`csv`, `statistics`, `math`, `random`,
`argparse`, `dataclasses`) — no numpy/pandas/scipy. Tests use `pytest`.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### `detect` — decompose a CSV column and flag anomalies

```bash
tsanomaly detect --input readings.csv --value-col reading --timestamp-col hour \
  --period 24 --method zscore
```

Real output, run against a synthetic 5-day hourly temperature series (daily
sinusoidal cycle, Gaussian noise, one injected +12 spike at hour 74):

```
method=zscore points=120 anomalies=1
stats: {'mean': -0.007313585069445144, 'std': 1.1422518471888226, 'threshold': 3.0}
  [74] t=74 value=35.1790 residual=8.7531
```

Pass `--format json` for machine-readable output. Use `--trend-window N`
instead of `--period N` for a series that drifts but has no fixed seasonal
cycle (subtracts a plain moving-average trend, no seasonal step). Choose the
scoring method with `--method {zscore,modified_zscore,iqr}` (defaults:
z-score threshold 3.0, modified-z threshold 3.5, IQR fence multiplier 1.5,
all overridable with `--threshold` / `--k`).

### `rolling-detect` — no decomposition, streaming-style

```bash
tsanomaly rolling-detect --input readings.csv --value-col reading --window 20 --threshold 3.0
```

Compares each point against the mean/std of the `window` points strictly
*before* it (never including itself), so it works online and needs no
seasonal period.

### `demo` — synthetic ground truth + precision/recall/F1

```bash
tsanomaly demo --length 200 --period 24 --anomalies 6 --seed 42 --method zscore
```

Real output:

```
synthetic demo: 200 points, period=24, method=zscore
  injected anomalies : [46, 73, 93, 137, 170, 174]
  detected anomalies : [46, 73, 93, 137, 170, 174]
  precision=1.000 recall=1.000 f1=1.000  (tp=6 fp=0 fn=0)
```

## How it works

1. **Trend**: a centered moving average of length `period` (or
   `--trend-window`). Even windows use the standard "2xM" centering trick so
   the estimate lands on an integer index instead of a half-step offset.
2. **Seasonal** (period mode only): detrend, then average the detrended
   values at each position within the period (e.g. every hour-0, every
   hour-1, ...), normalized so the seasonal indices sum to zero.
3. **Residual**: `value - trend - seasonal` (or `value - trend` in
   trend-only mode).
4. **Detection**: score the residuals with one of:
   - `zscore` — flag `|residual - mean| / std > threshold`.
   - `modified_zscore` — Iglewicz & Hoaglin's median/MAD version, more
     robust when the anomalies themselves are large enough to pull a plain
     mean/std around.
   - `iqr` — Tukey's fences: flag anything outside
     `[Q1 - k*IQR, Q3 + k*IQR]` (linear-interpolation percentiles, matching
     numpy's default method).
   - `rolling_zscore` (via `rolling-detect`) — causal trailing-window
     z-score directly on raw values, no decomposition.

## Current status

Functional v1: decomposition, all four detectors, CSV I/O, a synthetic data
generator with ground-truth anomaly indices, and a `demo` command that
scores a detector's precision/recall/F1 against that ground truth.
39 passing pytest tests covering the moving-average math (including a
hand-verified even-window centering case), decomposition correctness on a
noise-free synthetic signal, every detector's edge cases (`None`-skipping,
zero-variance/zero-MAD series, hand-computed IQR fences), the synthetic
generator's determinism, and CLI end-to-end runs including a
subprocess-level `python -m tsanomaly.cli` check.

Not yet implemented: multiplicative decomposition, STL (loess-based, as
opposed to this moving-average-based classical decomposition), multivariate
series, and change-point / level-shift detection (this only targets point
anomalies, not sustained regime shifts).

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
