"""Classical additive time series decomposition (moving-average trend +
period-averaged seasonal component), stdlib-only.

This mirrors the textbook "classical decomposition" method (as implemented
by e.g. R's ``decompose()``): the trend is a centered moving average of the
given period, the seasonal component is the average of the detrended values
for each position within the period (normalized to sum to zero), and the
residual is what's left over: ``value - trend - seasonal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


Number = float
MaybeNumber = Optional[float]


def moving_average(values: Sequence[Number], window: int) -> List[MaybeNumber]:
    """Centered moving average of the given window/period.

    Returns a list the same length as ``values`` with ``None`` at the
    positions too close to either edge to compute a full centered window.
    Handles both odd and even windows; for even windows this uses the
    standard "2xM" centering trick so the result lines up with integer
    indices (the same approach classical seasonal decomposition uses for
    an even period, e.g. period=12 for monthly-with-yearly-seasonality
    data).
    """
    n = len(values)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > n:
        return [None] * n

    result: List[MaybeNumber] = [None] * n
    half = window // 2

    if window % 2 == 1:
        for t in range(half, n - half):
            result[t] = sum(values[t - half : t + half + 1]) / window
    else:
        # Centered weighted average: half-weight on the two endpoints of a
        # (window + 1)-wide slice, full weight in between, so the result
        # sits on an integer index rather than a half-step offset.
        for t in range(half, n - half):
            slice_ = values[t - half : t + half + 1]
            s = 0.5 * slice_[0] + sum(slice_[1:-1]) + 0.5 * slice_[-1]
            result[t] = s / window

    return result


@dataclass
class DecomposeResult:
    trend: List[MaybeNumber]
    seasonal: List[Number]
    residual: List[MaybeNumber]
    seasonal_indices: List[Number]  # length == period, sums to ~0


def decompose(values: Sequence[Number], period: int) -> DecomposeResult:
    """Additive decomposition: value = trend + seasonal + residual.

    ``period`` is the number of observations per seasonal cycle (e.g. 24
    for hourly data with a daily cycle, 7 for daily data with a weekly
    cycle). Raises ValueError if there isn't enough data for at least two
    full cycles, since the seasonal average needs repeated observations at
    each position within the period.
    """
    n = len(values)
    if period < 2:
        raise ValueError("period must be >= 2")
    if n < 2 * period:
        raise ValueError(
            f"need at least 2 full periods of data ({2 * period} points) to "
            f"decompose with period={period}, got {n}"
        )

    trend = moving_average(values, period)

    detrended: List[MaybeNumber] = [
        (v - t) if t is not None else None for v, t in zip(values, trend)
    ]

    sums = [0.0] * period
    counts = [0] * period
    for i, d in enumerate(detrended):
        if d is not None:
            slot = i % period
            sums[slot] += d
            counts[slot] += 1

    if any(c == 0 for c in counts):
        raise ValueError("not enough data to estimate every seasonal position")

    seasonal_avg = [s / c for s, c in zip(sums, counts)]
    mean_seasonal = sum(seasonal_avg) / period
    seasonal_indices = [s - mean_seasonal for s in seasonal_avg]

    seasonal = [seasonal_indices[i % period] for i in range(n)]
    residual: List[MaybeNumber] = [
        (v - t - s) if t is not None else None
        for v, t, s in zip(values, trend, seasonal)
    ]

    return DecomposeResult(
        trend=trend, seasonal=seasonal, residual=residual, seasonal_indices=seasonal_indices
    )


def detrend_residuals(values: Sequence[Number], window: int) -> List[MaybeNumber]:
    """Non-seasonal residuals: value minus a centered moving-average trend.

    Useful when the series has no known seasonal period but still drifts
    (a trend) that would otherwise inflate a plain z-score's variance.
    """
    trend = moving_average(values, window)
    return [(v - t) if t is not None else None for v, t in zip(values, trend)]
