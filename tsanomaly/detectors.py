"""Residual-based anomaly detectors, stdlib-only.

Every detector takes a sequence that may contain ``None`` (positions where
no residual could be computed, e.g. the edges of a moving-average window)
and returns ``(flags, stats)`` where ``flags`` is a list of booleans the
same length as the input (``None`` positions are never flagged) and
``stats`` is a dict of the parameters the detector fit, for reporting.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

MaybeNumber = Optional[float]


def _valid(values: Sequence[MaybeNumber]) -> List[float]:
    return [v for v in values if v is not None]


def zscore_flags(
    residuals: Sequence[MaybeNumber], threshold: float = 3.0
) -> Tuple[List[bool], Dict[str, float]]:
    """Flag points whose (mean, population-std) z-score exceeds ``threshold``."""
    valid = _valid(residuals)
    if len(valid) < 2:
        raise ValueError("need at least 2 non-missing residuals")
    mean = sum(valid) / len(valid)
    variance = sum((v - mean) ** 2 for v in valid) / len(valid)
    std = math.sqrt(variance)

    flags: List[bool] = []
    for r in residuals:
        if r is None or std == 0:
            # std == 0 means every valid residual equals the mean, so
            # there's nothing to flag as an outlier.
            flags.append(False)
        else:
            flags.append(abs((r - mean) / std) > threshold)
    return flags, {"mean": mean, "std": std, "threshold": threshold}


def modified_zscore_flags(
    residuals: Sequence[MaybeNumber], threshold: float = 3.5
) -> Tuple[List[bool], Dict[str, float]]:
    """Iglewicz & Hoaglin's median/MAD-based modified z-score.

    More robust than :func:`zscore_flags` when the anomalies themselves are
    large enough to drag a plain mean/std estimate around.
    """
    valid = _valid(residuals)
    if len(valid) < 2:
        raise ValueError("need at least 2 non-missing residuals")
    median = statistics.median(valid)
    mad = statistics.median(abs(v - median) for v in valid)

    flags: List[bool] = []
    for r in residuals:
        if r is None:
            flags.append(False)
        elif mad == 0:
            flags.append(r != median)
        else:
            mz = 0.6745 * (r - median) / mad
            flags.append(abs(mz) > threshold)
    return flags, {"median": median, "mad": mad, "threshold": threshold}


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear' method)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("cannot take a percentile of an empty sequence")
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def iqr_flags(
    residuals: Sequence[MaybeNumber], k: float = 1.5
) -> Tuple[List[bool], Dict[str, float]]:
    """Tukey's fences: flag points outside ``[Q1 - k*IQR, Q3 + k*IQR]``."""
    valid = sorted(_valid(residuals))
    if len(valid) < 2:
        raise ValueError("need at least 2 non-missing residuals")
    q1 = _percentile(valid, 25)
    q3 = _percentile(valid, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    flags = [(r is not None) and (r < lower or r > upper) for r in residuals]
    return flags, {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper, "k": k}


def rolling_zscore_flags(
    values: Sequence[float], window: int = 20, threshold: float = 3.0
) -> Tuple[List[bool], List[Optional[Dict[str, float]]]]:
    """Causal (trailing-window) rolling z-score, for series with no fixed period.

    Each point is compared against the mean/std of the ``window`` points
    *before* it (never including itself), so this works as a streaming /
    online detector and doesn't require decomposition. The first ``window``
    points can't be scored (not enough history) and are never flagged.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    n = len(values)
    flags: List[bool] = []
    stats: List[Optional[Dict[str, float]]] = []
    for i in range(n):
        start = max(0, i - window)
        history = values[start:i]
        if len(history) < window:
            flags.append(False)
            stats.append(None)
            continue
        mean = sum(history) / len(history)
        variance = sum((v - mean) ** 2 for v in history) / len(history)
        std = math.sqrt(variance)
        if std == 0:
            flags.append(values[i] != mean)
        else:
            z = (values[i] - mean) / std
            flags.append(abs(z) > threshold)
        stats.append({"mean": mean, "std": std})
    return flags, stats
