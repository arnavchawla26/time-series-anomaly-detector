"""Synthetic time series generator with a known ground-truth anomaly set.

Used by the ``demo`` CLI command (and the test suite) to evaluate a
detector's precision/recall without needing a real dataset on hand: the
generator returns exactly which indices it injected as point anomalies, so
whatever the detector flags can be scored against ground truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SyntheticSeries:
    values: List[float]
    timestamps: List[int]
    anomaly_indices: List[int]
    period: int


def generate_series(
    length: int = 200,
    period: int = 24,
    base: float = 50.0,
    amplitude: float = 10.0,
    trend_slope: float = 0.05,
    noise_std: float = 1.0,
    n_anomalies: int = 6,
    anomaly_magnitude: Tuple[float, float] = (6.0, 10.0),
    seed: int = 42,
) -> SyntheticSeries:
    """Build ``base + trend + seasonal + noise``, then inject point anomalies.

    Anomalies are large additive spikes (``anomaly_magnitude`` std-devs of
    noise, sign chosen at random) placed at random indices at least one
    full period away from either edge, so decomposition has enough
    surrounding context on both sides. Fully deterministic given ``seed``.
    """
    if length < 4 * period:
        raise ValueError("length should be at least 4x period for a meaningful demo")
    if n_anomalies < 0:
        raise ValueError("n_anomalies must be >= 0")

    rng = random.Random(seed)

    values: List[float] = []
    for i in range(length):
        seasonal = amplitude * math.sin(2 * math.pi * (i % period) / period)
        trend = trend_slope * i
        noise = rng.gauss(0.0, noise_std)
        values.append(base + trend + seasonal + noise)

    margin = period
    candidates = list(range(margin, length - margin))
    rng.shuffle(candidates)
    anomaly_indices = sorted(candidates[: min(n_anomalies, len(candidates))])

    lo, hi = anomaly_magnitude
    for idx in anomaly_indices:
        sign = rng.choice([-1.0, 1.0])
        magnitude = rng.uniform(lo, hi)
        values[idx] += sign * magnitude * noise_std

    timestamps = list(range(length))
    return SyntheticSeries(
        values=values, timestamps=timestamps, anomaly_indices=anomaly_indices, period=period
    )
