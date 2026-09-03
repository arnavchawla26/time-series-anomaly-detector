import math

import pytest

from tsanomaly.decompose import decompose, moving_average, detrend_residuals


def test_moving_average_odd_window_on_linear_sequence():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ma = moving_average(values, 3)
    assert ma[0] is None
    assert ma[-1] is None
    for t in range(1, 8):
        assert ma[t] == pytest.approx(values[t])  # centered MA of a line recovers the line


def test_moving_average_even_window_recovers_linear_sequence():
    values = [float(i) for i in range(1, 10)]
    ma = moving_average(values, 4)
    assert ma[0] is None and ma[1] is None
    assert ma[-1] is None and ma[-2] is None
    for t in range(2, 7):
        assert ma[t] == pytest.approx(values[t])


def test_moving_average_window_larger_than_series_is_all_none():
    ma = moving_average([1, 2, 3], 10)
    assert ma == [None, None, None]


def test_moving_average_rejects_invalid_window():
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], 0)


def test_decompose_recovers_clean_seasonal_signal_with_no_noise():
    period = 8
    n = 64
    base = 10.0
    amplitude = 5.0
    slope = 0.25
    values = [
        base + slope * i + amplitude * math.sin(2 * math.pi * (i % period) / period)
        for i in range(n)
    ]

    result = decompose(values, period)

    # Seasonal indices should sum to ~0 by construction (additive normalization).
    assert sum(result.seasonal_indices) == pytest.approx(0.0, abs=1e-9)

    # With a clean, noise-free signal, residuals at every computable position
    # should be tiny (only floating point error).
    residual_magnitudes = [abs(r) for r in result.residual if r is not None]
    assert residual_magnitudes, "expected some non-None residuals"
    assert max(residual_magnitudes) < 1e-6


def test_decompose_even_period_also_recovers_clean_signal():
    period = 6
    n = 60
    amplitude = 3.0
    values = [
        20.0 + 0.1 * i + amplitude * math.sin(2 * math.pi * (i % period) / period)
        for i in range(n)
    ]
    result = decompose(values, period)
    residual_magnitudes = [abs(r) for r in result.residual if r is not None]
    assert max(residual_magnitudes) < 1e-6


def test_decompose_flags_a_single_injected_spike():
    period = 8
    n = 64
    values = [
        10.0 + 5.0 * math.sin(2 * math.pi * (i % period) / period) for i in range(n)
    ]
    values[32] += 25.0  # obvious spike, far from the moving-average edges

    result = decompose(values, period)
    assert result.residual[32] is not None
    assert abs(result.residual[32]) > 10.0  # the spike should dominate the residual there

    other_residuals = [
        abs(r) for i, r in enumerate(result.residual) if r is not None and i != 32
    ]
    assert max(other_residuals) < abs(result.residual[32])


def test_decompose_rejects_short_series():
    with pytest.raises(ValueError):
        decompose([1.0, 2.0, 3.0], 4)


def test_decompose_rejects_period_below_2():
    with pytest.raises(ValueError):
        decompose([1.0] * 20, 1)


def test_detrend_residuals_matches_value_minus_moving_average():
    values = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0]
    window = 3
    residuals = detrend_residuals(values, window)
    ma = moving_average(values, window)
    for v, t, r in zip(values, ma, residuals):
        if t is None:
            assert r is None
        else:
            assert r == pytest.approx(v - t)
