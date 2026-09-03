import pytest

from tsanomaly.synthetic import generate_series


def test_generate_series_is_deterministic_given_seed():
    a = generate_series(length=200, period=24, n_anomalies=5, seed=7)
    b = generate_series(length=200, period=24, n_anomalies=5, seed=7)
    assert a.values == b.values
    assert a.anomaly_indices == b.anomaly_indices


def test_generate_series_different_seeds_differ():
    a = generate_series(length=200, period=24, n_anomalies=5, seed=1)
    b = generate_series(length=200, period=24, n_anomalies=5, seed=2)
    assert a.values != b.values


def test_generate_series_anomaly_count_and_bounds():
    period = 24
    length = 200
    series = generate_series(length=length, period=period, n_anomalies=6, seed=42)
    assert len(series.anomaly_indices) == 6
    assert len(series.values) == length
    assert len(series.timestamps) == length
    assert series.timestamps == list(range(length))
    # anomalies should stay clear of the edges (one full period margin)
    for idx in series.anomaly_indices:
        assert period <= idx < length - period
    assert series.anomaly_indices == sorted(set(series.anomaly_indices))  # sorted, unique


def test_generate_series_rejects_too_short_length():
    with pytest.raises(ValueError):
        generate_series(length=10, period=24, n_anomalies=1, seed=1)


def test_generate_series_zero_anomalies():
    series = generate_series(length=200, period=24, n_anomalies=0, seed=1)
    assert series.anomaly_indices == []
