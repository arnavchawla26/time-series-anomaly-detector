import pytest

from tsanomaly.detectors import (
    iqr_flags,
    modified_zscore_flags,
    rolling_zscore_flags,
    zscore_flags,
)


def test_zscore_flags_isolates_single_outlier():
    residuals = [0.1, -0.2, 0.05, -0.1, 0.15, -0.05, 0.2, -0.15, 50.0, 0.1, -0.1, 0.05]
    flags, stats = zscore_flags(residuals, threshold=3.0)
    assert flags[8] is True
    assert sum(flags) == 1
    assert stats["mean"] == pytest.approx(sum(residuals) / len(residuals))


def test_zscore_flags_skips_none_positions():
    residuals = [None, 1.0, 1.1, 0.9, None, 1.0, 20.0, 1.0]
    flags, _ = zscore_flags(residuals, threshold=2.0)
    assert flags[0] is False
    assert flags[4] is False
    assert flags[6] is True


def test_zscore_flags_constant_series_flags_nothing():
    residuals = [5.0] * 10
    flags, stats = zscore_flags(residuals, threshold=3.0)
    assert not any(flags)
    assert stats["std"] == 0.0


def test_zscore_flags_requires_enough_data():
    with pytest.raises(ValueError):
        zscore_flags([1.0], threshold=3.0)


def test_modified_zscore_flags_isolates_outlier_with_zero_mad():
    residuals = [5.0, 5.0, 5.0, 5.0, 10.0]
    flags, stats = modified_zscore_flags(residuals)
    assert flags == [False, False, False, False, True]
    assert stats["mad"] == 0.0
    assert stats["median"] == 5.0


def test_modified_zscore_flags_typical_case():
    residuals = [1.0, 2.0, 1.5, 2.5, 1.8, 40.0, 2.2, 1.9]
    flags, _ = modified_zscore_flags(residuals, threshold=3.5)
    assert flags[5] is True
    assert sum(flags) == 1


def test_iqr_flags_matches_hand_computed_fences():
    residuals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    flags, stats = iqr_flags(residuals, k=1.5)
    assert stats["q1"] == pytest.approx(3.25)
    assert stats["q3"] == pytest.approx(7.75)
    assert stats["lower"] == pytest.approx(-3.5)
    assert stats["upper"] == pytest.approx(14.5)
    assert flags == [False] * 9 + [True]


def test_iqr_flags_skips_none():
    residuals = [None, 1.0, 2.0, 3.0, 100.0, None]
    flags, _ = iqr_flags(residuals, k=1.5)
    assert flags[0] is False
    assert flags[-1] is False


def test_rolling_zscore_flags_early_points_never_flagged():
    values = [10.0] * 10 + [1000.0] + [10.0] * 5
    flags, stats = rolling_zscore_flags(values, window=5, threshold=3.0)
    assert flags[:5] == [False] * 5
    assert stats[0] is None
    assert flags[10] is True  # the spike, once there's a full window of flat history


def test_rolling_zscore_flags_constant_series_flags_nothing():
    values = [3.0] * 30
    flags, _ = rolling_zscore_flags(values, window=5, threshold=3.0)
    assert not any(flags)


def test_rolling_zscore_flags_rejects_small_window():
    with pytest.raises(ValueError):
        rolling_zscore_flags([1.0, 2.0, 3.0], window=1)
