import pytest

from tsanomaly.io_utils import load_series


def test_load_series_reads_values_and_timestamps(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ts,value\n2024-01-01,1.5\n2024-01-02,2.5\n2024-01-03,3.5\n")

    values, timestamps = load_series(str(csv_path), value_col="value", timestamp_col="ts")
    assert values == [1.5, 2.5, 3.5]
    assert timestamps == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_load_series_without_timestamp_column(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\n2\n3\n")
    values, timestamps = load_series(str(csv_path), value_col="value")
    assert values == [1.0, 2.0, 3.0]
    assert timestamps == [None, None, None]


def test_load_series_missing_value_column_raises(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\n2\n")
    with pytest.raises(KeyError):
        load_series(str(csv_path), value_col="missing")


def test_load_series_missing_timestamp_column_raises(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\n2\n")
    with pytest.raises(KeyError):
        load_series(str(csv_path), value_col="value", timestamp_col="missing")


def test_load_series_bad_float_raises_with_row_number(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n1\nnot-a-number\n3\n")
    with pytest.raises(ValueError) as exc_info:
        load_series(str(csv_path), value_col="value")
    assert "series.csv:3" in str(exc_info.value)


def test_load_series_empty_file_raises(tmp_path):
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("value\n")
    with pytest.raises(ValueError):
        load_series(str(csv_path), value_col="value")
