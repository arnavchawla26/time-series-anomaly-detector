"""Minimal CSV loading, stdlib ``csv`` module only."""

from __future__ import annotations

import csv
from typing import List, Optional, Tuple


def load_series(
    path: str, value_col: str, timestamp_col: Optional[str] = None
) -> Tuple[List[float], List[Optional[str]]]:
    """Load a value column (and optional timestamp column) from a CSV file.

    Raises ``KeyError`` if the requested column isn't present, and
    ``ValueError`` if a value can't be parsed as a float.
    """
    values: List[float] = []
    timestamps: List[Optional[str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or value_col not in reader.fieldnames:
            raise KeyError(f"column {value_col!r} not found in {path}")
        if timestamp_col is not None and timestamp_col not in reader.fieldnames:
            raise KeyError(f"column {timestamp_col!r} not found in {path}")
        for row_num, row in enumerate(reader, start=2):
            raw = row[value_col]
            try:
                values.append(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row_num}: could not parse {value_col!r}={raw!r} as a float"
                ) from exc
            timestamps.append(row[timestamp_col] if timestamp_col else None)
    if not values:
        raise ValueError(f"{path}: no data rows found")
    return values, timestamps
