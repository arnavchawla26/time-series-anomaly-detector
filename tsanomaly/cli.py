"""``tsanomaly`` command-line interface.

Subcommands:
  detect          Decompose + flag anomalies in a CSV column.
  rolling-detect  Streaming-style rolling z-score on raw values (no decomposition).
  demo            Generate a synthetic series with known injected anomalies,
                   run a detector, and report precision/recall/F1 against the
                   ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import decompose, detectors, io_utils, synthetic

DEFAULT_THRESHOLDS = {"zscore": 3.0, "modified_zscore": 3.5}


def _run_detector(
    residuals: Sequence[Optional[float]],
    method: str,
    threshold: Optional[float],
    k: float,
):
    if method == "zscore":
        return detectors.zscore_flags(residuals, threshold if threshold is not None else 3.0)
    if method == "modified_zscore":
        return detectors.modified_zscore_flags(
            residuals, threshold if threshold is not None else 3.5
        )
    if method == "iqr":
        return detectors.iqr_flags(residuals, k)
    raise ValueError(f"unknown method {method!r}")


def _decompose_and_detect(
    values: Sequence[float],
    period: Optional[int],
    trend_window: Optional[int],
    method: str,
    threshold: Optional[float],
    k: float,
):
    if period is not None:
        result = decompose.decompose(values, period)
        residuals: List[Optional[float]] = result.residual
    elif trend_window is not None:
        residuals = decompose.detrend_residuals(values, trend_window)
    else:
        residuals = list(values)
    flags, stats = _run_detector(residuals, method, threshold, k)
    return residuals, flags, stats


def _print_text_report(values, timestamps, residuals, flags, stats, method):
    anomaly_indices = [i for i, f in enumerate(flags) if f]
    print(f"method={method} points={len(values)} anomalies={len(anomaly_indices)}")
    print(f"stats: {stats}")
    for i in anomaly_indices:
        ts = timestamps[i] if timestamps and timestamps[i] is not None else i
        r = residuals[i]
        r_str = f"{r:.4f}" if r is not None else "n/a"
        print(f"  [{i}] t={ts} value={values[i]:.4f} residual={r_str}")


def _build_output(values, timestamps, residuals, flags, stats, method) -> Dict[str, Any]:
    anomaly_indices = [i for i, f in enumerate(flags) if f]
    return {
        "method": method,
        "n_points": len(values),
        "stats": stats,
        "anomaly_count": len(anomaly_indices),
        "anomalies": [
            {
                "index": i,
                "timestamp": timestamps[i] if timestamps else None,
                "value": values[i],
                "residual": residuals[i],
            }
            for i in anomaly_indices
        ],
    }


def cmd_detect(args: argparse.Namespace) -> int:
    values, timestamps = io_utils.load_series(args.input, args.value_col, args.timestamp_col)
    residuals, flags, stats = _decompose_and_detect(
        values, args.period, args.trend_window, args.method, args.threshold, args.k
    )
    if args.format == "json":
        print(json.dumps(_build_output(values, timestamps, residuals, flags, stats, args.method), indent=2))
    else:
        _print_text_report(values, timestamps, residuals, flags, stats, args.method)
    return 0


def cmd_rolling_detect(args: argparse.Namespace) -> int:
    values, timestamps = io_utils.load_series(args.input, args.value_col, args.timestamp_col)
    flags, stats_list = detectors.rolling_zscore_flags(values, args.window, args.threshold)
    anomaly_indices = [i for i, f in enumerate(flags) if f]
    if args.format == "json":
        output = {
            "method": "rolling_zscore",
            "window": args.window,
            "threshold": args.threshold,
            "n_points": len(values),
            "anomaly_count": len(anomaly_indices),
            "anomalies": [
                {
                    "index": i,
                    "timestamp": timestamps[i] if timestamps else None,
                    "value": values[i],
                    "stats": stats_list[i],
                }
                for i in anomaly_indices
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(
            f"method=rolling_zscore window={args.window} points={len(values)} "
            f"anomalies={len(anomaly_indices)}"
        )
        for i in anomaly_indices:
            ts = timestamps[i] if timestamps and timestamps[i] is not None else i
            print(f"  [{i}] t={ts} value={values[i]:.4f} stats={stats_list[i]}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    series = synthetic.generate_series(
        length=args.length,
        period=args.period,
        n_anomalies=args.anomalies,
        seed=args.seed,
    )
    residuals, flags, stats = _decompose_and_detect(
        series.values, args.period, None, args.method, args.threshold, args.k
    )
    predicted = {i for i, f in enumerate(flags) if f}
    truth = set(series.anomaly_indices)

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    output = {
        "method": args.method,
        "n_points": len(series.values),
        "period": series.period,
        "true_anomaly_indices": sorted(truth),
        "predicted_anomaly_indices": sorted(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "stats": stats,
    }

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(f"synthetic demo: {len(series.values)} points, period={series.period}, method={args.method}")
        print(f"  injected anomalies : {sorted(truth)}")
        print(f"  detected anomalies : {sorted(predicted)}")
        print(f"  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsanomaly",
        description="Dependency-free time series anomaly detection: classical seasonal "
        "decomposition plus z-score / modified-z-score / IQR / rolling-z-score detectors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_io = argparse.ArgumentParser(add_help=False)
    common_io.add_argument("--input", required=True, help="CSV file with the series")
    common_io.add_argument("--value-col", required=True, help="name of the value column")
    common_io.add_argument("--timestamp-col", default=None, help="name of an optional timestamp column")
    common_io.add_argument("--format", choices=["text", "json"], default="text")

    p_detect = sub.add_parser(
        "detect", parents=[common_io], help="decompose (seasonal or trend-only) and flag anomalies"
    )
    group = p_detect.add_mutually_exclusive_group()
    group.add_argument("--period", type=int, default=None, help="seasonal period, e.g. 24 for hourly/daily")
    group.add_argument("--trend-window", type=int, default=None, help="moving-average window for a non-seasonal trend")
    p_detect.add_argument("--method", choices=["zscore", "modified_zscore", "iqr"], default="zscore")
    p_detect.add_argument("--threshold", type=float, default=None, help="z-score threshold (method-specific default if omitted)")
    p_detect.add_argument("--k", type=float, default=1.5, help="IQR fence multiplier (method=iqr only)")
    p_detect.set_defaults(func=cmd_detect)

    p_rolling = sub.add_parser(
        "rolling-detect", parents=[common_io], help="causal rolling z-score, no decomposition needed"
    )
    p_rolling.add_argument("--window", type=int, default=20)
    p_rolling.add_argument("--threshold", type=float, default=3.0)
    p_rolling.set_defaults(func=cmd_rolling_detect)

    p_demo = sub.add_parser(
        "demo", help="generate a synthetic series with known anomalies and score a detector against it"
    )
    p_demo.add_argument("--length", type=int, default=200)
    p_demo.add_argument("--period", type=int, default=24)
    p_demo.add_argument("--anomalies", type=int, default=6)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--method", choices=["zscore", "modified_zscore", "iqr"], default="zscore")
    p_demo.add_argument("--threshold", type=float, default=None)
    p_demo.add_argument("--k", type=float, default=1.5)
    p_demo.add_argument("--format", choices=["text", "json"], default="text")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
