from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting import plot_nlr_extrapolation_heatmap


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qoi",
        choices=["lift", "drag"],
        required=True,
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the aggregate metrics CSV. If omitted, inferred from --qoi.",
    )
    parser.add_argument(
        "--output-root",
        default="nlr",
        help="Root folder where results/ and plots/ live.",
    )
    parser.add_argument(
        "--model",
        choices=["TD-TF", "TD-DMD"],
        default="TD-TF",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse_mean", "rmse_std"],
        default="rmse_mean",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)

    if args.csv is None:
        csv_path = (
            output_root
            / "results"
            / f"nlr_{args.qoi}_extrapolation_agg_metrics.csv"
        )
    else:
        csv_path = Path(args.csv)

    plot_nlr_extrapolation_heatmap(
        csv_path,
        qoi=args.qoi,
        model=args.model,
        metric=args.metric,
        save_dir=output_root / "plots",
        show=args.show,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()