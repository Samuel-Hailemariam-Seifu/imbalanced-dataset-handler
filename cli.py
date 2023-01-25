"""CLI: resample a CSV classification dataset."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from imbalanced_handler import ImbalancedDatasetHandler, imbalance_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resample imbalanced classification data (CSV).")
    p.add_argument("input_csv", help="Input CSV path")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")
    p.add_argument("-t", "--target", required=True, help="Name of target column")
    p.add_argument(
        "-s",
        "--strategy",
        default="smote",
        choices=[
            "none",
            "smote",
            "adasyn",
            "random_over",
            "random_under",
            "smote_tomek",
        ],
        help="Resampling strategy",
    )
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_csv(args.input_csv)
    if args.target not in df.columns:
        print(f"Target column {args.target!r} not in columns: {list(df.columns)}", file=sys.stderr)
        return 1

    y = df[args.target]
    X = df.drop(columns=[args.target])
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        print(
            "Non-numeric feature columns require encoding before SMOTE-style resampling: "
            f"{non_numeric}",
            file=sys.stderr,
        )
        return 1

    print("Before:", imbalance_report(y))
    handler = ImbalancedDatasetHandler(
        strategy=args.strategy,
        random_state=args.random_state,
    )
    Xr, yr = handler.fit_resample(X, y)
    out = pd.concat([Xr, yr], axis=1)
    out.to_csv(args.output, index=False)
    print("After:", imbalance_report(yr))
    print(f"Wrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
