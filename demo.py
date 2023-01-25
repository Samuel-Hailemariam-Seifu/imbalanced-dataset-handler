"""Synthetic demo: imbalance report + SMOTE resampling."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from imbalanced_handler import ImbalancedDatasetHandler, imbalance_report


def main() -> None:
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.92, 0.08],
        random_state=0,
    )
    print("Original:", imbalance_report(y))
    h = ImbalancedDatasetHandler(strategy="smote", random_state=0)
    Xb, yb = h.fit_resample(X, y)
    print("After SMOTE:", imbalance_report(yb))
    print(f"Shapes: {X.shape} -> {Xb.shape}")


if __name__ == "__main__":
    main()
