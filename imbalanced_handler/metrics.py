from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def imbalance_report(y: np.ndarray | pd.Series) -> dict[str, Any]:
    """Summarize class counts and imbalance ratio (minority / majority)."""
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    y = np.asarray(y).ravel()
    counts = Counter(y.tolist())
    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    minority_count = sorted_items[0][1]
    majority_count = sorted_items[-1][1]
    ratio = float(minority_count / majority_count) if majority_count else 0.0
    return {
        "class_counts": dict(counts),
        "n_classes": len(counts),
        "minority_class": sorted_items[0][0],
        "majority_class": sorted_items[-1][0],
        "imbalance_ratio": ratio,
    }
