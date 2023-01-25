import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from sklearn.exceptions import NotFittedError

from imbalanced_handler import ImbalancedDatasetHandler, imbalance_report


def test_imbalance_report():
    y = np.array([0, 0, 0, 1])
    r = imbalance_report(y)
    assert r["class_counts"] == {0: 3, 1: 1}
    assert r["imbalance_ratio"] == pytest.approx(1 / 3)


def test_none_strategy_no_change():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        weights=[0.9, 0.1],
        random_state=1,
    )
    h = ImbalancedDatasetHandler(strategy="none")
    X2, y2 = h.fit_resample(X, y)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


@pytest.mark.parametrize(
    "strategy",
    ["smote", "adasyn", "random_over", "random_under", "smote_tomek"],
)
def test_resampling_balances_or_changes(strategy: str):
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        weights=[0.85, 0.15],
        random_state=2,
    )
    h = ImbalancedDatasetHandler(strategy=strategy, random_state=0)
    Xr, yr = h.fit_resample(X, y)
    assert len(Xr) == len(yr)
    counts = np.bincount(yr.astype(int))
    assert counts.min() > 0


def test_pandas_preserved():
    X, y = make_classification(n_samples=100, n_features=4, weights=[0.8, 0.2], random_state=3)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    ys = pd.Series(y, name="target")
    h = ImbalancedDatasetHandler(strategy="random_over", random_state=0)
    Xo, yo = h.fit_resample(Xdf, ys)
    assert isinstance(Xo, pd.DataFrame)
    assert isinstance(yo, pd.Series)
    assert list(Xo.columns) == list(Xdf.columns)
    assert yo.name == "target"


def test_resample_without_fit_raises():
    X, y = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=4)
    h = ImbalancedDatasetHandler(strategy="smote", random_state=0)
    with pytest.raises(NotFittedError):
        h.resample(X, y)
