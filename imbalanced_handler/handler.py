from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

# Resamplers that accept sampling_strategy and random_state where applicable
StrategyName = str


def _build_resampler(
    strategy: StrategyName,
    random_state: int | None,
    sampling_strategy: str | float | dict[Any, int] | None,
    **kwargs: Any,
) -> BaseEstimator:
    common: dict[str, Any] = {}
    if random_state is not None:
        common["random_state"] = random_state
    if sampling_strategy is not None:
        common["sampling_strategy"] = sampling_strategy
    merged = {**common, **kwargs}

    if strategy == "none":
        raise ValueError("strategy 'none' does not build a resampler")
    if strategy == "smote":
        return SMOTE(**merged)
    if strategy == "adasyn":
        return ADASYN(**merged)
    if strategy == "random_over":
        return RandomOverSampler(**merged)
    if strategy == "random_under":
        return RandomUnderSampler(**merged)
    if strategy == "smote_tomek":
        return SMOTETomek(**merged)
    raise ValueError(
        f"Unknown strategy {strategy!r}. Use one of: smote, adasyn, random_over, "
        "random_under, smote_tomek, none."
    )


class ImbalancedDatasetHandler(BaseEstimator):
    """
    Fit/transform interface for resampling imbalanced classification data.

    ``strategy``:
      - ``none``: no resampling (pass-through in ``fit_resample``)
      - ``smote``, ``adasyn``, ``random_over``: oversample minority
      - ``random_under``: undersample majority
      - ``smote_tomek``: SMOTE + Tomek links cleaning
    """

    _STRATEGIES = frozenset(
        {"none", "smote", "adasyn", "random_over", "random_under", "smote_tomek"}
    )

    def __init__(
        self,
        strategy: StrategyName = "smote",
        *,
        random_state: int | None = 42,
        sampling_strategy: str | float | dict[Any, int] | None = "auto",
        **resampler_kwargs: Any,
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.resampler_kwargs = resampler_kwargs

    def _get_resampler(self) -> BaseEstimator | None:
        if self.strategy == "none":
            return None
        return _build_resampler(
            self.strategy,
            self.random_state,
            self.sampling_strategy,
            **self.resampler_kwargs,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ImbalancedDatasetHandler:
        if self.strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(self._STRATEGIES)}")
        self._resampler = self._get_resampler()
        self._fitted = True
        return self

    def fit_resample(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        """Return resampled X, y. Preserves DataFrame/Series types when input is pandas."""
        self.fit(X, y)
        return self.resample(X, y)

    def resample(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.Series]:
        if self.strategy == "none":
            return X, y

        if not getattr(self, "_fitted", False):
            raise NotFittedError(
                "ImbalancedDatasetHandler is not fitted; call fit() before resample()."
            )
        assert self._resampler is not None

        x_is_df = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)

        X_arr = X.to_numpy() if x_is_df else check_array(X, accept_sparse=True)
        y_arr = y.to_numpy() if y_is_series else np.asarray(y).ravel()

        X_out, y_out = self._resampler.fit_resample(X_arr, y_arr)

        if x_is_df:
            X_out = pd.DataFrame(X_out, columns=X.columns, index=None)
        if y_is_series:
            y_out = pd.Series(y_out, name=y.name)

        return X_out, y_out
