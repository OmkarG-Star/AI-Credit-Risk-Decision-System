"""Fitted scorer: LightGBM probability of default plus isotonic calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from credit_risk.data import CATEGORICAL_FEATURES, FEATURES


def to_model_frame(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Fix categorical levels so training and scoring see identical codes."""
    X = X[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


@dataclass
class CreditScorer:
    booster: lgb.Booster
    calibrator: IsotonicRegression
    categories: dict[str, list[str]]
    grade_cutoffs: dict[str, float]
    grade_avg_pd: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(to_model_frame(X, self.categories))

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        return self.calibrator.predict(self.raw_score(X))

    def flag(self, X: pd.DataFrame, grades) -> np.ndarray:
        """True where the loan is riskier than the riskiest quarter of its LC grade."""
        cut = np.array([self.grade_cutoffs.get(str(g), np.inf) for g in grades])
        return self.predict_pd(X) >= cut
