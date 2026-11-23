"""Metrics used in the validation report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Max gap between the score CDFs of defaulters and non-defaulters."""
    order = np.argsort(score)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def score_metrics(y, score) -> dict:
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "ks": ks_statistic(y, score),
    }


def bootstrap_ci(y, score, metric, n=300, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(metric(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def calibration_table(y, pd_hat, rank_score=None, bins=10) -> pd.DataFrame:
    """Mean predicted vs actual default rate per decile of `rank_score`.

    Isotonic PDs are step-shaped with many ties, so deciles are cut on the
    continuous raw score when it is given.
    """
    rank = pd_hat if rank_score is None else rank_score
    df = pd.DataFrame({"y": y, "pd": pd_hat})
    df["bin"] = pd.qcut(pd.Series(rank).rank(method="first"), bins, labels=False)
    t = df.groupby("bin").agg(predicted=("pd", "mean"), actual=("y", "mean"), loans=("y", "size"))
    return t.reset_index(drop=True)


def brier(y, pd_hat) -> float:
    return brier_score_loss(y, pd_hat)


def psi(expected, actual, bins=10) -> float:
    """Population stability index of `actual` against `expected`'s deciles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def policy_outcome(pd_hat, default, net_cash, funded, threshold) -> dict:
    """Realised result of approving every loan with PD below the threshold."""
    approve = np.asarray(pd_hat) < threshold
    funded = np.asarray(funded, float)
    net_cash = np.asarray(net_cash, float)
    return {
        "threshold": float(threshold),
        "approval_rate": float(approve.mean()),
        "default_rate_approved": float(np.asarray(default)[approve].mean()),
        "net_cash": float(net_cash[approve].sum()),
        "return_on_funded": float(net_cash[approve].sum() / funded[approve].sum()),
    }


def best_threshold(pd_hat, net_cash, grid=None) -> float:
    """Threshold that maximises total realised net cash."""
    grid = np.round(np.arange(0.05, 0.51, 0.01), 2) if grid is None else grid
    pd_hat, net_cash = np.asarray(pd_hat), np.asarray(net_cash, float)
    totals = [net_cash[pd_hat < t].sum() for t in grid]
    return float(grid[int(np.argmax(totals))])
