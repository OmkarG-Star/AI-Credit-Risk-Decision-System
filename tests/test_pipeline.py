import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, LC_PRICING_COLUMNS, LEAKAGE_COLUMNS, SPLITS, build_features
from credit_risk.evaluate import best_threshold, ks_statistic, psi


def _row(**overrides):
    base = {
        "issue_d": "2012-05-01", "loan_amnt": 10000, "annual_inc": 50000, "dti": 15.0,
        "emp_length": "3 years", "earliest_cr_line": "2002-05-01", "delinq_2yrs": 0,
        "inq_last_6mths": 1, "mths_since_last_delinq": np.nan, "mths_since_last_record": np.nan,
        "open_acc": 8, "pub_rec": 0, "revol_bal": 5000, "revol_util": 0.4, "total_acc": 20,
        "pub_rec_bankruptcies": 0, "home_ownership": "RENT", "verification_status": "Verified",
        "purpose": "credit_card", "addr_state": "CA",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_no_leakage_or_pricing_columns_in_features():
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS)
    assert not set(FEATURES) & set(LC_PRICING_COLUMNS)


def test_splits_are_in_time_order_and_do_not_overlap():
    bounds = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in SPLITS.values()]
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end < start


def test_features_from_one_application():
    X = build_features(_row())
    assert list(X.columns) == FEATURES
    assert X.loc[0, "credit_history_months"] == pytest.approx(120, abs=1)
    assert X.loc[0, "loan_to_income"] == pytest.approx(0.2)
    assert X.loc[0, "emp_length_years"] == 3


def test_zero_months_with_no_record_is_treated_as_never():
    X = build_features(_row(mths_since_last_record=0, pub_rec=0))
    assert np.isnan(X.loc[0, "mths_since_last_record"])
    X = build_features(_row(mths_since_last_record=0, pub_rec=1))
    assert X.loc[0, "mths_since_last_record"] == 0


def test_less_than_one_year_employment_is_zero():
    assert build_features(_row(emp_length="< 1 year")).loc[0, "emp_length_years"] == 0


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    assert ks_statistic(y, rng.random(20000)) < 0.03


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(1).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_best_threshold_picks_the_profitable_cut():
    pd_hat = np.array([0.05, 0.1, 0.3, 0.4])
    cash = np.array([100, 100, -500, -500])
    assert 0.1 < best_threshold(pd_hat, cash) <= 0.3
