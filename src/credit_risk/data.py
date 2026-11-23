"""Load Lending Club loans and build the modelling table.

Only information known when the loan was approved is kept as a feature.
Payment history, recoveries and LC's own pricing (grade, interest rate)
are excluded from the features; grade is kept only as a benchmark column
and realised cash flow only for evaluating approval policies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/lending_club_2007_2017.parquet")

# Columns recorded after the loan was issued. Using any of them as a feature
# leaks the outcome (a charged-off loan has recoveries, a paid one has no
# outstanding principal). They are listed so tests can assert their absence.
LEAKAGE_COLUMNS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "hardship_flag", "debt_settlement_flag",
    "settlement_status", "settlement_amount", "loan_status",
]

# LC's own risk pricing. Not leakage, but it is the lender's existing score,
# so it is used as a benchmark rather than an input to the application model.
LC_PRICING_COLUMNS = ["grade", "sub_grade", "int_rate", "installment"]

NUMERIC_FEATURES = [
    "loan_amnt", "log_annual_inc", "loan_to_income", "dti", "emp_length_years",
    "credit_history_months", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_COLUMNS = [
    "issue_d", "term", "loan_status", "loan_amnt", "funded_amnt", "annual_inc",
    "dti", "emp_length", "earliest_cr_line", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies",
    "home_ownership", "verification_status", "purpose", "addr_state",
    "sub_grade", "grade", "int_rate", "total_pymnt",
]

# 36-month loans issued up to Oct 2014 had reached maturity by the Nov 2017
# data snapshot, so their outcome is known.
LAST_ISSUE_DATE = "2014-10-01"

SPLITS = {
    "train": ("2007-01-01", "2012-12-01"),
    "early_stop": ("2013-01-01", "2013-06-01"),
    "calibrate": ("2013-07-01", "2013-12-01"),
    "test": ("2014-01-01", LAST_ISSUE_DATE),
}


def _emp_years(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    years = s.str.extract(r"(\d+)")[0].astype(float)
    return years.where(~s.str.contains("<", na=False), 0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw application fields (works on one row too)."""
    out = pd.DataFrame(index=df.index)
    annual_inc = pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=1)
    out["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    out["log_annual_inc"] = np.log(annual_inc)
    out["loan_to_income"] = out["loan_amnt"] / annual_inc
    out["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    out["emp_length_years"] = _emp_years(df["emp_length"])
    issue = pd.to_datetime(df["issue_d"])
    first_line = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    out["credit_history_months"] = (issue - first_line).dt.days / 30.44
    for col in [
        "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
        "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
        "revol_util", "total_acc", "pub_rec_bankruptcies",
    ]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    # 2007-08 files store "never" as 0 months. Zero months since a record (or a
    # delinquency) is impossible when the borrower has none on file, so treat
    # those zeros as missing, matching how later years code "never".
    out.loc[(out["mths_since_last_record"] == 0) & (out["pub_rec"] == 0), "mths_since_last_record"] = np.nan
    out.loc[(out["mths_since_last_delinq"] == 0) & (out["delinq_2yrs"] == 0), "mths_since_last_delinq"] = np.nan
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string").fillna("missing").astype(str)
    return out[FEATURES]


def load_modelling_table(path: Path = RAW_PATH) -> pd.DataFrame:
    """Matured 36-month loans with a known outcome, plus split labels."""
    df = pd.read_parquet(path, columns=RAW_COLUMNS)
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df = df[(df["term"].astype(str).str.strip() == "36") & (df["issue_d"] <= LAST_ISSUE_DATE)]
    # "Does not meet the credit policy" loans were approved under an older
    # policy and the rest are still unresolved; both are left out.
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["default"] = (df["loan_status"] != "Fully Paid").astype(int)
    df["net_cash"] = pd.to_numeric(df["total_pymnt"]) - pd.to_numeric(df["funded_amnt"])
    grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
    df["sub_grade_rank"] = df["sub_grade"].astype(str).map({g: i for i, g in enumerate(grades)})
    df["split"] = None
    for name, (start, end) in SPLITS.items():
        mask = df["issue_d"].between(start, end)
        df.loc[mask, "split"] = name
    return df.reset_index(drop=True)
