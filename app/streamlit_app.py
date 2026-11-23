"""Credit risk demo: probability of default for one Lending Club-style application."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data import FEATURES, build_features  # noqa: E402
from credit_risk.model import to_model_frame  # noqa: E402

st.set_page_config(page_title="Credit Risk Scorer", layout="wide")


@st.cache_resource
def load():
    scorer = joblib.load(ROOT / "models" / "credit_scorer.joblib")
    return scorer, shap.TreeExplainer(scorer.booster)


scorer, explainer = load()
cats = scorer.categories

LABELS = {
    "loan_amnt": "Loan amount", "log_annual_inc": "Annual income",
    "loan_to_income": "Loan-to-income ratio", "dti": "Debt-to-income",
    "emp_length_years": "Years employed", "credit_history_months": "Length of credit history",
    "delinq_2yrs": "Delinquencies (2 yrs)", "inq_last_6mths": "Credit enquiries (6 mths)",
    "mths_since_last_delinq": "Months since last delinquency",
    "mths_since_last_record": "Months since last public record", "open_acc": "Open accounts",
    "pub_rec": "Public records", "revol_bal": "Revolving balance",
    "revol_util": "Revolving utilisation", "total_acc": "Total accounts",
    "pub_rec_bankruptcies": "Bankruptcies", "home_ownership": "Home ownership",
    "verification_status": "Income verification", "purpose": "Loan purpose",
    "addr_state": "State",
}

st.title("Credit Risk Scorer")
st.caption(
    "LightGBM trained on 72,566 real Lending Club 36-month loans (2007-2012), calibrated on "
    "late-2013 loans and tested on 136,967 loans from 2014. Uses only information known at "
    "application. A portfolio demo, not a lending tool."
)

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("Application")
    c1, c2 = st.columns(2)
    loan = c1.number_input("Loan amount ($)", 1000, 35000, 12000, step=500)
    income = c2.number_input("Annual income ($)", 4000, 1_000_000, 60000, step=1000)
    dti = c1.number_input("Debt-to-income (%)", 0.0, 40.0, 17.0, step=0.5)
    emp = c2.selectbox("Years employed", ["< 1 year"] + [f"{i} years" for i in range(1, 10)] + ["10+ years"], 6)
    home = c1.selectbox("Home ownership", [c for c in cats["home_ownership"] if c in {"RENT", "MORTGAGE", "OWN"}])
    ver = c2.selectbox("Income verification", cats["verification_status"])
    purpose = c1.selectbox("Purpose", cats["purpose"], cats["purpose"].index("debt_consolidation"))
    state = c2.selectbox("State", cats["addr_state"], cats["addr_state"].index("CA"))
    hist = c1.number_input("Years since first credit line", 1, 50, 14)
    inq = c2.number_input("Credit enquiries, last 6 months", 0, 10, 1)
    delinq = c1.number_input("Delinquencies, last 2 years", 0, 15, 0)
    last_delinq = c2.number_input("Months since last delinquency (0 = never)", 0, 150, 0)
    open_acc = c1.number_input("Open accounts", 0, 60, 10)
    total_acc = c2.number_input("Total accounts", 1, 100, 24)
    revol_bal = c1.number_input("Revolving balance ($)", 0, 500_000, 14000, step=500)
    revol_util = c2.number_input("Revolving utilisation (%)", 0.0, 150.0, 55.0, step=1.0)
    pub_rec = c1.number_input("Public records", 0, 10, 0)
    bankr = c2.number_input("Bankruptcies", 0, 5, 0)
    grade = st.selectbox("Lending Club grade for this loan", list("ABCDEFG"), 1)

issue = pd.Timestamp("2014-06-01")
raw = pd.DataFrame([{
    "issue_d": issue, "loan_amnt": loan, "annual_inc": income, "dti": dti, "emp_length": emp,
    "earliest_cr_line": issue - pd.DateOffset(years=int(hist)), "delinq_2yrs": delinq,
    "inq_last_6mths": inq, "mths_since_last_delinq": last_delinq or np.nan,
    "mths_since_last_record": np.nan if pub_rec == 0 else 60, "open_acc": open_acc,
    "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util / 100,
    "total_acc": total_acc, "pub_rec_bankruptcies": bankr, "home_ownership": home,
    "verification_status": ver, "purpose": purpose, "addr_state": state,
}])
X = build_features(raw)
pd_hat = float(scorer.predict_pd(X)[0])
cut = scorer.grade_cutoffs[grade]
avg = scorer.grade_avg_pd[grade]
flagged = pd_hat >= cut

with right:
    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of default", f"{pd_hat:.1%}")
    m2.metric(f"Average for grade {grade}", f"{avg:.1%}")
    m3.metric("Flag cutoff for this grade", f"{cut:.1%}")
    if flagged:
        st.error(f"Flag: riskier than most grade {grade} loans. In 2014, flagged loans returned "
                 "7.6% over three years against 9.5% for the rest.")
    else:
        st.success(f"No flag: risk is in line with grade {grade}.")

    sv = explainer.shap_values(to_model_frame(X, cats))
    sv = (sv[1] if isinstance(sv, list) else sv)[0]
    contrib = (pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "effect": sv})
               .assign(size=lambda d: d["effect"].abs()).nlargest(8, "size"))
    st.markdown("**What moved this score** (SHAP, log-odds; positive = riskier)")
    st.bar_chart(contrib.set_index("feature")["effect"], horizontal=True, color="#0B6E4F")
    st.caption(
        "The flag compares this applicant with other loans of the same grade. Every grade was "
        "profitable on average in 2014 because riskier loans paid higher interest, so the model "
        "is most useful for spotting loans priced too cheaply for their risk."
    )
