import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="AI Credit Risk Decision System",
    page_icon="💳",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E3A8A;
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin-bottom: 12px;
}
.card {
    background-color: #F8FAFC;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    margin-bottom: 12px;
}
.stButton>button {
    background: linear-gradient(90deg, #2563EB, #4F46E5);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #4338CA);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

model = load_model()

# =========================
# Helper Functions
# =========================
def create_input_dataframe(
    gender, married, dependents, education, self_employed,
    applicant_income, coapplicant_income, loan_amount,
    loan_amount_term, credit_history, property_area
):
    """Create input dataframe with the same features used during model training."""
    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / (loan_amount + 1)
    log_total_income = np.log1p(total_income)
    log_loan_amount = np.log1p(loan_amount)

    return pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "total_income": total_income,
        "income_to_loan_ratio": income_to_loan_ratio,
        "log_total_income": log_total_income,
        "log_loan_amount": log_loan_amount
    }])


def get_risk_decision(default_probability):
    """Convert probability into business-friendly risk level and decision."""
    if default_probability >= 0.70:
        return "High Risk", "Loan Rejected / Strict Review", "error"
    elif default_probability >= 0.40:
        return "Medium Risk", "Manual Review Required", "warning"
    else:
        return "Low Risk", "Loan Approved", "success"


def get_feature_importance(model):
    """Extract feature importance from pipeline model."""
    if not hasattr(model, "named_steps"):
        return None

    try:
        preprocessor = model.named_steps["preprocessor"]
        final_model = model.named_steps["model"]

        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][2]

        onehot_features = preprocessor.named_transformers_["cat"] \
            .named_steps["onehot"] \
            .get_feature_names_out(categorical_features)

        all_features = list(numeric_features) + list(onehot_features)
        importances = final_model.feature_importances_

        return pd.DataFrame({
            "Feature": all_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
    except Exception:
        return None


def generate_business_recommendation(risk_level, credit_history, income_to_loan_ratio):
    """Simple business recommendation based on risk and applicant profile."""
    if risk_level == "High Risk":
        return "Applicant shows high default risk. Bank should reject or request stronger documents, collateral, or guarantor."
    if risk_level == "Medium Risk":
        return "Applicant should be manually reviewed. Bank can verify income documents and repayment capacity before approval."
    if credit_history == 1 and income_to_loan_ratio >= 20:
        return "Applicant looks financially stable with good credit history. Loan can be considered for approval."
    return "Applicant is low risk, but final decision should include document verification and policy checks."

# =========================
# App Header
# =========================
st.markdown("<div class='main-title'>💳 AI-Powered Credit Risk Decision System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning app for loan default risk prediction, risk scoring, explainability, and business recommendations</div>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📌 Project Highlights")
    st.write("✅ End-to-end ML project")
    st.write("✅ Feature engineering")
    st.write("✅ Random Forest model")
    st.write("✅ Risk score logic")
    st.write("✅ Feature importance")
    st.write("✅ Downloadable report")
    st.info("Tip: Use realistic income and loan values for better prediction testing.")

# =========================
# Input Form
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.subheader("💰 Financial Details")
    applicant_income = st.number_input("Applicant Income", min_value=0, step=500, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=500, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=1, step=10, value=120)
    loan_amount_term = st.slider("Loan Term (months)", 12, 480, 360)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# =========================
# Prediction Section
# =========================
if st.button("🚀 Predict Credit Risk"):
    try:
        input_df = create_input_dataframe(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        )

        prediction = model.predict(input_df)[0]
        default_probability = model.predict_proba(input_df)[0][1]

        risk_level, decision, message_type = get_risk_decision(default_probability)
        recommendation = generate_business_recommendation(
            risk_level,
            credit_history,
            input_df.loc[0, "income_to_loan_ratio"]
        )

        st.markdown("## 📊 Prediction Result")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Default Probability", f"{default_probability * 100:.2f}%")
        with r2:
            st.metric("Risk Level", risk_level)
        with r3:
            st.metric("Decision", decision)

        if message_type == "success":
            st.success(f"✅ {decision} | {risk_level}")
        elif message_type == "warning":
            st.warning(f"⚠️ {decision} | {risk_level}")
        else:
            st.error(f"🚨 {decision} | {risk_level}")

        st.progress(int(default_probability * 100))

        st.markdown("## 💼 Business Recommendation")
        st.info(recommendation)

        with st.expander("📄 View Applicant Input Data"):
            st.dataframe(input_df, use_container_width=True)

        # Feature Importance
        st.markdown("## 🔍 Model Explainability: Feature Importance")
        st.write("This section shows which features are most important for the model decision.")

        fi_df = get_feature_importance(model)
        if fi_df is not None:
            top_fi = fi_df.head(10)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(top_fi["Feature"], top_fi["Importance"])
            ax.invert_yaxis()
            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

            with st.expander("📌 Feature Importance Table"):
                st.dataframe(top_fi, use_container_width=True)
        else:
            st.warning("Feature importance is not available for this model structure.")

        # Download Report
        report_df = input_df.copy()
        report_df["Default_Probability"] = round(default_probability, 4)
        report_df["Risk_Level"] = risk_level
        report_df["Decision"] = decision
        report_df["Business_Recommendation"] = recommendation
        report_df["Report_Generated_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        csv = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="credit_risk_decision_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error while making prediction: {e}")
        st.write("Please check that credit_risk_model.pkl is available in the same folder as app.py.")
