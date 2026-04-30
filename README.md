# 💳 AI-Powered Credit Risk Decision System with Explainable ML

## 📌 Project Overview
This project is an end-to-end machine learning system that predicts loan default risk and supports lending decisions using customer financial and demographic data.

The project goes beyond basic prediction by adding risk scoring, model explainability, feature importance, business recommendations, and a Streamlit dashboard.

---

## 🧠 Business Problem
Banks and financial institutions need to identify risky loan applicants before approving loans. Wrong approvals can increase financial loss and bad loans.

This project helps lenders answer:
- Is the applicant low risk, medium risk, or high risk?
- What is the probability of default?
- Which factors are influencing the decision?
- What business action should be taken?

---

## ✅ Key Features
- Data cleaning and missing value handling
- Duplicate checking and outlier treatment
- Exploratory Data Analysis using charts
- Feature engineering for better model performance
- Model comparison using multiple ML algorithms
- Hyperparameter tuning
- Final Random Forest model deployment
- Risk score and decision logic
- Feature importance for model explainability
- SHAP explainability in notebook
- Streamlit web app for real-time prediction
- Downloadable prediction report

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Streamlit
- Joblib

---

## 🔄 Project Workflow
1. Data loading and understanding
2. Missing value handling
3. Duplicate checking
4. Outlier detection and treatment
5. Exploratory Data Analysis
6. Feature engineering
7. Data preprocessing using pipeline
8. Model building
9. Model comparison
10. Hyperparameter tuning
11. Model explainability using feature importance and SHAP
12. Streamlit deployment

---

## 🤖 Models Used
- Logistic Regression
- Random Forest
- XGBoost / Gradient Boosting

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 🏆 Final Model
Random Forest was selected as the final model based on overall performance and interpretability.

---

## 🔍 Explainable AI
This project includes explainability using feature importance and SHAP.

Explainability helps understand:
- Why a loan applicant is classified as risky or safe
- Which features increase or decrease risk
- How credit history, income, loan amount, and property area affect predictions
- How the model can support transparent business decisions

---

## 🚀 Streamlit App Features
- User-friendly applicant input form
- Real-time credit risk prediction
- Default probability score
- Low / Medium / High risk classification
- Business decision output
- Business recommendation
- Feature importance chart
- Downloadable prediction report

---

## 📁 Project Structure
```text
AI-Credit-Risk-Decision-System/
│
├── Advanced_Credit_Risk_and_Loan_Default_Prediction.ipynb
├── app.py
├── credit_risk_model.pkl
├── loan_data.csv
├── requirements.txt
├── README.md
└── screenshots/
    ├── app_home.png
    ├── prediction_result.png
    └── feature_importance.png
```

---

## ▶️ How to Run Project

### Step 1: Clone the repository
```bash
git clone <your-github-repository-link>
cd AI-Credit-Risk-Decision-System
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit app
```bash
streamlit run app.py
```

---

## 💼 Business Impact
This project can help financial institutions reduce default risk by identifying high-risk applicants early and supporting data-driven loan decisions.

---

## 🧑‍💻 Author
**Omkar Gadhave**  
Aspiring Data Scientist | Python | SQL | Machine Learning | Streamlit
