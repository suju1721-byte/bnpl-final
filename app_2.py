import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

# =========================
# CREATE DUMMY MODEL (TEMP FIX)
# =========================
@st.cache_resource
def load_model():
    model = LogisticRegression()

    # Dummy training data (so app doesn't crash)
    X = np.random.rand(100, 16)
    y = np.random.randint(0, 2, 100)

    model.fit(X, y)
    return model

model = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="BNPL Prediction")

st.title("💳 BNPL Loan Default Prediction")
st.markdown("### Quick Demo Version (Working)")

# Inputs
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 100)
    Income = st.number_input("Income")
    LoanAmount = st.number_input("Loan Amount")
    CreditScore = st.number_input("Credit Score")

with col2:
    MonthsEmployed = st.number_input("Months Employed")
    NumCreditLines = st.number_input("Credit Lines")
    InterestRate = st.slider("Interest Rate", 0.0, 30.0)
    LoanTerm = st.slider("Loan Term", 1, 60)

DTIRatio = st.slider("DTI Ratio", 0.0, 1.0)

Education = st.selectbox("Education", [0,1,2])
EmploymentType = st.selectbox("Employment Type", [0,1,2])
MaritalStatus = st.selectbox("Marital Status", [0,1,2])
HasMortgage = st.selectbox("Has Mortgage", [0,1])
HasDependents = st.selectbox("Has Dependents", [0,1])
LoanPurpose = st.selectbox("Loan Purpose", [0,1,2,3])
HasCoSigner = st.selectbox("Has Co-Signer", [0,1])

# =========================
# Prediction
# =========================
if st.button("🔍 Predict"):
    features = np.array([[Age, Income, LoanAmount, CreditScore,
                          MonthsEmployed, NumCreditLines, InterestRate,
                          LoanTerm, DTIRatio, Education,
                          EmploymentType, MaritalStatus,
                          HasMortgage, HasDependents,
                          LoanPurpose, HasCoSigner]])

    pred = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    st.subheader("Result")

    if pred[0] == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk (Safe)")

    st.progress(float(prob))
    st.write(f"📊 Default Probability: {prob:.2f}")
