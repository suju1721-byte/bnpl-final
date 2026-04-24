import base64
import io
import numpy as np
import joblib
import streamlit as st

# =========================

# Load Model from Base64

# =========================

def load_model(pkl_string):
    return joblib.load(io.BytesIO(base64.b64decode(pkl_string)))

# =========================

# 🔴 PASTE YOUR FULL STRINGS BELOW (UNCHANGED)

# =========================

_PKL_LR = "PASTE_FULL_LR_STRING_HERE"
_PKL_RF = "PASTE_FULL_RF_STRING_HERE"

# Load models

model_lr = load_model(_PKL_LR)
model_rf = load_model(_PKL_RF)

# =========================

# Streamlit UI

# =========================

st.set_page_config(page_title="BNPL Prediction", layout="centered")

st.title("💳 BNPL Loan Default Prediction")

st.write("Fill all details:")

# =========================

# Inputs (MATCHES YOUR MODEL)

# =========================

Age = st.number_input("Age", 18, 100)
Income = st.number_input("Income")
LoanAmount = st.number_input("Loan Amount")
CreditScore = st.number_input("Credit Score")
MonthsEmployed = st.number_input("Months Employed")
NumCreditLines = st.number_input("Number of Credit Lines")
InterestRate = st.number_input("Interest Rate")
LoanTerm = st.number_input("Loan Term")
DTIRatio = st.number_input("DTI Ratio")

Education = st.selectbox("Education", [0,1,2])
EmploymentType = st.selectbox("Employment Type", [0,1,2])
MaritalStatus = st.selectbox("Marital Status", [0,1,2])
HasMortgage = st.selectbox("Has Mortgage", [0,1])
HasDependents = st.selectbox("Has Dependents", [0,1])
LoanPurpose = st.selectbox("Loan Purpose", [0,1,2,3])
HasCoSigner = st.selectbox("Has Co-Signer", [0,1])

model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

# =========================

# Prediction

# =========================

if st.button("Predict"):
try:
features = np.array([[Age, Income, LoanAmount, CreditScore,
MonthsEmployed, NumCreditLines, InterestRate,
LoanTerm, DTIRatio, Education,
EmploymentType, MaritalStatus,
HasMortgage, HasDependents,
LoanPurpose, HasCoSigner]])

```
    if model_choice == "Logistic Regression":
        pred = model_lr.predict(features)
    else:
        pred = model_rf.predict(features)

    result = "⚠️ Default Risk" if pred[0] == 1 else "✅ Safe"

    st.subheader("Result:")
    st.success(result)

except Exception as e:
    st.error(f"Error: {e}")
```
