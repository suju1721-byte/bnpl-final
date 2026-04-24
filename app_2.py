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
# 🔴 PASTE YOUR LR MODEL STRING HERE
# =========================
_PKL_LR = "PASTE_YOUR_REAL_LR_STRING_HERE"

# Load model safely
try:
    model_lr = load_model(_PKL_LR)
except Exception as e:
    model_lr = None
    st.error(f"Model failed to load: {e}")

# =========================
# UI
# =========================
st.set_page_config(page_title="BNPL Prediction", layout="centered")

st.title("💳 BNPL Loan Default Prediction")
st.markdown("### Enter customer details below")

# Layout
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

Education = st.selectbox("Education", [0, 1, 2])
EmploymentType = st.selectbox("Employment Type", [0, 1, 2])
MaritalStatus = st.selectbox("Marital Status", [0, 1, 2])
HasMortgage = st.selectbox("Has Mortgage", [0, 1])
HasDependents = st.selectbox("Has Dependents", [0, 1])
LoanPurpose = st.selectbox("Loan Purpose", [0, 1, 2, 3])
HasCoSigner = st.selectbox("Has Co-Signer", [0, 1])

# =========================
# Prediction
# =========================
if st.button("🔍 Predict"):
    if model_lr is None:
        st.error("Model not loaded. Check your base64 string.")
        st.stop()

    try:
        features = np.array([[Age, Income, LoanAmount, CreditScore,
                              MonthsEmployed, NumCreditLines, InterestRate,
                              LoanTerm, DTIRatio, Education,
                              EmploymentType, MaritalStatus,
                              HasMortgage, HasDependents,
                              LoanPurpose, HasCoSigner]])

        pred = model_lr.predict(features)

        # Result display
        st.subheader("Prediction Result")

        if pred[0] == 1:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk (Safe)")

        # Probability (if supported)
        try:
            prob = model_lr.predict_proba(features)[0][1]
            st.progress(float(prob))
            st.write(f"📊 Default Probability: {prob:.2f}")
        except:
            pass

    except Exception as e:
        st.error(f"Prediction error: {e}")
