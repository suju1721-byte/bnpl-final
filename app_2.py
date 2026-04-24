import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

# =========================
# TEMP MODEL (WORKING)
# =========================
@st.cache_resource
def load_model():
    model = LogisticRegression()

    # dummy training (so app runs)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    model.fit(X, y)
    return model

model = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="BNPL Risk Checker", layout="centered")

st.title("💳 BNPL Risk Prediction")
st.markdown("### Enter Financial Details")

# Inputs (only what you asked)
credit_score = st.slider("Credit Score", 300, 900, 650)
income = st.number_input("Income", min_value=0.0)
interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0)
dti_ratio = st.slider("DTI Ratio", 0.0, 1.0)

cutoff = st.slider("Risk Cutoff (%)", 0, 100, 50)

# =========================
# Prediction
# =========================
if st.button("🔍 Check Risk"):

    # Prepare input
    features = np.array([[credit_score, income, interest_rate, dti_ratio]])

    # Predict probability
    prob = model.predict_proba(features)[0][1] * 100  # convert to %

    st.subheader("📊 Risk Analysis")

    # Show probability
    st.progress(prob / 100)
    st.write(f"Default Probability: **{prob:.2f}%**")

    # Apply cutoff logic
    if prob >= cutoff:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk (Safe)")

    # Simple insights
    st.markdown("### 💡 Insights")
    if credit_score < 600:
        st.warning("Low credit score increases risk")
    if dti_ratio > 0.4:
        st.warning("High DTI ratio increases risk")
    if interest_rate > 15:
        st.warning("High interest rate may indicate risky profile")
