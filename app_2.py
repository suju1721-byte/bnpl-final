import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

# =========================
# Improved Demo Model
# =========================
@st.cache_resource
def load_model():
    model = LogisticRegression()

    X = []
    y = []

    # Generate smarter synthetic data
    for _ in range(300):
        credit = np.random.randint(300, 900)
        income = np.random.randint(10000, 100000)
        interest = np.random.uniform(1, 25)
        dti = np.random.uniform(0, 1)

        risk = 0
        if credit < 600: risk += 1
        if income < 40000: risk += 1
        if interest > 15: risk += 1
        if dti > 0.4: risk += 1

        label = 1 if risk >= 2 else 0

        X.append([credit, income, interest, dti])
        y.append(label)

    model.fit(np.array(X), np.array(y))
    return model

model = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="BNPL Risk Checker", layout="centered")

st.title("💳 BNPL Risk Prediction")
st.markdown("### Enter Financial Details")

# Inputs
credit_score = st.slider("Credit Score", 300, 900, 650)
income = st.number_input("Income", min_value=0.0, value=50000.0)
interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
cutoff = st.slider("Risk Cutoff (%)", 0, 100, 50)

# =========================
# Prediction
# =========================
if st.button("🔍 Check Risk"):

    features = np.array([[credit_score, income, interest_rate, dti_ratio]])

    prob = float(model.predict_proba(features)[0][1] * 100)
    prob = min(max(prob, 0), 100)  # safety clamp

    st.subheader("📊 Risk Analysis")

    # Progress bar
    st.progress(prob / 100)

    # Show probability
    st.write(f"Default Probability: **{prob:.2f}%**")

    # Risk decision
    if prob >= cutoff:
        st.error(f"⚠️ High Risk ({prob:.1f}%)")
    elif prob >= cutoff - 20:
        st.warning(f"⚠️ Medium Risk ({prob:.1f}%)")
    else:
        st.success(f"✅ Low Risk ({prob:.1f}%)")

    # =========================
    # Insights
    # =========================
    st.markdown("### 💡 Insights")

    if credit_score < 600:
        st.warning("Low credit score increases risk")

    if income < 40000:
        st.warning("Low income increases risk")

    if dti_ratio > 0.4:
        st.warning("High DTI ratio increases risk")

    if interest_rate > 15:
        st.warning("High interest rate indicates riskier profile")
