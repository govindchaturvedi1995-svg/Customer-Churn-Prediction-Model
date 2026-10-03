"""
app.py
------
Customer Churn Prediction System - Interactive Streamlit App

Run with:  streamlit run app.py

Requires the following files to exist in the same folder
(created by running train_model.py first):
    model_rf.pkl, model_ann.keras, scaler.pkl, encoders.pkl,
    target_encoder.pkl, metadata.pkl
"""

import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ----------------------------------------------------------------------
# LOAD SAVED MODEL + PREPROCESSING OBJECTS (cached so it loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    metadata = joblib.load("metadata.pkl")
    rf_model = joblib.load("model_rf.pkl")
    ann_model = load_model("model_ann.keras")
    return scaler, encoders, metadata, rf_model, ann_model

try:
    scaler, encoders, metadata, rf_model, ann_model = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to train and save the model, then relaunch this app."
    )
    st.stop()

categorical_cols = metadata["categorical_cols"]
numeric_cols = metadata["numeric_cols"]
feature_order = metadata["feature_order"]
category_display_values = metadata["category_display_values"]
numeric_ranges = metadata["numeric_ranges"]
best_model_type = metadata["best_model_type"]

best_model = rf_model if best_model_type == "random_forest" else ann_model

def get_probability(input_scaled):
    if best_model_type == "random_forest":
        return best_model.predict_proba(input_scaled)[0][1]
    else:
        return float(best_model.predict(input_scaled, verbose=0).ravel()[0])

# ----------------------------------------------------------------------
# MAIN TITLE
# ----------------------------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Enter the customer's details below and click **Predict** to find out "
    "whether they're likely to churn."
)
st.markdown("---")

# ----------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------
st.subheader("📝 Customer Details")

user_input = {}

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    all_cols = categorical_cols + numeric_cols
    half = len(all_cols) // 2 + 1

    for i, col in enumerate(all_cols):
        target_col = col1 if i < half else col2

        with target_col:
            if col in categorical_cols:
                options = category_display_values.get(col, ["Unknown"])
                user_input[col] = st.selectbox(f"{col}", options)
            else:
                low, high, med = numeric_ranges[col]
                if col.lower() in ["tenure", "seniorcitizen"]:
                    user_input[col] = st.number_input(
                        f"{col}", min_value=float(low), max_value=float(high),
                        value=float(med), step=1.0
                    )
                else:
                    user_input[col] = st.number_input(
                        f"{col}", min_value=float(low), max_value=float(high * 1.5),
                        value=float(med)
                    )

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ----------------------------------------------------------------------
# PREDICTION LOGIC
# ----------------------------------------------------------------------
if submitted:
    # Build a single-row dataframe in the correct column order
    input_df = pd.DataFrame([user_input])[feature_order]

    # Encode categorical columns using the SAME encoders from training
    for col in categorical_cols:
        le = encoders[col]
        val = input_df.at[0, col]
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            input_df[col] = 0  # fallback for an unseen category

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    churn_prob = get_probability(input_scaled)
    churn_percentage = churn_prob * 100
    will_churn = churn_prob >= 0.5

    if churn_percentage >= 70:
        risk_label, risk_color = "Very High Risk", "#c0392b"
    elif churn_percentage >= 50:
        risk_label, risk_color = "High Risk", "#e74c3c"
    elif churn_percentage >= 30:
        risk_label, risk_color = "Moderate Risk", "#f39c12"
    else:
        risk_label, risk_color = "Low Risk", "#2ecc71"

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    # ---- Headline statement ----
    if will_churn:
        st.error("⚠️ This customer is **likely to churn**.")
    else:
        st.success("✅ This customer is **likely to stay**.")

    # ---- Probability + risk metric ----
    m1, m2 = st.columns(2)
    m1.metric("Churn Probability", f"{churn_percentage:.1f}%")
    m2.metric("Risk Level", risk_label)

    # ---- Gauge chart ----
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [100], color="#ecf0f1", height=0.5)
    ax.barh([0], [churn_percentage], color=risk_color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Churn Probability (%)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig)

    # ---- Pie chart: Stay vs Churn ----
    st.markdown("#### Stay vs. Churn Likelihood")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.pie(
        [100 - churn_percentage, churn_percentage],
        labels=["Stay", "Churn"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        explode=(0, 0.08),
        textprops={"fontsize": 11}
    )
    ax2.axis("equal")
    st.pyplot(fig2)

    # ---- Plain-language takeaway ----
    st.markdown("#### Summary")
    if will_churn:
        st.write(
            f"Based on the details provided, this customer has a "
            f"**{churn_percentage:.1f}% probability of churning** — "
            f"consider proactive retention outreach (offers, support "
            f"check-in, or a loyalty incentive)."
        )
    else:
        st.write(
            f"Based on the details provided, this customer has only a "
            f"**{churn_percentage:.1f}% probability of churning** and "
            f"appears likely to remain with the service."
        )

else:
    st.info("👆 Fill in the customer details above and click **Predict Churn** to see the result.")

st.markdown("---")
st.caption("Built with Python, Pandas, Scikit-learn, TensorFlow & Streamlit")
