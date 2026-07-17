import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="📚",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Exam_score_Predictor.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("📚 Exam Score Predictor")
st.markdown(
    """
Predict a student's **exam score** using the trained Machine Learning model.

Enter the student details below and click **Predict Score**.
"""
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
    )

    previous_scores = st.number_input(
        "Previous Score",
        min_value=0,
        max_value=100,
        value=70,
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
    )

with col2:
    sample_papers = st.number_input(
        "Sample Papers Practiced",
        min_value=0,
        max_value=50,
        value=10,
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=90,
    )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🎯 Predict Score", use_container_width=True):

    features = np.array([[
        hours_studied,
        previous_scores,
        sleep_hours,
        sample_papers,
        attendance
    ]])

    prediction = model.predict(features)[0]

    st.success(f"### Predicted Exam Score: **{prediction:.2f}**")

    if prediction >= 90:
        st.balloons()
        st.info("🌟 Excellent Performance Expected!")

    elif prediction >= 75:
        st.info("👍 Good Performance Expected!")

    elif prediction >= 50:
        st.warning("📘 Average Performance Expected!")

    else:
        st.error("📖 More preparation is recommended.")

st.divider()

st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
