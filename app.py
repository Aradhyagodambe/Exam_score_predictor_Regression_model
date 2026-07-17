import streamlit as st
import pickle
import numpy as np
import time
import plotly.graph_objects as go
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EduPredict | Exam Score Forecaster",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Styling
# -----------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.title("About EduPredict")
    st.markdown("""
    This application uses a Machine Learning model to predict a student's final exam score based on their study habits and historical data.
    
    **How to use:**
    1. Enter your daily study and sleep hours.
    2. Input your current attendance percentage.
    3. Enter your previous exam score.
    4. Click Predict!
    
    ---
    *Built for educational purposes.*
    """)

# -----------------------------
# Load Model (with Error Handling)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "Exam_score_Predictor.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# Main Header
# -----------------------------
st.markdown('<p class="main-header">🎓 EduPredict: Exam Score Forecaster</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leverage Machine Learning to estimate your academic performance.</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file (`Exam_score_Predictor.pkl`) not found. Please ensure it is in the same directory as this app.")
    st.stop()

# -----------------------------
# Input Section (Using Cards/Containers)
# -----------------------------
with st.container():
    st.markdown("### 📊 Student Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input(
            "📚 Hours Studied (Daily)",
            min_value=0.0, max_value=24.0, value=5.0, step=0.5,
            help="Average number of hours dedicated to studying per day."
        )
        
        sleep_hours = st.number_input(
            "😴 Sleep Hours (Daily)",
            min_value=0.0, max_value=24.0, value=7.0, step=0.5,
            help="Average number of hours of sleep per night."
        )

    with col2:
        attendance = st.number_input(
            "🏫 Attendance (%)",
            min_value=0, max_value=100, value=90, step=1,
            help="Percentage of classes attended during the semester."
        )
        
        previous_scores = st.number_input(
            "📝 Previous Exam Score",
            min_value=0, max_value=100, value=70, step=1,
            help="Your score in the most recent major examination."
        )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Prediction & Visualization
# -----------------------------
if st.button("🎯 Predict My Score", use_container_width=True):
    with st.spinner("Analyzing student metrics..."):
        time.sleep(1) # Simulating processing time for UX
        
        # Ensure array order matches your original training feature order
        features = np.array([[hours_studied, sleep_hours, attendance, previous_scores]])
        
        # Predict
        prediction = model.predict(features)[0]
        # Cap prediction between 0 and 100 just in case
        prediction = max(0.0, min(100.0, prediction)) 

    # Layout for results
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("### Prediction Results")
        if prediction >= 90:
            st.balloons()
            st.success(f"## **{prediction:.1f} / 100**")
            st.info("🌟 **Outstanding!** Keep up the fantastic work. You're on track for top-tier results.")
        elif prediction >= 75:
            st.success(f"## **{prediction:.1f} / 100**")
            st.info("👍 **Great Job!** You have a solid grasp of the material. A little more push could get you to the top!")
        elif prediction >= 50:
            st.warning(f"## **{prediction:.1f} / 100**")
            st.warning("📘 **Average Performance.** You're passing, but increasing your study hours or attendance could yield better results.")
        else:
            st.error(f"## **{prediction:.1f} / 100**")
            st.error("📖 **Needs Improvement.** Consider reviewing your study habits, getting more sleep, and attending more classes.")

    with res_col2:
        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Score", 'font': {'size': 24}},
            number = {'suffix': "%", 'font': {'size': 40, 'color': "#1E3A8A"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2563EB"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "#FCA5A5"},   # Red
                    {'range': [50, 75], 'color': "#FDE047"},  # Yellow
                    {'range': [75, 90], 'color': "#86EFAC"},  # Light Green
                    {'range': [90, 100], 'color': "#22C55E"}   # Dark Green
                ],
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with ❤️ using Streamlit, Plotly & Scikit-Learn | © 2026")
