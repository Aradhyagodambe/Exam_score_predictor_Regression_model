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
# Custom CSS for Modern Aesthetics
# -----------------------------
st.markdown("""
    <style>
    /* Soften the background for a modern look */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Elegant Typography */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Modern Interactive Button */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    .stButton>button:active {
        transform: translateY(1px);
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
    This application uses Machine Learning to predict a student's final exam score based on study habits and historical data.
    
    **How to use:**
    1. Adjust the sliders to match your daily habits.
    2. Input your current academic standing.
    3. Click **Predict My Score!**
    
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
st.markdown('<p class="main-header">🎓 EduPredict Forecaster</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Adjust the interactive sliders below to see how your habits impact your final score.</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file (`Exam_score_Predictor.pkl`) not found. Please ensure it is in the same directory as this app.")
    st.stop()

# -----------------------------
# Input Section (Modern Interactive Cards)
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    # Card 1: Daily Habits
    with st.container(border=True):
        st.markdown("### 🕒 Daily Habits")
        st.caption("Drag the sliders to reflect your daily routine.")
        
        hours_studied = st.slider(
            "📚 Hours Studied",
            min_value=0.0, max_value=24.0, value=5.0, step=0.5,
            help="Average hours dedicated to studying per day."
        )
        
        sleep_hours = st.slider(
            "😴 Sleep Hours",
            min_value=0.0, max_value=12.0, value=7.0, step=0.5,
            help="Average hours of sleep per night. (Rest is crucial!)"
        )

with col2:
    # Card 2: Academic History
    with st.container(border=True):
        st.markdown("### 📊 Academic Record")
        st.caption("Drag the sliders to reflect your current standing.")
        
        attendance = st.slider(
            "🏫 Attendance (%)",
            min_value=0, max_value=100, value=90, step=1,
            format="%d%%",
            help="Percentage of classes attended during the semester."
        )
        
        previous_scores = st.slider(
            "📝 Previous Exam Score",
            min_value=0, max_value=100, value=70, step=1,
            help="Your score in the most recent major examination."
        )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Prediction & Visualization
# -----------------------------
# Center the button for better aesthetics
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("✨ Predict My Score", use_container_width=True)

if predict_clicked:
    st.divider()
    with st.spinner("Running Machine Learning algorithms..."):
        time.sleep(1) # Simulating processing time for UX
        
        features = np.array([[hours_studied, sleep_hours, attendance, previous_scores]])
        prediction = model.predict(features)[0]
        prediction = max(0.0, min(100.0, prediction)) 

    # Layout for results
    res_col1, res_col2 = st.columns([1.2, 1], gap="large")
    
    with res_col1:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        if prediction >= 90:
            st.balloons()
            st.success(f"## Predicted Score: **{prediction:.1f}**")
            st.info("🌟 **Outstanding!** Keep up the fantastic work. You're on track for top-tier results.")
        elif prediction >= 75:
            st.success(f"## Predicted Score: **{prediction:.1f}**")
            st.info("👍 **Great Job!** You have a solid grasp of the material. A little more push could get you to the top!")
        elif prediction >= 50:
            st.warning(f"## Predicted Score: **{prediction:.1f}**")
            st.warning("📘 **Average Performance.** You're passing, but increasing your study hours or attendance could yield better results.")
        else:
            st.error(f"## Predicted Score: **{prediction:.1f}**")
            st.error("📖 **Needs Improvement.** Consider reviewing your study habits, getting more sleep, and attending more classes.")

    with res_col2:
        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 42, 'color': "#1E3A8A", 'family': "Arial Black"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': "#2563EB", 'thickness': 0.25},
                'bgcolor': "#F1F5F9",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': "#FECACA"},   # Soft Red
                    {'range': [50, 75], 'color': "#FEF08A"},  # Soft Yellow
                    {'range': [75, 90], 'color': "#BBF7D0"},  # Soft Light Green
                    {'range': [90, 100], 'color': "#86EFAC"}  # Soft Dark Green
                ],
            }
        ))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#1E3A8A", 'family': "sans-serif"}
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with ❤️ using Streamlit, Plotly & Scikit-Learn | © 2026")
