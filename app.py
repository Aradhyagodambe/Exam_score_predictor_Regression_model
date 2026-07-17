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
    page_title="EduPredict | Academic Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sleek Minimalist CSS
# -----------------------------
st.markdown("""
    <style>
    /* Base typography and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Elegant Typography */
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    /* Modern Interactive Button */
    .stButton>button {
        background-color: #111827;
        color: #FFFFFF;
        border-radius: 6px;
        border: 1px solid #111827;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.5rem 2rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #374151;
        border-color: #374151;
        color: #FFFFFF;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Insight Cards */
    .insight-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #111827;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #111827; font-weight: 600;'>System Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    EduPredict utilizes machine learning regression models to project end-of-term academic performance based on historical baselines and current behavioral metrics.
    
    **Input Parameters:**
    *   **Behavioral:** Daily study volume and rest intervals.
    *   **Academic:** Current attendance rate and previous assessment benchmarks.
    
    ---
    *Model Version: 1.2.0*  
    *Execution: Local Environment*
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
st.markdown('<div class="main-header">Academic Forecasting Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Adjust behavioral and academic parameters below to generate performance projections and analytical insights.</div>', unsafe_allow_html=True)

if model is None:
    st.error("System Error: Predictive model (Exam_score_Predictor.pkl) not located in the execution directory. Initialization halted.")
    st.stop()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">Behavioral Metrics</div>', unsafe_allow_html=True)
    
    hours_studied = st.slider(
        "Study Volume (Hours/Day)",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5,
        help="Average daily hours dedicated to active learning and review."
    )
    
    sleep_hours = st.slider(
        "Rest Intervals (Hours/Night)",
        min_value=0.0, max_value=12.0, value=7.0, step=0.5,
        help="Average nightly sleep duration, critical for memory consolidation."
    )

with col2:
    st.markdown('<div class="section-header">Academic Baselines</div>', unsafe_allow_html=True)
    
    attendance = st.slider(
        "Course Attendance (%)",
        min_value=0, max_value=100, value=90, step=1,
        help="Percentage of scheduled lectures/sessions attended to date."
    )
    
    previous_scores = st.slider(
        "Historical Benchmark (Score)",
        min_value=0, max_value=100, value=70, step=1,
        help="Performance metric from the most recent standardized assessment."
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Prediction Trigger & Analytics
# -----------------------------
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("Initialize Forecast")

if predict_clicked:
    st.markdown("---")
    with st.spinner("Compiling metrics and running projection algorithms..."):
        time.sleep(0.8) # Simulating data processing
        
        # Make Prediction
        features = np.array([[hours_studied, sleep_hours, attendance, previous_scores]])
        prediction = model.predict(features)[0]
        prediction = max(0.0, min(100.0, prediction)) 

    # Layout for results
    res_col1, res_col2 = st.columns([1.2, 1], gap="large")
    
    with res_col1:
        st.markdown(f"<h3 style='color: #111827; font-weight: 700; margin-bottom: 0;'>Projected Assessment Score</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='font-size: 4rem; color: #111827; margin-top: 0; line-height: 1;'>{prediction:.1f}<span style='font-size: 2rem; color: #6B7280;'>/100</span></h1>", unsafe_allow_html=True)
        
        st.markdown("<br><div class='section-header' style='margin-bottom: 1rem;'>Algorithmic Insights</div>", unsafe_allow_html=True)
        
        # Dynamic Insight Generation
        if sleep_hours < 6.5:
            st.markdown("""
            <div class='insight-box'>
                <strong>Cognitive Risk:</strong> Sub-optimal rest detected. Memory consolidation requires ~7+ hours. Increasing sleep duration is mathematically likely to yield higher retention rates than equivalent extra study hours.
            </div>
            """, unsafe_allow_html=True)
            
        if attendance < 80:
            st.markdown("""
            <div class='insight-box'>
                <strong>Engagement Deficit:</strong> Attendance below the 80% threshold correlates strongly with instructional gaps. Prioritize lecture attendance to stabilize baseline understanding.
            </div>
            """, unsafe_allow_html=True)
            
        if hours_studied > 9.0:
            st.markdown("""
            <div class='insight-box'>
                <strong>Diminishing Returns:</strong> Extreme study volume observed. Ensure qualitative focus (active recall, practice testing) rather than purely quantitative hour accumulation to avoid burnout.
            </div>
            """, unsafe_allow_html=True)
            
        if previous_scores >= 85 and prediction < previous_scores:
            st.markdown("""
            <div class='insight-box'>
                <strong>Trajectory Warning:</strong> Your projected score is trending below your historical baseline. Adjust current behavioral metrics to maintain peak performance.
            </div>
            """, unsafe_allow_html=True)
            
        if not (sleep_hours < 6.5 or attendance < 80 or hours_studied > 9.0 or (previous_scores >= 85 and prediction < previous_scores)):
            st.markdown("""
            <div class='insight-box'>
                <strong>System Optimization:</strong> Current parameters indicate a balanced equilibrium between rest, attendance, and study volume. Maintain current operational cadence.
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Minimalist Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'font': {'size': 1, 'color': "rgba(0,0,0,0)"}}, # Hide default number to use custom HTML above
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9CA3AF", 'tickfont': {'color': "#6B7280", 'family': "Inter"}},
                'bar': {'color': "#111827", 'thickness': 0.15},
                'bgcolor': "#F3F4F6",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 60], 'color': "#E5E7EB"},
                    {'range': [60, 85], 'color': "#D1D5DB"},
                    {'range': [85, 100], 'color': "#9CA3AF"}
                ],
            }
        ))
        
        fig.update_layout(
            height=320, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            title={'text': "Performance Index", 'font': {'color': '#374151', 'size': 16, 'family': 'Inter'}}
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>EduPredict Analytical Infrastructure | Enterprise Data Solutions © 2026</div>", unsafe_allow_html=True)
