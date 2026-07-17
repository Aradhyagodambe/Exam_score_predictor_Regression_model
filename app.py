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
    page_title="EDUPREDICT | COIN-OP FORECASTER",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Retro Arcade CSS
# -----------------------------
st.markdown("""
    <style>
    /* Import Retro Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
    
    /* Base arcade cabinet background */
    .stApp {
        background-color: #080811;
        background-image: 
            linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
            linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        background-size: 100% 4px, 6px 100%;
    }
    
    html, body, [class*="css"] {
        font-family: 'VT323', monospace;
        color: #E0E0E0;
    }
    
    /* Neon Headers */
    .main-header {
        font-family: 'Press Start 2P', cursive;
        font-size: 2.2rem;
        color: #00FFFF;
        text-align: center;
        text-shadow: 3px 3px 0px #FF00FF;
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sub-header {
        font-family: 'VT323', monospace;
        font-size: 1.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 3rem;
        text-shadow: 1px 1px 5px #FFFFFF;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #05050A;
        border-right: 2px solid #00FFFF;
        box-shadow: inset -5px 0 15px rgba(0, 255, 255, 0.1);
    }
    
    /* Input Labels */
    label, .stSlider > div > div > div > div {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 0.75rem !important;
        color: #00FF00 !important;
        text-shadow: 1px 1px 2px rgba(0, 255, 0, 0.5);
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Press Start 2P', cursive;
        font-size: 1rem;
        color: #FF00FF;
        border-bottom: 2px solid #00FFFF;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 5px rgba(255, 0, 255, 0.5);
        box-shadow: 0px 5px 5px -5px rgba(0, 255, 255, 0.5);
    }
    
    /* The Big Arcade Button */
    .stButton>button {
        font-family: 'Press Start 2P', cursive;
        background-color: #000000;
        color: #00FF00;
        border: 4px solid #00FF00;
        border-radius: 10px;
        font-weight: 400;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        transition: all 0.1s ease-in-out;
        box-shadow: 0 0 15px #00FF00, inset 0 0 10px #00FF00;
        width: 100%;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background-color: #00FF00;
        color: #000000;
        box-shadow: 0 0 25px #00FF00, inset 0 0 15px #000000;
        transform: scale(1.02);
    }
    .stButton>button:active {
        transform: scale(0.95);
        box-shadow: 0 0 5px #00FF00;
    }
    
    /* Insight Cards (Retro Terminal Style) */
    .insight-box {
        background-color: #000000;
        border: 2px dashed #00FFFF;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        font-family: 'VT323', monospace;
        font-size: 1.3rem;
        color: #00FF00;
        box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.1);
    }
    .insight-warning {
        border: 2px solid #FF00FF;
        color: #FF00FF;
        box-shadow: inset 0 0 10px rgba(255, 0, 255, 0.1);
    }
    .insight-title {
        font-family: 'Press Start 2P', cursive;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<h2 style='font-family: \"Press Start 2P\", cursive; color: #00FFFF; font-size: 1.2rem; text-shadow: 0 0 5px #00FFFF;'>SYSTEM OS</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 1.3rem; color: #E0E0E0;'>
    > INITIALIZING EDUPREDICT...<br>
    > LOAD ML REGRESSION CORE...<br>
    > READY.<br><br>
    PROJECT END-OF-TERM SCORES USING BASELINE BEHAVIORAL AND ACADEMIC METRICS.<br><br>
    INPUT PARAMETERS:<br>
    [+] DAILY STUDY VOLUME<br>
    [+] REST INTERVALS<br>
    [+] ATTENDANCE RATE<br>
    [+] HISTORICAL BENCHMARK<br>
    <br>
    --<br>
    VER: 1.2.0-ARCADE<br>
    CREDIT: 1 (FREE PLAY)<br>
    </div>
    """, unsafe_allow_html=True)

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
st.markdown('<div class="main-header">EDUPREDICT FORECASTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">INSERT PARAMETERS TO INITIALIZE PROJECTION ALGORITHM.</div>', unsafe_allow_html=True)

if model is None:
    st.markdown("<div style='font-family: \"Press Start 2P\", cursive; color: #FF0000; text-align: center;'>FATAL ERROR: Exam_score_Predictor.pkl MISSING.<br>INSERT COIN TO RESTART.</div>", unsafe_allow_html=True)
    st.stop()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">PLAYER 1: HABITS</div>', unsafe_allow_html=True)
    
    hours_studied = st.slider(
        "STUDY VOLUME [HRS/DAY]",
        min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
    
    sleep_hours = st.slider(
        "REST CYCLE [HRS/NIGHT]",
        min_value=0.0, max_value=12.0, value=7.0, step=0.5
    )

with col2:
    st.markdown('<div class="section-header">PLAYER 1: STATS</div>', unsafe_allow_html=True)
    
    attendance = st.slider(
        "ATTENDANCE RATE [%]",
        min_value=0, max_value=100, value=90, step=1
    )
    
    previous_scores = st.slider(
        "PREVIOUS HI-SCORE",
        min_value=0, max_value=100, value=70, step=1
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# -----------------------------
# Prediction Trigger & Analytics
# -----------------------------
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("PRESS START TO PREDICT")

if predict_clicked:
    st.markdown("<br><hr style='border: 1px dashed #00FFFF;'>", unsafe_allow_html=True)
    with st.spinner("PROCESSING DATA..."):
        time.sleep(1.2) # Simulating loading screen
        
        # Make Prediction
        features = np.array([[hours_studied, sleep_hours, attendance, previous_scores]])
        prediction = model.predict(features)[0]
        prediction = max(0.0, min(100.0, prediction)) 

    # Layout for results
    res_col1, res_col2 = st.columns([1.2, 1], gap="large")
    
    with res_col1:
        st.markdown(f"<div style='font-family: \"Press Start 2P\", cursive; color: #FFFFFF; font-size: 1rem; margin-bottom: 0.5rem;'>PROJECTED HI-SCORE:</div>", unsafe_allow_html=True)
        
        # Color coding the score
        score_color = "#00FF00" if prediction >= 75 else ("#FFFF00" if prediction >= 50 else "#FF0000")
        
        st.markdown(f"<div style='font-family: \"Press Start 2P\", cursive; font-size: 3.5rem; color: {score_color}; text-shadow: 4px 4px 0px rgba(0,0,0,0.5); margin-top: 0; line-height: 1;'>{prediction:.1f}<span style='font-size: 1.5rem; color: #FFFFFF;'>/100</span></div>", unsafe_allow_html=True)
        
        st.markdown("<br><div class='section-header' style='margin-bottom: 1rem;'>SYSTEM DIAGNOSTICS</div>", unsafe_allow_html=True)
        
        # Dynamic Insight Generation (Retro Themed)
        if sleep_hours < 6.5:
            st.markdown("""
            <div class='insight-box insight-warning'>
                <span class='insight-title'>[WARNING: REST DEPLETION]</span><br>
                RAM clearing protocol (sleep) critically low. Memory consolidation algorithms require 7+ hours. Increase sleep to prevent data loss.
            </div>
            """, unsafe_allow_html=True)
            
        if attendance < 80:
            st.markdown("""
            <div class='insight-box insight-warning'>
                <span class='insight-title'>[WARNING: CONNECTION LOST]</span><br>
                Server attendance below 80%. Packet loss detected in knowledge base. Reconnect to main servers (classes) to stabilize baseline.
            </div>
            """, unsafe_allow_html=True)
            
        if hours_studied > 9.0:
            st.markdown("""
            <div class='insight-box insight-warning'>
                <span class='insight-title'>[WARNING: CPU OVERHEATING]</span><br>
                Extreme grind detected. Grind yields diminishing returns. Shift focus to qualitative processing (active recall) to prevent system burnout.
            </div>
            """, unsafe_allow_html=True)
            
        if previous_scores >= 85 and prediction < previous_scores:
            st.markdown("""
            <div class='insight-box insight-warning'>
                <span class='insight-title'>[ALERT: SCORE DEGRADATION]</span><br>
                Trajectory trending below historical hi-score. Current input parameters are insufficient to maintain peak leaderboard status.
            </div>
            """, unsafe_allow_html=True)
            
        if not (sleep_hours < 6.5 or attendance < 80 or hours_studied > 9.0 or (previous_scores >= 85 and prediction < previous_scores)):
            st.markdown("""
            <div class='insight-box' style='border-color: #00FF00; color: #00FF00; box-shadow: inset 0 0 10px rgba(0,255,0,0.1);'>
                <span class='insight-title' style='color: #00FF00;'>[SYSTEM NOMINAL]</span><br>
                Parameters indicate a balanced equilibrium. Mechanics optimized for current level. Maintain operational cadence to secure victory.
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Retro Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'font': {'size': 1, 'color': "rgba(0,0,0,0)"}}, # Hide default number
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': "#FFFFFF", 'tickfont': {'color': "#00FFFF", 'family': "Courier New", 'size': 14}},
                'bar': {'color': "#00FFFF", 'thickness': 0.3},
                'bgcolor': "#05050A",
                'borderwidth': 2,
                'bordercolor': "#FF00FF",
                'steps': [
                    {'range': [0, 50], 'color': "#330000"},
                    {'range': [50, 75], 'color': "#333300"},
                    {'range': [75, 100], 'color': "#003300"}
                ],
            }
        ))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=30, r=30, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title={'text': "POWER METER", 'font': {'color': '#FF00FF', 'size': 20, 'family': '"Press Start 2P", cursive'}}
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #444; font-family: \"Press Start 2P\", cursive; font-size: 0.6rem;'>EDUPREDICT SYSTEM | (C) 2026<br>INSERT COIN TO CONTINUE</div>", unsafe_allow_html=True)
