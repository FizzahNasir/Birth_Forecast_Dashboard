import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

# Change this near the top of dashboard.py:
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
model_path = os.path.join(BASE_DIR, "app-folder", "best_model.pkl")
features_path = os.path.join(BASE_DIR, "app-folder", "model_features.pkl")
# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Scotland Birth Forecast",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stNumberInput {margin-bottom: 15px;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .success-prediction {
            font-size: 1.5rem;
            text-align: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e6f7ee;
            margin-top: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 1️⃣ Load Model & Feature List
# -------------------------------
st.title("👶 Scotland Birth Forecast Dashboard")
st.markdown("Predict monthly birth rates based on key socioeconomic and health indicators")

with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    This dashboard uses an XGBoost model to predict monthly births in Scotland.
    Adjust the inputs on the right to see how they affect the forecast.
    """)
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# -------------------------------
# 2️⃣ Model Loading Section
# -------------------------------
@st.cache_resource
def load_model_artifacts():
    """Cache the model and features to avoid reloading on every interaction"""
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    features_path = os.path.join(BASE_DIR, "model_features.pkl")
    
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    except Exception as e:
        st.error(f"❌ Error loading model or features: {e}")
        st.error("Please ensure both 'best_model.pkl' and 'model_features.pkl' are in the same directory as this script.")
        st.stop()

model, feature_names = load_model_artifacts()
st.sidebar.success("✅ Model & features loaded successfully")

# -------------------------------
# 3️⃣ User Input Section
# -------------------------------
st.subheader("📊 Input Parameters")
st.markdown("Adjust these key indicators to see their impact on birth forecasts")

col1, col2 = st.columns(2)

with col1:
    cpi = st.number_input(
        "CPI Index (Consumer Price Index)", 
        value=100.0,
        min_value=50.0,
        max_value=200.0,
        step=0.1,
        help="Measure of inflation and cost of living"
    )
    
    unemployment = st.number_input(
        "Unemployment Rate (%)", 
        value=5.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Percentage of working-age population unemployed"
    )
    
    births_last_month = st.number_input(
        "Births Last Month", 
        value=5000,
        min_value=3000,
        max_value=8000,
        step=100,
        help="Number of births in the previous month"
    )

with col2:
    diabetes = st.number_input(
        "Diabetes Prevalence (%)", 
        value=8.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Percentage of population with diabetes"
    )
    
    induction = st.number_input(
        "Labor Induction Rate (%)", 
        value=20.0,
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        help="Percentage of births involving labor induction"
    )

# -------------------------------
# 4️⃣ Create Feature Vector
# -------------------------------
input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Map user inputs to features
input_mapping = {
    "cpi index": cpi,
    "UnemploymentRate": unemployment,
    "Births_lag1": births_last_month,
    "Diabetes %": diabetes,
    "Induction %": induction
}

for feature, value in input_mapping.items():
    if feature in input_data.columns:
        input_data[feature] = value

# -------------------------------
# 5️⃣ Prediction Section
# -------------------------------
st.markdown("---")
pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    if st.button("🔮 Predict Monthly Births", type="primary"):
        try:
            with st.spinner("Making prediction..."):
                prediction = model.predict(input_data)[0]
                
                # Display with animation
                st.balloons()
                st.markdown(
                    f"<div class='success-prediction'>"
                    f"📈 Predicted Births: <strong>{prediction:,.0f}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Show interpretation
                st.markdown("""
                    **Interpretation:**
                    - Typical range for Scotland: 4,500-6,000 births/month
                    - Values above 6,000 indicate unusually high birth rates
                    - Values below 4,500 suggest declining birth rates
                """)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

with pred_col2:
    st.markdown("**How to use this dashboard:**")
    st.markdown("""
        1. Adjust the input parameters using the sliders
        2. Click the "Predict Monthly Births" button
        3. View the predicted birth count and interpretation
        
        **Key Factors Affecting Birth Rates:**
        - Economic conditions (CPI, Unemployment)
        - Previous month's birth count
        - Maternal health indicators
    """)

# -------------------------------
# 6️⃣ Footer
# -------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
        Scotland Birth Forecast Dashboard • National Health Statistics • 
        <a href="#" target="_blank">Data Sources</a>
    </div>
""", unsafe_allow_html=True)
