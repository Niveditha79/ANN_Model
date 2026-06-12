import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS for modern card styling and layout
st.markdown("""
    <style>
    .reportview-container {
        background: #f5f7f9;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #eef2f6;
    }
    .metric-title {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 5px;
    }
    .risk-low {
        border-left: 5px solid #10b981;
    }
    .risk-medium {
        border-left: 5px solid #f59e0b;
    }
    .risk-high {
        border-left: 5px solid #ef4444;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Asset Loader
@st.cache_resource
def load_assets():
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    model_path = os.path.join('models', 'ann_model.keras')
    
    if not (os.path.exists(preprocessor_path) and os.path.exists(model_path)):
        return None, None
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    model = tf.keras.models.load_model(model_path)
    return preprocessor, model

# Load database sample for background context plot
@st.cache_data
def load_sample_data():
    data_path = os.path.join('data', 'Churn_Modelling.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path).sample(500, random_state=42)
    return None

preprocessor, model = load_assets()
sample_df = load_sample_data()

# 3. Sidebar Information panel
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3121/3121768.png", width=80)
    st.title("Model Information")
    st.markdown("""
    This analyzer utilizes an **Artificial Neural Network (ANN)** trained to identify patterns correlated with customer departures.
    
    ### Key Churn Factors:
    * **Age Group**: Older demographics show higher churn rates in this segment.
    * **Number of Products**: Customers with 3 or more products exhibit higher exit rates.
    * **Active Status**: Active members are significantly more loyal.
    """)
    st.divider()
    st.caption("v2.0.0 • Deep Learning Engine Enabled")

# 4. Main Application Interface
st.title("🔮 Customer Churn Analytics Portal")
st.markdown("Optimize customer retention by calculating real-time risk assessments using advanced deep learning.")

if preprocessor is None or model is None:
    st.error("⚠️ Model files not detected. Please verify that `model_training.py` has been executed successfully and artifacts are located in the `models/` directory.")
else:
    # Organize screen into Input Column and Visual Outputs Column
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.subheader("📋 Customer Demographics & Profile")
        
        # Grid layout for inputs to save screen space
        g1, g2 = st.columns(2)
        with g1:
            geography = st.selectbox("Geographic Region", options=["France", "Germany", "Spain"], help="Country of customer registration.")
            gender = st.radio("Gender Profile", options=["Male", "Female"], horizontal=True)
            age = st.slider("Age (Years)", min_value=18, max_value=90, value=38, step=1)
            tenure = st.slider("Account Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
            credit_score = st.slider("Credit Score Rating", min_value=300, max_value=850, value=650, step=5)

        with g2:
            balance = st.number_input("Current Account Balance ($)", min_value=0.0, value=75000.0, step=5000.0, format="%.2f")
            estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=90000.0, step=5000.0, format="%.2f")
            num_products = st.selectbox("Number of Active Products", options=[1, 2, 3, 4], index=1)
            has_cr_card = st.checkbox("Holds active Credit Card", value=True)
            is_active = st.checkbox("Is actively engaged Member", value=True)

    # Prepare user input data
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_cr_card else 0,
        'IsActiveMember': 1 if is_active else 0,
        'EstimatedSalary': estimated_salary
    }])

    # Preprocess and execute predictions reactive to any parameter change
    processed_input = preprocessor.transform(input_data)
    churn_probability = float(model.predict(processed_input, verbose=0)[0][0])
    churn_percentage = churn_probability * 100

    # Classify Risk Levels
    if churn_probability < 0.25:
        risk_class = "risk-low"
        risk_status = "Low Risk Profile"
        risk_color = "#10b981"
        recommendation = "Customer demonstrates stable retention metrics. Maintain standard engagement strategies."
    elif churn_probability < 0.55:
        risk_class = "risk-medium"
        risk_status = "Moderate Risk Profile"
        risk_color = "#f59e0b"
        recommendation = "Elevated risk signals detected. Target with loyalty promotions or review product usage satisfaction."
    else:
        risk_class = "risk-high"
        risk_status = "High Risk Profile"
        risk_color = "#ef4444"
        recommendation = "Critical churn indicators present. Immediate outreach by dedicated customer success team advised."

    with col2:
        st.subheader("🔍 Prediction Insights & Evaluation")
        
        # Display Styled Results Card
        st.markdown(f"""
            <div class="card {risk_class}">
                <div class="metric-title">Risk Assessment Evaluation</div>
                <div class="metric-value" style="color: {risk_color};">{risk_status}</div>
                <p style="margin-top: 10px; color: #475569; font-size: 14px;">{recommendation}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 5. Interactive Gauge Plot (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_percentage, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 44, 'color': '#1e293b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': risk_color},
                'bgcolor': "#f1f5f9",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 25], 'color': '#e6f4ea'},
                    {'range': [25, 55], 'color': '#fef3c7'},
                    {'range': [55, 100], 'color': '#fee2e2'}
                ],
            }
        ))
        
        fig_gauge.update_layout(
            height=260, 
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 6. Interactive Comparative Analysis Plot (Plotly Scatter)
        if sample_df is not None:
            st.markdown("### Profile Comparison")
            st.caption("This visualization maps the currently evaluated profile against a sample cohort of existing customers.")
            
            # Map values for better readability in plot
            plot_sample = sample_df.copy()
            plot_sample['Churn Status'] = plot_sample['Exited'].map({0: 'Retained', 1: 'Churned'})
            
            # Create a reference row for the current customer
            current_cust = pd.DataFrame([{
                'Age': age,
                'Balance': balance,
                'Churn Status': 'Current Customer'
            }])
            
            combined_plot_data = pd.concat([plot_sample, current_cust], ignore_index=True)
            
            fig_compare = px.scatter(
                combined_plot_data, 
                x="Age", 
                y="Balance", 
                color="Churn Status",
                color_discrete_map={'Retained': '#94a3b8', 'Churned': '#f87171', 'Current Customer': risk_color},
                size=combined_plot_data['Churn Status'].apply(lambda x: 18 if x == 'Current Customer' else 6),
                labels={"Balance": "Account Balance ($)", "Age": "Age (Years)"},
                opacity=combined_plot_data['Churn Status'].apply(lambda x: 1.0 if x == 'Current Customer' else 0.4)
            )
            
            fig_compare.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)