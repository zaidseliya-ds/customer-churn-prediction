import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction & Analytics Dashboard")

if not os.path.exists('src/churn_model.pkl'):
    st.error("⚠️ Please run model.py first to generate the trained model file.")
else:
    model = joblib.load('src/churn_model.pkl')
    st.sidebar.header("🔮 Predict New Customer Churn")
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 150.0, 65.0)
    total_charges = age * monthly_charges
    contract = st.sidebar.selectbox("Contract Type", [0, 1, 2])
    tech_support = st.sidebar.selectbox("Has Tech Support? (1=Yes, 0=No)", [1, 0])
    
    input_data = pd.DataFrame([[age, tenure, monthly_charges, total_charges, contract, tech_support]], 
                              columns=['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'TechSupport'])
    
    if st.sidebar.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.sidebar.error("🚨 High Risk! Customer is likely to Churn.")
        else:
            st.sidebar.success("✅ Safe! Customer is likely to Stay.")

    # Mock Charts for dashboard
    st.subheader("📉 Historical Churn Insights")
    mock_df = pd.DataFrame({
        'Tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'Churn': np.random.choice(['Loyal', 'Churned'], 100)
    })
    fig = px.histogram(mock_df, x="Tenure", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
