import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Streamlit Page Setup

st.set_page_config(page_title="SmartPremium", layout="wide")

# Loading the trained model

def load_model():
    if os.path.exists("XGBoost_model.joblib"):
        st.info("Loaded XGBoost model")
        return joblib.load("XGBoost_model.joblib")
    elif os.path.exists("Random_Forest_model.joblib"):
        st.info("Loaded Random Forest model")
        return joblib.load("Random_Forest_model.joblib")
    elif os.path.exists("Linear_Regression_model.joblib"):
        st.info("Loaded Linear Regression model")
        return joblib.load("Linear_Regression_model.joblib")
    else:
        st.error("No model file found! Please train and save a model first.")
        st.stop()

model = load_model()

# Streamlit Page Layout

st.title("💰 SmartPremium: Insurance Cost Predictor")
st.markdown("Enter customer details to get a real-time estimate of the insurance premium.")

#Input Form

with st.form("insurance_form"):
    st.header("Customer & Policy Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=500000)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
        health_score = st.slider("Health Score (1-100)", min_value=1, max_value=100, value=75)

    with col3:
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
        smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
        exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

    submit_button = st.form_submit_button("Predict Premium")

# Prediction

if submit_button:
    input_data = pd.DataFrame([{
        'Age': age,
        'Annual Income': annual_income,
        'Number of Dependents': num_dependents,
        'Health Score': health_score,
        'Gender': gender,
        'Marital Status': marital_status,
        'Education Level': education_level,
        'Occupation': occupation,
        'Location': location,
        'Policy Type': policy_type,
        'Smoking Status': smoking_status,
        'Exercise Frequency': exercise_frequency,
        # Default values for other features
        'Previous Claims': 0,
        'Vehicle Age': 5,
        'Credit Score': 700,
        'Insurance Duration': 5,
        'Policy Start Date': '2022-01-01',
        'Customer Feedback': 'No feedback',
        'Property Type': 'House'
    }])

    try:
        predicted_premium = model.predict(input_data)[0]
        st.success(f"Estimated Insurance Premium: **${predicted_premium:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed. Check the input data and model compatibility. Error: {e}")


st.markdown("---")
