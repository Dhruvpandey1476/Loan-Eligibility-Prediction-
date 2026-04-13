import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set page config
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

# Load model and scaler
try:
    model = joblib.load('loan_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except:
    st.error("Error loading model files. Make sure loan_prediction_model.pkl, scaler.pkl, and feature_columns.pkl are in the same directory.")
    st.stop()

st.title("🏠 Loan Eligibility Predictor")
st.write("Predict whether your loan application will be approved using Machine Learning")

# Create sidebar with instructions
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses a K-Nearest Neighbors (KNN) model to predict loan eligibility.
    Enter your details below to get a prediction.
    """)

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=0)
    loan_term = st.number_input("Loan Amount Term (months)", min_value=12, max_value=480, step=12, value=360)
    credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    
with col2:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], format_func=lambda x: "3+" if x == 3 else str(x))
    education = st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 1 else "Undergraduate")
    self_employed = st.selectbox("Self Employed", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    property_area = st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: {0: "Rural", 1: "Semiurban", 2: "Urban"}[x])

# Prediction button
if st.button("🔮 Predict Loan Eligibility", use_container_width=True):
    try:
        # Create feature vector in the correct order
        # Encoding for categorical features
        dependents_0 = 1 if dependents == 0 else 0
        dependents_1 = 1 if dependents == 1 else 0
        dependents_2 = 1 if dependents == 2 else 0
        dependents_3_plus = 1 if dependents == 3 else 0
        
        property_rural = 1 if property_area == 0 else 0
        property_semiurban = 1 if property_area == 1 else 0
        property_urban = 1 if property_area == 2 else 0
        
        # Apply transformations (same as training data)
        applicant_income_sqrt = np.sqrt(applicant_income) if applicant_income > 0 else 0
        coapplicant_income_sqrt = np.sqrt(coapplicant_income) if coapplicant_income > 0 else 0
        loan_amount_sqrt = np.sqrt(loan_amount) if loan_amount > 0 else 0
        
        # Create sample data in the correct feature order
        sample_data = [[
            applicant_income_sqrt,
            coapplicant_income_sqrt,
            loan_amount_sqrt,
            loan_term,
            credit_history,
            gender,
            married,
            dependents_0,
            dependents_1,
            dependents_2,
            dependents_3_plus,
            education,
            self_employed,
            property_rural,
            property_semiurban,
            property_urban
        ]]
        
        # Scale the data
        sample_scaled = scaler.transform(sample_data)
        
        # Make prediction
        prediction = model.predict(sample_scaled)[0]
        
        # Display result
        st.divider()
        if prediction == 1:
            st.success("✅ **Loan Application: APPROVED**")
            st.balloons()
        else:
            st.error("❌ **Loan Application: NOT APPROVED**")
        
        st.divider()
        
        # Display input summary
        st.subheader("Your Application Summary")
        
        summary_data = {
            'Applicant Income': f"${applicant_income:,}",
            'Coapplicant Income': f"${coapplicant_income:,}",
            'Loan Amount': f"${loan_amount}k",
            'Loan Term (months)': loan_term,
            'Credit History': "Good" if credit_history == 1 else "Bad",
            'Gender': "Male" if gender == 1 else "Female",
            'Marital Status': "Yes" if married == 1 else "No",
            'Number of Dependents': "3+" if dependents == 3 else dependents,
            'Education': "Graduate" if education == 1 else "Undergraduate",
            'Self Employed': "Yes" if self_employed == 1 else "No",
            'Property Area': {0: "Rural", 1: "Semiurban", 2: "Urban"}[property_area]
        }
        
        for key, value in summary_data.items():
            st.write(f"**{key}:** {value}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.divider()
st.write("*Built with Streamlit and Machine Learning*")
