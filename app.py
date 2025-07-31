import streamlit as st
import streamlit as st
import numpy as np
import joblib
import pandas as pd
# Load model
model = joblib.load('loan_model.pkl')
st.markdown(
    """
    <style>
        .stApp {
            background-color:#f3e5f5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# UI
st.title("Loan Approval Prediction")
st.write("Fill out the details to check loan approval status.")
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
no_of_dependents = st.number_input("Number of Dependents", min_value=0)
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Asset Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

if st.button("Predict Loan Approval"):
    # Encode categorical inputs 
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0

    # Create array
    input_data = np.array([[
        no_of_dependents, income_annum, loan_amount, loan_term,
        cibil_score, residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value, education, self_employed
    ]])

    # Convert to DataFrame
    columns = [
        ' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
        ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
        ' luxury_assets_value', ' bank_asset_value', ' education', ' self_employed'
    ]

    
    input_df = pd.DataFrame(input_data, columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]

    result = "Approved " if prediction == 0 else "Rejected "
    st.success(f"Loan Status: {result}")
