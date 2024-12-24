import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('email_open_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title('Email Open Prediction for Marketing Campaigns')

st.write("""
### Enter the customer details below to predict whether they will open the email or not.
""")

# Input fields for the customer information (matching features used during training)
customer_age = st.number_input('Customer Age:', min_value=2, max_value=100, value=30)
purchase_history = st.number_input('Purchase History:', min_value=35.3, max_value=3121.5, value=100.0)

# Button to trigger prediction
predict_button = st.button('Predict')

if predict_button:
    # Store inputs into a DataFrame (only keep features the model expects)
    input_data = pd.DataFrame({
        'Customer_Age': [customer_age],
        'Purchase_History': [purchase_history]
    })

    # Scaling the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(scaled_data)

    # Output prediction
    if prediction == 1:
        st.success("The customer will likely open the email!")
    else:
        st.error("The customer is unlikely to open the email.")

# # Show the input data for reference
# st.write("### Customer Information:")
# st.write(input_data)