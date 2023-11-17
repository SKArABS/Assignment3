import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
import pickle

# Load the Keras model
model = load_model("churn_predictor.keras")

# Load the label encoder
with open("encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the standard scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app
st.title("Predicting Customer Churn")

# User input
user_input_senior_citizen = st.radio("Is the customer a senior citizen?", ["Yes", "No"])
user_input_phone_service = st.radio("Does the customer have phone service?", ["Yes", "No"])
user_input_multiple_lines = st.selectbox("Select multiple lines status:", ["No phone service", "No", "Yes"])
user_input_streaming_tv = st.radio("Does the customer have streaming TV?", ["No", "Yes"])
user_input_streaming_movies = st.radio("Does the customer have streaming movies?", ["No", "Yes"])
user_input_paperless_billing = st.radio("Does the customer use paperless billing?", ["No", "Yes"])
user_input_payment_method = st.selectbox("Select a payment method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

user_input_monthly_charges = st.number_input("Enter a numerical value:", min_value=0.0)

# Preprocess user input
encoded_user_input_senior_citizen = label_encoder.transform([user_input_senior_citizen])
encoded_user_input_phone_service = label_encoder.transform([user_input_phone_service])
encoded_user_input_multiple_lines = label_encoder.transform([user_input_multiple_lines])
encoded_user_input_streaming_tv = label_encoder.transform([user_input_streaming_tv])
encoded_user_input_streaming_movies = label_encoder.transform([user_input_streaming_movies])
encoded_user_input_paperless_billing = label_encoder.transform([user_input_paperless_billing])
encoded_user_input_payment_method = label_encoder.transform([user_input_payment_method])
scaled_user_monthly_charges = scaler.transform([[user_input_monthly_charges]])

# Make predictions
predictions = model.predict([encoded_user_input_senior_citizen, encoded_user_input_phone_service, 
                             encoded_user_input_multiple_lines, encoded_user_input_streaming_tv,
                             encoded_user_input_streaming_movies, encoded_user_input_paperless_billing,
                             encoded_user_input_payment_method, scaled_user_monthly_charges ,
                             scaled_user_monthly_charges])

# Display results
st.header("Prediction Results:")
st.write("Probability of class 0:", predictions[0, 0])
st.write("Probability of class 1:", predictions[0, 1])