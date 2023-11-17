import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
import pickle

# Load the Keras model
model = load_model("path/to/your_model.h5")

# Load the label encoder
with open("path/to/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the standard scaler
with open("path/to/standard_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app
st.title("Your Model Deployment with Streamlit")

# User input
user_input_categorical = st.selectbox("Select a category:", ["Yes", "bird"])
user_input_numerical = st.number_input("Enter a numerical value:", min_value=0.0)

# Preprocess user input
encoded_user_input = label_encoder.transform([user_input_categorical])
scaled_user_input = scaler.transform([[user_input_numerical]])

# Make predictions
predictions = model.predict([encoded_user_input, scaled_user_input])

# Display results
st.header("Prediction Results:")
st.write("Probability of class 0:", predictions[0, 0])
st.write("Probability of class 1:", predictions[0, 1])