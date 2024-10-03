import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model and scaler
model = load_model('temperature_predictor.h5')
scaler = joblib.load('scaler.pkl')  # Assuming you saved your scaler as a pickle file

# Define a function to predict
def predict(temp_c, solar_rad, precip, day_of_year):
    input_data = np.array([[temp_c, solar_rad, precip, day_of_year]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit interface
st.title("GBC Temperature Predictor")

st.write("Enter the following details:")

# User inputs
temp_c = st.text_input("Air Temperature (°C)")
solar_rad = st.text_input("Solar Radiation (W/m²)")
precip = st.text_input("Precipitation (mm)")
day_of_year = st.text_input("Day of Year")

if st.button("Predict"):
    predicted_temp = predict(temp_c, solar_rad, precip, day_of_year)
    gbc_temp = predicted_temp[0]
    subgrade_layer = int(predicted_temp[1])

    # Map subgrade layer back to categories
    layer_category = {0: "Low", 1: "Medium", 2: "High"}
    layer_name = layer_category.get(subgrade_layer, "Unknown")

    st.write(f"**Predicted GBC Temperature (°C):** {gbc_temp:.4f}")
    st.write(f"**Predicted Subgrade Layer:** {layer_name}")
