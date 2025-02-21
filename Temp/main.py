import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import pickle

with open('scaler_texas.pkl','rb') as file:
    scaler_texas = pickle.load(file) 

with open('scaler_newYork.pkl','rb') as file:
    scaler_newyork = pickle.load(file) 

def model_loding(model_name):
    models = {
        'New York': 'ann_newYork.h5',
        'Texas': 'ann_texas.h5'
    }
    return load_model(models[model_name], custom_objects={'mse': MeanSquaredError})

model_choice = st.selectbox(
    'Choose a model to use:',
    ('New York', 'Texas')
)

model = model_loding(model_choice)

def predict(input1, input2, input3):
    # scaler = MinMaxScaler()
    inputs = np.array([[input1, input2, input3]])
    if model_choice=='Texas':
        new_data_scaled = scaler_texas.transform(inputs)
    else:
        new_data_scaled=scaler_newyork.transform(inputs)
    predicted_values = model.predict(new_data_scaled)
    return list(predicted_values)

input1 = st.number_input('Air_Temp_C')
input2 = st.number_input('Rainfall (inch)')
input3 = st.number_input('Day_of_year')

if st.button('Predict'):
    outputs = predict(input1, input2, input3)
    option = st.selectbox(
        'Select the output to display:',
        options=[f'Output {i+1}: {outputs[i]}' for i in range(len(outputs))],
        index=0
    )
    st.write('You selected:', option)
