# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

port = int(os.environ.get('PORT', 8501))

# Load the model
with open('Models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title('Predictor App')

# Input form
st.header('Input')
year = st.number_input('Year', min_value=2000, max_value=2021, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)

# Prediction
if st.button('Predict'):
    input_data = {'year': year, 'month': month}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f'The predicted value is: {prediction}')