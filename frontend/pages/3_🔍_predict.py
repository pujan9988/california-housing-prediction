import sys
sys.path.append("../..")
import numpy as np 


import streamlit as st 
from models.predict import lamda_values, make_input, make_prediction, load_model, load_std_scaler

st.set_page_config(
    page_title="Prediction",
    page_icon="🔍"
)

st.title("Prediction")

# Input fields
longitude = st.number_input("Enter the value of block group longitude:", value=0.0)
latitude = st.number_input("Enter the value of block group latitude:", value=0.0)
housingMedianAge = st.number_input("Enter the Housing Median Age in block group:", value=0.0)
totalRooms = st.number_input("Enter the average no of rooms per household:", value=0.0)
totalBedrooms = st.number_input("Enter the average no of bedrooms per household:", value=0.0)
population = st.number_input("Enter the block group population", value=0.0)
households = st.number_input("Enter the average no of household members", value=0.0)
medianIncome = st.number_input("Enter the median income in block group", value=0.0)

# Button to trigger prediction
if st.button("Predict"):

    new_data = [longitude, latitude, housingMedianAge, totalRooms,
                totalBedrooms, population, households, medianIncome]

    scaler = load_std_scaler()
    transformed_data = make_input(new_data, lamda_values=lamda_values, scaler=scaler)


    model = load_model()
    pred_value = make_prediction(transformed_data, model=model)
    pred_value = pred_value.item()
    
    st.success(f"Predicted Housing Value: ${pred_value:.2f}")
