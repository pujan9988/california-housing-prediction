import sys
sys.path.append("..")

from src.preprocessing import boxplot,histogram
from models.predict import make_input,load_model,load_std_scaler,lamda_values


import streamlit as st 

st.set_page_config(
    page_title="California Housing",
    page_icon="🏡",
)

st.title("California Housing Prediction🏘️")


st.markdown(
    """
    California Housing Prediction is a supervised machine learning algorithm model built to predict the California Housing Values.

    **👈 Select a option from the sidebar** to explore about different functions to inspect, visualize and analyze the dataset which was used to train this model and to make predictions based on your certain given values.

"""
)

st.write(
    """
    ## Project Overview

    This project focuses on predicting housing values in California using a Random Forest regression model. Below are the key components of the project:

    - **Dataset:** The model is trained on a dataset containing various features related to California housing, such as median income, housing median age, average rooms, etc.

    - **Model Training:** The Random Forest regression model is utilized for predicting housing values. The model is trained on historical data to learn patterns and relationships.

    - **Functionality:**
        - **Dataframe:** Explore the dataset and its statistics.
        - **Visualization:** Visualize data distribution through histograms and boxplots.
        - **Predict:** Make predictions using the trained model based on user input.

    Feel free to navigate through the different pages to get more insights!

    ## How to Use

    1. **Dataframe Page:** Explore the dataset and view summary statistics.

    2. **Visualization Page:** Visualize the distribution of key features using histograms and boxplots.

    3. **Predict Page:** Input your values and let the model predict the housing value for you.

    Enjoy exploring and understanding the California housing market!

    """
)