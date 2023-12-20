import streamlit as st

import sys
sys.path.append("..")

from src.preprocessing import boxplot,histogram
from models.predict import make_input

st.header("California Housing Prediction")

