import sys
sys.path.append("../..")

import streamlit as st 
from models.predict import lamda_values,make_input,make_prediction,load_model,load_std_scaler

st.set_page_config(
    page_title="Prediction",
    page_icon="🔍"
)

st.title("Prediction")

actual_output = 140000
new_data = [-122.26,37.85,50,1120,283,697,264,2.125]
scaler = load_std_scaler()

tranformed_data = make_input(new_data,lamda_values=lamda_values,scaler=scaler)

model = load_model()
# pred_original = make_prediction(tranformed_data,model=model)
pred_value = make_prediction(tranformed_data,model=model)

st.write(f"original value: {actual_output}")
st.write(f"predicted value: {pred_value}")
