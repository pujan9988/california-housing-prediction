import os
import sys
sys.path.append('..')

import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.special import inv_boxcox

from models.model import lambda_values
from src.preprocessing import boxcox

current_scirpt_dir = os.path.dirname(__file__)

model_path = os.path.join(current_scirpt_dir,"ranfor_model.joblib")
scaler_path = os.path.join(current_scirpt_dir,"scaler.joblib")

def load_model():
    model = joblib.load(model_path)
    return model

def load_std_scaler():
    scaler = joblib.load(scaler_path)
    return scaler


constant = 1e-5
min_value_longitude = -124.35

columns = ["longitude","latitude","housingMedianAge","totalRooms","totalBedrooms","population","households","medianIncome"]

lamda_values = [value for value in lambda_values.values()]


def make_input(input_data:list,lamda_values:list,scaler) -> list:

    input_data[0] += abs(min_value_longitude) + constant
    transformed_data = []

    #since input_data has 8 elements, the element of lamda_values(for medianHouseValue)
    #will be ignored in for loop due to the zip function 

    for data,lamda_value in zip(input_data,lamda_values):
        transformed_data.append(boxcox(data,lmbda=lamda_value))

    transformed_data = np.array(transformed_data).reshape(1,-1)
    transformed_data_scaled = scaler.transform(transformed_data)

    return transformed_data_scaled

def make_prediction(input_data:list,model) -> list:
    pred_value_transformed = model.predict(input_data)
    pred_value = inv_boxcox(pred_value_transformed,lamda_values[-1])
    return pred_value



if __name__ =="__main__":
    actual_output = 52900
    new_data = [-1.1906e+02,  3.6150e+01,  2.5000e+01,  2.4020e+03,  4.7800e+02,
         1.5270e+03,  4.6100e+02,  2.3194e+00]
    scaler = load_std_scaler()

    tranformed_data = make_input(new_data,lamda_values=lamda_values,scaler=scaler)

    model = load_model()

    pred_value = make_prediction(tranformed_data,model=model)
    print(pred_value)
    print(actual_output)
