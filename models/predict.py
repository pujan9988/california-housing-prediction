import sys
sys.path.append('..')
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.special import inv_boxcox

from model import lambda_values
from src.preprocessing import boxcox

def load_model():
    model = joblib.load("ranfor_model.joblib")
    return model

def load_std_scaler():
    scaler = joblib.load("scaler.joblib")
    return scaler


constant = 1e-5
min_value_longitude = -124.35

columns = ["longitude","latitude","housingMedianAge","totalRooms","totalBedrooms","population","households","medianIncome"]

lamda_values = [value for value in lambda_values.values()]


def make_input(input_data:list,lamda_values:list) -> list:

    input_data[0] += abs(min_value_longitude) + constant
    transformed_data = np.empty(shape=(1,8))

    #since input_data has 8 elements, the element of lamda_values(for medianHouseValue)
    #will be ignored in for loop due to the zip function 

    for data,lamda_value in zip(input_data,lamda_values):
        np.append(transformed_data,boxcox(data,lmbda=lamda_value))

    scaler = load_std_scaler()
    transformed_data_scaled = scaler.transform(transformed_data)

    return transformed_data_scaled


if __name__ =="__main__":
    actual_output = 358500.000000
    new_data = [-122.220000,37.860000,21.000000,7099.000000,1106.000000,2401.000000,1138.000000,8.301400]

    tranformed_data = make_input(new_data,lamda_values=lamda_values)

    model = load_model()
    pred = model.predict(tranformed_data)

    pred_original = inv_boxcox(pred,lamda_values[-1])
    print(pred_original)
    print(actual_output)
