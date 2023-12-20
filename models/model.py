import sys
sys.path.append('..')

import pandas as pd
import os
import random
import joblib
from time import perf_counter

from src.preprocessing import outlier_detection,box_cox_transformation
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler

current_script_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_script_dir,"../data/cal_housing.csv")

df = pd.read_csv(csv_path)
columns = [column for column in df.columns]

new_df,lambda_values = box_cox_transformation(df,df["longitude"],columns=columns)
df_tuned = outlier_detection(new_df,columns)

# print(new_df.head())
def main():

    X = df_tuned.drop("medianHouseValue",axis=1).values
    y = df_tuned["medianHouseValue"].values


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44)



    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    LinReg = LinearRegression()
    svm = LinearSVR(max_iter=10000,C=11,random_state=42)
    ranfor = RandomForestRegressor(n_estimators=102, random_state=42)
    gradboost = GradientBoostingRegressor(n_estimators=90, learning_rate=0.2, max_depth=3, random_state=42)

    models = [LinReg,svm,ranfor,gradboost]

    for model in models:

        model.fit(X_train_scaled,y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Performance of {model} :- ")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}\n")


    # scalers = StandardScaler()
    # X_scaled = scalers.fit_transform(X)

    # ran = RandomForestRegressor(n_estimators=102, random_state=42)

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # # Performing k-fold cross-validation
    # mse_scores = cross_val_score(ran, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    # r2_scores = cross_val_score(ran, X_scaled, y, cv=kf, scoring='r2')

    # mse_scores = -mse_scores

    # for fold, (mse, r2) in enumerate(zip(mse_scores, r2_scores), 1):
    #     print(f"Fold {fold}:-")
    #     print(f"Mean Squared Error: {mse}")
    #     print(f"R-squared: {r2}\n")

    # print("Average Performance Across Folds:")
    # print(f"Mean Squared Error: {mse_scores.mean()}")
    # print(f"R-squared: {r2_scores.mean()}")

    model_filename = "ranfor_model.joblib"
    scaler_filename = "scaler.joblib"

    # saving the model and the scaler object
    joblib.dump(ranfor,model_filename)
    joblib.dump(scaler,scaler_filename)

    print(f"Random forest model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

if __name__ == "__main__":
    main()