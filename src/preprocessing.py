import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import boxcox,zscore
from scipy.special import inv_boxcox
import os

#checking for null values
# df.isna().sum()

# df.info()

def histogram(df:pd.DataFrame,nrows:int,ncols:int,figsize:tuple,columns:list):
  fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
  axes = axes.flatten()
  for i,ax in enumerate(axes):
    ax.hist(df[columns[i]],bins="auto")
    ax.set_xlabel(f"{columns[i]}")

  plt.subplots_adjust(hspace=0.2, wspace=0.6)

  return plt.show()

def boxplot(df:pd.DataFrame,nrows:int,ncols:int,figsize:tuple,columns:list):
  fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
  axes = axes.flatten()
  for i,ax in enumerate(axes):
    ax.boxplot(df[columns[i]])
    ax.set_xlabel(f'{columns[i]}')
  
  plt.subplots_adjust(hspace=0.2, wspace=0.6)
  return plt.show()



def box_cox_transformation(df: pd.DataFrame,negative_or_zero_column,columns:list) -> pd.DataFrame:
  min_value = negative_or_zero_column.min()
  constant = 1e-5
  # print(1+constant)
  negative_or_zero_column = pd.Series(negative_or_zero_column + abs(min_value) + constant)

  box_cox_data = {}
  lambda_values = {}

  box_cox_data["longitude"],lambda_values["longitude"] = \
   boxcox(negative_or_zero_column)

  for column in columns:
    if column =="longitude":
      continue
    box_cox_data[column],lambda_values[column] = boxcox(df[column])

  transformed_df = pd.DataFrame(box_cox_data)
  return (transformed_df,lambda_values)



def inverse_box_cox(transformed_data,lambda_value):
  
  return inv_boxcox(transformed_data,lambda_value)


def single_value_boxcox(datas:list,lambda_values:list) -> list:
  transformed_values = list()
  for data,lambda_value in zip(datas,lambda_values):
    transformed_values.append(boxcox(data,lmbda=lambda_value))
  return transformed_values

      



# z_scores = zscore(new_df["totalRooms"])
# outliers = (z_scores > 3) | (z_scores < -3)
# print(f"Total outliers for totalRooms : {outliers.sum()}")

# outlier_indices = outliers[outliers==True].index.values

def outlier_detection(df:pd.DataFrame,columns:list) -> pd.DataFrame:
  outliers_dict = {}
  for column in columns:
    z_scores = zscore(df[column])
    outliers = (z_scores > 3) | (z_scores < -3)
    #print(f"Total outliers for {column} : {outliers.sum()}")
    if not outliers.sum():
      continue
    outliers_dict[column] = list(outliers[outliers==True].index.values)
  indices = np.array([value for sublist in outliers_dict.values() for value in sublist])
  unique_elements, counts = np.unique(indices,return_counts=True)
  new_df = df.drop(unique_elements)
  return new_df
  # print(df1["medianHouseValue"].mean(),df1["medianHouseValue"].std())
  # print(df1.shape)
  # df1.head()
# df1.to_csv("cal_housing_tuned.csv",index=False)
