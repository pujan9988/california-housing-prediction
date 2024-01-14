import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

current_script_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_script_dir,"../data/cal_housing.csv")

df = pd.read_csv(csv_path)

def show_head(df:pd.DataFrame):
   df.head()

df.isna().sum()

df.info()

df.describe()

random_indices = random.sample(range(len(df)),100)
sample_df = df.iloc[random_indices]
sample_df.head()


columns = [column for column in sample_df.columns if not column=="medianHouseValue"]
vertical = sample_df["medianHouseValue"]
colors = ["red","green","blue","indigo","magenta","cyan","black","crimson"]
markers = ["x","o","X","*","+","D","d","P"]
fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(10,10))
axes = axes.flatten()
for i,ax in enumerate(axes):
  ax.scatter(sample_df[columns[i]],vertical,marker=markers[i],color=colors[i])
  # ax.set_title(f"{columns[i]} vs median_house_value")
  ax.set_xlabel(f"{columns[i]}")
  ax.tick_params(axis="both",labelsize=8)


plt.subplots_adjust(hspace=0.2, wspace=0.6)
plt.suptitle("medianHouseValue in y-axis",fontsize=15)
plt.show()


corr =df.corr()
corr.style.background_gradient()

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

predictors = df.drop('medianHouseValue', axis=1)

X = sm.add_constant(predictors)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

if (vif_data["VIF"]>10).any():
  print(f"The null hypothesis is rejected !!\n")

print(vif_data)

plt.hist(df["medianHouseValue"],bins="auto");

import scipy.stats as stats

stat, p_value = stats.shapiro(df["medianHouseValue"]);

alpha = 0.05
if p_value > alpha:
    print("Dependent variable is normally distributed (fail to reject the null hypothesis)")
else:
    print("Dependent variable is not normally distributed (reject the null hypothesis)")

