import os,sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st 

sys.path.append("../..")

from models.model import df,df_tuned


def draw_hist(feature_column,figsize:tuple = (10,10),
              color:str="blue",edgecolor:str='red',
              xlabel:str="Value",ylabel:str="Frequency",
              bins:str="auto"):
    
    fig,axes = plt.subplots(figsize=figsize)
    sns.histplot(df[feature_column],color=color,
             edgecolor=edgecolor,kde=True,ax=axes);
    axes.set_xlabel(xlabel=xlabel)
    axes.set_ylabel(ylabel=ylabel)
    axes.set_title(f"Histogram of {feature_column}")
    st.pyplot(fig)
    
def draw_boxplot(feature_column,figsize:tuple=(10,10)):
    
    fig,axes = plt.subplots(figsize=figsize)
    axes.boxplot(df[feature_column])
    axes.set_xlabel(f"{feature_column}")
    st.pyplot(fig)

st.set_page_config(page_title="visualization", 
                   page_icon="📊")


st.title("visualizations")

st.markdown("""
        ### Data Visualization

        Visualize the distribution of key features in the dataset using histograms and boxplots. This can help you understand the data better and identify potential patterns.
            """)


st.write("### Histograms")
selected_feature_histogram = st.selectbox("Select a feature for histogram:",df.columns)
if selected_feature_histogram:
    draw_hist(selected_feature_histogram)

st.write("### Boxplots")
selected_feature_boxplot = st.selectbox("Select a feature for boxplot:",df.columns)
if selected_feature_boxplot:
    draw_boxplot(selected_feature_boxplot)
    
