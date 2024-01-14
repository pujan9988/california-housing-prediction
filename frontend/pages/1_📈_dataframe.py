import streamlit as st 
import os,sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../..")

from models.model import df,df_tuned

st.set_page_config(page_title="Dataframe", 
                   page_icon="📈")

# current_script_dir = os.path.dirname(__file__)
# csv_path = os.path.join(current_script_dir,"../../data/cal_housing.csv")
# df = pd.read_csv(csv_path)

st.markdown("""
## Dataset Overview
**Attribute Information**:

        - longitude          block group longitude
        - latitude           block group latitude
        - housingMedianAge   median house age in block group
        - totalRooms         average number of rooms per household
        - totalBedrooms      average number of bedrooms per household
        - population         block group population
        - households         average number of household members
        - medianIncome       median income in block group
        - medianHouseValue   median house value


The **target** variable is the **medianHouseValue** for California districts,
expressed in dollars ($1).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.
            


**Explore the dataset used to train the model. This page provides a glimpse of the dataset, including summary statistics and key information about each feature.**

### Sample Data

        """
    )
st.dataframe(df.head(20))
st.write("### Data Summary")
st.dataframe(df.describe())

st.write("### Data Types")
st.dataframe(df.dtypes)

st.write('### Correlation Matrix')
cor = df.corr()
st.dataframe(cor.style.background_gradient())

st.write("### Tuned/Modified dataframe after performing some data processing to improve the model performance :-")
st.dataframe(df_tuned.head(10))

st.write("""## Interactive column selection
         """)
selected_columns = st.multiselect("Select Columns:", df.columns)
if selected_columns:
    st.dataframe(df[selected_columns])

