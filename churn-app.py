import numpy as np
import pandas as pd 
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt

st.write("""

# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers

who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with the company. 

This app predicts the probability of a customer churning using Telco Customer data. Here customer churn means the customer does not make another purchase after a period of time. 

""")

df_selected = pd.read_csv("telco_churn.csv")

df_selected_all = df_selected[['gender', 'Partner', 'PaymentMethod', 'Dependents', 'PhoneService','tenure', 'MonthlyCharges', 'Churn']].copy()

st.write(df_selected_all)


uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender',('Male','Female'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 18.0)
        tenure = st.sidebar.slider('Tenure', 0.0,72.0, 0.0)
        data = {'Gender':[gender], 'PaymentMethod':[PaymentMethod], 'MonthlyCharges':[MonthlyCharges], 'Tenure':[tenure],}
        features = pd.DataFrame(data)
        return features
     
    input_df = user_input_features()

@st.cache
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_selected_all)
st.download_button(label="Download data as CSV",
    data=csv,
    file_name='DataFrame.csv',
    mime='text/csv')

load_clf = pickle.load(open('churn_clf.pkl', 'rb'))