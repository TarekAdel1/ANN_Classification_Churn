import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = load_model("model.keras", compile=False)

# Load encoders & scaler
with open('gender_label_encoder.pkl', 'rb') as file:
    gender_label_encoder = pickle.load(file)

with open('geo_onehotencoder.pkl', 'rb') as file:
    geo_onehotencoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction App')

st.write("Fill in the customer details below to predict the likelihood of churn:")

# User input
geography = st.selectbox('Geography', geo_onehotencoder.categories_[0])
gender = st.selectbox('Gender', gender_label_encoder.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Balance', min_value=0.0, step=100.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, step=1)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0)
tenure = st.slider('Tenure (years)', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = geo_onehotencoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=geo_onehotencoder.get_feature_names_out(['Geography'])
)

# Combine input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Show results
st.subheader("Prediction Result")
st.write(f"Churn Probability: {prediction_proba:.2f}")

# Progress bar
st.progress(float(prediction_proba))

# Bar chart for visualization
st.bar_chart(pd.DataFrame({
    'Probability': [prediction_proba, 1 - prediction_proba]
}, index=['Churn', 'Not Churn']))

# Text result
if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')