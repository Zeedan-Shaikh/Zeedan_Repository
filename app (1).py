
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model2 = tf.keras.models.load_model('model2.h5')

# Placeholder DataFrame (same structure as your training data)
X_train = pd.DataFrame(columns=['Num_children', 'Num_family', 'Account_length', 'Total_income', 
                                'Age', 'Years_employed', 'Gender_Male', 'Own_car_Yes', 
                                'Own_property_Yes', 'Work_phone_Yes', 'Phone_Yes', 'Email_Yes', 
                                'Unemployed_Yes', 'Income_type_Pensioner', 'Income_type_State servant',
                                'Income_type_Student', 'Income_type_Working', 'Education_type_Higher education',
                                'Education_type_Incomplete higher', 'Education_type_Lower secondary',
                                'Education_type_Secondary_secondary special', 'Family_status_Married',
                                'Family_status_Separated', 'Family_status_Single_unmarried', 
                                'Family_status_Widow', 'Housing_type_House_apartment', 
                                'Housing_type_Municipal apartment', 'Housing_type_Office apartment', 
                                'Housing_type_Rented apartment', 'Housing_type_With parents', 
                                'Occupation_type_Cleaning staff', 'Occupation_type_Cooking staff', 
                                'Occupation_type_Core staff', 'Occupation_type_Drivers', 
                                'Occupation_type_High skill tech staff', 'Occupation_type_IT staff', 
                                'Occupation_type_Laborers', 'Occupation_type_Low-skill Laborers', 
                                'Occupation_type_Managers', 'Occupation_type_Medicine staff', 
                                'Occupation_type_Other', 'Occupation_type_Private service staff', 
                                'Occupation_type_Realty agents', 'Occupation_type_Sales staff', 
                                'Occupation_type_Secretaries', 'Occupation_type_Security staff', 
                                'Occupation_type_Waiters_barmen staff'])

# Prediction function
def predict_credit_risk(
    Gender, Work_phone, Own_car, Own_property,Phone, Email, Unemployed, Num_children, Num_family,
    Total_income, Age, Years_employed, Income_type, Account_length, Housing_type,
    Occupation_type, Education_type, Family_status
    ):
    
    new_data = pd.DataFrame(0, index=[0], columns=X_train.columns)

    # The rest of your function goes here...
    # Copy the code you've provided for assigning values to `new_data`
    # followed by the prediction part

    # Predict credit risk
    pred2 = model2.predict(new_data)

    # Convert prediction to binary class
    prediction = (pred2 > 0.5).astype(int)

    if prediction == 1:
        prediction_final = "high"
    else:
        prediction_final = "low"

    return prediction_final

# Streamlit app
st.title("Credit Risk Predictor")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Work_phone = st.selectbox("Work Phone", ["Yes", "No"])
Own_car = st.selectbox("Own Car", ["Yes", "No"])
Own_property = st.selectbox("Own Property", ["Yes", "No"])
Phone = st.selectbox("Phone", ["Yes", "No"])
Email = st.selectbox("Email", ["Yes", "No"])
Unemployed = st.selectbox("Unemployed", ["Yes", "No"])
Num_children = st.number_input("Number of Children", 0, 20, 0)
Num_family = st.number_input("Number of Family Members", 1, 20, 1)
Total_income = st.number_input("Total Income", 0.0, 1000000.0, 0.0)
Age = st.number_input("Age", 18, 100, 18)
Years_employed = st.number_input("Years Employed", 0.0, 50.0, 0.0)
Income_type = st.selectbox("Income Type", ["Working", "Pensioner", "State servant", "Student"])
Account_length = st.number_input("Account Length (Years)", 0, 100, 0)
Housing_type = st.selectbox("Housing Type", ["Rented apartment", "With parents", "Municipal apartment", 
                                             "House_apartment", "Office apartment"])
Occupation_type = st.selectbox("Occupation Type", ["Laborers", "Core staff", "Managers", "Sales staff", "Drivers", 
                                                   "High skill tech staff", "Medicine staff", "IT staff", 
                                                   "Cleaning staff", "Cooking staff", "HR staff", "Low-skill Laborers", 
                                                   "Realty agents", "Security staff", "Waiters_barmen staff", 
                                                   "Other", "Private service staff", "Secretaries"])
Education_type = st.selectbox("Education Type", ["Higher education", "Secondary_secondary special", 
                                                 "Incomplete higher", "Lower secondary"])
Family_status = st.selectbox("Family Status", ["Married", "Separated", "Single_unmarried", "Widow"])

# Predict button
if st.button("Predict"):
    result = predict_credit_risk(Gender, Work_phone, Own_car, Own_property, Phone, Email, Unemployed, 
                                 Num_children, Num_family, Total_income, Age, Years_employed, 
                                 Income_type, Account_length, Housing_type, Occupation_type, 
                                 Education_type, Family_status)
    st.write(f"The predicted credit risk is: {result}")

