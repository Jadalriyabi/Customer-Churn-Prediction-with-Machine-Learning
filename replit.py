import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import plotly.graph_objs as go
import utils  

# Function to load models
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load models
xgboost_model = load_model('Data/xgb_model.pkl')
native_bayes_model = load_model('Data/nb_model.pkl')
random_forest_model = load_model('Data/rf_model.pkl')
knn_model = load_model('Data/knn_model.pkl')

# Function to prepare input data for prediction
def prepare_input(credit_score, location, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCreditCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Georgraphy_France': 1 if location == "France" else 0,
        'Georgraphy_Germany': 1 if location == "Germany" else 0,
        'Georgraphy_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Function to align input features with model
def align_features(input_df, model):
    """Align input features with the model's expected features."""
    if hasattr(model, 'get_booster') and model.get_booster().feature_names:
        expected_features = model.get_booster().feature_names
        
        # Add missing features with default value 0
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match model's training order
        input_df = input_df[expected_features]
    
    return input_df

# Function to make predictions
def make_prediction(input_df, input_dict):
    # Align features for each model
    input_df_xgb = align_features(input_df.copy(), xgboost_model)
    input_df_rf = input_df.copy()  # Assuming random_forest_model doesn't need alignment
    input_df_knn = input_df.copy()  # Assuming knn_model doesn't need alignment

    # Fill missing values and ensure float type
    input_df_xgb = input_df_xgb.fillna(0).astype(float)
    input_df_rf = input_df_rf.fillna(0).astype(float)
    input_df_knn = input_df_knn.fillna(0).astype(float)

    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df_xgb)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df_rf)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df_knn)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))
    return avg_probability, probabilities

# Function to generate email
def generate_email(probability, input_dict, explanation, surname):
    # Simulated email generation code
    email = f"""
    Dear {surname},

    Based on our analysis, we believe there are opportunities to improve your banking experience with us. 
    Here are some incentives we are offering:

    - Lower interest rates for loyal customers
    - Personalized financial advice
    - Special rewards for account activity

    Please don't hesitate to reach out for further assistance.

    Best regards,
    HS Bank
    """
    return email

# Streamlit app title
st.title("🧮 Customer Churn Prediction")

# Load dataset
df = pd.read_csv("Data/churn.csv")

# Create list of customers for dropdown
customers = [f"{row['CustomerId']} - {row['Surname']} " for _, row in df.iterrows()]

# Dropdown for customer selection
selected_customer_option = st.selectbox("Select a customer", customers, key='customer_selectbox')

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)

    # First column for input fields
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore']),
            key='credit_score'
        )
        location = st.selectbox(
            "Location", 
            ["Spain", "France", "Germany"], 
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
            key='location_selectbox'
        )
        gender = st.radio(
            "Gender", 
            ["Male", "Female"], 
            index=0 if selected_customer['Gender'] == "Male" else 1,
            key='gender_radio'
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age']),
            key='age_input'
        )
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure']),
            key='tenure_input'
        )

    # Second column for input fields
    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0, 
            value=float(selected_customer['Balance']),
            key='balance_input'
        )
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts']),
            key='products_input'
        )
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard']),
            key='credit_card_checkbox'
        )
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']),
            key='active_member_checkbox'
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']),
            key='salary_input'
        )

    # Add a button to trigger the prediction and results display
    if st.button('Show Churn Prediction'):

        # Prepare input data for prediction
        input_df, input_dict = prepare_input(
            credit_score, location, gender, age, tenure, balance, num_products, 
            has_credit_card, is_active_member, estimated_salary
        )

        # Make predictions and display results
        avg_probability, probabilities = make_prediction(input_df, input_dict)

        # Display results
        st.subheader("📊 Churn Probability and Model Comparison")
        st.write(f"Average Probability: {avg_probability:.2%}")
        st.write("Model Probabilities:", probabilities)

        # Generate and display email
        email = generate_email(avg_probability, input_dict, "", selected_customer['Surname'])
        st.subheader("📧 Personalized Email")
        st.markdown(email)
