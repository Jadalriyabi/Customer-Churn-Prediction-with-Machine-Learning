import streamlit as st
import pandas as pd

st.title("Customer Churn Prediction")

# Load the data
df = pd.read_csv("churn.csv")

# Create a list of customer options for selection
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Dropdown for selecting a customer
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    # Extract the selected customer ID and Surname
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    st.write("Selected Customer ID:", selected_customer_id)
    st.write("Selected Customer Surname:", selected_customer_option.split(" - ")[1])

    # Filter the DataFrame to get details of the selected customer
    selected_customer = df[df["CustomerId"] == selected_customer_id]

    if not selected_customer.empty:
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=int(selected_customer['CreditScore'].iloc[0])
            )

            location = st.selectbox(
                "Location",
                ["Spain", "France", "Germany"],
                index=["Spain", "France", "Germany"].index(selected_customer['Geography'].iloc[0])
            )

            gender = st.radio(
                "Gender", 
                ["Male", "Female"],
                index=0 if selected_customer['Gender'].iloc[0] == 'Male' else 1
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=int(selected_customer['Age'].iloc[0])
            )

            tenure = st.number_input(
                "Tenure (years)",
                min_value=0,
                max_value=50,
                value=int(selected_customer['Tenure'].iloc[0])
            )

        with col2:
            balance = st.number_input(
                "Balance",
                min_value=0.0,
                max_value=1_000_000.0,  # Adjust max value as appropriate
                value=float(selected_customer['Balance'].fillna(0).iloc[0])  # Handle NaN values
            )

            num_products = st.number_input(
                "Number of Products",
                min_value=0,
                max_value=10,
                value=int(selected_customer['NumOfProducts'].iloc[0])
            )

            has_credit_card = st.checkbox(
                "Has Credit Card",
                value=bool(selected_customer['HasCrCard'].iloc[0])
            )

            is_active_member = st.checkbox(
                "Is Active Member",
                value=bool(selected_customer['IsActiveMember'].iloc[0])
            )

            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                max_value=1_000_000.0,  # Adjust max value as appropriate
                value=float(selected_customer['EstimatedSalary'].fillna(0).iloc[0])  # Handle NaN values
            )
    else:
        st.write("Selected customer data not found.")
