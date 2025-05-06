import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# ---------- Load Assets ----------
def download_from_drive(file_id, filename):
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)

# Model Files
download_from_drive('1hgQybaepQfZ_6ZqcJ8kP48-ndzNbPBcX', 'regressor.pkl')
model = joblib.load('regressor.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
pca = joblib.load('pca.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# ---------- UI Header ----------
st.set_page_config(page_title="AI Financial Coach", page_icon="💡", layout="wide")
st.markdown("<h1 style='text-align: center; color: #3B82F6;'>💡 AI Financial Coach</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart suggestions to help you improve your financial habits.</h4>", unsafe_allow_html=True)

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("🧾 Enter Your Information")
    income = st.number_input("Income ($)", min_value=1000.0, max_value=100000.0, value=5000.0, step=500.0, help="Enter your monthly income.")
    age = st.slider("Age", 18, 65, 30, help="Enter your age.")
    dependents = st.slider("Dependents", 0, 10, 1, help="Number of dependents.")
    occupation = st.selectbox("Occupation", ["Employed", "Self_Employed", "Student", "Retired"], help="What is your occupation?")
    city_tier = st.selectbox("City Tier", ["Tier_1", "Tier_2", "Tier_3"], help="Select the tier of your city.")
    desired_savings = st.number_input("Desired Savings ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=100.0, help="Enter your desired savings goal.")
    actual_savings = st.number_input("Actual Savings ($)", min_value=0.0, value=2000.0, step=100.0, help="Enter the savings you currently have.")

# ---------- Spending Inputs ----------
st.subheader("📊 Monthly Expenses")
with st.expander("🏠 Fixed Expenses", expanded=True):
    col1, col2, col3 = st.columns(3)
    rent = col1.number_input("Rent ($)", min_value=0.0)
    loan_repay = col2.number_input("Loan Repayment ($)", min_value=0.0)
    insurance = col3.number_input("Insurance ($)", min_value=0.0)

with st.expander("🛒 Essential Expenses", expanded=True):
    col1, col2, col3 = st.columns(3)
    groceries = col1.number_input("Groceries ($)", min_value=0.0)
    transport = col2.number_input("Transport ($)", min_value=0.0)
    utilities = col3.number_input("Utilities ($)", min_value=0.0)
    col4, col5 = st.columns(2)
    healthcare = col4.number_input("Healthcare ($)", min_value=0.0)
    education = col5.number_input("Education ($)", min_value=0.0)

with st.expander("🎉 Discretionary Expenses", expanded=True):
    col1, col2, col3 = st.columns(3)
    eating_out = col1.number_input("Eating Out ($)", min_value=0.0)
    entertainment = col2.number_input("Entertainment ($)", min_value=0.0)
    misc = col3.number_input("Miscellaneous ($)", min_value=0.0)

# ---------- Derived Metrics ----------
fixed = {'Rent': rent, 'Loan_Repayment': loan_repay, 'Insurance': insurance}
essential = {'Groceries': groceries, 'Transport': transport, 'Utilities': utilities, 'Healthcare': healthcare, 'Education': education}
discretionary = {'Eating_Out': eating_out, 'Entertainment': entertainment, 'Miscellaneous': misc}

total_expenses = sum(fixed.values()) + sum(essential.values()) + sum(discretionary.values())
discretionary_spend = sum(discretionary.values())
discretionary_to_income = discretionary_spend / income
disposable_utilization = total_expenses / income
savings_vs_desired = (actual_savings - desired_savings) / income
rent_to_income = rent / income
loan_to_income = loan_repay / income
overspending_alert = disposable_utilization > 1.0
savings_rate = actual_savings / income

# ---------- Assemble Data ----------
user_data = {
    'Income': income,
    'Age': age,
    'Dependents': dependents,
    'Desired_Savings': desired_savings,
    'Actual_Savings': actual_savings,
    'Total_Expenses': total_expenses,
    'Disposable_Utilization': disposable_utilization,
    'Discretionary_to_Income': discretionary_to_income,
    'Savings_vs_Desired': savings_vs_desired,
    'Rent_to_Income': rent_to_income,
    'Loan_to_Income': loan_to_income,
    'Overspending_Alert': overspending_alert,
    'Savings_Rate': savings_rate,
    **fixed, **essential, **discretionary
}

# One-hot encoding
for val in ['Employed', 'Self_Employed', 'Student', 'Retired']:
    user_data[f"Occupation_{val}"] = 1 if occupation == val else 0
for val in ['Tier_1', 'Tier_2', 'Tier_3']:
    user_data[f"City_Tier_{val}"] = 1 if city_tier == val else 0
for col in feature_cols:
    if col not in user_data:
        user_data[col] = 0

input_df = pd.DataFrame([user_data])[feature_cols]
scaled_input = scaler.transform(input_df)

# ---------- Profile Insights ----------
profile_map = {
    0: "🧾 Likely high spenders with low savings",
    1: "💼 Balanced spenders with moderate savings",
    2: "📈 Frugal individuals with strong savings habits"
}

# ---------- Generate Nudges ----------
def generate_nudges(row):
    nudges = []
    # Overspending alert
    if row['Disposable_Utilization'] > 1.0:
        nudges.append("❌ You’ve overspent this month. Consider cutting expenses.")
    elif row['Disposable_Utilization'] > 0.85:
        nudges.append("⚠️ Close to overspending. Monitor your expenses.")
    # Savings gap
    if row['Savings_vs_Desired'] < -0.1:
        nudges.append("💰 You are significantly below your savings goal.")
    elif row['Savings_vs_Desired'] < -0.05:
        nudges.append("💰 Below your savings target. Consider increasing savings.")
    # High discretionary spending
    if row['Discretionary_to_Income'] > 0.35:
        nudges.append("🛍️ Very high discretionary spending.")
    elif row['Discretionary_to_Income'] > 0.25:
        nudges.append("🛍️ Consider reducing discretionary expenses.")
    # Rent burden
    if row['Rent_to_Income'] > 0.4:
        nudges.append("🏠 Rent is too high relative to your income.")
    elif row['Rent_to_Income'] > 0.3:
        nudges.append("🏠 Consider lowering your housing costs.")
    # Loan burden
    if row['Loan_to_Income'] > 0.3:
        nudges.append("📉 High loan repayment burden detected.")
    return nudges

def classify_nudge_priority(nudges):
    high = any("❌" in n or "💰 You are significantly" in n or "🏠 Rent is too high" in n for n in nudges)
    medium = any("⚠️" in n or "💰 Below your savings" in n or "🛍️ Very high" in n for n in nudges)

    if high:
        return "High"
    elif medium or len(nudges) >= 3:
        return "Medium"
    elif nudges:
        return "Low"
    return "None"

# ---------- Analyze Button ----------
# -- Predict when the "Analyze" button is clicked --
if st.button("Analyze"):
    try:
        # --- Your processing logic ---
        scaled_input = scaler.transform(input_df)
        savings_prediction = model.predict(scaled_input)[0] * 1e6
        cluster = kmeans.predict(scaled_input)[0]
        pca_result = pca.transform(scaled_input)[0]

        nudges = generate_nudges(user_data)
        priority = classify_nudge_priority(nudges)

        # ---------- Results Display ----------
        st.markdown("---")
        st.subheader("📈 Your Financial Snapshot")
        col1, col2, col3 = st.columns(3)
        col1.metric("💸 Suggested Adjustment", f"${savings_prediction:,.2f}")
        col2.metric("🔢 Cluster", f"{cluster}")
        col3.metric("🎯 Priority", priority)

        st.markdown(f"🧠 **Profile Insight**: {profile_map.get(cluster, 'N/A')}")

        if nudges:
            st.subheader("🔔 Nudges for You")
            for n in nudges:
                st.markdown(f"- {n}")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

