import streamlit as st
import pandas as pd
import joblib

#Load

artifact = joblib.load("artifacts/best_model.pkl")

preprocessor = artifact["preprocessor"]
model = artifact["model"]
feature_names = artifact["feature_names"]

credit_score_map = {
    0: "Poor",
    1: "Standard",
    2: "Good"
}

st.set_page_config(page_title="Credit Score Prediction", layout="wide")
st.header("Credit Score Predictor")

col1, col2, col3 = st.columns(3)

#personal info
with col1:

    st.markdown("### Personal Information")

    month = st.selectbox("Month of Observation", ["January","February","March","April","May","June","July","August"])
    age = st.slider("Age",min_value=18,max_value=100,value=30)
    occupation = st.selectbox(
        "Occupation",
        ["Lawyer","Architect","Scientist","Developer","Mechanic","Engineer","Accountant","Doctor",
         "Teacher","Media_Manager","Entrepreneur","Manager","Journalist","Musician","Writer"],index=5)
    
    st.markdown("### Finance")

    annual_income = st.slider("Annual Income",min_value=100.0,max_value=100000.0)
    monthly_balance = st.slider("Monthly Balance",min_value=0.0,max_value=1000.0)
    amount_invested_monthly = st.slider("Amount Invested Monthly",min_value=0.0,max_value=600.0)
    monthly_inhand_salary = st.slider("Monthly Inhand Salary", min_value=10.0,max_value=20000.0)
    num_bank_accounts = st.number_input("Number of Bank Accounts",min_value=0,max_value=10)

#credit
with col2:

    st.markdown("### Credit")

    credit_mix = st.radio("Credit Mix",["Bad", "Standard", "Good"],horizontal=True)
    spending_level = st.radio("Spending Level",["Low", "High"],horizontal=True)
    payment_size = st.radio("Payment Size",["Small", "Medium", "Large"],horizontal=True)
    payment_behaviour = (f"{spending_level}_spent_{payment_size}_value_payments")
    num_credit_card = st.number_input("Number of Credit Cards",min_value=0,value=4)
    years = st.number_input("Credit History Years",min_value=0,value=5)
    months = st.number_input("Credit History Months",min_value=0,max_value=11,value=6)
    credit_history_age = (f"{years} Years and {months} Months")
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=2)
    credit_utilization_ratio = st.slider( "Credit Utilization Ratio", min_value=0.0,value=100.0)
    interest_rate = st.slider( "Interest Rate",min_value=0.0,value=100.0)
    changed_credit_limit = st.number_input("Changed Credit Limit",value=5.0)



# ==========================================
# COLUMN 3 - LOANS & FINANCES
# ==========================================

with col3:

    st.markdown("### Loans")

    type_of_loan = st.selectbox("Type of loan", ["Credit-Builder Loan","Payday Loan",
                                                 "Personal Loan","Auto Loan","Debt Consolidation Loan",
                                                 "Home Equity Loan","Student Loan ","Mortgage Loan"])

    total_emi_per_month = st.slider("Total EMI Per Month",min_value=0.0,value=500.0)
    num_of_loan = st.number_input("Number of Loans",min_value=0,value=2)
    num_of_delayed_payment = st.number_input("Number of Delayed Payments",min_value=0,value=20)
    payment_of_min_amount = st.radio("Payment of Minimum Amount",["Yes", "No"],horizontal=True)
    delay_from_due_date = st.slider("Delay From Due Date (amount of days)",min_value=0,max_value=100)
    outstanding_debt = st.number_input("Outstanding Debt",min_value=0.0,value=1000.0)

if st.button("Predict My Credit Score"):

    data = pd.DataFrame(
        [{
            "Month": month,
            "Occupation": occupation,
            "Type_of_Loan": type_of_loan,
            "Credit_Mix": credit_mix,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Delay_from_due_date": delay_from_due_date,
            "Num_Credit_Inquiries": num_credit_inquiries,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Total_EMI_per_month": total_emi_per_month,
            "Age": age,
            "Annual_Income": annual_income,
            "Num_of_Loan": num_of_loan,
            "Num_of_Delayed_Payment": num_of_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Outstanding_Debt": outstanding_debt,
            "Amount_invested_monthly": amount_invested_monthly,
            "Monthly_Balance": monthly_balance,
            "Credit_History_Age": credit_history_age
        }]
    )

    processed = preprocessor.transform(data)
    processed = processed.reindex(columns=feature_names,fill_value=0)
    prediction = model.predict(processed)[0]
    probs = model.predict_proba(processed)[0]

    st.subheader("Prediction Probability")

    st.progress(float(probs[0]))
    st.write(f"Poor: {probs[0]:.1%}")

    st.progress(float(probs[1]))
    st.write(f"Standard: {probs[1]:.1%}")

    st.progress(float(probs[2]))
    st.write(f"Good: {probs[2]:.1%}")

    if prediction == 0:
        st.error("Poor Credit Score")
    elif prediction == 1:
        st.warning("Standard Credit Score")
    else:
        st.success("Good Credit Score")

    st.subheader("Main Factors")
    importance_df = pd.DataFrame({"Feature": processed.columns,"Importance": model.feature_importances_})
    importance_df = (importance_df.sort_values("Importance", ascending=False).head(10))

    st.bar_chart(importance_df.set_index("Feature"))