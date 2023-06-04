#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import pickle
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder


# In[4]:


cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']


# In[5]:


# Computing missing values with -2 (dummy variable)
def encodeData(cols, data):
    le = LabelEncoder()
    for col in cols:
        if data[col].dtype == object:  
            data[col] = data[col].fillna('-1').astype(str)  
            data[col] = le.fit_transform(data[col])
    
    return data



# In[15]:



def main(model, cols):
    st.title("Bank Marketing Prediction App")

    age = st.number_input("Age", min_value=0, max_value=120, value=18, step=1)

    job_options = [
        "blue-collar", "services", "admin.", "entrepreneur", "entrepreneur",
        "self-employed", "technician", "management", "student", "retired",
        "housemaid", "unemployed"
    ]
    selected_job = st.selectbox("Select job", job_options)

    marital_options = ["married", "single", "divorced"]
    selected_marital = st.selectbox("Select Marital Status", marital_options)

    education_options = [
        "basic.9y", "basic.4y", "high.school", "university.degree",
        "professional.course", "basic.6y", "illiterate"
    ]
    selected_education = st.selectbox("Select Education", education_options)

    default_options = ["no", "yes"]
    selected_default = st.selectbox("Select Has Credit in Default?", default_options)

    housing_options = ["no", "yes"]
    selected_housing = st.selectbox("Select Has Housing Loan?", housing_options)

    loan_options = ["no", "yes"]
    selected_loan = st.selectbox("Select Has Personal Loan?", loan_options)

    contact_options = ["cellular", "telephone"]
    selected_contact = st.selectbox("Select Contact Communication Type", contact_options)

    month_options = ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"]
    selected_month = st.selectbox("Select Last Contact Month", month_options)

    day_of_week_options = ["mon", "tue", "wed", "thu", "fri"]
    selected_day_of_week = st.selectbox("Select Last Contact Day of the Week", day_of_week_options)

    duration = st.number_input("Enter Last Contact Duration", min_value=0, max_value=9999, value=0, step=1)

    campaign = st.number_input("Enter Number of Contacts Performed", min_value=0, max_value=9999, value=0, step=1)

    pdays = st.number_input("Enter Number of Days Passed After Last Contact", min_value=0, max_value=9999, value=0, step=1)

    previous = st.number_input(
        "Enter Number of Contacts Performed Before this Campaign",
        min_value=0,
        max_value=9999,
        value=0,
        step=1
    )

    poutcome_options = ["nonexistent", "failure", "success"]
    selected_poutcome = st.selectbox("Select Outcome of Previous Marketing Campaign", poutcome_options)

    emp_var_rate = st.number_input("Enter Employment Variation Rate", max_value=999.0, value=0.0, step=0.01)

    cons_price_idx = st.number_input("Enter Consumer Price Index", min_value=0.0, max_value=999.0, value=0.0, step=0.01)

    cons_conf_idx = st.number_input("Enter Consumer Confidence Index", value=0.0, step=0.01)

    euribor3m = st.number_input("Enter Euribor 3 Month Rate", min_value=0.0, max_value=999.0, value=0.0, step=0.01)

    nr_employed = st.number_input("Enter Number of Employees", min_value=0.0, value=0.0, step=0.01)

    if st.button("Predict"):
        st.write("Prediction:")
        data = pd.DataFrame({
            'age': [age],
            'job': [selected_job],
            'marital': [selected_marital],
            'education': [selected_education],
            'default': [selected_default],
            'housing': [selected_housing],
            'loan': [selected_loan],
            'contact': [selected_contact],
            'month': [selected_month],
            'day_of_week': [selected_day_of_week],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [selected_poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed]
        })

        data_encoded = encodeData(cols, data)  
        prediction = model.predict(data_encoded)
        if prediction >= 0.5:
            st.success("Yes")
        else:
            st.success("No")
# In[16]:


if __name__ == '__main__':
    model = joblib.load('best_model.joblib')
    main(model,cols)








