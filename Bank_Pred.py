
import streamlit as st
import pandas as pd
import pickle
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
# Computing missing values with -2 (dummy variable)
def encodeData(cols, data):
    # computing missing values with -2 (dummy variable)
    data['job'] = data['job'].fillna(-2).astype(str)
    data['marital'] = data['marital'].fillna(-2).astype(str)
    data['education'] = data['education'].fillna(-2).astype(str)
    data['default'] = data['default'].fillna(-2).astype(str)
    data['housing'] = data['housing'].fillna(-2).astype(str)
    data['loan'] = data['loan'].fillna(-2).astype(str)

    le = LabelEncoder()
    for col in cols:
        data[col] = le.fit_transform(data[col])
    
    return data
def main(model,cols):
    st.title("Bank Marketing Prediction App")
    age = st.number_input("Age", min_value=0, max_value=120, value=18, step=1)

    job_options = [
    "blue-collar", "services", "admin.", "entrepreneur", "entrepreneur",
    "self-employed", "technician", "management", "student", "retired",
    "housemaid", "unemployed"
]

    marital_options = ["married", "single", "divorced"]

    education_options = [
    "basic.9y", "basic.4y", "high.school", "university.degree",
    "professional.course", "basic.6y", "illiterate"
]
    selected_job = st.selectbox("Select job", job_options)
    selected_marital = st.selectbox("Select Marital Status", marital_options)
    selected_education = st.selectbox("Select Education", education_options)


    default = ["no", "yes"]
    selected_name = st.selectbox("Select Has Credit in Default?", default)

    housing = ["no", "yes"]
    selected_name = st.selectbox("Select Has Housing Loan?", housing)

    loan = ["no", "yes"]
    selected_name = st.selectbox("Select Has Personal Loan?", loan)

    contact = ["cellular", "telephone"]
    selected_name = st.selectbox("Select Contact Communication Type", contact)

    month = ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"]
    selected_name = st.selectbox("Select Last Contact Month", month)

    day_of_week = ["mon", "tue", "wed", "thu", "fri"]
    selected_name = st.selectbox("Select Last Contact Day of the Week", day_of_week)

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

    poutcome = ["nonexistent", "failure", "success"]
    selected_name = st.selectbox("Select Outcome of Previous Marketing Campaign", poutcome)

    emp_var_rate = st.number_input("Enter Employment Variation Rate", min_value=0.0, max_value=999.0, value=0.0, step=0.01)

    cons_price_idx = st.number_input("Enter Consumer Price Index", min_value=0.0, max_value=999.0, value=0.0, step=0.01)
    cons_conf_idx = st.number_input("Enter Consumer Confidence Index",value=0.0, step=0.01)
    euribor3m = st.number_input("Enter Euribor 3 Month Rate", min_value=0.0, max_value=999.0, value=0.0, step=0.01)

    nr_employed = st.number_input("Enter Number of Employees", min_value=0.0, value=0.0, step=0.01)

    if st.button("Predict"):
        st.write("Prediction:")
        data = pd.DataFrame({
    'age': [age],
    'job': [selected_job],
    'marital': [selected_marital],
    'education': [selected_education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed]
})

        data_encoded = encodeData(cols, data)
        # Use the best_model object directly to make predictions
        prediction = model.predict(data_encoded)
        if prediction >= 0.5:
            st.success("Yes")
        else:
            st.success("No")
if __name__ == '__main__':
    model = joblib.load('best_model.joblib')
    main(model,cols)
