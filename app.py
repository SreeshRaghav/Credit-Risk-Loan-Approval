import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

from fuzzy_logic.fuzzy_module import fuzzy_credit_score
from genetic_algorithm.ga_module import optimize_rules

model = tf.keras.models.load_model('models/nn_credit_risk_model.h5')
scaler = joblib.load('models/scaler.save')

housing_mapping     = {'own':0, 'rent':1, 'free':2}
saving_acc_mapping  = {'rich':0, 'quite rich':1, 'moderate':2, 'little':3, 'missing':4}
checking_acc_mapping= {'rich':0, 'moderate':1, 'little':2, 'missing':3}
purpose_mapping     = {'radio/TV':0, 'car':1, 'furniture/equipment':2, 'business':3, 'domestic appliances':4}

st.title("Intelligent Credit Risk Scoring & Loan Approval")

age          = st.number_input('Age', min_value=18, max_value=80, value=30)
sex          = st.selectbox('Sex', ['Male', 'Female'])
job          = st.number_input('Job Level (0-3)', min_value=0, max_value=3, value=1)
housing      = st.selectbox('Housing status', list(housing_mapping.keys()))
saving_acc   = st.selectbox('Savings Account', list(saving_acc_mapping.keys()))
checking_acc = st.selectbox('Checking Account', list(checking_acc_mapping.keys()))
credit_amount= st.number_input('Credit Amount', min_value=0.0, max_value=40000.0, value=5000.0)
duration     = st.number_input('Duration (months)', min_value=1, max_value=60, value=24)
purpose      = st.selectbox('Purpose', list(purpose_mapping.keys()))

if st.button('Evaluate Loan'):

    sex_val = 1 if sex == 'Male' else 0
    # Prepare feature vector for NN (must match training order)
    X = np.array([[age, sex_val, job, housing_mapping[housing], 
                   saving_acc_mapping[saving_acc], checking_acc_mapping[checking_acc], 
                   credit_amount, duration, purpose_mapping[purpose]]])
    
    # Only scale expected numeric columns!
    X[:, [0, 6, 7]] = scaler.transform(X[:, [0, 6, 7]])

    nn_pred = model.predict(X)[0][0]
    fuzzy_score = fuzzy_credit_score(age, credit_amount, duration, saving_acc, checking_acc)
    ga_features = optimize_rules()

    st.write(f"Neural Network Risk Probability: {nn_pred:.2f}")
    st.write(f"Fuzzy Credit Score: {fuzzy_score if fuzzy_score is not None else 'N/A'}")
    st.write(f"GA Optimized Features: {ga_features}")

    # Adopt your domain decision logic
    approval_threshold = 0.5
    fuzzy_threshold = 6
    if nn_pred < approval_threshold and (fuzzy_score is not None and fuzzy_score > fuzzy_threshold):
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")
