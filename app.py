import streamlit as st
import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.title("Diabetes Prediction System")

preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    result = model.predict(data_pca)

    if result[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")
