import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction App 💓", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of **heart disease** based on user inputs using a trained Machine Learning model.")

# Collect user input
st.subheader("Enter Patient Details")

age = st.number_input("Age", 18, 100, 40)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Convert categorical values if needed
sex = 1 if sex == "Male" else 0

# Prepare input for model
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts that this patient **has a high risk of heart disease**.")
    else:
        st.success("✅ The model predicts that this patient **is unlikely to have heart disease.**")
