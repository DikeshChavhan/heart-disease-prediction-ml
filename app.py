import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="💓 Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

# --- Custom CSS ---
page_style = """
<style>
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-15px); }
  to { opacity: 1; transform: translateY(0); }
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f6fa !important;
    animation: fadeIn 1.2s ease-in-out;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16213e, #1a1a2e);
    color: white;
}
[data-testid="stSidebar"] * {color: white !important;}
.main-card {
    background-color: rgba(255,255,255,0.08);
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    max-width: 850px;
    margin: 40px auto;
    backdrop-filter: blur(8px);
}
h1,h2,h3{color:#f8bbd0!important;text-align:center;font-weight:700;}
label{font-weight:600!important;color:#f1f1f1!important;}
div.stButton>button:first-child{
    background-color:#e91e63;color:white;border:none;border-radius:12px;
    font-size:18px;padding:10px 24px;transition:0.3s;
}
div.stButton>button:first-child:hover{background-color:#ad1457;}
input,select,textarea{background-color:#1e1e2f!important;color:white!important;}
.footer{
    position:fixed;left:0;bottom:0;width:100%;text-align:center;
    background:linear-gradient(90deg,#e91e63,#ad1457);
    padding:10px;font-size:14px;color:#fff;font-weight:500;
}
.contact-info{
    position:absolute;top:15px;right:25px;font-size:15px;
    background-color:rgba(255,255,255,0.1);border-radius:10px;
    padding:10px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.3);
    text-align:right;line-height:1.5;display:flex;align-items:center;gap:6px;
}
.contact-info a{color:#00acee;text-decoration:none;font-weight:600;}
.contact-info a:hover{text-decoration:underline;}
.linkedin-icon {
  width: 20px; height: 20px; vertical-align: middle;
}
.mode-box {
    background: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
    cursor: pointer;
    transition: 0.3s;
    border: 2px solid transparent;
}
.mode-box:hover {
    border: 2px solid #e91e63;
    transform: scale(1.03);
    background: rgba(255,255,255,0.15);
}
.mode-selected {
    border: 2px solid #e91e63;
    background: rgba(255,255,255,0.18);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Top-right contact ---
st.markdown("""
<div class="contact-info">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" class="linkedin-icon">
    <a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank">LinkedIn</a>
    | 📞 +91 8591531092
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧮 Predict", "💡 Health Tips", "📊 Dataset Info"])

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("""
    <div style='background: linear-gradient(90deg, #ff4081, #ec407a, #f06292);
    color: white; text-align: center; padding: 18px 10px; font-size: 24px;
    font-weight: bold; letter-spacing: 1px; border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); animation: fadeIn 2s ease-in-out;'>
    💓 Welcome to the <span style='color:#fff59d;'>Heart Disease Prediction App</span> 💻
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("💓 Heart Disease Prediction App")
    st.markdown("""
    ### 🤖 Overview
    This web app helps you **predict the likelihood of heart disease** using machine learning.  
    Built by **Dikesh Chavhan** with ❤️ using **Python, Scikit-learn, and Streamlit** 🚀  

    ---
    ### 🧠 Model
    - Dataset: UCI Heart Disease  
    - Algorithm: Random Forest Classifier  
    - Accuracy: ~89%

    ---
    ### 💡 Modes
    - 🧍 **Smart Mode:** Simple lifestyle-based questions (no medical report needed)  
    - 🩺 **Expert Mode:** Enter your medical parameters

    ❤️ *Your heart matters — check your health status today!* 🫀
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICTION PAGE ---
elif page == "🧮 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🩺 Heart Disease Risk Prediction")

    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        if st.button("🧍 Smart Mode (Easy)", key="smart"):
            st.session_state.mode = "Smart"
    with col_mode2:
        if st.button("🩺 Expert Mode (Full)", key="expert"):
            st.session_state.mode = "Expert"

    mode = st.session_state.get("mode", "Smart")
    st.markdown("---")

    if mode == "Smart":
        st.subheader("🧍 Easy Mode: Lifestyle & Symptoms")
        age = st.slider("Age", 18, 100, 40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoke = st.selectbox("Do you smoke?", ["No", "Sometimes", "Regularly"])
        exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
        chest_pain = st.selectbox("Do you feel chest pain?", ["No", "Sometimes", "Often"])
        tired = st.selectbox("Do you get tired easily?", ["No", "Yes"])
        overweight = st.selectbox("Are you overweight?", ["No", "Yes"])

        sex = 1 if gender == "Male" else 0
        cp = 2 if chest_pain == "Often" else (1 if chest_pain == "Sometimes" else 0)
        fbs = 1 if smoke == "Regularly" else 0
        exang = 1 if tired == "Yes" else 0
        chol = 250 if overweight == "Yes" else 180
        trestbps = 145 if overweight == "Yes" else 120
        thalach = 140 if exercise == "No" else 170
        oldpeak = 2.0 if chest_pain == "Often" else 0.5
        slope = 1
        ca = 0
        thal = 2
        restecg = 1

    else:
        st.subheader("🩺 Expert Mode: Medical Inputs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 40)
            sex = st.selectbox("Sex", ("Male", "Female"))
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        with col2:
            restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
            slope = st.selectbox("Slope (0–2)", [0, 1, 2])
            ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])
        sex = 1 if sex == "Male" else 0

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    if st.button("🔍 Predict Risk"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** ❤️‍🩹")
            st.session_state.pred = "High"
        else:
            st.success("✅ **Low Risk of Heart Disease. Stay Healthy! 💪**")
            st.session_state.pred = "Low"

    st.markdown("</div>", unsafe_allow_html=True)

# --- HEALTH TIPS PAGE ---
elif page == "💡 Health Tips":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("💡 Personalized Health Tips")

    pred = st.session_state.get("pred", "Low")

    if pred == "High":
        st.error("⚠️ Based on your inputs, you might have a **higher risk** of heart disease.")
        st.markdown("""
        ### 🩺 Recommended Lifestyle Changes:
        - 🥗 **Adopt a Heart-Healthy Diet:** Eat fruits, veggies, and whole grains.
        - 🏃‍♂️ **Exercise Regularly:** Aim for 30 minutes daily.
        - 🚭 **Quit Smoking:** It greatly increases heart risk.
        - 😌 **Manage Stress:** Meditation and yoga help.
        - 💊 **Regular Checkups:** Track cholesterol, BP, and sugar.
        """)
    else:
        st.success("✅ Your risk appears **low**, keep maintaining a healthy lifestyle!")
        st.markdown("""
        ### 🌟 Maintain Good Heart Health:
        - 🍎 **Eat Balanced Meals:** Avoid processed foods.
        - 🧘 **Stay Active:** Exercise & mindfulness.
        - 💤 **Sleep Well:** 7–8 hours daily.
        - 💧 **Stay Hydrated:** Drink enough water.
        - 🤝 **Routine Checkups:** Prevention is better than cure!
        """)

    st.markdown("</div>", unsafe_allow_html=True)

# --- DATASET INFO PAGE ---
elif page == "📊 Dataset Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📘 Dataset Information")
    st.markdown("""
    - **Rows:** 303  
    - **Target Variable:** `1` = disease, `0` = no disease  
    - **Features:**  
      Age, Sex, Chest Pain Type, Resting BP, Cholesterol, ECG, Heart Rate, etc.  
    ---
    ⚙️ Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class='footer'>
    Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | 💻 Machine Learning Project
</div>
""", unsafe_allow_html=True)
