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

# --- Custom CSS + Animation ---
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
    text-align:right;line-height:1.5;
}
.contact-info a{color:#00acee;text-decoration:none;font-weight:600;}
.contact-info a:hover{text-decoration:underline;}
/* Animated Banner */
.banner {
  background: linear-gradient(90deg, #ff4081, #ec407a, #f06292);
  color: white;
  text-align: center;
  padding: 18px 10px;
  font-size: 24px;
  font-weight: bold;
  letter-spacing: 1px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  animation: fadeIn 2s ease-in-out;
}
.typing {
  overflow: hidden;
  white-space: nowrap;
  border-right: 3px solid white;
  animation: typing 4s steps(40, end), blink 1s step-end infinite;
}
@keyframes typing {
  from { width: 0; }
  to { width: 100%; }
}
@keyframes blink {
  50% { border-color: transparent; }
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Animated Welcome Banner ---
st.markdown("""
<div class="banner">
  💓 <span class="typing">Welcome to the AI-Powered Heart Disease Prediction App!</span> 💻
</div>
""", unsafe_allow_html=True)

# --- Top-right contact ---
st.markdown("""
<div class="contact-info">
    <a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank">🔗 LinkedIn</a><br>
    📞 +91 8591531092
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 About", "🧮 Predict", "📊 Dataset Info", "📈 Model Insights"])

# --- About Page ---
if page == "🏠 About":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("💓 Heart Disease Prediction App")
    st.markdown("""
    ### 🤖 Overview
    This AI-powered app predicts the **likelihood of heart disease** using machine learning.  
    Built by **Dikesh Chavhan** with ❤️ using **Python, Scikit-learn, and Streamlit** 🚀  

    ---
    ### 🧠 Model
    - Dataset: UCI Heart Disease  
    - Algorithm: Random Forest Classifier  
    - Accuracy: ~89%

    ---
    ### 💡 Modes
    - 🧍 **Smart Mode:** Simple lifestyle-based questions (no medical report needed)  
    - 🩺 **Expert Mode:** Enter actual medical parameters

    ❤️ *Your health matters — use AI for awareness!* 🫀
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Page ---
elif page == "🧮 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🩺 Heart Disease Risk Prediction")

    mode = st.radio("Select Input Mode:", ["🧍 Smart Mode (Easy)", "🩺 Expert Mode (Full)"])
    st.markdown("---")

    if mode == "🧍 Smart Mode (Easy)":
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
        else:
            st.success("✅ **Low Risk of Heart Disease. Stay Healthy! 💪**")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Dataset Info Page ---
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

# --- Model Insights Page ---
elif page == "📈 Model Insights":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📈 Model Insights")

    try:
        importances = model.feature_importances_
        features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
        df_imp = df_imp.sort_values(by="Importance", ascending=False)

        st.subheader("🌟 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_imp["Feature"], df_imp["Importance"], color="#e91e63")
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Top Features Impacting Predictions")
        st.pyplot(fig)

        st.markdown("""
        ### 🧠 Interpretation
        - **cp (Chest Pain Type):** Major heart stress indicator  
        - **thalach (Max Heart Rate):** Lower rates = higher risk  
        - **oldpeak (ST Depression):** High = likely disease  
        - **chol (Cholesterol):** High = risk factor  
        - **age:** Older = higher probability  
        """)
    except Exception:
        st.warning("⚠️ Feature importances unavailable for this model type.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | 💻 Machine Learning Project
</div>
""", unsafe_allow_html=True)
