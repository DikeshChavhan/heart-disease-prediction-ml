import streamlit as st
import numpy as np
import joblib

# Load your trained model
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
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f6fa !important;
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
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Top-right contact ---
st.markdown("""
<div class="contact-info">
    <a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank">🔗 LinkedIn</a><br>
    📞 +91 8591531092
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 About", "🧮 Predict", "📊 Dataset Info"])

# --- About ---
if page == "🏠 About":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("💓 Heart Disease Prediction App")
    st.markdown("""
    ### 🤖 Overview
    This app predicts the **likelihood of heart disease** using machine learning.  
    Built with **Python, Scikit-learn, and Streamlit** by *Dikesh Chavhan* 🚀  

    ---
    ### 🧠 Model
    - Dataset: UCI Heart Disease  
    - Algorithm: Random Forest Classifier  
    - Accuracy: ~89%

    ---
    ### 💡 Modes
    - 🧍 **Smart Mode:** Simple lifestyle-based questions (no medical report needed)  
    - 🩺 **Expert Mode:** Enter actual medical report values

    ❤️ *Health is wealth — stay fit, stay aware!* 🫀
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction ---
elif page == "🧮 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🩺 Heart Disease Risk Prediction")

    # Mode toggle
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

        # Estimate approximate values
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

    # Prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** ❤️‍🩹")
        else:
            st.success("✅ **Low Risk of Heart Disease. Stay Healthy! 💪**")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Dataset Info ---
elif page == "📊 Dataset Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📊 Dataset Information")
    st.markdown("""
    ### 📘 UCI Heart Disease Dataset
    - **Rows:** 303  
    - **Target:** `1` = disease, `0` = no disease  
    - **Features:** Age, Sex, Chest Pain, BP, Cholesterol, ECG, Heart Rate, etc.  
    ---
    ⚙️ Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | 💻 Machine Learning Project
</div>
""", unsafe_allow_html=True)
