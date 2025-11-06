import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("best_model.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="💓 Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

# --- Custom Dark Theme CSS ---
page_style = """
<style>
/* Background Gradient (Dark Theme) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: #f5f6fa !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Card */
.main-card {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    max-width: 800px;
    margin: 50px auto;
    backdrop-filter: blur(10px);
    color: #f5f6fa;
}

/* Title Styling */
h1, h2, h3 {
    color: #f8bbd0 !important;
    text-align: center;
    font-weight: 700;
}

/* Labels */
label {
    font-weight: 600 !important;
    color: #f1f1f1 !important;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #e91e63;
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 18px;
    padding: 10px 24px;
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #ad1457;
}

/* Input fields */
input, select, textarea {
    background-color: #1e1e2f !important;
    color: white !important;
}

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    background: linear-gradient(90deg, #e91e63, #ad1457);
    padding: 10px;
    font-size: 14px;
    color: #fff;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Top-right contact info */
.contact-info {
    position: absolute;
    top: 15px;
    right: 25px;
    font-size: 15px;
    background-color: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    text-align: right;
    line-height: 1.5;
}
.contact-info a {
    color: #00acee;
    text-decoration: none;
    font-weight: 600;
}
.contact-info a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Top Right Contact ---
st.markdown("""
<div class="contact-info">
    <a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank">🔗 LinkedIn</a><br>
    📞 +91 8591531092
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 About", "🧮 Predict", "📊 Dataset Info"])

# --- About Page ---
if page == "🏠 About":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("💓 Heart Disease Prediction App")
    st.markdown("""
    ### 🤖 Project Overview
    Welcome to the **Heart Disease Prediction App**!  
    This ML-powered web app predicts the **risk of heart disease** using patient health data.  
    Built with **Python, Scikit-learn, and Streamlit** — deployed by *Dikesh Chavhan* 🚀

    ---
    ### 🧠 Model Details
    - **Dataset:** UCI Heart Disease  
    - **Algorithm:** Random Forest Classifier 🌲  
    - **Accuracy:** ~89%  
    - **Tech Stack:** Python | Pandas | Scikit-learn | Streamlit  

    ---
    ### 💡 How to Use
    1. Navigate to the **Predict** tab  
    2. Enter patient medical details  
    3. Click **Predict** to view results instantly  

    ❤️ *Stay fit. Stay heart-healthy!* 🫀
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Page ---
elif page == "🧮 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🩺 Heart Disease Risk Prediction")
    st.markdown("### Please enter the details below:")

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
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
        ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    sex = 1 if sex == "Male" else 0
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    st.write("")
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** ❤️‍🩹")
        else:
            st.success("✅ **Low Risk of Heart Disease. Stay Healthy! 💪**")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Dataset Info Page ---
elif page == "📊 Dataset Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📊 Dataset Information")
    st.markdown("""
    ### 📘 UCI Heart Disease Dataset Overview
    - **Rows:** 303  
    - **Target Variable:** `target` (1 = heart disease, 0 = no disease)  
    - **Features Include:**
        - Age, Sex, Chest Pain Type, Resting BP, Cholesterol, ECG Results,  
          Max Heart Rate, Exercise Angina, ST Depression, Slope, CA, Thal  
    ---
    ⚙️ **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | 💻 Machine Learning Project
</div>
""", unsafe_allow_html=True)
