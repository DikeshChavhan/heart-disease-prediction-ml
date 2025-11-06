import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load("best_model.pkl")

# --- Page Configuration ---
st.set_page_config(
    page_title="💓 Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"
)

# --- Custom CSS Styling ---
page_style = """
<style>
/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ffebee, #e3f2fd);
}

/* Center Card */
.main-card {
    background-color: #ffffffcc;
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    max-width: 800px;
    margin: 50px auto;
    backdrop-filter: blur(10px);
}

/* Title Styling */
h1, h2, h3 {
    color: #d81b60;
    text-align: center;
    font-weight: 700;
}

/* Input labels */
label {
    font-weight: 600 !important;
    color: #37474f !important;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #d81b60;
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

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    background-color: #f8bbd0;
    padding: 10px;
    font-size: 14px;
    color: #4a148c;
    font-weight: 500;
}

/* Top-right contact section */
.contact-info {
    position: absolute;
    top: 15px;
    right: 25px;
    font-size: 15px;
    background-color: #fff;
    border-radius: 10px;
    padding: 10px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    text-align: right;
    line-height: 1.5;
}
.contact-info a {
    color: #0a66c2;
    text-decoration: none;
    font-weight: 600;
}
.contact-info a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Top Right Contact Info ---
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
    This machine learning web app predicts the **likelihood of heart disease** based on medical parameters.  
    Built with **Python, Scikit-learn, and Streamlit**, this app demonstrates end-to-end ML deployment.  
    
    ---
    ### 🧠 Model Info
    - **Dataset:** UCI Heart Disease  
    - **Algorithm Used:** Random Forest Classifier  
    - **Accuracy:** ~89%  
    - **Developer:** Dikesh Chavhan 🚀  
    
    ---
    ### 💡 How to Use
    1. Go to the **Predict** page.  
    2. Enter patient medical details.  
    3. Click **Predict** to see the result instantly.  
    
    ❤️ Stay healthy — prevention is better than cure!
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Page ---
elif page == "🧮 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🩺 Heart Disease Risk Prediction")
    st.markdown("### Please enter patient details below:")

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
            st.error("⚠️ The model predicts a **High Risk of Heart Disease.** Stay safe! ❤️‍🩹")
        else:
            st.success("✅ The model predicts a **Low Risk of Heart Disease.** Keep it up! 💪")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Dataset Info Page ---
elif page == "📊 Dataset Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📊 Dataset Information")
    st.markdown("""
    ### 📘 UCI Heart Disease Dataset Overview
    - **Rows:** 303  
    - **Target Variable:** `target` (1 = heart disease, 0 = no disease)  
    - **Features:**
        - Age, Sex, Chest Pain Type (cp)
        - Resting Blood Pressure (trestbps)
        - Serum Cholesterol (chol)
        - Fasting Blood Sugar (fbs)
        - Resting ECG (restecg)
        - Max Heart Rate (thalach)
        - Exercise Induced Angina (exang)
        - ST Depression (oldpeak)
        - Slope, Number of Vessels (ca)
        - Thalassemia (thal)
    
    ---
    ⚙️ Data Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | 💻 Machine Learning Project
</div>
""", unsafe_allow_html=True)
