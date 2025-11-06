import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from deep_translator import GoogleTranslator

# Load trained model
model = joblib.load("best_model.pkl")

# Page configuration
st.set_page_config(
    page_title="HeartCheck | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Available languages
languages = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "मराठी (Marathi)": "mr",
    "தமிழ் (Tamil)": "ta"
}

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=60)
st.sidebar.title("💓 HeartCheck")

selected_lang = st.sidebar.selectbox("🌐 Choose Language", list(languages.keys()))
target_lang = languages[selected_lang]

# Translation function
def tr(text):
    """Translate text to the selected language."""
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

# Sidebar Navigation
page = st.sidebar.radio(
    tr("Navigate"),
    [tr("🏠 Home"), tr("🧮 Predict"), tr("💡 Health Tips"), tr("📊 Dataset Info"), tr("🗣 Virtual Assistant")]
)

# Contact info (top-right)
st.markdown("""
<div style="position:absolute;top:15px;right:25px;font-size:15px;
background-color:rgba(255,255,255,0.1);border-radius:10px;
padding:10px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.3);
display:flex;align-items:center;gap:6px;">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20">
<a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank" style="color:#00acee;text-decoration:none;font-weight:600;">
LinkedIn</a> | 📞 +91 8591531092
</div>
""", unsafe_allow_html=True)

# ---------------- HOME ----------------
if "🏠" in page:
    st.markdown(f"""
    <div style='background: linear-gradient(90deg,#ff4081,#ec407a,#f06292);
    color:white;text-align:center;padding:18px;font-size:24px;font-weight:bold;
    border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);'>
    💓 {tr("Welcome to HeartCheck — Predict. Prevent. Protect.")} 💻
    </div>
    """, unsafe_allow_html=True)

    st.title(tr("About HeartCheck"))
    st.write(tr("""
    HeartCheck helps you estimate the likelihood of heart disease
    using simple lifestyle inputs or full medical data.
    Built by Dikesh Chavhan using Python & Machine Learning.
    """))

    st.markdown("### " + tr("Model Details"))
    st.markdown(f"""
    - 📘 {tr('Dataset')}: UCI Heart Disease  
    - 🤖 {tr('Model')}: Random Forest Classifier  
    - 🎯 {tr('Accuracy')}: ~89%  
    - 💾 {tr('Tech Stack')}: Streamlit | Scikit-Learn | Plotly  
    """)

# ---------------- PREDICT ----------------
elif "🧮" in page:
    st.title(tr("🩺 Heart Disease Risk Prediction"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button(tr("🧍 Smart Mode")):
            st.session_state.mode = "Smart"
    with col2:
        if st.button(tr("🩺 Expert Mode")):
            st.session_state.mode = "Expert"

    mode = st.session_state.get("mode", "Smart")
    st.markdown("---")

    if mode == "Smart":
        age = st.slider(tr("Age"), 18, 100, 40)
        gender = st.selectbox(tr("Gender"), [tr("Male"), tr("Female")])
        smoke = st.selectbox(tr("Do you smoke?"), [tr("No"), tr("Sometimes"), tr("Regularly")])
        exercise = st.selectbox(tr("Do you exercise regularly?"), [tr("Yes"), tr("No")])
        chest_pain = st.selectbox(tr("Do you feel chest pain?"), [tr("No"), tr("Sometimes"), tr("Often")])
        tired = st.selectbox(tr("Do you get tired easily?"), [tr("No"), tr("Yes")])
        overweight = st.selectbox(tr("Are you overweight?"), [tr("No"), tr("Yes")])

        sex = 1 if gender in ["Male", "पुरुष", "आदमी"] else 0
        cp = 2 if chest_pain == tr("Often") else (1 if chest_pain == tr("Sometimes") else 0)
        fbs = 1 if smoke == tr("Regularly") else 0
        exang = 1 if tired == tr("Yes") else 0
        chol = 250 if overweight == tr("Yes") else 180
        trestbps = 145 if overweight == tr("Yes") else 120
        thalach = 140 if exercise == tr("No") else 170
        oldpeak = 2.0 if chest_pain == tr("Often") else 0.5
        slope, ca, thal, restecg = 1, 0, 2, 1

    else:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(tr("Age"), 18, 100, 40)
            sex = st.selectbox(tr("Sex"), ("Male", "Female"))
            cp = st.selectbox(tr("Chest Pain Type (0-3)"), [0, 1, 2, 3])
            trestbps = st.number_input(tr("Resting BP (mm Hg)"), 80, 200, 120)
            chol = st.number_input(tr("Cholesterol (mg/dl)"), 100, 600, 200)
            fbs = st.selectbox(tr("Fasting Blood Sugar >120"), [0, 1])
        with c2:
            restecg = st.selectbox(tr("Resting ECG (0-2)"), [0, 1, 2])
            thalach = st.number_input(tr("Max Heart Rate"), 60, 220, 150)
            exang = st.selectbox(tr("Exercise Angina"), [0, 1])
            oldpeak = st.number_input(tr("ST Depression"), 0.0, 10.0, 1.0)
            slope = st.selectbox(tr("Slope (0-2)"), [0, 1, 2])
            ca = st.number_input(tr("Major Vessels (0-3)"), 0, 3, 0)
            thal = st.selectbox(tr("Thal (0-3)"), [0, 1, 2, 3])
        sex = 1 if sex == "Male" else 0

    X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]])

    if st.button(tr("🔍 Predict Risk")):
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
        pred = 1 if prob > 0.5 else 0
        st.session_state.pred = "High" if pred == 1 else "Low"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': tr("Risk Probability (%)")},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': '#e91e63'},
                   'steps': [
                       {'range': [0, 50], 'color': '#4caf50'},
                       {'range': [50, 75], 'color': '#ffb300'},
                       {'range': [75, 100], 'color': '#f44336'}
                   ]}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        if pred == 1:
            st.error(tr("⚠️ High Risk of Heart Disease!"))
        else:
            st.success(tr("✅ Low Risk of Heart Disease. Stay Healthy!"))

# ---------------- HEALTH TIPS ----------------
elif "💡" in page:
    st.title(tr("💡 Personalized Health Tips"))
    pred = st.session_state.get("pred", "Low")
    if pred == "High":
        st.error(tr("⚠️ You might have a higher risk of heart disease."))
        tips = ["🥗 Eat fruits and whole grains", "🏃‍♂️ Exercise 30 mins daily",
                "🚭 Quit smoking", "😌 Manage stress", "💊 Regular health checkups"]
    else:
        st.success(tr("✅ Your risk appears low. Keep these habits strong!"))
        tips = ["🍎 Eat balanced meals", "🧘 Stay active and calm",
                "💤 Sleep 7–8 hours", "💧 Stay hydrated"]
    for t in tips:
        st.markdown(f"- {tr(t)}")

# ---------------- DATASET INFO ----------------
elif "📊" in page:
    st.title(tr("📘 Dataset Information"))
    st.markdown(tr("""
    - Rows: 303  
    - Target: 1 = disease, 0 = no disease  
    - Features: Age, Sex, Chest Pain, BP, Cholesterol, ECG, Heart Rate, etc.
    """))

# ---------------- VIRTUAL ASSISTANT ----------------
elif "🗣" in page:
    st.title(tr("🗣 Virtual Health Assistant"))
    st.markdown(tr("Ask me anything about heart health ❤️"))
    user_input = st.chat_input(tr("Type your question here..."))

    faq = {
        "exercise": "🏃‍♂️ Regular exercise strengthens your heart and improves blood flow.",
        "cholesterol": "🥗 Eat oats, nuts, and fruits to lower cholesterol.",
        "smoking": "🚭 Smoking increases heart disease risk. Quitting helps quickly.",
        "stress": "😌 Meditation and yoga help reduce stress.",
        "diet": "🍎 Eat less sugar, more greens, and lean proteins.",
        "hello": "👋 Hello! How can I help you with your heart health today?"
    }

    if user_input:
        # find matching answer
        english_reply = next((ans for key, ans in faq.items() if key in user_input.lower()), 
                             "I'm not sure, but try to eat healthy and stay active! 💪")
        # translate the reply into selected language
        reply = tr(english_reply)
        st.markdown("💬 " + reply)

# ---------------- FOOTER ----------------
st.markdown("""
<div style='position:fixed;left:0;bottom:0;width:100%;text-align:center;
background:linear-gradient(90deg,#e91e63,#ad1457);padding:10px;
font-size:14px;color:#fff;font-weight:500;'>
Developed with ❤️ by <b>Dikesh Chavhan</b> | © 2025 | HeartCheck
</div>
""", unsafe_allow_html=True)
