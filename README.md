🩺 HeartCheck – Heart Disease Prediction App

💓 Predict. Prevent. Protect.
A machine learning–powered web application that predicts heart disease risk, provides personalized health tips, and supports multiple Indian languages — all inside an interactive Streamlit interface.

🚀 Live Demo

👉 Try the App (https://heart-disease-prediction-ml-daktckecmvdr472ursaf4f.streamlit.app/#heart-disease-prediction-app)

---
📘 Project Overview

HeartCheck is an intelligent health assistant built using Machine Learning and Streamlit that helps users:

Estimate their risk of heart disease

Get personalized lifestyle recommendations

Interact with a Virtual Health Assistant that answers common heart-health questions

Use the app in English, Hindi, Marathi, or Tamil

It’s designed for both medical users (Expert Mode) and general users (Smart Mode) — making health prediction simple and accessible for everyone.

--
🧠 Key Features

| Feature                         | Description                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------ |
| 🧍 **Smart Mode**               | For users without medical data — enter simple lifestyle answers                |
| 🩺 **Expert Mode**              | For users who know their detailed medical readings                             |
| 🌐 **Multilingual Support**     | English, Hindi, Marathi, Tamil using real-time translation                     |
| 💬 **Virtual Health Assistant** | Chatbot that understands 10+ health topics & replies in your selected language |
| 📊 **Prediction Gauge**         | Dynamic meter showing probability of heart disease                             |
| 💡 **Health Tips Page**         | Displays personalized tips based on model prediction                           |
| 💻 **Modern Dark UI**           | Built with Streamlit + custom CSS for a clean, professional look               |
| 🔗 **Developer Info Bar**       | LinkedIn and contact info directly inside the app                              |

---
🧩 Tech Stack

Python 3.13

Streamlit – Web Framework

Scikit-Learn – Model Training & Prediction

Pandas / NumPy – Data Preprocessing

Plotly – Interactive Gauge Visualization

Deep Translator – Language Translation

Joblib – Model Serialization

---
🧮 Dataset Information

Dataset Source: UCI Machine Learning Repository – Heart Disease Dataset

Rows: 303

Target Variable: 1 = heart disease, 0 = no disease

Features: Age, Sex, Chest Pain, Blood Pressure, Cholesterol, ECG, Heart Rate, etc.

---
🧰 Model Details

Algorithm Used: Random Forest Classifier

Training Accuracy: ~89%

Saved Model File: best_model.pkl

Feature Scaling & Encoding: Performed via preprocessing pipeline

---
🌐 App Structure
heart-disease-prediction-ml/
│
├── app.py                  # Main Streamlit app
├── best_model.pkl          # Trained ML model
├── requirements.txt        # Dependencies
├── UCI_Heart_Disease_Dataset.ipynb   # Notebook for model training
└── README.md               # Project Documentation
---
🗣️ Virtual Health Assistant Topics

The assistant can answer questions about:

Exercise and fitness 🏃‍♂️

Cholesterol & diet 🥗

Smoking 🚭

Stress management 😌

Sleep & hydration 💤💧

Heart attack emergencies 🚨

You can chat in English, and it will reply in your selected language.

---
🌍 Multilingual Support
Language	Code	Status
English	en	✅ Default
Hindi	hi	✅ Supported
Marathi	mr	✅ Supported
Tamil	ta	✅ Supported

Translations are handled using deep-translator for real-time language support.

---
⭐ If you like this project...

Give it a ⭐ on GitHub — it helps others find and use this open-source project!

“Data is the new medicine. Let’s make it accessible to everyone.” – Dikesh Chavhan ❤️
