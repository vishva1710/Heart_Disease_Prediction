import streamlit as st
import requests

# -------------------------------------------------
# Backend API URL
# For local testing: http://127.0.0.1:8000/predict
# After deploying backend (EC2/Render), replace with the live URL
# -------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient details below to predict heart disease risk.")

# -------------------------------------------------
# Input widgets - one per feature, same 13 columns used in training
# -------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=52)
sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=125)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=212)
fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG (0-2)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=168)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", options=[1, 2, 3])

# -------------------------------------------------
# Predict button - sends data to FastAPI backend instead of loading model here
# -------------------------------------------------
if st.button("Predict Risk"):
    # 1. Build the payload - keys MUST match FastAPI's PatientData field names
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # 2. Call the FastAPI backend
    with st.spinner("Contacting backend..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            # 3. Show the result returned by the backend
            if result["prediction"] == 1:
                st.error(f"⚠️ {result['result']}  (Risk: {result['risk_probability']})")
            else:
                st.success(f"✅ {result['result']}  (Safe: {result['safe_probability']})")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Is the FastAPI server running?")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")