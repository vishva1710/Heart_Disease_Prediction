from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os


# App create
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "notebookfile", "heart_disease_rfc_pipeline.joblib"))


# Input format define
class PatientData(BaseModel):
    age: int
    sex: int          # 1=Male, 0=Female
    cp: int            # chest pain type (0-3)
    trestbps: int      # resting blood pressure
    chol: int          # serum cholesterol
    fbs: int           # fasting blood sugar > 120 (1=Yes, 0=No)
    restecg: int       # resting ECG result (0-2)
    thalach: int       # max heart rate achieved
    exang: int         # exercise induced angina (1=Yes, 0=No)
    oldpeak: float     # ST depression
    slope: int         # slope of ST segment (0-2)
    ca: int            # number of major vessels (0-3)
    thal: int          # thalassemia (1/2/3)

# Home route
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}

# Predict route
@app.post("/predict")
def predict(data: PatientData):
    # Input array create
    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]])

    # Predict (pipeline handles scaling internally)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    risk_prob = round(probability[0][1] * 100, 2)
    safe_prob = round(probability[0][0] * 100, 2)

    return {
        "prediction": int(prediction[0]),
        "result": "High risk of heart disease" if prediction[0] == 1 else "Low risk of heart disease",
        "risk_probability": f"{risk_prob}%",
        "safe_probability": f"{safe_prob}%"
    }