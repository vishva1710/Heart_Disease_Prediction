# Heart Disease Risk Prediction

This is an end-to-end machine learning project I built to predict whether a person is at risk of heart disease based on their clinical data (age, cholesterol, blood pressure, etc.). I wanted to go beyond just training a model in a notebook and actually take it all the way to a working app that anyone can use.

## Live Demo

- Frontend (Streamlit): https://heartdiseaseprediction-wcywwdxkpnnqz5qshezupt.streamlit.app
- Backend API (FastAPI docs): https://heart-disease-prediction-2-4kss.onrender.com/docs

Note: the backend is hosted on Render's free tier, so the first request after some inactivity can take 30-50 seconds to respond while the server wakes up. After that it's fast.

## What it does

You enter some basic health details (age, sex, chest pain type, resting blood pressure, cholesterol, etc.) and the model tells you whether you're at high or low risk of heart disease, along with a probability.

## How I built it

I used the UCI Heart Disease dataset for this. The process was roughly:

1. Did EDA to understand the data and check for class imbalance
2. Split the data using GroupShuffleSplit / StratifiedGroupKFold instead of a plain train_test_split, since some rows in this dataset are near-duplicates and I wanted to avoid data leakage between train and test sets
3. Tried a few models - Logistic Regression, SVM, Random Forest, XGBoost - and compared them
4. Tuned the best one (Random Forest) using RandomizedSearchCV
5. Evaluated it properly - not just accuracy, but recall, F1, and the confusion matrix too, since missing an actual heart disease case matters more than a false alarm here
6. Saved the final pipeline (scaler + model together) with joblib
7. Wrapped it in a FastAPI backend so it can be called like a normal API
8. Built a Streamlit frontend so people don't have to hit the API manually
9. Deployed the backend on Render and the frontend on Streamlit Cloud

## Tech stack

- Python, pandas, scikit-learn, XGBoost
- FastAPI (backend/API)
- Streamlit (frontend)
- joblib (model persistence)
- Render (backend hosting)
- Streamlit Community Cloud (frontend hosting)

## Project structure

```
├── backend/
│   ├── main.py              # FastAPI app
│   └── requirements.txt
├── frontend/
│   └── app.py                # Streamlit app
├── notebookfile/
│   ├── heart_disease_risk_prediction.ipynb
│   └── heart_disease_rfc_pipeline.joblib
├── dataset/
│   └── heart.csv
└── README.md
```

## Running it locally

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend (in a separate terminal):
```bash
cd frontend
streamlit run app.py
```

The frontend calls the backend over an API, so make sure the backend is running first if you're testing locally. If you're running both locally, update the `API_URL` in `app.py` to point to `http://127.0.0.1:8000/predict` instead of the Render link.


## Dataset

UCI Heart Disease dataset (Cleveland subset).

---


