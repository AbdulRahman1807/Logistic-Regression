import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("Ten-Year CHD Risk Predictor")

DATA_PATH = Path("framingham_heart_disease.csv")
MODEL_PATH = Path("heart_model.pkl")
SCALER_PATH = Path("scaler.pkl")

model = None
scaler = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error(f"Model file not found: {MODEL_PATH}. Run `logistic_regression.py` first to create it.")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    st.error(f"Scaler file not found: {SCALER_PATH}. Run `logistic_regression.py` first to create it.")

if model is None or scaler is None:
    st.stop()


if not DATA_PATH.exists():
    st.warning("Dataset `framingham_heart_disease.csv` not found — app will still run if you provide all inputs manually.")
    feature_cols = [
        "male",
        "age",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    ]
    defaults = {}
else:
    df = pd.read_csv(DATA_PATH)
    feature_cols = df.drop(["education", "TenYearCHD"], axis=1).columns.tolist()
    defaults = {c: float(df[c].median(skipna=True)) for c in feature_cols}

st.sidebar.header("Patient features")
inputs = {}

binary_like = {"male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"}
for col in feature_cols:
    default = defaults.get(col, 0)
    if col in binary_like:
        inputs[col] = st.sidebar.selectbox(col, options=[0, 1], index=int(default) if default in (0, 1) else 0)
    else:
        if col in ("age",):
            min_v, max_v, step = 20, 120, 1
        elif col in ("cigsPerDay",):
            min_v, max_v, step = 0, 100, 1
        elif col in ("sysBP", "diaBP"):
            min_v, max_v, step = 50, 300, 1
        elif col in ("totChol",):
            min_v, max_v, step = 50, 500, 1
        elif col in ("BMI",):
            min_v, max_v, step = 10, 80, 0.1
        elif col in ("heartRate",):
            min_v, max_v, step = 30, 200, 1
        elif col in ("glucose",):
            min_v, max_v, step = 10, 500, 1
        else:
            min_v, max_v, step = 0, 1000, 1
        inputs[col] = st.sidebar.number_input(
    col,
    min_value=float(min_v),
    max_value=float(max_v),
    value=float(default),
    step=float(step)
)
if st.sidebar.button("Predict"):
    x = np.array([inputs[c] for c in feature_cols], dtype=float).reshape(1, -1)
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        st.error(f"Error scaling input: {e}")
    else:
        proba = model.predict_proba(x_scaled)[0, 1]
        pred = int(proba >= 0.5)
        st.subheader("Prediction")
        st.metric(label="Predicted 10-year CHD (0=no, 1=yes)", value=pred)
        st.write(f"Estimated risk (probability): {proba:.3f}")
        st.info("Use this tool for exploratory purposes only — not a medical diagnosis.")

st.markdown("---")
st.write("Model expects features in this order:")
st.write(feature_cols)
