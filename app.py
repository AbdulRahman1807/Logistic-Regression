import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

def yes_no_select(label, default=0):
    return st.selectbox(
        label,
        options=[0, 1],
        index=default,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown(
    """
    <h1 style='text-align: center;'>🫀 10-Year Heart Disease Risk Predictor</h1>
    <p style='text-align: center; color: gray;'>
    Logistic Regression Model • Educational Use Only
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🩺 Patient Features")

st.sidebar.subheader("Basic Information")
male = yes_no_select("Male")
age = st.sidebar.number_input("Age", 18.0, 100.0, 49.0, 1.0)

st.sidebar.subheader("Lifestyle")
currentSmoker = yes_no_select("Current Smoker")
cigsPerDay = st.sidebar.number_input("Cigarettes per Day", 0.0, 60.0, 0.0, 1.0)

st.sidebar.subheader("Medical History")
BPmeds = yes_no_select("On BP Medication")
prevalentStroke = yes_no_select("Previous Stroke")
prevalentHyp = yes_no_select("Hypertension")
diabetes = yes_no_select("Diabetes")

st.sidebar.subheader("Clinical Measurements")
totChol = st.sidebar.number_input("Total Cholesterol (mg/dL)", 100.0, 400.0, 234.0, 1.0)
sysBP = st.sidebar.number_input("Systolic BP (mmHg)", 80.0, 250.0, 128.0, 1.0)
diaBP = st.sidebar.number_input("Diastolic BP (mmHg)", 50.0, 150.0, 82.0, 1.0)
BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
heartRate = st.sidebar.number_input("Heart Rate (bpm)", 40.0, 200.0, 72.0, 1.0)
glucose = st.sidebar.number_input("Glucose (mg/dL)", 40.0, 400.0, 90.0, 1.0)

features = np.array([[
    male,
    age,
    currentSmoker,
    cigsPerDay,
    BPmeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose
]])

scaled_features = scaler.transform(features)
prediction = model.predict(scaled_features)[0]
probability = model.predict_proba(scaled_features)[0][1]

st.subheader("Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted CHD (10 years)", "Yes" if prediction == 1 else "No")

with col2:
    st.metric("Estimated Risk", f"{probability * 100:.1f}%")

if probability < 0.2:
    st.success("Low estimated risk")
elif probability < 0.4:
    st.warning("Moderate estimated risk")
else:
    st.error("High estimated risk")

st.info("This tool is for exploratory and educational purposes only. Not a medical diagnosis.")

st.subheader("Model Input Order")
st.json([
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPmeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose"
])
