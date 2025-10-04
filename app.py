import streamlit as st
import pandas as pd
import torch
import joblib
from models.models import DiabetesModel

# --- Load trained model ---
num_features = 28
model = DiabetesModel(num_features=num_features)
model.load_state_dict(torch.load("models/diabetes_model.pth"))
model.eval()

# --- Sidebar links ---
github_url = "https://github.com/nirireads/diabities-health-indicator/tree/main"
website_url = "https://www.nirsingh.com.np"

st.markdown(
    f"""
    <div style='display: flex; gap: 15px;'>
        <a href="{github_url}" target="_blank">💻 GitHub</a>
        <a href="{website_url}" target="_blank">🌐 Website</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Stage Prediction")

# --- Columns ---
categorical_cols = ['gender','ethnicity','education_level','income_level','employment_status','smoking_status']
binary_cols = ['family_history_diabetes','hypertension_history','cardiovascular_history']
numeric_cols = [
    'age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
    'diet_score','sleep_hours_per_day','screen_time_hours_per_day',
    'bmi','waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
    'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides',
    'glucose_fasting','glucose_postprandial','insulin_level','hba1c'
]

# --- Load encoders and scaler ---
encoders = {col: joblib.load(f"models/encoders_scaler/{col}_encoder.pkl") for col in categorical_cols}
scaler = joblib.load("models/encoders_scaler/scaler.pkl")

# --- Stage labels ---
stage_labels = {0: "Gestational", 1: "No Diabetes", 2: "Pre-Diabetes", 3: "Type 1", 4: "Type 2"}

# --- Default values ---
default_categorical = {col: encoders[col].classes_[0] for col in categorical_cols}
default_binary = {col: "No" for col in binary_cols}
default_numeric = {
    'age': 50, 'alcohol_consumption_per_week': 0, 'physical_activity_minutes_per_week': 30,
    'diet_score': 5, 'sleep_hours_per_day': 7, 'screen_time_hours_per_day': 5,
    'bmi': 25, 'waist_to_hip_ratio': 0.85, 'systolic_bp': 120, 'diastolic_bp': 80,
    'heart_rate': 70, 'cholesterol_total': 180, 'hdl_cholesterol': 50, 'ldl_cholesterol': 100,
    'triglycerides': 150, 'glucose_fasting': 90, 'glucose_postprandial': 120,
    'insulin_level': 10, 'hba1c': 5.5
}

user_input = {}

# --- 3-column layout ---
col1, col2, col3 = st.columns(3)

# Categorical inputs with defaults
for i, col in enumerate(categorical_cols):
    column = [col1, col2, col3][i % 3]
    default_value = default_categorical[col]
    default_index = list(encoders[col].classes_).index(default_value)
    user_input[col] = column.selectbox(
        f"{col.replace('_',' ').title()}",
        encoders[col].classes_,
        index=default_index
    )

# Binary inputs with defaults
for i, col in enumerate(binary_cols):
    column = [col1, col2, col3][i % 3]
    default_value = default_binary[col]
    default_index = ["No","Yes"].index(default_value)
    user_input[col] = column.selectbox(
        f"{col.replace('_',' ').title()}",
        ["No","Yes"],
        index=default_index
    )

# Numeric inputs with realistic defaults
cols = st.columns(3)
for i, col in enumerate(numeric_cols):
    column = cols[i % 3]
    user_input[col] = column.number_input(
        f"{col.replace('_',' ').title()}",
        value=float(default_numeric[col]),
        format="%.2f"
    )

# --- Predict button ---
if st.button("Predict"):
    # Encode categorical
    for col in categorical_cols:
        user_input[col] = encoders[col].transform([user_input[col]])[0]
    # Binary to 0/1
    for col in binary_cols:
        user_input[col] = 1 if user_input[col]=="Yes" else 0

    # Create dataframe
    input_df = pd.DataFrame([user_input])
    input_df = pd.concat([input_df[categorical_cols + binary_cols], input_df[numeric_cols]], axis=1)

    # Scale numeric features
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_label = stage_labels[pred_class]

    # --- Color-coded display ---
    if pred_label in ["Gestational", "No Diabetes"]:
        st.success(f"Predicted Diabetes Stage: {pred_label}")
    elif pred_label == "Pre-Diabetes":
        st.warning(f"Predicted Diabetes Stage: {pred_label}")
    else:  # Type 1 or Type 2
        st.error(f"Predicted Diabetes Stage: {pred_label}")
