import streamlit as st
import joblib
import pandas as pd
import os

# Page config
st.set_page_config(page_title="Stroke Prediction", layout="centered")

# --- Load model & expected columns ---
try:
    model = joblib.load("xgb_stroke_model.pkl")
except Exception:
    st.error("Could not load model file 'xgb_stroke_model.pkl'. Make sure it's in the app folder.")
    st.stop()

try:
    expected_cols = joblib.load("model_columns.pkl")
except Exception:
    st.error("Could not load 'model_columns.pkl'. Save the one-hot encoded column list during training and place it here.")
    st.stop()

# Optional scaler
scaler = None
if os.path.exists("scaler.pkl"):
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        scaler = None

st.title("🧠 Stroke Prediction")
st.write("Fill out the form below to predict stroke risk.")

# --- Form UI ---
with st.form("stroke_form"):
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)
    hypertension = st.selectbox("Hypertension", ["", "No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["", "No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["", "No", "Yes"])
    work_type = st.selectbox("Work Type", ["", "Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["", "Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, step=0.1, value=100.0)

    # Height and Weight inputs
    height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=0.1, value=170.0)
    weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, step=0.1, value=70.0)
    
    smoking_status = st.selectbox("Smoking Status", ["", "formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict Stroke Risk")

# --- Run prediction ---
if submitted:
    # Validate input
    if "" in [gender, hypertension, heart_disease, ever_married, work_type, Residence_type, smoking_status]:
        st.warning("⚠️ Please fill out all fields.")
    else:
        hypertension_map = {"No": 0, "Yes": 1}
        heart_disease_map = {"No": 0, "Yes": 1}

        # Calculate BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        # Prepare input dictionary
        input_dict = {
            'age': [int(age)],
            'hypertension': [hypertension_map[hypertension]],
            'heart_disease': [heart_disease_map[heart_disease]],
            'avg_glucose_level': [float(avg_glucose_level)],
            'bmi': [float(bmi)],
            'gender': [gender],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [Residence_type],
            'smoking_status': [smoking_status]
        }

        input_df = pd.DataFrame(input_dict)

        # One-hot encoding
        X_encoded = pd.get_dummies(
            input_df,
            columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
            dtype=int
        )

        for c in expected_cols:
            if c not in X_encoded.columns:
                X_encoded[c] = 0
        X_encoded = X_encoded[expected_cols]

        if scaler is not None:
            try:
                X_encoded = pd.DataFrame(scaler.transform(X_encoded), columns=expected_cols)
            except Exception:
                X_encoded = X_encoded.values

        try:
            pred = int(model.predict(X_encoded)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # --- Display summary table ---
        st.subheader("📋 Input Summary")
        st.table(input_df.T.rename(columns={0: "Value"}))

        # --- Display prediction ---
        st.subheader("🩺 Prediction Result")
        if pred == 1:
            st.error("⚠️ Chances of getting stroke detected. It is recommended to consult a doctor.")
        else:
            st.success("✅ No chances of stroke detected. Keep up a healthy lifestyle.")

st.caption("This tool is for informational purposes only and is not a substitute for professional medical advice.")
