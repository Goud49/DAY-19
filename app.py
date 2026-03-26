import streamlit as st
import pickle
import pandas as pd

st.title("🩺 Medical Prediction App")

# 🔹 Select dataset
option = st.selectbox(
    "Select Prediction Type",
    ["Heart Disease", "Medical Diagnosis"]
)

# ================= HEART MODEL =================
if option == "Heart Disease":
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))

    st.header("💓 Heart Disease Prediction")

    age = st.number_input("Age", 20, 100)
    trestbps = st.number_input("Blood Pressure")
    chol = st.number_input("Cholesterol")
    thalach = st.number_input("Max Heart Rate")

    if st.button("Predict Heart Disease"):
        input_data = pd.DataFrame([[age, trestbps, chol, thalach]],
                                 columns=["age", "trestbps", "chol", "thalch"])

        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[columns]
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ Heart Disease Risk")
        else:
            st.success("✅ No Heart Disease")


# ================= DIAGNOSIS MODEL =================
elif option == "Medical Diagnosis":
    model = pickle.load(open("lr_diag.pkl", "rb"))
    scaler = pickle.load(open("scaler_diag.pkl", "rb"))
    columns = pickle.load(open("columns_diag.pkl", "rb"))

    st.header("🧪 Medical Diagnosis Prediction")

    # Example inputs (adjust based on your dataset)
    inputs = {}
    for col in columns[:5]:  # take first few columns for UI
        inputs[col] = st.number_input(col)

    if st.button("Predict Diagnosis"):
        input_data = pd.DataFrame([inputs])

        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[columns]
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)

        st.success(f"Prediction: {pred[0]}")