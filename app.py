import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Neurova Shield – Manual Input", layout="centered")

st.title("🧠 Neurova Shield – Stress Prediction from Sensor Input")
st.markdown("Enter your current wearable sensor readings to get real-time stress level prediction and wellness recommendation.")

# -------------------------------
# Load Model
# -------------------------------
try:
    with open("models/stress_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error("🚫 Failed to load the model.")
    st.stop()

# -------------------------------
# Load Dataset to Get Ranges
# -------------------------------
try:
    df = pd.read_csv("all_subjects_labeled_final.csv")
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'acc_x': 'ACC_x',
        'acc_y': 'ACC_y',
        'acc_z': 'ACC_z',
        'bvp': 'BVP',
        'eda': 'EDA',
        'temp': 'TEMP',
        'hr': 'HR'
    })
except Exception as e:
    st.error("🚫 Could not load dataset to extract valid input ranges.")
    st.text(f"Error: {e}")
    st.stop()

# -------------------------------
# Define Label Mapping
# -------------------------------
label_map = {
    0: 'low',
    1: 'medium',
    2: 'high',
    3: 'very high',
    4: 'extreme'
}

rec_map = {
    'low': "🙂 You're doing great! Keep up your wellness routine.",
    'medium': "😐 Take a short break. Try breathing exercises.",
    'high': "😟 Consider stepping away and speaking with a counselor if this continues.",
    'very high': "⚠️ High stress detected. Please take immediate rest and consult a therapist if needed.",
    'extreme': "🚨 Extreme stress detected! Step away and seek help immediately."
}

# -------------------------------
# Get Min-Max for Inputs
# -------------------------------
def get_range(feature):
    if feature in df.columns:
        return float(df[feature].min()), float(df[feature].max())
    else:
        return 0.0, 1.0  # fallback

ranges = {
    'ACC_x': get_range('ACC_x'),
    'ACC_y': get_range('ACC_y'),
    'ACC_z': get_range('ACC_z'),
    'BVP': get_range('BVP'),
    'EDA': get_range('EDA'),
    'TEMP': get_range('TEMP'),
    'HR': get_range('HR')
}

# -------------------------------
# Manual Feature Input
# -------------------------------
st.subheader("📥 Input Sensor Readings")

acc_x = st.number_input("ACC_x", value=np.mean(ranges['ACC_x']), min_value=ranges['ACC_x'][0], max_value=ranges['ACC_x'][1])
acc_y = st.number_input("ACC_y", value=np.mean(ranges['ACC_y']), min_value=ranges['ACC_y'][0], max_value=ranges['ACC_y'][1])
acc_z = st.number_input("ACC_z", value=np.mean(ranges['ACC_z']), min_value=ranges['ACC_z'][0], max_value=ranges['ACC_z'][1])
bvp   = st.number_input("BVP", value=np.mean(ranges['BVP']), min_value=ranges['BVP'][0], max_value=ranges['BVP'][1])
eda   = st.number_input("EDA", value=np.mean(ranges['EDA']), min_value=ranges['EDA'][0], max_value=ranges['EDA'][1])
temp  = st.number_input("TEMP (°C)", value=np.mean(ranges['TEMP']), min_value=ranges['TEMP'][0], max_value=ranges['TEMP'][1])
hr    = st.number_input("HR (bpm)", value=np.mean(ranges['HR']), min_value=ranges['HR'][0], max_value=ranges['HR'][1])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict Stress Level"):
    input_data = {
        'ACC_x': acc_x, 'ACC_y': acc_y, 'ACC_z': acc_z,
        'BVP': bvp, 'EDA': eda, 'TEMP': temp, 'HR': hr
    }

    # Validate Ranges
    errors = []
    for feature, val in input_data.items():
        min_val, max_val = ranges[feature]
        if not (min_val <= val <= max_val):
            errors.append(f"{feature} must be between {min_val:.2f} and {max_val:.2f}.")

    if errors:
        st.error("🚫 Invalid input values:")
        for err in errors:
            st.write(f"- {err}")
    else:
        try:
            input_array = np.array([[acc_x, acc_y, acc_z, bvp, eda, temp, hr]])
            prediction = model.predict(input_array)[0]
            label = label_map.get(prediction, str(prediction))
            recommendation = rec_map.get(label, "No recommendation available.")

            st.success(f"🎯 **Predicted Stress Level: {label.upper()}**")
            st.markdown(f"💡 **Recommendation:** {recommendation}")
        except Exception as e:
            st.error("🚫 Failed to make prediction.")
            st.text(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Created by Ved Shukla & Parshav Shah – Neurova Shield Project")
