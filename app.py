import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dataset
data = {
    "fever": [1,1,0,1,0,1,0,0,1,0],
    "cough": [1,0,1,1,0,1,0,0,1,0],
    "fatigue": [1,1,0,1,0,0,1,0,1,0],
    "headache": [1,0,1,1,0,0,1,0,1,0],
    "disease": ["flu","malaria","healthy","flu","healthy","malaria","flu","healthy","malaria","healthy"]
}

df = pd.DataFrame(data)

X = df.drop("disease", axis=1)
y = df["disease"]

model = RandomForestClassifier()
model.fit(X, y)

# App UI
st.title("🩺 Medical AI Disease Predictor")

st.write("Enter symptoms below:")

fever = st.number_input("Fever (1=yes, 0=no)")
cough = st.number_input("Cough (1=yes, 0=no)")
fatigue = st.number_input("Fatigue (1=yes, 0=no)")
headache = st.number_input("Headache (1=yes, 0=no)")

if st.button("Predict"):
    prediction = model.predict([[fever, cough, fatigue, headache]])
    st.success(f"Predicted Disease: {prediction[0]}")
