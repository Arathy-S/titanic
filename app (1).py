import streamlit as st
import pickle
import numpy as np

# Load the model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 50.0)

# Encode 'sex'
sex_encoded = 1 if sex == "male" else 0

# Prepare input
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")
