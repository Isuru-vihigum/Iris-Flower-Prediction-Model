import streamlit as st
import numpy as np
import joblib

model =joblib.load("model/model.pkl")

st.title("Iris Flower Prediction Model")
sepal_length = st.text_input("Enter Sepal length","")
sepal_width = st.text_input("Enter Sepal width","")
petal_length = st.text_input("Enter Petal length","")
petal_width = st.text_input("Enter Petal width","")

# input_data = np.array([float(sepal_length) ,float(sepal_width), float(petal_length), float(petal_width)]).reshape(1, -1)
# prediction = model.predict(input_data)
# predicted_class = {
#     0:"Iris-setosa",
#     1:"Iris-versicolor",
#     2:"Iris-virginica"
# }[prediction[0]]
#
# st.write(f"Predicted Species: {predicted_class}")

if st.button("Predict"):
    if all([sepal_length, sepal_width, petal_length, petal_width]):  # Ensure all fields are filled
        try:
            # Convert inputs to float and reshape to match model input shape
            input_data = np.array([
                float(sepal_length),
                float(sepal_width),
                float(petal_length),
                float(petal_width)
            ]).reshape(1, -1)  # Convert to 2D array

            # Make prediction
            prediction = model.predict(input_data)

            # Map prediction to class
            predicted_class = {
                0: "Iris-setosa",
                1: "Iris-versicolor",
                2: "Iris-virginica"
            }[prediction[0]]

            # Display result
            st.success(f"Predicted Species: {predicted_class}")
        except ValueError:
            st.error("Invalid input! Please enter numeric values.")
    else:
        st.warning("Please enter all values before predicting.")