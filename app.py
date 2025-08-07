import streamlit as st
import joblib
import numpy as np

# Load your model
model = joblib.load("model.pkl")

st.title("ML Model Web App")

bedrooms = st.number_input("Enter the number of bedrooms:", min_value=0)
bathrooms = st.number_input("Enter the number of bathrooms:", min_value=0.0)
sqft_living = st.number_input("Enter the living area:", min_value=0)
sqft_lot = st.number_input("Enter the lot area:", min_value=0)
floors = st.number_input("Enter the number of floors:", min_value=0.0)
waterfront = st.number_input("Enter the waterfront:", min_value=0, max_value=1)
view = st.number_input("Enter the view:", min_value=0, max_value=4)
condition = st.number_input("Enter the condition:", min_value=1, max_value=5)
sqft_above = st.number_input("Enter the above area:", min_value=0)
sqft_basement = st.number_input("Enter the basement area:", min_value=0)
yr_built = st.number_input("Enter the year built:", min_value=0)
yr_renovated = st.number_input("Enter the year renovated:", min_value=0)


if st.button("Predict"):
    # Note: Handling categorical features (city and statezip) requires more complex
    # processing to match the model's training data. This example assumes
    # a simplified input where city and statezip are not directly used in prediction
    # in this form. You would need to apply the same one-hot encoding here
    # as was done during training.
    input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
