import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the model
with open('best_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Load the test data
X_test = joblib.load('Test Kompas\data\X_test.pkl')
y_test = joblib.load('Test Kompas\data\y_test.pkl')

# Function to predict category
def predict_category(text):
    prediction = model.predict([text])
    return prediction[0]

# Streamlit UI
def main():
    st.title("Text Category Prediction")
    st.write("Select a text to predict its category:")

    # Dropdown to select text
    selected_text = st.selectbox("Choose a text:", X_test['FullText'])

    if st.button("Predict"):
        prediction = predict_category(selected_text)
        st.write(f"Predicted category: {prediction}")

if __name__ == "__main__":
    main()
