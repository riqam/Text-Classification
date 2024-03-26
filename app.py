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

# Streamlit UI
def main():
    st.title("Text Category Prediction")
    st.write("Select a title to predict its associated text category:")

    # Dropdown to select title
    selected_title = st.selectbox("Choose a title:", X_test['Title'])

    # Find the corresponding FullText for the selected title
    selected_fulltext = X_test.loc[X_test['Title'] == selected_title, 'FullText'].iloc[0]

    if st.button("Predict"):
        prediction = model.predict([selected_fulltext])
        st.write(f"Predicted category for '{selected_title}': {prediction[0]}")

if __name__ == "__main__":
    main()
