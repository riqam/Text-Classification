import streamlit as st
import pandas as pd
import nltk
import json
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load the best model from the .pkl file
best_model = joblib.load('best_model.pkl')

nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('stopwords')  # Download the NLTK stop words
stop_words = set(stopwords.words('indonesian'))

# Load the model
with open('best_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Load the test data
X_test = pd.read_pickle('X_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

# Function to make prediction
def predict_title(title):
    # Preprocess the input
    title = [title]  # Convert to list for consistency with model input format
    
    # Predict
    prediction = model.predict(title)
    
    return prediction[0]

# Streamlit app
def main():
    st.title("Simple Text Classification App")
    
    # Ask user for input
    title_input = st.text_input("Enter the title to predict:")
    
    if st.button("Predict"):
        # Perform prediction
        prediction = predict_title(title_input)
        st.success(f"The predicted class is: {prediction}")

if __name__ == "__main__":
    main()
