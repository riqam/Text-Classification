import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

# Load the model
model = joblib.load('best_model.pkl')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian')) 

# Streamlit UI
def main():
    st.title("Text Category Prediction")

    X_test = joblib.load('Test Kompas/data/X_test.pkl')
    X_test_clean = joblib.load('Test Kompas/data/X_test_clean.pkl')

    st.write("Select a title to predict its associated text category:")

    # Dropdown to select title
    selected_title = st.selectbox("Choose a title:", X_test['Title'])

    # Find the corresponding FullText for the selected title
    selected_fulltext = X_test_clean.loc[X_test['Title'] == selected_title, 'FullText'].iloc[0]


    if st.button("Predict"):

        X_test_clean['tokens'] = word_tokenize(selected_fulltext)
        X_text_test = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform(X_test_clean['tokens'].apply(lambda x: ' '.join(x)))

        prediction = model.predict([X_text_test])
        st.write(f"Predicted category for '{selected_title}': {prediction[0]}")

if __name__ == "__main__":
    main()
