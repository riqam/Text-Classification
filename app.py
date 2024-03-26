import streamlit as st
import joblib
from nltk.tokenize import word_tokenize

# Load the model
model = joblib.load('best_model.pkl')

# Streamlit UI
def main():
    st.title("Text Category Prediction")

    X_test = joblib.load('Test Kompas\data\X_test.pkl')
    X_test_clean = joblib.load('Test Kompas\data\X_test_clean.pkl')

    st.write("Select a title to predict its associated text category:")

    # Dropdown to select title
    selected_title = st.selectbox("Choose a title:", X_test['Title'])

    # Find the corresponding FullText for the selected title
    selected_fulltext = X_test.loc[X_test['Title'] == selected_title, X_test_clean['FullText']].iloc[0]

    if st.button("Predict"):

        X_test_clean['tokens'] = selected_fulltext.apply(lambda x: word_tokenize(x))
        X_text_test = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform(X_test_clean['tokens'].apply(lambda x: ' '.join(x)))

        prediction = model.predict([X_text_test])
        st.write(f"Predicted category for '{selected_title}': {prediction[0]}")

if __name__ == "__main__":
    main()
