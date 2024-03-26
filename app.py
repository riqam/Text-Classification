import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the model
model = joblib.load('best_model.pkl')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('indonesian')) 

# Streamlit UI
def main():
    st.title("Text Analysis")

    X_test = joblib.load('Test Kompas/data/X_test.pkl')
    X_test_clean = joblib.load('Test Kompas/data/X_test_clean.pkl')

    st.write("Select a title to analyze:")

    # Dropdown to select title
    selected_title = st.selectbox("Choose a title:", X_test['Title'])

    # Find the corresponding FullText for the selected title
    selected_fulltext = X_test_clean.loc[X_test['Title'] == selected_title, 'FullText'].iloc[0]

    if st.button("Analyze"):
        # Sentiment Analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(selected_fulltext)

        # Predict the category
        tokens = word_tokenize(selected_fulltext)
        tokens_filtered = [word for word in tokens if word.lower() not in stop_words]
        text_for_prediction = ' '.join(tokens_filtered)
        X_text_test = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform([text_for_prediction])
        prediction = model.predict(X_text_test)

        st.write(f"Predicted category for '{selected_title}': {prediction[0]}")
        st.write("Sentiment Analysis:")
        st.write(sentiment_scores)

if __name__ == "__main__":
    main()
