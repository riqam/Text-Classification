import streamlit as st
import time
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
    st.title("TEXT ANALYSIS AND CLASSIFICATION")

    X_test = joblib.load('Test Kompas/data/X_test.pkl')
    X_test_clean = joblib.load('Test Kompas/data/X_test_clean.pkl')

    st.subheader("Please, select the title of the article you want to analyze!")

    # Dropdown to select title
    selected_title = st.selectbox("Choose a title:", X_test['Title'])

    # Find the corresponding FullText for the selected title
    selected_fulltext = X_test_clean.loc[X_test['Title'] == selected_title, 'FullText'].iloc[0]

    def analyze_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        # Mengembalikan label sentimen berdasarkan nilai komponen sentimen
        if sentiment['compound'] == 0:
            return 'Neutral'
        elif sentiment['compound'] < 0 :
            return 'Negative'
        else:
            return 'Positive'
    
    if st.button("Analyze"):
        with st.spinner('Analyze the article...'):
            time.sleep(3)
        # Sentiment Analysis
        sentiment_scores = analyze_sentiment(selected_fulltext)

        # Predict the category
        tokens = word_tokenize(selected_fulltext)
        tokens_filtered = [word for word in tokens if word.lower() not in stop_words]
        text_for_prediction = ' '.join(tokens_filtered)
        X_text_test = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform([text_for_prediction])
        prediction = model.predict(X_text_test)

        st.write(f"Predicted Site News of the content: {prediction[0]}")
        st.write("Results of sentiment analysis from articles:", sentiment_scores)

if __name__ == "__main__":
    main()
