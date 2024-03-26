import joblib
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    X_train = joblib.load('Test Kompas\X_train_clean.pkl')
    X_valid = joblib.load('Test Kompas\X_valid_clean.pkl')
    X_test = joblib.load('Test Kompas\X_test_clean.pkl')

    return X_train, X_valid, X_test

def tokenization():
    X_train['tokens'] = X_train['FullText'].apply(lambda x: word_tokenize(x))
    X_valid['tokens'] = X_valid['FullText'].apply(lambda x: word_tokenize(x))
    X_test['tokens'] = X_test['FullText'].apply(lambda x: word_tokenize(x))

    return X_train, X_valid, X_test

def vectorizer():
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_clean['tokens'].apply(lambda x: ' '.join(x)))

    joblib.dump(tfidf_vectorizer, 'Test Kompas/tfidf_vectorizer.pkl')

    return tfidf_vectorizer

def applying():
    X_text_train = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform(X_train_clean['tokens'].apply(lambda x: ' '.join(x)))
    X_text_valid = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform(X_valid_clean['tokens'].apply(lambda x: ' '.join(x)))
    X_text_test = joblib.load('Test Kompas/tfidf_vectorizer.pkl').transform(X_test_clean['tokens'].apply(lambda x: ' '.join(x)))

    joblib.dump(X_text_train, 'Test Kompas\X_train_vect.pkl')
    joblib.dump(X_text_valid, 'Test Kompas\X_valid_vect.pkl')
    joblib.dump(X_text_test, 'Test Kompas\X_test_vect.pkl')

    return X_text_train, X_text_valid, X_text_test

if __name__ == '__main__':
    X_train, X_valid, X_test = load_data()
    X_train, X_valid, X_test = tokenization()

    X_train_clean = X_train[['FullText', 'tokens']]
    X_valid_clean = X_valid[['FullText', 'tokens']]
    X_test_clean = X_test[['FullText', 'tokens']]

    vectorizer()

    X_text_train, X_text_valid, X_text_test = applying()