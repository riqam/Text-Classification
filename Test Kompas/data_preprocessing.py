import joblib
import re
import html as ihtml
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

# Load data
def load_data():
    X_train = joblib.load('Test Kompas\X_train.pkl')
    X_valid = joblib.load('Test Kompas\X_valid.pkl')
    X_test = joblib.load('Test Kompas\X_test.pkl')

    return X_train, X_valid, X_test

# Hapus kolom yang tidak diperlukan
columns_to_drop = ['Url', 'UrlShort', 'SiteID', 'SectionID', 'AuthorID', 'Photo', 'Lipsus', 'Video', 'EmbedSocial', 'Related', 'PublishedDate']
def drop_columns(return_file=False):
    X_train.drop(columns=columns_to_drop, inplace=True)
    X_valid.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)
    
    if return_file:
        return X_train, X_valid, X_test
    
# Fungsi untuk mengonversi list menjadi string
def list_to_string(text):
    if isinstance(text, list):
        return ' '.join(text)
    else:
        return text

# Fungsi pra-pemrosesan teks
def preprocess_text(text):
    # Memastikan bahwa nilai bukan None
    if text is not None:
        # Menghapus HTML tags
        text = BeautifulSoup(ihtml.unescape(text)).text
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\s+", " ", text) 

        # Lowercasing
        text = text.lower()

        # Menghapus karakter khusus
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # ^a-zA-Z\s

        # Tokenisasi
        tokens = word_tokenize(text)

        # Menghapus stop words
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Menggabungkan token kembali menjadi teks
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    else:
        return ''  # Mengembalikan string kosong untuk nilai None

if __name__ == '__main__':
    # Load Data
    X_train, X_valid, X_test = load_data()

    # Drop kolom yang tidak diperlukan
    drop_columns()

    # Convert list menjadi string pada kolom tag
    X_train['Tag'] = X_train['Tag'].apply(list_to_string)
    X_valid['Tag'] = X_valid['Tag'].apply(list_to_string)
    X_test['Tag'] = X_test['Tag'].apply(list_to_string)

    # Download resources nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Inisialisasi stemmer dan stopwords
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # stemmer = PorterStemmer()
    
    stop_words = set(stopwords.words('indonesian'))  # Stopwords dalam bahasa Indonesia

    # Apply preprocessing ke data training
    X_train = X_train.applymap(preprocess_text)
    X_valid = X_valid.applymap(preprocess_text)
    X_test = X_test.applymap(preprocess_text)

    # Simpan data clean
    joblib.dump(X_train, 'Test Kompas\X_train_clean.pkl')
    joblib.dump(X_valid, 'Test Kompas\X_valid_clean.pkl')
    joblib.dump(X_test, 'Test Kompas\X_test_clean.pkl')