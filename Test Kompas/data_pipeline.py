import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def load_data(return_file=False):
    # Load data
    data = joblib.load('Test Kompas\kompascom_article_jan.pickle')

    # Ambil data artikel dari dalam list dan konversi ke DataFrame
    df = pd.DataFrame([article[0] for article in data])

    # Menggabungkan kolom Title, SubTitle, dan Content menjadi kolom baru bernama FullText
    df['FullText'] = df['SupTitle'] + ' ' + df['Title']  + ' ' + df['SubTitle'] + ' ' + df['Description'] + ' ' + df['Keyword'] + ' ' + df['Content']

    joblib.dump(df, 'Test Kompas\kompascom_df.pkl')

    if return_file:
        return df
    
def split_input_output(return_file=False):
    # Load data
    data = joblib.load('Test Kompas\kompascom_df.pkl')


    
    y = data['SiteName']
    X = data.drop(['SiteName'], axis=1)

    print('Input shape  :', X.shape)
    print('Output shape :', y.shape)

    joblib.dump(X, 'Test Kompas\input.pkl')
    joblib.dump(y, 'Test Kompas\output.pkl')

    if return_file:
        return X, y
    
def split_train_test(return_file=True):
    # Load data
    X = joblib.load('Test Kompas\input.pkl')
    y = joblib.load('Test Kompas\output.pkl')

    # Split test & rest (train & valid)
    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y,
                                            test_size = 0.2,
                                            random_state = 42
                                        )
    
    # Split train & valid
    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size = 0.2,
                                            random_state = 42
                                        )
    
    # Print splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump file
    joblib.dump(X_train, 'Test Kompas\X_train.pkl')
    joblib.dump(y_train, 'Test Kompas\y_train.pkl')
    joblib.dump(X_valid, 'Test Kompas\X_valid.pkl')
    joblib.dump(y_valid, 'Test Kompas\y_valid.pkl')
    joblib.dump(X_test, 'Test Kompas\X_test.pkl')
    joblib.dump(y_test, 'Test Kompas\y_test.pkl')

    if return_file:
        return X_train, X_valid, X_test, y_train, y_valid, y_test

if __name__ == '__main__':
    load_data()
    split_input_output()
    split_train_test()