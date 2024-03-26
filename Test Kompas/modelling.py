import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import copy

def load_data():
    X_train_vect = joblib.load('Test Kompas\X_train_vect.pkl')
    X_valid_vect = joblib.load('Test Kompas\X_valid_vect.pkl')
    X_test_vect = joblib.load('Test Kompas\X_test_vect.pkl')
    y_train = joblib.load('Test Kompas\y_train.pkl')
    y_valid = joblib.load('Test Kompas\y_valid.pkl')
    y_test = joblib.load('Test Kompas\y_test.pkl')

    return X_train_vect, X_valid_vect, X_test_vect, y_train, y_valid, y_test

def create_model_param(return_file=False):
    nb_params = {
    'alpha': [0.1, 0.5, 1.0]
    }
    
    lgr_params = {
        'penalty': ['l2'],
        'C': [0.01, 0.1],
        'max_iter': [100, 300, 500]
    }

    svc_params = {
    'C': [0.1, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
    }

    # Create model params
    list_of_param = {
        'LogisticRegression': lgr_params,
        'SVC': svc_params,
        'MultinomialNB': nb_params
    }

    if return_file:
        return list_of_param

def create_model_object(return_file=False):
    # Buat model object
    lgr = LogisticRegression()
    nb = MultinomialNB()
    svc = SVC()

    # Buat list model
    list_of_model = [
        {'model_name': lgr.__class__.__name__, 'model_object': lgr},
        {'model_name': nb.__class__.__name__, 'model_object': nb},
        {'model_name': svc.__class__.__name__, 'model_object': svc}
    ]

    if return_file:
        return list_of_model
    
def train_model(return_file=False):
    # Buat list params dan object model
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # Buat dictionary kosong untuk model yg sudah dilatih
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])
        model_param = list_of_param[model_name]

        print('Training model :', model_name)

        model = GridSearchCV(estimator = model_obj,
                             param_grid = model_param,
                             cv = 5,
                             n_jobs=1,
                             verbose=10,
                             scoring = 'accuracy')

        # Train model
        model.fit(X_train_vect, y_train)

        # Predict
        y_pred_train = model.predict(X_train_vect)
        y_pred_valid = model.predict(X_valid_vect)

        # Get score
        train_score = accuracy_score(y_train, y_pred_train)
        valid_score = accuracy_score(y_valid, y_pred_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model,
            'train_auc': train_score,
            'valid_auc': valid_score,
            'best_params': model.best_params_
        }

    if return_file:
        return list_of_param, list_of_model, list_of_tuned_model
    
def get_best_model(return_file=False):
    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = -99999
    best_model_param = None

    for model_name, model in list_of_tuned_model.items():
        if model['valid_auc'] > best_performance:
            best_model_name = model_name
            best_model = model['model']
            best_performance = model['valid_auc']
            best_model_param = model['best_params']

    # Dump the best model
    joblib.dump(best_model, 'best_model.pkl')

    if return_file:
        return best_model

if __name__ == '__main__':
    X_train_vect, X_valid_vect, X_test_vect, y_train, y_valid, y_test = load_data()

    create_model_param()
    create_model_object()
    list_of_param, list_of_model, list_of_tuned_model = train_model()
    get_best_model(list_of_tuned_model=list_of_tuned_model)
