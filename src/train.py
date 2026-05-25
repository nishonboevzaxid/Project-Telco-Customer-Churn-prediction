from sklearn.linear_model import LogisticRegression
from preprocessing import preprocessing_logreg, DATA_PATH, MODEL_PATH
from joblib import dump



def train_model_logreg(DATA_PATH):
    X_train, _, y_train, _ = preprocessing_logreg(DATA_PATH)
    model = LogisticRegression(random_state=42,
                               class_weight='balanced',
                               C=0.1,
                               solver='lbfgs',
                               penalty='l2')
    model.fit(X_train, y_train)

    dump(model, MODEL_PATH / 'logreg_model.pkl')

    return model

if __name__ == '__main__':
    train_model_logreg(DATA_PATH)