from joblib import load
from preprocessing import preprocessing_logreg, DATA_PATH
from train import MODEL_PATH
import pandas as pd
import numpy as np

def predict(data_path):
    _, X_test, _, y_test = preprocessing_logreg(data_path)
    model = load(MODEL_PATH / 'logreg_model.pkl')

    y_probability = model.predict_proba(X_test)[:, 1]
    y_predict = (y_probability > 0.5).astype(int)

    result = pd.DataFrame({ 'y_test' : y_test.copy(),
                            'y_test_probability, %' : np.round(y_probability.copy(), 2) * 100,
                            'y_test_predicted' : y_predict.copy()})
    print(result.head(5))


if __name__ == '__main__':
    predict(DATA_PATH)