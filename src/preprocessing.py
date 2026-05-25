from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from joblib import dump

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models'

def load_data(DATA_PATH):
    data = pd.read_csv(DATA_PATH / 'Telco_Customer_Churn.csv')
    return data

def preprocessing_logreg(DATA_PATH):
    df = load_data(DATA_PATH).drop(columns=['customerID'], axis=1)
    df['TotalCharges'] = df['TotalCharges'].map(lambda x: float(x) if x != ' ' else 0)

    num_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']
    bool_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    multiclass_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    passthrough_column = ['SeniorCitizen']

    data_transform = ColumnTransformer([
        ('num_cols', StandardScaler(), num_columns),
        ('pass', 'passthrough', passthrough_column),
        ('le', OrdinalEncoder(), bool_columns),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), multiclass_columns)
    ], n_jobs=-1)

    X = df.drop(columns=['Churn'], axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['Churn'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                               test_size=0.25,
                                                               random_state=42,
                                                               stratify=y)

    X_train = data_transform.fit_transform(X_train)
    X_test = data_transform.transform(X_test)

    feature_names = data_transform.get_feature_names_out()
    feature_names = [col.split('__')[1] for col in feature_names]

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    columns_to_drop = ['PhoneService',
                       'OnlineBackup_Yes',
                       'InternetService_DSL',
                       'MultipleLines_No phone service',
                       'StreamingMovies_No']

    X_train = X_train.drop(columns=columns_to_drop, axis=1)
    X_test = X_test.drop(columns=columns_to_drop, axis=1)


    dump(data_transform, MODEL_PATH / 'data_transform.pkl')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    preprocessing_logreg(DATA_PATH)