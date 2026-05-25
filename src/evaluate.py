from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from preprocessing import preprocessing_logreg, DATA_PATH
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import load
from train import MODEL_PATH

def evaluate(DATA_PATH):
    model = load(MODEL_PATH / 'logreg_model.pkl')
    X_train, X_test, _, y_test = preprocessing_logreg(DATA_PATH)
    y_predict, y_probability = model.predict(X_test), model.predict_proba(X_test)[:, 1]

    clf_report = classification_report(y_test, y_predict)
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_predict)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(cm, annot=True, fmt='.2f')
    plt.show()

    print(f'Classification report | Test:\n{clf_report}')
    print(f'ROC-AUC | Test: {roc_auc: .2f}')
    print(f'Precision Recall | Test: {pr_auc: .2f}')

    # Интерпретируемость модели
    coef = pd.DataFrame({'Coef' : model.coef_[0],
                         'Features' : X_test.columns}).sort_values(ascending=True, by='Coef')

    plt.figure(figsize = (10, 9))
    sns.barplot(data=coef, x='Coef', y='Features')
    plt.xlabel('Коэффициенты модели по признакам')
    plt.ylabel('Признаки')
    plt.tight_layout()
    plt.show()

    import shap
    explainer_shap = shap.LinearExplainer(model, masker=X_test)
    shap_values = explainer_shap(X_test)
    shap.summary_plot(shap_values, X_test, plot_type='bar')

if __name__ == '__main__':
    evaluate(DATA_PATH)