from joblib import load
from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd

app = Flask(__name__)
MODEL_PATH = Path(__name__).resolve().parent.parent / 'models'

@app.route('/')
def home_page():
    return render_template('home_page.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def solve():
    if request.method == 'POST':
        file = request.files['file']
        n = request.form['n_row']

        if not file:
            file_error = 'Файл не загружен!'
            return render_template('predict.html', file_error=file_error)
        if not n:
            text_error = 'Заполните данное поле'
            return render_template('predict.html', text_error=text_error)
        if file and n:
            df = pd.read_csv(file).drop('Churn', axis=1)
            rows_count = df.shape[0]
            n = int(n)

            if n > rows_count or n < 0:
                text_error = f'Ошибка! Введено недействительное значение.\nПодсказка: диапазон строк от {0} до {rows_count}'
                return render_template('predict.html', text_error=text_error)
            else:
                model = load(MODEL_PATH / 'logreg_model.pkl')
                df = df.iloc[[n]]
                if not request.form.get('checkbox1'):
                    print('prep data')
                    pass
                elif request.form.get('checkbox1'):
                    print('raw data')
                    data_transform = load(MODEL_PATH / 'data_transform.pkl')
                    columns_to_drop = ['PhoneService',
                                       'OnlineBackup_Yes',
                                       'InternetService_DSL',
                                       'MultipleLines_No phone service',
                                       'StreamingMovies_No']
                    df = data_transform.transform(df)
                    feature_names = [feature_name.split('__')[1] for feature_name in data_transform.get_feature_names_out()]
                    df = pd.DataFrame(df, columns=feature_names)
                    df = df.drop(columns_to_drop, axis=1)

                predicted_probability = model.predict_proba(df)[0, 1]
                predicted_class = (predicted_probability > 0.5).astype(int)

                if predicted_class == 1:
                    prediction_text = (f'Модель предсказала, что данный клиент\n откажется от услуг '
                                      f'компании с вероятностью {predicted_probability * 100: .2f}%')
                    color = 'red'
                    return render_template('predict.html', color=color, prediction_text=prediction_text)

                elif predicted_class == 0:
                    prediction_text = (f'Модель предсказала, что данный клиент продолжит\n'
                                     f'пользоваться услугами компании с вероятностью {(1 - predicted_probability) * 100: .2f}%')
                    color = 'green'
                    return render_template('predict.html', color=color, prediction_text=prediction_text)


        return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)