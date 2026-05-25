import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from pathlib import Path
from joblib import load
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
def load_file():
    global filepath
    filepath = filedialog.askopenfilename()
    if filepath:
        choose_file.config(text=f'Путь до файла:\n{filepath}')
    else:
        messagebox.showerror(title='Ошибка', message='Выберите файл')
        return None
    data = pd.read_csv(filepath)
    n_label.config(text=f'Введите номер строки данных для предсказания -\n({0} - {len(data)})')


def predict():
    model = load(MODEL_DIR / 'logreg_model.pkl')
    try:
        df =  pd.read_csv(filepath).drop('Churn', axis=1)
        n = int(choose_row.get())
        if n > df.shape[0] or n < 0:
            messagebox.showerror(title='Ошибка', message=f'Введите номер строки, находящийся в диапазоне {0} - {df.shape[0]}')
        else:
            if not checkbutton1():
                df = df.iloc[[n]]
            elif checkbutton1():
                df = df.iloc[[n]]
                data_transform = load(MODEL_DIR/ 'data_transform.pkl')
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
                prediction_label.config(bg='red', text=f'Модель предсказала, что данный клиент\n откажется от услуг '
                                                       f'компании с вероятностью {predicted_probability * 100: .2f}%')
            elif predicted_class == 0:
                prediction_label.config(bg='green', text=f'Модель предсказала, что данный клиент продолжит\n'
                                        f'пользоваться услугами компании с вероятностью {(1 - predicted_probability) * 100: .2f}%')
    except NameError:
        messagebox.showerror('Ошибка', f'Сначала укажите путь до файла')
    except FileNotFoundError:
        messagebox.showerror('Ошибка', f'Сначала укажите путь до файла')
    except Exception as error:
        messagebox.showerror('Неизвестная ошибка', f'Произошла ошибка\n{error}')

def checkbutton1():
    return var1.get()

root = tk.Tk()

root.geometry("800x400")
root.resizable(width=False, height=False)
root.title("Telco Customer Churn Analyzer")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

font_1 = ('Times New Roman', 14)
font_2 = ('Times New Roman', 14, 'bold')
# -------------------------------------------------------------------------------------------------------------

hello_label = tk.Label(root, text='Здравствуйте! Загрузите файл для осуществления предсказания: ', font=font_1)
hello_label.place(x=160, y=25)

choose_file = tk.Label(root, text='Выберите файл для загрузки . . .', font=font_2)
choose_file.place(x=25, y=100)

load_file_button = tk.Button(text='Выбрать файл', command=load_file)
load_file_button.place(x=550, y=100)

model_label = tk.Label(text='Используется модель логистической регрессии', font=font_1)
model_label.place(x=25, y=150)

n_label = tk.Label(text='Введите номер строки данных для предсказания - ', font=font_1)
n_label.place(x=25, y=200)

choose_row = tk.Entry(root, width=10)
choose_row.place(x=450, y=200)
choose_row.insert(0, '0')

predict_button = tk.Button(text='Предсказать', command=predict)
predict_button.place(x=550, y=200)

prediction_label = tk.Label(text='Модель предсказала, что данный клиент ...', font=font_1)
prediction_label.place(x=150, y=300)

var1 = tk.BooleanVar()
data_type = tk.Checkbutton(root, command=checkbutton1, variable=var1)
data_type.place(x=25, y=250)

data_type_label = tk.Label(text='- Raw data', font=font_1)
data_type_label.place(x=50, y=250)

root.mainloop()