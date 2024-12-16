import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns


# чтение дата сетов
df = pd.read_csv('train.csv')
df1 = pd.read_csv('valid.csv')




# удаление дубликатов и столбцов с более чем 25% пропущенных значений
df = df.drop_duplicates()

threshold = len(df) * 0.75
res = df.dropna(thresh=threshold, axis=1)
res.to_csv("res.csv", index=False)
print(res.info())


# определяем категориальные столбцы которые должны закодировать
categorical_columns = res.select_dtypes(include=['object']).columns

# кодируем данные
res_encoded = pd.get_dummies(res, columns=categorical_columns)

# заполнение оставшихся пропущенных значений 0
res_encoded = res_encoded.fillna(0)


# создание матрицы корреляции
corr_matrix = res_encoded.corr()

# выявление и удаление столбцов сильно коррелирующих между собой
threshold = 0.9
high_corr_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold]
to_drop = [pair[1] for pair in high_corr_pairs]
clean = res_encoded.drop(columns=to_drop)
clean.to_csv('clean.csv', index=False)


# вывод тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Тепловая карта корреляции')
plt.show()



# удаление болевых колонок появившихся из-за выше указаных преобразований
for column in df.columns:
    if df[column].dtype == 'bool':
        df = df.drop(columns=column)

df.to_csv("clean1.csv", index=False)
print(df.info())


# приведение valid к виду train
valid1 = df1.reindex(columns=df.columns)
valid1.to_csv("valid1.csv", index=False)
valid1 = pd.read_csv('valid1.csv')
valid1.info()

# чтение данных
clean1 = pd.read_csv('clean1.csv')
valid1 = pd.read_csv('valid1.csv')

# разделение на признаки и целевую переменную train
X = clean1.drop('target', axis=1)
y = clean1['target']

# сохранение признаков и целевой переменной train
X.to_csv('features.csv', index=False)
y.to_csv('target.csv', index=False)


# разделение на признаки и целевую переменную valid
x_valid = valid1.drop('target', axis=1)
y_valid = valid1['target']

# сохранение признаков и целевой переменной valid
x_valid.to_csv('features_val.csv', index=False)
y_valid.to_csv('target_val.csv', index=False)

# загрузка данных train
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv')

# разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# загрузка данных valid
X_val = pd.read_csv('features_val.csv')
y_val = pd.read_csv('target_val.csv')



# создание модели регрессии и установка гиперпараметров
base_model = RandomForestRegressor(
    n_estimators=5,       # количество деревьев
    max_depth=5,           # глубина деревьев
    min_samples_split=25,  # минимальное количество образцов для разделения
    min_samples_leaf=15,    # минимальное количество образцов в листе
    random_state=42
)

model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=42)
model.fit(X_train, y_train)


# предсказание на тестовых данных
y_pred = model.predict(X_test)



# вычисление ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')

# вывод графика ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")


mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")


# предсказание на данных valid
y_val_pred = model.predict(X_val)


# вычисление ROC-AUC
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'ROC-AUC Score: {roc_auc}')

# вывод графика ROC-AUC
fpr, tpr, thresholds1 = roc_curve(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


mse = mean_squared_error(y_val, y_val_pred)
print(f"MSE: {mse}")

rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

mae = mean_absolute_error(y_val, y_val_pred)
print(f"MAE: {mae}")

