import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import mlflow
import matplotlib.pyplot as plt

# Устанавливаем URI для отслеживания в MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Загружаем данные из CSV-файла
df = pd.read_csv('data/df.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop('discount_applied', axis=1)
y = df['discount_applied']

# Кодируем целевую переменную с помощью LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Масштабируем возрастные данные с использованием StandardScaler
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Кодируем категориальные признаки с помощью OrdinalEncoder
ordinal = OrdinalEncoder()
categorical_columns = ['gender', 'category', 'size', 'subscription_status']
X_train[categorical_columns] = ordinal.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = ordinal.transform(X_test[categorical_columns])

# Параметры для первой модели
model_params_1 = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
}

# Обучение первой модели и логирование результатов в MLflow
with mlflow.start_run() as run_1:
    model_1 = RandomForestClassifier(**model_params_1)
    model_1.fit(X_train, y_train)

    # Предсказание и вычисление F1-score
    y_pred_1 = model_1.predict(X_test)
    f1_1 = f1_score(y_test, y_pred_1)

    # Логируем параметры и метрики первой модели
    mlflow.log_params(model_params_1)
    mlflow.log_metric("f1_score_model_1", f1_1)

    # Сохраняем первую модель
    joblib.dump(model_1, 'model_1.pkl')

# Параметры для второй модели (альтернативные параметры)
model_params_2 = {
    'n_estimators': 250,
    'max_depth': None,
    'min_samples_split': 15,
    'min_samples_leaf': 7,
}

# Обучение второй модели и логирование результатов в MLflow
with mlflow.start_run() as run_2:
    model_2 = RandomForestClassifier(**model_params_2)
    model_2.fit(X_train, y_train)

    # Предсказание и вычисление F1-score для второй модели
    y_pred_2 = model_2.predict(X_test)
    f1_2 = f1_score(y_test, y_pred_2)

    # Логируем параметры и метрики второй модели
    mlflow.log_params(model_params_2)
    mlflow.log_metric("f1_score_model_2", f1_2)

    # Сохраняем вторую модель
    joblib.dump(model_2, 'model_2.pkl')

# Вывод F1-score для обеих моделей
print(f"F1-score Model 1: {f1_1}")
print(f"F1-score Model 2: {f1_2}")

# Создаем график для сравнения производительности моделей
models = ['Model 1', 'Model 2']
f1_scores = [f1_1, f1_2]

plt.bar(models, f1_scores)
plt.ylabel('F1 Score')
plt.title('Comparison of Model Performance')
plt.savefig('performance_comparison.png')  # Сохраняем график в файл
mlflow.log_artifact('performance_comparison.png')  # Логируем график как артефакт

plt.show()

# Логируем текстовый отчет о производительности моделей
with open('model_performance_report.txt', 'w') as f:
    f.write(f"F1-score Model 1: {f1_1}\n")
    f.write(f"F1-score Model 2: {f1_2}\n")

mlflow.log_artifact('model_performance_report.txt')  # Логируем отчет как артефакт
