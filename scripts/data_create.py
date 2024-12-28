import pandas as pd

# Открываем файл CSV с данными о покупках
df = pd.read_csv('data/shopping_trends.csv')

# Приводим имена колонок к нижнему регистру и заменяем пробелы на символы подчеркивания
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Оставляем только нужные колонки для дальнейшей работы
columns_to_keep = ['age', 'gender', 'category', 'size', 'subscription_status', 'discount_applied']
df = df[columns_to_keep]

# Сохраняем обработанные данные в новое CSV файле без индекса
df.to_csv('data/df.csv', index=False)
