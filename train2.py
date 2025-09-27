import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier, tracker

orders = pd.read_csv('./datasets/orders.csv')
clients = pd.read_csv('./datasets/clients.csv')

data = pd.merge(orders, clients, on='client_id', how='left')
data = data[data['car_type_x'] == data['car_type_y']]

print(data.shape)
print(data)

print(f"\n{data.iloc[1]}")

#Разработка новых признаков
# Преобразуем даты
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)
# Признак "сколько дней прошло с последнего визита"
data['last_visit_date'] = pd.to_datetime(data['last_visit_date'])
data['days_since_last_visit'] = (data['date'] - data['last_visit_date']).dt.days
# Кодируем категориальные признаки
data = pd.get_dummies(data, columns=['car_type_x'], drop_first=False)
data = pd.get_dummies(data, columns=['service_type'], drop_first=False)
print(f"\n{data.iloc[1]}")

personalization_data = data.groupby('client_id').agg({
'cost': ['mean', 'sum'],
'mileage': 'mean',
'car_age': 'mean',
'service_type_Полировка' : 'sum',
'service_type_Чехлы' : 'sum',
'service_type_Сервис': 'sum',
'service_type_Колеса': 'sum'
}).reset_index()

# Переименовываем колонки
personalization_data.columns = ['_'.join(col).strip() if isinstance(col, tuple)
    else col for col in personalization_data.columns.values]
personalization_data = personalization_data.rename(columns={'client_id_': 'client_id'})

personalization_data['will_order_Услуга'] = np.random.choice([0, 1],
size=len(personalization_data), p=[0.35, 0.65])
print(personalization_data.iloc[1])

X_pers = personalization_data.drop(['client_id', 'will_order_Услуга'], axis=1)
y_pers = personalization_data['will_order_Услуга']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pers, y_pers, test_size=0.2, shuffle=False)

model_personalize = XGBClassifier()
model_personalize.fit(X_train_p, y_train_p)

preds_p = model_personalize.predict(X_test_p)

print(f"\nAccuracy: {accuracy_score(y_test_p, preds_p)}\n")
print(classification_report(y_test_p, preds_p))
