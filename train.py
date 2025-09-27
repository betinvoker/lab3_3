import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Целевая переменная:количество заказов на услугу -service_type_Чехлы
daily_orders1 = data.groupby(['date', 'service_type_Чехлы'])['order_id'].count().reset_index()
daily_orders1 = daily_orders1.rename(columns={'order_id': 'order_id'})
print(f"\n{daily_orders1.head()}")

X = daily_orders1.drop(['date', 'order_id'], axis=1)
y = daily_orders1['order_id']
print(X.head())
print(y.head(), end='\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_demand1 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_demand1.fit(X_train, y_train)

preds1 = model_demand1.predict(X_test)

print(f"\nMAE: {mean_squared_error(y_test, preds1)}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds1)))
print(f"MAE: {r2_score(y_test, preds1)}\n")

# Целевая переменная:количество заказов на услугу -service_type_Полировка
daily_orders1 = data.groupby(['date', 'service_type_Полировка'])['order_id'].count().reset_index()
daily_orders1 = daily_orders1.rename(columns={'order_id': 'order_id'})
print(f"\n{daily_orders1.head()}")

X = daily_orders1.drop(['date', 'order_id'], axis=1)
y = daily_orders1['order_id']
print(X.head())
print(y.head(), end='\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_demand1 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_demand1.fit(X_train, y_train)

preds1 = model_demand1.predict(X_test)

print(f"\nMAE: {mean_squared_error(y_test, preds1)}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds1)))
print(f"R^2: {r2_score(y_test, preds1)}\n")

# Целевая переменная:количество заказов на услугу -service_type_Сервис
daily_orders1 = data.groupby(['date', 'service_type_Сервис'])['order_id'].count().reset_index()
daily_orders1 = daily_orders1.rename(columns={'order_id': 'order_id'})
print(f"\n{daily_orders1.head()}")

X = daily_orders1.drop(['date', 'order_id'], axis=1)
y = daily_orders1['order_id']
print(X.head())
print(y.head(), end='\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_demand1 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_demand1.fit(X_train, y_train)

preds1 = model_demand1.predict(X_test)

print(f"\nMAE: {mean_squared_error(y_test, preds1)}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds1)))
print(f"R^2: {r2_score(y_test, preds1)}\n")

# Целевая переменная:количество заказов на услугу -service_type_Колеса
daily_orders1 = data.groupby(['date', 'service_type_Колеса'])['order_id'].count().reset_index()
daily_orders1 = daily_orders1.rename(columns={'order_id': 'order_id'})
print(f"\n{daily_orders1.head()}")

X = daily_orders1.drop(['date', 'order_id'], axis=1)
y = daily_orders1['order_id']
print(X.head())
print(y.head(), end='\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_demand1 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_demand1.fit(X_train, y_train)

preds1 = model_demand1.predict(X_test)

print(f"\nMAE: {mean_squared_error(y_test, preds1)}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds1)))
print(f"R^2: {r2_score(y_test, preds1)}\n")