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