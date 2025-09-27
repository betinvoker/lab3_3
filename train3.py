import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier, tracker
from lightfm import LightFM #install
from scipy.sparse import coo_matrix
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

def evaluate_model(model, train_data, test_data, k=10):
    prec = precision_at_k(model, test_data, k=k).mean()
    rec = recall_at_k(model, test_data, k=k).mean()
    auc = auc_score(model, test_data).mean()
    print(f"Precision@{k}: {prec:.4f}")
    print(f"Recall@{k}: {rec:.4f}")
    print(f"AUC: {auc:.4f}")

# Загрузка из CSV
orders = pd.read_csv('./datasets/orders.csv')
clients = pd.read_csv('./datasets/clients.csv')

# Объединяем данные и фильтруем по условию равенства car_type_x и car_type_y
data1 = pd.merge(orders, clients, on='client_id', how='left')
data1 = data1[data1['car_type_x'] == data1['car_type_y']]
interactions = data1.pivot_table(index='client_id',
                                 columns='service_type', values='order_id', aggfunc='count').fillna(0)

interactions = interactions.astype(int)
print(f"{interactions}\n")

interaction_matrix = coo_matrix(interactions.values)
print(f"{interaction_matrix}\n")

train_interactions, test_interactions = random_train_test_split(
    interaction_matrix, test_percentage=0.2
)

model = LightFM(loss='logistic', no_components=2)
model.fit(train_interactions, epochs=2, num_threads=1)

evaluate_model(model, train_interactions, test_interactions, k=10)

def get_recommendations(client_id, model, interactions_df, n=4):
    # Проверяем, что client_id есть в индексах interactions
    if client_id not in interactions_df.index:
        print(f"Клиент с id={client_id} отсутствует в данных.")
        return 0
    else:
        # Получаем индекс клиента в матрице
        client_idx = interactions_df.index.get_loc(client_id)
        # Предсказываем оценки для всех сервисов
        #scores = model.predict(client_idx, np.arange(interactions_df.shape[1]))
        n_items = interactions.shape[1]
        scores = model.predict(user_ids=np.repeat(client_idx, n_items),
        item_ids=np.arange(n_items),
        num_threads=2)
        print(scores)
        # Выбираем топ-n сервисов
        top_items = np.argsort(-scores)[:n]
        print(top_items)
        # Возвращаем названия сервисов
        return interactions_df.columns[top_items].tolist()
    
user_id = 16
recommendations = get_recommendations(user_id, model, interactions)
print(f"\nРекомендации для клиента {user_id}: {recommendations}")

user_id = 0
recommendations = get_recommendations(user_id, model, interactions)
print(f"\nРекомендации для клиента {user_id}: {recommendations}")

def get_positive_recommendations(model, user_id, items_list, interactions, 
                                 num_threads=1):
    n_items = interactions.shape[1]
    # Предсказываем рейтинги для всех услуг
    scores = model.predict(
    user_ids=np.repeat(user_id, n_items),
    item_ids=np.arange(n_items),
    num_threads=num_threads
    )
    # Формируем пары (item_id, score)
    scored_items = list(zip(range(n_items), scores))
    # Фильтруем только положительные
    positive_items = [(item_id, score) for item_id, score in 
                      scored_items if score > 0]
    # Сортируем по убыванию рейтинга
    positive_items.sort(key=lambda x: x[1], reverse=True)
    # Возвращаем список названий услуг и их рейтингов
    return [(items_list[item_id], score) 
            for item_id, score in positive_items]
# Список услуг в порядке item_id
items_list = ['Колеса','Полировка', 'Сервис','Чехлы']

positive_recs = get_positive_recommendations(model, user_id=16,
                                             items_list=items_list, 
                                             interactions=test_interactions)
for service, score in positive_recs:
    print(f"- {service} (рейтинг: {score:.2f})")

model1 = LightFM(
    no_components=10,
    loss='warp',
    item_alpha=1e-6,
    user_alpha=1e-6,
)
# Добавляем веса
sample_weights = train_interactions.copy()
sample_weights.data = np.ones_like(sample_weights.data)

# Шумим непросмотренные позиции
uninteracted_mask = (sample_weights.data == 0)
sample_weights.data[uninteracted_mask] = 0.2 # уменьшаем штраф за отсутствие взаимодействия

model1.fit(train_interactions, sample_weight=sample_weights, 
           epochs=2, num_threads=2)

# Не работает
evaluate_model(model1, train_interactions, test_interactions, k=10)
