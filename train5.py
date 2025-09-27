import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

def load_data():
    return pd.read_csv('./datasets/data.csv')

data = load_data()

app = dash.Dash(__name__)
app.title = "A/B Тест | Дитейлинг"

app.layout = html.Div([
    html.H1("A/B тестирование рекомендаций в дитейлинге", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Выберите тип автомобиля:"),
        dcc.Dropdown(
            id='car-type-filter',
            options=[{'label': car, 'value': car} for car in
                     data['car_type'].unique()],
                     value=data['car_type'].unique().tolist(),
                     multi=True
                     )
                     ], style={'width': '80%', 'margin': 'auto', 'padding': '10px'}),
                     html.Div(id='metrics-summary', style={'textAlign': 'center', 'margin': '20px'}),
                     html.Div([
                         dcc.Graph(id='ctr-chart'),
                         dcc.Graph(id='cr-chart'),
                         dcc.Graph(id='avg-check-chart'),
                         dcc.Graph(id='revenue-per-user-chart')
                         ], style={'width': '90%', 'margin': 'auto'})
                         ], style={'fontFamily': 'Arial'})

@app.callback([
    Output('ctr-chart', 'figure'), # Обновляет график CTR (Click-Through Rate)
    Output('cr-chart', 'figure'), # Обновляет график CR (Conversion Rate)
    Output('avg-check-chart', 'figure'), # Обновляет график среднего чека
    Output('revenue-per-user-chart', 'figure'), # Обновляет график дохода на пользователя
    Output('metrics-summary', 'children')], # Обновляет текстовый компонент с итоговыми метриками
    [Input('car-type-filter', 'value')] # Получает текущее значение фильтра по типу автомобиля
)

def update_figures(selected_car_types):
    # Фильтруем данные на основе выбранных типов автомобилей
    filtered_data = data[data['car_type'].isin(selected_car_types)]
    # Расчет CTR (Click-Through Rate) для каждой группы
    # Группируем данные по 'group' и вычисляем среднее значение кликов
    ctr_data = filtered_data.groupby('group')['clicked'].agg(['mean']).reset_index()
    # Преобразуем среднее значение в проценты
    ctr_data['mean'] *= 100
    # Создаем столбчатую диаграмму для CTR
    fig_ctr = px.bar(ctr_data, x='group', y='mean', title="CTR (%)", 
                     labels={'mean': 'CTR (%)'})
    # Расчет Conversion Rate (коэффициента конверсии) для каждой группы
    # Фильтруем данные, где клики равны 1, затем группируем по 'group' и вычисляем 
    # среднее значение покупок
    cr_data = filtered_data[filtered_data['clicked'] ==
    1].groupby('group')['purchased'].agg(['mean']).reset_index()
    # Преобразуем среднее значение в проценты
    cr_data['mean'] *= 100
    # Создаем столбчатую диаграмму для Conversion Rate
    fig_cr = px.bar(cr_data, x='group', y='mean', title="Conversion Rate (%)", 
                    labels={'mean': 'CR (%)'})
    # Расчет среднего чека для каждой группы
    # Фильтруем данные, где покупки равны 1, затем группируем по 'group' и 
    # вычисляем среднее значение суммы покупки
    avg_check_data = filtered_data[filtered_data['purchased'] == 1].groupby('group')['purchase_amount'].mean().reset_index()
    # Создаем столбчатую диаграмму для среднего чека
    fig_avg_check = px.bar(avg_check_data, x='group', y='purchase_amount', title="Средний чек (₽)")
    # Расчет дохода на пользователя для каждой группы
    # Суммируем сумму покупок и делим на количество пользователей в каждой группе
    revenue_per_user = filtered_data.groupby('group')['purchase_amount'].sum() / filtered_data.groupby('group').size()
    # Преобразуем результат в DataFrame
    revenue_df = revenue_per_user.reset_index()
    revenue_df.columns = ['group', 'revenue_per_user']
    # Создаем столбчатую диаграмму для дохода на пользователя
    fig_revenue = px.bar(revenue_df, x='group', y='revenue_per_user', title="Доход на пользователя (₽)")
    # Создаем текстовый отчет с основными метриками для каждой группы
    metrics_text = [
        html.P(f"CTR Group A: {ctr_data[ctr_data['group']=='A']['mean'].values[0]:.2f}% | "
               f"CTR Group B: {ctr_data[ctr_data['group']=='B']['mean'].values[0]:.2f}%"),
        html.P(f"Conversion Rate Group A: {cr_data[cr_data['group']=='A']['mean'].values[0]:.2f}% | "
               f"Conversion Rate Group B: {cr_data[cr_data['group']=='B']['mean'].values[0]:.2f}%"),
        html.P(f"Средний чек Group A: {avg_check_data[avg_check_data['group']=='A']['purchase_amount'].values[0]:.2f} ₽ | "
               f"Средний чек Group B: {avg_check_data[avg_check_data['group']=='B']['purchase_amount'].values[0]:.2f} ₽"),
        html.P(f"Доход на пользователя Group A: {revenue_df[revenue_df['group']=='A']['revenue_per_user'].values[0]:.2f} ₽ | "
               f"Доход на пользователя Group B: {revenue_df[revenue_df['group']=='B']['revenue_per_user'].values[0]:.2f} ₽")
    ]
    # Возвращаем обновленные графики и текстовый отчет
    return fig_ctr, fig_cr, fig_avg_check, fig_revenue, metrics_text

app.run(debug=False)