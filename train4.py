import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./datasets/data.csv')

def calculate_metrix(group):
    group_data = data[data['group'] == group]
    clicks = group_data['clicked'].sum() / len(group_data)
    purchases = group_data['purchased'].sum() / len(group_data)
    avg_check = group_data['purchase_amount'].mean()
    return {
        'CTR': clicks,
        'Conversion Rate': purchases,
        'Avg Check': avg_check
    }

metrics_a = calculate_metrix('A')
metrics_b = calculate_metrix('B')

print("Метрики по группам:")
print(f"Группа A: {metrics_a}")
print(f"Группа B: {metrics_b}")

# Визуализация CTR и конверсии
metrics_df = pd.DataFrame([metrics_a, metrics_b], index=['A', 'B'])
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=metrics_df.index, y=metrics_df['CTR'], ax=ax[0])
ax[0].set_title('CTR по группам')
ax[0].set_ylabel('CTR (%)')

sns.barplot(x=metrics_df.index, y=metrics_df['Conversion Rate'], ax=ax[1])
ax[1].set_title('Conversion Rate по группам')
ax[1].set_ylabel('CR (%)')

plt.tight_layout()
plt.show()

group_a_sales = data[(data['group'] == 'A') 
                     & (data['purchase_amount'] > 0)]['purchase_amount']

group_b_sales = data[(data['group'] == 'B') 
                     & (data['purchase_amount'] > 0)]['purchase_amount']
t_stat, p_val = ttest_ind(group_a_sales, group_b_sales, equal_var=False)
print(f"\nT-тест: t={t_stat:.2f}, p={p_val:.4f}")
if p_val < 0.05:
    print("Различие в среднем чеке статистически значимо!")
else:
    print("Нет значимого различия.")
# Сравнение среднего чека
clicks_a = data[(data['group'] == 'A') & (data['clicked'] == 1)].shape[0]
non_clicks_a = data[(data['group'] == 'A') & (data['clicked'] == 0)].shape[0]
clicks_b = data[(data['group'] == 'B') & (data['clicked'] == 1)].shape[0]
non_clicks_b = data[(data['group'] == 'B') & (data['clicked'] == 0)].shape[0]

contingency_table = [[clicks_a, non_clicks_a],
                     [clicks_b, non_clicks_b]]

chi2, p, _, _ = chi2_contingency(contingency_table)

print(f"Хи-квадрат тест: χ²={chi2:.2f}, p={p:.4f}\n")

print("\nИТОГОВЫЙ АНАЛИЗ")
print("------------------------")
print(f"CTR Group A: {metrics_a['CTR']:.2%}")
print(f"CTR Group B: {metrics_b['CTR']:.2%}")
print(f"Рост CTR: {((metrics_b['CTR'] - metrics_a['CTR']) / metrics_a['CTR']) * 100:.1f}%")
print(f"\nConversion Rate Group A: {metrics_a['Conversion Rate']:.2%}")
print(f"Conversion Rate Group B: {metrics_b['Conversion Rate']:.2%}")
print(f"Рост CR: {((metrics_b['Conversion Rate'] - metrics_a['Conversion Rate']) / metrics_a['Conversion Rate']) * 100:.1f}%")
print(f"\nСредний чек Group A: {metrics_a['Avg Check']:.2f} ₽")
print(f"Средний чек Group B: {metrics_b['Avg Check']:.2f} ₽")
print(f"Рост среднего чека: {((metrics_b['Avg Check'] - metrics_a['Avg Check']) / metrics_a['Avg Check']) * 100:.1f}%")

