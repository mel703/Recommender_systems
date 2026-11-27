"""
Мелине, гипотеза Г1: чем больше покупок мы знаем о человеке, тем точнее совет.

Способ В: обучаем алгоритм «Товар + покупатель» (он учитывает привычки человека)
и смотрим его ошибку на проверочной части отдельно для групп покупателей
по числу их оценок в обучающей части: 1, 2, 3–4, 5 и больше.
Для сравнения — алгоритм «Оценка товара», который о человеке ничего не знает.

Правило вывода (записано до проверки):
    подтверждена, если у покупателей с 5+ оценками ошибка (MAE) меньше,
    чем у покупателей с 1 оценкой, хотя бы на 0,1 звезды.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import FIGURES_PATH, Bias, ItemMean, load_data, split

THRESHOLD = 0.1

df = load_data()
train, test = split(df)
history = train.groupby('user_id').size()
groups = pd.cut(test['user_id'].map(history), [0, 1, 2, 4, np.inf], labels=['1', '2', '3–4', '5+'])

personal = Bias().fit(train)
general = ItemMean().fit(train)
test = test.assign(
    группа=groups,
    err_personal=(test['rating'] - personal.predict(test['user_id'].to_numpy(), test['item_id'].to_numpy())).abs(),
    err_general=(test['rating'] - general.predict(test['user_id'].to_numpy(), test['item_id'].to_numpy())).abs(),
)
table = test.groupby('группа', observed=True).agg(
    покупателей=('rating', 'size'),
    ошибка_с_учётом_человека=('err_personal', 'mean'),
    ошибка_без_учёта_человека=('err_general', 'mean'),
).round(3)
print('Г1: ошибка (MAE, звёзд) по числу известных покупок человека')
print(table.to_string())

gain = table.loc['1', 'ошибка_с_учётом_человека'] - table.loc['5+', 'ошибка_с_учётом_человека']
print(f'\nОшибка у покупателей с 5+ оценками меньше, чем с 1 оценкой, на {gain:.3f} звезды')
print('Г1:', 'подтверждена' if gain > THRESHOLD else 'опровергнута')

# Картинка для слайда
FIGURES_PATH.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(table))
ax.bar(x - 0.2, table['ошибка_без_учёта_человека'], 0.4, label='не знаем человека', color='#9CB3D9')
ax.bar(x + 0.2, table['ошибка_с_учётом_человека'], 0.4, label='учитываем его покупки', color='#2B5CA8')
for i, v in enumerate(table['ошибка_с_учётом_человека']):
    ax.text(i + 0.2, v, f'{v:.2f}', ha='center', va='bottom')
ax.set_xticks(x, table.index)
ax.set_xlabel('Сколько оценок человека мы знаем')
ax.set_ylabel('Средняя ошибка, звёзд (MAE)')
ax.set_title('Г1: чем больше знаем о человеке, тем точнее прогноз')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'g1_gabrielyan.png', dpi=150)
plt.close(fig)
print('Картинка: figures/g1_gabrielyan.png')
