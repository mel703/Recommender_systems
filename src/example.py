"""
Образец скрипта. Скопируй его, назови g<номер гипотезы>_<фамилия>.py и оставь нужный способ.
Запуск: правой кнопкой на файле -> Run (в PyCharm) или из папки src: python example.py
Данные читаются около 5 секунд.
"""

from common import Bias, GlobalMean, compare_groups, compare_models, load_data, split

df = load_data()

# ---------- Способ А: сравнить оценки групп ----------
# Гипотеза (пример): «в выходные люди ставят оценки так же, как в будни».
# Правило вывода (пишем ДО проверки): подтверждена, если разница средних меньше 0,1 звезды.
df['выходной'] = df['date'].dt.dayofweek >= 5
table = compare_groups(df, 'выходной', name='example_a')
print('Способ А — средняя оценка по группам:')
print(table)
diff = table['средняя оценка'].max() - table['средняя оценка'].min()
print(f'Разница: {diff:.3f} звезды ->', 'подтверждена' if diff < 0.1 else 'опровергнута')

# ---------- Способ Б: что покупают вместе ----------
# Гипотеза (пример): «вторую покупку человек чаще делает в той же категории, что и первую».
# Сравниваем с тем, как часто это совпадало бы случайно.
# Правило: подтверждена, если доля совпадений хотя бы в 2 раза выше случайной.
first = df[df['user_order'] == 0].set_index('user_id')['category']
second = df[df['user_order'] == 1].set_index('user_id')['category']
pairs = first.to_frame('первая').join(second.rename('вторая'), how='inner')
share = (pairs['первая'] == pairs['вторая']).mean()
p = df['category'].value_counts(normalize=True)
chance = (p ** 2).sum()
print(f'\nСпособ Б — вторая покупка в той же категории: {share:.1%}, случайно было бы {chance:.1%}')
print('->', 'подтверждена' if share >= 2 * chance else 'опровергнута')

# ---------- Способ В: сравнить точность алгоритмов ----------
# Гипотеза (пример): «учёт привычек покупателя уменьшает ошибку хотя бы на 0,05 звезды».
train, test = split(df)
table = compare_models([GlobalMean(), Bias()], train, test, name='example_c')
print('\nСпособ В — ошибка на проверочной части:')
print(table.to_string(index=False))
gain = table['MAE'].iloc[0] - table['MAE'].iloc[1]
print(f'Ошибка меньше на {gain:.3f} звезды ->', 'подтверждена' if gain >= 0.05 else 'опровергнута')

print('\nКартинки: figures/example_a.png, figures/example_c.png')
