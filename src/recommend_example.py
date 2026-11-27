"""
Мелине: пример работы рекомендательной системы (п. 2.10 методички).

Берём покупателя, у которого в обучающей части несколько оценок, показываем его историю
и два варианта рекомендаций (алгоритм «Товар + покупатель»):
    1) топ-10 из всех товаров;
    2) топ-10 из тех категорий, которые он уже покупал.
Затем смотрим, что он на самом деле оценил на проверочной части.

Запуск без параметров — пример для заранее выбранного покупателя.
Свой покупатель: python recommend_example.py <user_id>
"""

import sys

from common import Bias, load_data, recommend, split

df = load_data()
train, test = split(df)

if len(sys.argv) >= 2:
    user = int(sys.argv[1])
else:
    # покупатель с 10+ оценками и обычными, не завышенными оценками (средняя 3,5–4,2),
    # у которого есть оценка в проверочной части
    stats = train.groupby('user_id')['rating'].agg(['size', 'mean'])
    pool = stats[(stats['size'] >= 10) & stats['mean'].between(3.5, 4.2) & stats.index.isin(test['user_id'])]
    user = int(pool.sort_values('size', ascending=False).index[0])

model = Bias().fit(train)
history = train[train['user_id'] == user][['date', 'item_id', 'category', 'brand', 'rating']]
print(f'Покупатель {user}: {len(history)} оценок в обучающей части')
print(history.tail(10).to_string(index=False))

print('\nРекомендации — топ-10 из всех товаров:')
print(recommend(model, user, train).to_string(index=False))

cats = history['category'].unique().tolist()
print(f'\nРекомендации — топ-10 из его категорий ({", ".join(cats)}):')
print(recommend(model, user, train, categories=cats).to_string(index=False))

print('\nЧто он оценил потом (проверочная часть):')
print(test[test['user_id'] == user][['date', 'item_id', 'category', 'brand', 'rating']].to_string(index=False))
