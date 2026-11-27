"""
Общий файл ЛР 4 «Рекомендательные системы» (common.py). Пишет и меняет только Мелине.

Задача: по прошлым оценкам угадать, какую оценку (1–5) пользователь поставил бы товару,
и по этим прогнозам рекомендовать ему топ-10 товаров.
Данные: data/df_electronics.zip — оценки товаров раздела «Электроника» Amazon (Wan и др., 2020).

Все скрипты команды начинаются со строки, например:
    from common import load_data, split, compare_groups, compare_models, ...

Что здесь есть:
    load_data()        — таблица оценок + удобные столбцы (номер покупки по времени и т. п.)
    split()            — обучение (split = 0) и проверка (split = 2), как задали авторы данных
    алгоритмы          — GlobalMean, ItemMean, Bias, SVD: у всех .fit(train) и .predict(users, items)
    evaluate()         — ошибка прогноза: MAE и RMSE (в звёздах)
    compare_models()   — способ В: ошибка нескольких алгоритмов на проверочной части + картинка
    compare_groups()   — способ А: средняя оценка по группам + картинка
    recommend()        — топ-10 товаров для пользователя
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'df_electronics.zip'
FIGURES_PATH = ROOT / 'figures'

COLUMNS = ['item_id', 'user_id', 'rating', 'timestamp', 'model_attr', 'category',
           'brand', 'year', 'user_attr', 'split']

# Настройки алгоритмов: подобраны по наименьшей RMSE на части split = 1 (для проверки она не используется)
LAMBDA_ITEM = 25    # «сглаживание» оценки товара: товару с 1–2 оценками не доверяем полностью
LAMBDA_USER = 3     # то же для привычки пользователя ставить выше / ниже
SVD_FACTORS = 20    # число скрытых «вкусов» в SVD
RANDOM_STATE = 42


# ---------------- данные ----------------

def load_data(path=DATA_PATH):
    """
    Все оценки (1 292 954 строки) и дополнительные столбцы:
        date          — дата оценки (timestamp как дата)
        user_n        — сколько всего оценок у этого пользователя
        user_order    — номер покупки пользователя по времени (0 — первая, 1 — вторая, ...)
        item_n        — сколько всего оценок у этого товара
        item_order    — номер оценки товара по времени (0 — самый первый покупатель)
        brand         — пустой бренд заменён на 'не указан'
    """
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS or len(df) != 1_292_954:
        raise ValueError(f'Неожиданные данные: {df.shape}. Тот ли файл?')
    if df[['item_id', 'user_id', 'rating', 'timestamp', 'category', 'split']].isna().any().any():
        raise ValueError('В основных столбцах есть пропуски.')

    df['date'] = pd.to_datetime(df['timestamp'])
    df['brand'] = df['brand'].fillna('не указан')
    df = df.sort_values(['date', 'user_id', 'item_id'], kind='stable').reset_index(drop=True)
    df['user_n'] = df.groupby('user_id')['rating'].transform('size')
    df['user_order'] = df.groupby('user_id').cumcount()
    df['item_n'] = df.groupby('item_id')['rating'].transform('size')
    df['item_order'] = df.groupby('item_id').cumcount()
    return df


def split(df):
    """Обучение — split = 0, проверка — split = 2. Одинаково у всей команды."""
    return df[df['split'] == 0], df[df['split'] == 2]


# ---------------- алгоритмы ----------------

class GlobalMean:
    """Всем одна и та же оценка — средняя по обучающей части. Точка отсчёта."""
    name = 'Средняя оценка'

    def fit(self, train):
        self.mu = train['rating'].mean()
        return self

    def predict(self, users, items):
        return np.full(len(users), self.mu)


class ItemMean:
    """Оценка товара: как его оценивают другие люди (со сглаживанием LAMBDA_ITEM)."""
    name = 'Оценка товара'

    def __init__(self, lam=LAMBDA_ITEM):
        self.lam = lam

    def fit(self, train):
        self.mu = train['rating'].mean()
        g = train.groupby('item_id')['rating'].agg(['sum', 'count'])
        self.bi = (g['sum'] - self.mu * g['count']) / (g['count'] + self.lam)
        return self

    def predict(self, users, items):
        return self.mu + pd.Series(items).map(self.bi).fillna(0).to_numpy()


class Bias(ItemMean):
    """Товар + покупатель: оценка товара плюс привычка человека ставить выше или ниже других."""
    name = 'Товар + покупатель'

    def __init__(self, lam_item=LAMBDA_ITEM, lam_user=LAMBDA_USER):
        self.lam = lam_item
        self.lam_user = lam_user

    def fit(self, train):
        super().fit(train)
        rest = train['rating'] - self.mu - train['item_id'].map(self.bi)
        g = rest.groupby(train['user_id']).agg(['sum', 'count'])
        self.bu = g['sum'] / (g['count'] + self.lam_user)
        return self

    def predict(self, users, items, clip=True):
        pred = super().predict(users, items) + pd.Series(users).map(self.bu).fillna(0).to_numpy()
        return np.clip(pred, 1, 5) if clip else pred


class SVD(Bias):
    """
    Матричное разложение: к модели «товар + покупатель» добавляются скрытые «вкусы»
    пользователей и «свойства» товаров, найденные разложением таблицы остатков (SVD_FACTORS штук).
    """
    name = 'SVD'

    def __init__(self, k=SVD_FACTORS):
        super().__init__()
        self.k = k

    def fit(self, train):
        super().fit(train)
        self.users = {u: i for i, u in enumerate(train['user_id'].unique())}
        self.items = {it: i for i, it in enumerate(train['item_id'].unique())}
        rest = (train['rating'] - Bias.predict(self, train['user_id'].to_numpy(),
                                                   train['item_id'].to_numpy())).to_numpy()
        m = csr_matrix((rest, (train['user_id'].map(self.users), train['item_id'].map(self.items))),
                       shape=(len(self.users), len(self.items)))
        u, s, vt = svds(m, k=self.k, random_state=RANDOM_STATE)
        self.P = u * s
        self.Q = vt.T
        return self

    def predict(self, users, items, clip=True):
        base = Bias.predict(self, users, items, clip=False)
        ui = pd.Series(users).map(self.users)
        ii = pd.Series(items).map(self.items)
        ok = (ui.notna() & ii.notna()).to_numpy()
        extra = np.zeros(len(base))
        extra[ok] = np.einsum('ij,ij->i', self.P[ui[ok].astype(int)], self.Q[ii[ok].astype(int)])
        return np.clip(base + extra, 1, 5) if clip else base + extra


# ---------------- проверка ----------------

def evaluate(model, test):
    """Ошибка прогноза на проверочной части: MAE и RMSE (в звёздах)."""
    pred = model.predict(test['user_id'].to_numpy(), test['item_id'].to_numpy())
    err = test['rating'].to_numpy() - pred
    return {'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err ** 2).mean()))}


def compare_models(models, train, test, name=None):
    """
    Способ В: обучаем алгоритмы на train и сравниваем ошибку на test.
    Возвращает таблицу: алгоритм, MAE, RMSE. name — имя картинки в figures/.
    """
    rows = []
    for m in models:
        m.fit(train)
        r = evaluate(m, test)
        rows.append({'алгоритм': m.name, 'MAE': round(r['mae'], 3), 'RMSE': round(r['rmse'], 3)})
    table = pd.DataFrame(rows)
    if name:
        _bar(table['алгоритм'], table['MAE'], 'Средняя ошибка, звёзд (MAE)',
             'Ошибка прогноза на проверочной части', name, fmt='{:.3f}')
    return table


def compare_groups(df, column, name=None, min_count=1):
    """
    Способ А: средняя оценка по группам. column — столбец df (можно свой, добавленный в скрипте).
    Возвращает таблицу: группа, сколько оценок, средняя оценка. Группы меньше min_count не показываются.
    """
    table = (df.groupby(column)['rating']
               .agg(['count', 'mean'])
               .rename(columns={'count': 'оценок', 'mean': 'средняя оценка'})
               .round(3))
    table = table[table['оценок'] >= min_count]
    if name:
        _bar(table.index.astype(str), table['средняя оценка'], 'Средняя оценка, звёзд',
             f'Средняя оценка по группам: {column}', name, fmt='{:.2f}')
    return table


def recommend(model, user_id, df, n=10, categories=None):
    """
    Топ-n товаров для пользователя: прогноз оценки для всех товаров, которые он ещё не оценивал,
    сортировка по убыванию. categories — если задан список, советуем только из этих категорий.
    Возвращает таблицу: товар, категория, бренд, прогноз.
    """
    items = df.drop_duplicates('item_id').set_index('item_id')[['category', 'brand']]
    seen = set(df.loc[df['user_id'] == user_id, 'item_id'])
    cand = items[~items.index.isin(seen)]
    if categories is not None:
        cand = cand[cand['category'].isin(categories)]
    users = np.full(len(cand), user_id)
    try:
        score = model.predict(users, cand.index.to_numpy(), clip=False)   # для сортировки — без обрезки до 5
    except TypeError:
        score = model.predict(users, cand.index.to_numpy())
    out = cand.assign(прогноз=np.round(np.clip(score, 1, 5), 2), _score=score)
    out = out.sort_values('_score', ascending=False).head(n).drop(columns='_score')
    return out.rename_axis('товар').reset_index()


def _bar(labels, values, ylabel, title, name, fmt='{:.2f}'):
    FIGURES_PATH.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(x) for x in labels], values, color='#2B5CA8')
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha='center', va='bottom')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{name}.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    # Быстрая проверка, что всё работает
    df = load_data()
    train, test = split(df)
    print(f'Оценок: {len(df)} | пользователей: {df.user_id.nunique()} | товаров: {df.item_id.nunique()}')
    print(f'Обучение: {len(train)} | проверка: {len(test)}')
    print(compare_models([GlobalMean(), ItemMean(), Bias(), SVD()], train, test).to_string(index=False))
