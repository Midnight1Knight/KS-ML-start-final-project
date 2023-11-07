import hashlib
import os
from datetime import datetime


import lightgbm as lgb
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine


from typing import List
from fastapi import FastAPI
from schema import PostGet, Response


# узнаем путь до модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        if path == 'control':
            path = '/workdir/user_input/model_control'
        elif path == 'test':
            path = '/workdir/user_input/model_test'
    else:
        if path == 'control':
            path = 'models/catboost_model_weak'
        elif path == 'test':
            path = 'models/catboost_model'
    return path


# загружаем нашу модель в память кода
def load_models():
    model_path_control = get_model_path('control')
    model_path_test = get_model_path('test')

    loaded_model_control = CatBoostClassifier()  # Создаем пустую модель
    loaded_model_test = CatBoostClassifier()
    # Загружаем сохраненную модель
    loaded_model_control.load_model(model_path_control)
    loaded_model_test.load_model(model_path_control)
    return loaded_model_control, loaded_model_test


def load_features() -> pd.DataFrame:
    query1 = 'SELECT * FROM andre_karasev_user_features_lesson_22'
    query2 = 'SELECT * FROM andre_karasev_post_features_lesson_22'
    return batch_load_sql(query1), batch_load_sql(query2)


# Ваша функция для загрузки данных по частям
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# загрузка модел
model_control, model_test = load_models()
# загрузка данных, они распределены так: слева от user_id до exp_group_4 идут данные о всех пользователях
# справа от post_id до tech идут данные о всех постах
# данные об юзерах и о постах несвязаны, я бы загрузил 2 таблицы, но по заданию должна быть одна
user_df, post_df = load_features()
# post_test_part_data = data[
#     ['post_id', 'text', 'TF-IDF_max', 'TF-IDF_mean', 'covid', 'entertainment', 'movie', 'politics', 'sport', 'tech']
# ]
# post_test_part_data = post_test_part_data.dropna(subset='post_id')
# post_test_part_data = post_test_part_data.reset_index()
# user_test_part_data = data[
#     ['user_id', 'gender', 'age', 'country', 'city', 'exp_group_1', 'exp_group_2', 'exp_group_3', 'exp_group_4']
# ]
# user_test_part_data = user_test_part_data.dropna(subset='user_id')
# user_test_part_data = user_test_part_data.reset_index()
# val_of_posts = post_test_part_data.shape[0]


app = FastAPI()


def get_5_posts(users, posts, id, model):
    this_user_data = users.copy().loc[users['user_id'] == id]
    all_post_features_df = posts.copy()
    this_user_data['key'] = 1
    all_post_features_df['key'] = 1
    result = this_user_data.merge(all_post_features_df, on='key').drop(['key', 'text'], axis=1)
    result['prediction'] = model.predict_proba(result)[:, 1]
    return result.sort_values('prediction', ascending=False).head(5)


def get_exp_group(user_id: int) -> str:
    part = int(hashlib.md5((str(user_id) + 'salt').encode()).hexdigest(), 16) % 100
    if part >= 50:
        exp_group = 'test'
    else:
        exp_group = 'control'
    return exp_group


@app.get("/post/recommendations/")
def recommended_posts(id: int, limit: int = 5) -> Response:

    exp_group = get_exp_group(id)

    if exp_group == 'control':
        model = model_control
    elif exp_group == 'test':
        model = model_test
    else:
        raise ValueError('unknown group')

    # # создаем таблицу, в которой к одному пользователя присоеденияем все посты
    # # сначала создаем таблицу размером (количество постов) с одинаковыми данными (данными о пользователе)
    # user_head = pd.concat([user_test_part_data[user_test_part_data['user_id'] == id].head(1)] * val_of_posts)
    # user_head = user_head.reset_index()
    # # конкатим таблицу с пользователем к постам и получаем таблицу, которую можно предиктить
    # user_head_for_posts = pd.concat([user_head, post_test_part_data], axis=1)
    # user_head_for_posts = user_head_for_posts.drop(['level_0', 'index', 'index'], axis=1)
    top_5 = get_5_posts(user_df, post_df, id=id, model=model)

    result = list()
    for i in range(5):
        post_id = int(top_5['post_id'].iloc[i])
        # post_id = top_5[i][0]
        # так как у меня ohe для topic, я делаю условия, чтобы проверить какой topic
        if top_5['entertainment'].iloc[i]:
            topic = 'entertainment'
        elif top_5['covid'].iloc[i]:
            topic = 'covid'
        elif top_5['sport'].iloc[i]:
            topic = 'sport'
        elif top_5['tech'].iloc[i]:
            topic = 'tech'
        elif top_5['politics'].iloc[i]:
            topic = 'politics'
        elif top_5['movie'].iloc[i]:
            topic = 'movie'
        else:
            topic = 'business'

        result.append(
            {'id': post_id,
             'text': str(post_df.text[post_df['post_id'] == post_id].item()),
             'topic': topic})
    return Response(**{'exp_group': exp_group, 'recommendations': result})
