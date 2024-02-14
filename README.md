# KS-ML-start-final-project

## Техническое задание
Представим, что мы построили социальную сеть, которая обладает следующим функционалом: можно отправлять друг другу письма, создавать сообщества, аналогичные группам в известных сетях, и в этих сообществах публиковать посты.

В текущем проекте построил рекомендательную систему постов в социальной сети. В качестве базовых сырых данных я использовал подготовленные заранее таблицы.

С точки зрения разработки я реализовал сервис, который будет для каждого юзера в любой момент времени возвращать посты, которые пользователю покажут в его ленте соцсети.

## Описание данных
### Таблица user_data
| Field name	| Overview |
| - | - |
|age	| Возраст пользователя (в профиле) |
|city	| Город пользователя (в профиле) |
|country |	Страна пользователя (в профиле) |
|exp_group |	Экспериментальная группа: некоторая зашифрованная категория |
|gender	| Пол пользователя |
|user_id |	Уникальный идентификатор пользователя | 
|os	| Операционная система устройства, с которого происходит пользование соц.сетью |
|source |	Пришел ли пользователь в приложение с органического трафика или с рекламы |

### Таблица post_text_df
| Field name	| Overview |
| - | - |
|id	| Уникальный идентификатор поста |
|text	| Текстовое содержание поста |
|topic |	Основная тематика |


### Таблица feed_data
| Field name	| Overview |
| - | - |
|timestamp |	Время, когда был произведен просмотр |
|user_id |	id пользователя, который совершил просмотр |
|post_id |	id просмотренного поста |
|action	| Тип действия: просмотр или лайк |
|target |	1 у просмотров, если почти сразу после просмотра был совершен лайк, иначе 0. У действий like пропущенное значение |

## Стэк
fastapi, uvicorn, pandas, sqlalchemy, scikit_learn, requests, catboost, numpy, pydantic, psycopg2-binary

## Пайплайн
1. Загрузка данных из БД в Jupyter Hub, обзор данных.
2. Создание признаков и обучающей выборки.
3. Тренировка модели на Jupyter Hub.
4. Сохранение модели и таблиц для рекоментательного сервиса.
5. Написание сервиса: загрузка модели -> получение признаков для модели по user_id -> предсказание постов, которые лайкнут -> возвращение ответа.
6. Загрузка другой модели, и проведение A/B теста между разными моделями
   
