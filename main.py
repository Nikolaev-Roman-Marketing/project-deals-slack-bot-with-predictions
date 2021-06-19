from datetime import datetime, timedelta

import pandas as pd
from fast_bitrix24 import *

from sqlalchemy import create_engine
from sqlalchemy.sql import text

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from slack_loader import SlackLoader
from transformer import Transformer
from models import AllInSARIMA, AllInLSTM


def create_delivery_dates_list(**kwargs):
    """
    Находим даты ближайшей Субботы, Воскресенья, Понедельника
    На эти даты в CRM создаются сделки текущей недели
    """

    ti = kwargs['ti']

    current_weekday = datetime.now().weekday()
    current_date = datetime.now()

    delivery_weekdays = [5, 6, 7]
    delivery_dates = []

    for weekday in delivery_weekdays:
        delta = weekday - current_weekday
        delivery_date = (current_date + timedelta(days=delta)).strftime('%d.%m.%y')
        delivery_dates.append(delivery_date)

    ti.xcom_push(key='dates_list', value=delivery_dates)


def extract_data(**kwargs):
    """
    Получаем сделки и товары этих сделок из CRM, используя API
    """

    ti = kwargs['ti']

    dates_list = ti.xcom_pull(key='dates_list', task_ids=['create_delivery_dates_list'])[0]

    bitrix_token = Variable.get("bitrix_token")

    webhook = f"https://uzhin-doma.bitrix24.ru/rest/128/{bitrix_token}/"
    bitrix = Bitrix(webhook)

    deals = bitrix.get_all('crm.deal.list',
                           params={
                               'select': ['id'],
                               'filter': {'CLOSED': 'N', 'TITLE': dates_list}
                           })

    products = bitrix.get_by_ID('crm.deal.productrows.get', [d['ID'] for d in deals])

    ti.xcom_push(key='data_json', value=products)


def transform_data(**kwargs):
    """
    Трансформируем данные
    Считаем количество сделок, промо-наборов
    Составляем табличку проданных товаров
    """

    ti = kwargs['ti']

    raw_data = ti.xcom_pull(key='data_json', task_ids=['extract_data'])[0]

    transformer = Transformer(raw_data)

    deals_amount, promo_amount, dishes_table = transformer.run()

    ti.xcom_push(key='deals_amount', value=deals_amount)
    ti.xcom_push(key='promo_amount', value=promo_amount)
    ti.xcom_push(key='dishes_table', value=dishes_table)


def load_data_to_mysql(**kwargs):
    """
    Сохраняем кол-во сделок на основном сайте в MySQL
    Для Субботы, Воскресенья и Понедельника сохраняем только вечерние данные
    """

    current_weekday = int(datetime.now().weekday())
    current_hour = datetime.now().hour + 3

    if (current_weekday in [0, 6, 7]) and (current_hour < 21):

        print("we don't need data for this time")

    else:

        ti = kwargs['ti']

        deals_amount = ti.xcom_pull(key='deals_amount', task_ids=['transform_data'])[0]
        promo_amount = ti.xcom_pull(key='promo_amount', task_ids=['transform_data'])[0]

        main_site_deals = deals_amount - promo_amount

        new_data = pd.DataFrame([{'input_date': datetime.now() + timedelta(hours=3), 'deals': main_site_deals}])

        local_mysql_login_pass = Variable.get("local_mysql_login_pass")

        conn = create_engine(f'mysql+pymysql://{local_mysql_login_pass}@host.docker.internal/projects', echo=False)

        query = text("""
                CREATE TABLE IF NOT EXISTS deals_stats(
                    input_date DATETIME, 
                    deals INT UNSIGNED
                )""")

        conn.execute(query)

        # new_data.to_sql(name='deals_stats', con=conn, if_exists='append', index=False)


def define_steps_to_end_of_week(**kwargs):
    """
    Считаем шаги до конца недели
    То есть насколько шагов в будущее мы хотим сделать прогноз
    (Шагов именно до конца текущей недели)
    """

    ti = kwargs['ti']

    current_weekday = datetime.now().weekday()
    last_weekday = 6
    days_left = last_weekday - current_weekday
    steps = days_left * 3

    ti.xcom_push(key='steps_to_end_of_week', value=steps)


def load_train_data(**kwargs):
    """
    Загружаем из БД данные для обучения :
    последние 90 наблюдений, отсортированные по дате добавления
    """

    ti = kwargs['ti']

    local_mysql_login_pass = Variable.get("local_mysql_login_pass")

    conn = create_engine(f'mysql+pymysql://{local_mysql_login_pass}@host.docker.internal/projects', echo=False)

    query = """
        SELECT * FROM(
            SELECT deals, input_date
            FROM deals_stats
            ORDER BY input_date DESC
            LIMIT 90) q1
        ORDER BY input_date
    """

    train_data = pd.read_sql(query, con=conn)[['deals']]

    ti.xcom_push(key='train_data', value=train_data)


def load_min_max_values(**kwargs):
    """
    Загружаем из БД min и max значения, их мы сохранили, когда обучали LSTM
    Они нужные, чтобы нормализовать данных и привести их к дапозону [-1;1]
    """

    ti = kwargs['ti']

    local_mysql_login_pass = Variable.get("local_mysql_login_pass")

    conn = create_engine(f'mysql+pymysql://{local_mysql_login_pass}@host.docker.internal/projects', echo=False)

    query = """
        SELECT * FROM deals_min_max
    """

    data = pd.read_sql(query, con=conn)
    min_value = data['min_value'][0]
    max_value = data['max_value'][0]

    ti.xcom_push(key='min_max_values', value=[min_value, max_value])


def send_data_to_slack(**kwargs):
    """
    Используя API, отправляем в нужный Slack чат всю статистику
    И прогнозы моделей по сделкам
    """

    ti = kwargs['ti']

    deals_amount = ti.xcom_pull(key='deals_amount', task_ids=['transform_data'])[0]
    promo_amount = ti.xcom_pull(key='promo_amount', task_ids=['transform_data'])[0]
    dishes_table = ti.xcom_pull(key='dishes_table', task_ids=['transform_data'])[0]
    sarima_prediction = ti.xcom_pull(key='sarima_prediction', task_ids=['sarima_fit_and_predict'])[0]
    lstm_prediction = ti.xcom_pull(key='lstm_prediction', task_ids=['lstm_fit_and_predict'])[0]

    loader = SlackLoader()

    weekdays = {0: "Понедельник", 1: "Вторник", 2: "Среда",
                3: "Четверг", 4: "Пятница", 5: "Суббота",
                6: "Воскресенье"}
    current_time = str((datetime.now() + timedelta(hours=3)).strftime('%H:%M'))
    current_weekday_string = weekdays[datetime.now().weekday()]

    loader.send_message(f"{current_weekday_string} {current_time}")

    loader.send_statistic(deals_amount, promo_amount)

    loader.send_predictions(sarima_prediction, lstm_prediction)

    loader.send_products_table_file(dishes_table)


def bridge(**kwargs):
    """
    Пустой таск, чтобы Граф выглядил красиво
    """
    pass


args = {
    'owner': 'roma',
    'start_date': days_ago(2),
    'task_concurency': 1,
    'provide_context':  True
}


with DAG('Deals_Slack_Bot_With_Predictions', description='Bot_predict_deals',
         schedule_interval='45 4,13,18 * * *', catchup=False,
         default_args=args) as dag:

    create_delivery_dates_list = PythonOperator(task_id='create_delivery_dates_list',
                                                python_callable=create_delivery_dates_list)

    extract_data = PythonOperator(task_id='extract_data',
                                  python_callable=extract_data)

    transform_data = PythonOperator(task_id='transform_data',
                                    python_callable=transform_data)

    load_data_to_mysql = PythonOperator(task_id='load_data_to_mysql',
                                        python_callable=load_data_to_mysql)

    define_steps_to_end_of_week = PythonOperator(task_id='define_steps_to_end_of_week',
                                                 python_callable=define_steps_to_end_of_week)

    load_train_data = PythonOperator(task_id='load_train_data',
                                     python_callable=load_train_data)

    load_min_max_values = PythonOperator(task_id='load_min_max_values',
                                         python_callable=load_min_max_values)

    bridge = PythonOperator(task_id='bridge',
                            python_callable=bridge)

    send_data_to_slack = PythonOperator(task_id='send_data_to_slack',
                                        python_callable=send_data_to_slack)

    sarima = AllInSARIMA()
    lstm = AllInLSTM()

    models = [
        {'model_object': sarima, 'name': 'sarima'},
        {'model_object': lstm, 'name': 'lstm'}
    ]

    create_delivery_dates_list >> extract_data >> transform_data >> load_data_to_mysql >> \
    define_steps_to_end_of_week >> [load_train_data, load_min_max_values] >> bridge

    for model in models:

        name = model['name']
        model_object = model['model_object']

        prepare_data = PythonOperator(task_id=f'{name}_prepare_data', python_callable=model_object.prepare_data)

        fit_and_predict = PythonOperator(task_id=f'{name}_fit_and_predict', python_callable=model_object.fit_and_predict)

        bridge >> prepare_data >> fit_and_predict >> send_data_to_slack





