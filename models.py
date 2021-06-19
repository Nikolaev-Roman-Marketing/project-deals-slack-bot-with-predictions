import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

import statsmodels.api as sm
import torch, torch.nn as nn

from scipy import stats
import scipy.stats as scs


class AllInSARIMA:

    def __init__(self):

        self.lambda_ = 0.5
        self.sarima_order = (0, 1, 0)
        self.sarima_seasonal_order = (1, 1, 1, 15)

    def prepare_data(self, **kwargs):
        """
        Проводим преобразование Бокса-Кокса над обучающими данными,
        результат передем дальше
        """

        ti = kwargs['ti']
        train_data = ti.xcom_pull(key='train_data', task_ids=['load_train_data'])[0]
        train_data['deals'] = scs.boxcox(train_data['deals'], lmbda=self.lambda_)

        ti.xcom_push(key='prepared_train_data', value=train_data)

    def fit_and_predict(self, **kwargs):
        """
        Обучаем модель на преобразованных обучающих данных
        Делаем прогноз на конец недели,
        над предсказанными значениями проводим преобразование обратное преобразованию Бокса-Кокса,
        чтобы вернуть данные к исходному масштабу/размерности
        """

        ti = kwargs['ti']

        train_data = ti.xcom_pull(key='prepared_train_data', task_ids=['sarima_prepare_data'])[0]
        steps_to_end_of_week = ti.xcom_pull(key='steps_to_end_of_week', task_ids=['define_steps_to_end_of_week'])[0]

        N = len(train_data)

        sarima_model = sm.tsa.statespace.SARIMAX(train_data,
                                                 order=self.sarima_order,
                                                 seasonal_order=self.sarima_seasonal_order).fit()

        prediction = sarima_model.predict(start=N, end=N+int(steps_to_end_of_week))

        # проводим преобразование обратное преобразованию Бокса-Кокса
        prediction = (np.exp(np.log(self.lambda_ * prediction + 1) / self.lambda_)).max()

        ti.xcom_push(key='sarima_prediction', value=prediction)


class AllInLSTM:

    def __init__(self):

        self.path_to_trained_model = '/usr/local/airflow/dags/Deals_Slack_Bot_With_Predictions/lstm_model_95'
        self.epochs = 15
        self.lr = 0.0001
        self.week_window = 15

    def prepare_data(self, **kwargs):
        """
        Берем сохраненные min и max значения,
        Нормализуем данные и приводим их к диапозону [-1;1]
        """

        ti = kwargs['ti']
        train_data = ti.xcom_pull(key='train_data', task_ids=['load_train_data'])[0]
        min_max_values = ti.xcom_pull(key='min_max_values', task_ids=['load_min_max_values'])[0]
        min_value, max_value = min_max_values[0], min_max_values[1]

        train_data['deals'] = (2 * (train_data['deals'] - min_value) / (max_value - min_value)) - 1

        ti.xcom_push(key='prepared_train_data', value=train_data)

    def fit_and_predict(self, **kwargs):
        """
        Формируем обучающую выборку
        Уже обученную LSTM немного дообучаем на самых последних данных, проходя self.epochs эпох
        Делаем прогноз на конец недели
        """

        ti = kwargs['ti']
        train_data = ti.xcom_pull(key='prepared_train_data', task_ids=['lstm_prepare_data'])[0]
        steps_to_end_of_week = ti.xcom_pull(key='steps_to_end_of_week', task_ids=['define_steps_to_end_of_week'])[0]

        min_max_values = ti.xcom_pull(key='min_max_values', task_ids=['load_min_max_values'])[0]
        min_value, max_value = min_max_values[0], min_max_values[1]

        X, Y = sliding_windows(train_data['deals'], self.week_window)
        X = torch.FloatTensor(X).reshape(X.shape[0], X.shape[1], 1)
        Y = torch.FloatTensor(Y).reshape(Y.shape[0], 1)

        model = LSTM()
        model.load_state_dict(torch.load(self.path_to_trained_model))

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for i in range(self.epochs):

            model.train()

            optimizer.zero_grad()

            Y_predict = model.forward(X)

            loss = loss_func(Y_predict, Y)
            loss.backward()

            optimizer.step()

        model.eval()

        predictions = self.predict_step_by_step(train_data, model, steps_to_end_of_week)

        # делаем обратную нормализацию
        # прогноз из диапозона [-1;1] переводим в истинный масштаб

        predictions = ((predictions+1)*(max_value - min_value)/2) + min_value

        ti.xcom_push(key='lstm_prediction', value=predictions.max())

    def predict_step_by_step(self, train_data, model, steps):
        """
        Делаем прогноз на steps шагов в будущее
        Чтобы получить прогноз на конец недели
        """

        last_sequence = torch.FloatTensor(
            train_data['deals'].loc[
            len(train_data)-self.week_window-1:
            len(train_data)-2].values).\
            reshape(1, self.week_window, 1)

        predictions = []
        for _ in range(steps):
            new_y = model.forward(last_sequence)
            predictions.append(new_y[0][0].detach().numpy())
            last_sequence = torch.cat([last_sequence[0][1:15], new_y]).reshape(1, self.week_window, 1)

        predictions = np.array(predictions)

        return predictions


def sliding_windows(data, seq_length):
    """
    Делим все данные на группы, где :
    _x - группа из наблюдений длинной в seq_length
    _y - одно наблюдение, идущие сразу после группы наблюдений _x
    LSTM будет из последовательности _x восстанавливать значение _y
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


class LSTM(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=400, num_layers=2, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, 1)

        self.act = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc1(out[:, -1, :])
        out = self.act(out)
        out = self.fc2(out)

        return out


