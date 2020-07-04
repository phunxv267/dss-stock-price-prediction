""" load stock data, preprocessing """
from os.path import join

import pandas as pd
from torch.utils.data import Dataset

from sources.configs import *


def generate_features(df_stock, time_period):
    # rename columns
    df_stock = df_stock.rename({'Open': 'open', 'High': 'high', 'Low': 'low',
                                'Close': 'close', 'Volume': 'volume', 'OpenInt': 'open_int'})
    """Overlap Studies Functions"""

    # BBANDS - Bollinger Bands
    df_stock = pd.read_csv('aabs.csv', header=0)
    df_stock = abtract.BBANDS(df_stock, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df_stock.head()

    # DEMA - Double Exponential Moving Average
    df_stock = abtract.DEMA(df_stock, timeperiod=30)

    # EMA - Exponential Moving Average
    df_stock = abtract.EMA(df_stock, timeperiod=30)

    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    df_stock = abtract.HT_TRENDLINE(df_stock)

    # KAMA - Kaufman Adaptive Moving Average
    df_stock = abtract.KAMA(df_stock, timeperiod=30)


def load_stock_file(stock_file):
    """Loading stock price file """
    stock_code = stock_file.replace('.txt', '')
    df_stock = pd.read_csv(join(DATA_PATH, stock_file), delimiter=',', header=0)
    df_stock['Label'] = stock_code
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values('Date', ascending=True)
    df_stock = df_stock.drop('Date', axis=1)
    # TODO infer more features, feature scaling
    # input [20, 6] output [1]
    X_train, y_train = [], []
    for i in range(WINDOW_SIZE, len(df_stock)):
        row = df_stock.loc[(i - WINDOW_SIZE):(i - 1), :]
        X_train.append(row)
        y_train.append(df_stock.loc[i, 'Close'])

    return X_train, y_train


class StockDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.data_size = len(X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.data_size


df_stock = pd.read_csv(join(DATA_PATH, 'aaba.us.txt'), delimiter=',', header=0)
df_stock.drop(['Date'], axis=1).iloc[:100, :].to_csv('aabs.csv', index=False,
                                                     header=['open', 'high', 'low', 'close', 'volume', 'open_int'])
