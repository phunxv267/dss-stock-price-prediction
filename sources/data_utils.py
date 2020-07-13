""" load stock data, preprocessing """
import math
import os
from os.path import join, isfile
import talib
import torch
from talib import abstract
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from sources.configs import *


class Helpers:
    @staticmethod
    def price_transform(column):
        # do the log-return transformation, and exponential smoothing
        column = column.ptc_change()
        column = column.ewm(span=SPAN, alpha=SMOOTH_ALPHA)

        return column

    @staticmethod
    def non_price_transform(column):
        scaler = MinMaxScaler()
        column = scaler.fit_transform(column)

        return column

    @staticmethod
    def transform_features(df_stock):
        proc_cols = df_stock.columns.drop(['datetime', 'ticker', 'volume'])
        for col in proc_cols:
            scaler = MinMaxScaler()
            df_stock[col] = scaler.fit_transform(np.array(df_stock[col]).reshape(-1, 1))
        df_stock['volume'] = np.log1p(df_stock['volume'])

        return df_stock

    @staticmethod
    def gen_label(df_stock):
        df_stock = df_stock.sort_values('datetime', ascending=True)
        df_stock['is_rise'] = 0
        label_idx = df_stock.columns.get_loc('is_rise')
        close_idx = df_stock.columns.get_loc('close')
        for i in range(1, len(df_stock) - 1):
            df_stock.iloc[i, label_idx] = 1 if df_stock.iloc[i, close_idx] > df_stock.iloc[i - 1, close_idx] else 0
        df_stock.drop(df_stock.index[0], axis=0, inplace=True)
        return df_stock

    @staticmethod
    def gen_features(df_stock):
        args_list = ['open', 'high', 'low', 'close', 'volume']
        for func_group in func_groups:
            for func_name in talib.get_function_groups()[func_group]:
                func = abstract.Function(func_name)
                if func_name == 'MAVP':
                    continue
                else:
                    results = func(df_stock[args_list])
                if isinstance(results, pd.Series):
                    df_stock[func_name] = results
                else:
                    results.columns = [func_name + '_' + w for w in results.columns]
                    df_stock = pd.concat((df_stock, results), axis=1)
        df_stock = Helpers.normalize_df(df_stock)

        return df_stock

    @staticmethod
    def normalize_df(df_stock):
        """fill na for input and do transformation """
        df_stock = df_stock.drop(['ASIN', 'ACOS'], axis=1)
        df_stock = df_stock.fillna(0, axis=0)
        df_stock = df_stock.replace([float('inf'), -float('inf')], 0)
        df_stock = Helpers.transform_features(df_stock)
        return df_stock


class DataUtils:
    def __init__(self):
        """
            stock file format:
            csv file with 5 columns ['datetime', 'ticker', 'open', 'close', 'high', 'low', volume]
            after preprocess: stock file with ~180 features
        """
        if not isfile(STOCK_TRAIN):
            self.file_process(DF_STOCK)

    def file_process(self, file_in):
        print('Pre-processing file ...')
        df_stock = pd.read_csv(file_in)

        for ticker, sframe in tqdm(df_stock.groupby('ticker')):
            if len(sframe) > 250:
                sframe = Helpers.gen_label(sframe)
                sframe = Helpers.gen_features(sframe)
                sframe.to_csv(join('../../resources', ticker + '.csv'), index=False)

    @staticmethod
    def split_stock_file(stock_path):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for _file in os.listdir(stock_path):
            frame = pd.read_csv(os.path.join(stock_path, _file), header=0)
            mark = math.floor(len(frame) * 0.9)
            df_train = pd.concat((df_train, frame.iloc[:mark, :]))
            df_test = pd.concat((df_test, frame.iloc[mark:, :]))
        df_train.to_csv(STOCK_TRAIN, index=False)
        df_test.to_csv(STOCK_TEST, index=False)

    @staticmethod
    def load_feature(stock_file):
        df_stock = pd.read_csv(stock_file, header=0)
        X_feature = df_stock.drop(labels=['datetime', 'ticker', 'is_rise'], axis=1)
        y_feature = df_stock['is_rise']

        return torch.from_numpy(X_feature), torch.from_numpy(y_feature)


utils = DataUtils()
DataUtils.split_stock_file('../../resources')


class StockDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.data_size = len(X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.data_size
