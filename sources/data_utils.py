""" load stock data, preprocessing """
import h5py
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
        df_stock['is_rise'] = 0
        label_idx = df_stock.columns.get_loc('is_rise')
        close_idx = df_stock.columns.get_loc('close')
        for i in range(1, len(df_stock) - 2):
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

    @staticmethod
    def convert_feature(feature_tsr, dtype, device='cuda:0'):
        feature_tsr = feature_tsr.type(dtype)
        return feature_tsr.to(device)


class DataUtils:
    def __init__(self):
        """
            stock file format:
            csv file with 5 columns ['datetime', 'ticker', 'open', 'close', 'high', 'low', volume]
            after preprocess: stock file with ~180 features
        """
        self.file_process(DF_STOCK, topk=10)

    def file_process(self, file_in, topk=None):
        """
        get top k most longest stock
        :param file_in:
        :param topk:
        :return:
        """
        print('Pre-processing file ...')
        df_stock = pd.read_csv(file_in)
        stock_count = df_stock.groupby('ticker').count().sort_values('volume', ascending=False)
        if topk:
            stock_labels = list(stock_count.index[:topk]) + PREDICTED_TICKERS
            stock_labels = set(stock_labels)
        else:
            stock_labels = stock_count.index
        df_stock = df_stock[df_stock['ticker'].isin(stock_labels)]
        print('dataset length %s ' % len(df_stock))
        df_stock['datetime'] = pd.to_datetime(df_stock['datetime'])

        train_feature, train_label, test_feature, test_label = [], [], [], []

        for ticker, sframe in tqdm(df_stock.groupby('ticker')):
            sframe = sframe.sort_values('datetime', ascending=True)
            sframe = Helpers.gen_label(sframe)
            sframe = Helpers.gen_features(sframe)
            features = []
            labels = []
            for i in range(WINDOW_SIZE, len(sframe) - 2):
                feature = sframe.iloc[i - WINDOW_SIZE:i, :]
                x_feature = feature.drop(labels=['datetime', 'ticker', 'is_rise'], axis=1).to_numpy()
                y_label = sframe['is_rise'].iloc[i + 1].astype(int)
                features.append(x_feature)
                labels.append(y_label)
            mark = math.floor(len(features) * 0.95)
            train_feature.extend(features[:mark])
            test_feature.extend(features[mark:])
            train_label.extend(labels[:mark])
            test_label.extend(labels[mark:])

        with h5py.File(STOCK_H5, 'w') as h5_file:
            train_group = h5_file.create_group('train')
            train_group.create_dataset('x_feature', data=np.array(train_feature))
            train_group.create_dataset('y_feature', data=np.array(train_label))

            test_group = h5_file.create_group('test')
            test_group.create_dataset('x_feature', data=np.array(test_feature))
            test_group.create_dataset('y_feature', data=np.array(test_label))


class StockDataset(Dataset):
    def __init__(self, input_type='train'):
        assert input_type in ['train', 'test']
        self.stock_h5 = h5py.File(STOCK_H5, 'r')
        group_feature = self.stock_h5.get(input_type)
        self.X_feature = group_feature.get('x_feature')
        self.y_feature = group_feature.get('y_feature')

        self.data_size = self.X_feature.shape[0]

    def __getitem__(self, index):
        return self.X_feature[index], self.y_feature[index]

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    utils = DataUtils()
