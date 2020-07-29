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
        # proc_cols = df_stock.columns.drop('volume')
        for col in proc_cols:
            scaler = MinMaxScaler()
            try:
                df_stock[col] = scaler.fit_transform(np.array(df_stock[col]).reshape(-1, 1))
            except ValueError:
                print()
        df_stock['volume'] = np.log1p(df_stock['volume'])

        return df_stock

    @staticmethod
    def gen_label(df_stock):
        label = (df_stock['close'] > df_stock['close'].shift(1)).apply(lambda x: 1 if x else 0)
        df_stock['is_rise'] = label
        df_stock.drop(df_stock.index[0], axis=0, inplace=True)
        return df_stock

    @staticmethod
    def gen_rollings_labels(df_stock):
        df_stock['Log_Ret_1d'] = np.log(df_stock['close'] / df_stock['close'].shift(1))
        df_stock['Return_Label'] = pd.Series(df_stock['Log_Ret_1d']).shift(-21).rolling(window=21).sum()
        df_stock['is_rise'] = np.where(df_stock['Return_Label'] > 0, 1, 0)
        df_stock = df_stock.drop(["Return_Label", "Log_Ret_1d"], axis=1)

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
    def gen_rolling_features(df_stock):
        df_stock['Log_Ret_1d'] = np.log(df_stock['close'] / df_stock['close'].shift(1))

        # Compute logarithmic returns using the pandas rolling mean function
        df_stock['Log_Ret_1w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=5).sum()
        df_stock['Log_Ret_2w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=10).sum()
        df_stock['Log_Ret_3w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=15).sum()
        df_stock['Log_Ret_4w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=20).sum()
        df_stock['Log_Ret_8w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=40).sum()
        df_stock['Log_Ret_12w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=60).sum()
        df_stock['Log_Ret_16w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=80).sum()
        df_stock['Log_Ret_20w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=100).sum()
        df_stock['Log_Ret_24w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=120).sum()
        df_stock['Log_Ret_28w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=140).sum()
        df_stock['Log_Ret_32w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=160).sum()
        df_stock['Log_Ret_36w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=180).sum()
        df_stock['Log_Ret_40w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=200).sum()
        df_stock['Log_Ret_44w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=220).sum()
        df_stock['Log_Ret_48w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=240).sum()
        df_stock['Log_Ret_52w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=260).sum()
        df_stock['Log_Ret_56w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=280).sum()
        df_stock['Log_Ret_60w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=300).sum()
        df_stock['Log_Ret_64w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=320).sum()
        df_stock['Log_Ret_68w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=340).sum()
        df_stock['Log_Ret_72w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=360).sum()
        df_stock['Log_Ret_76w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=380).sum()
        df_stock['Log_Ret_80w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=400).sum()

        # Compute Volatility using the pandas rolling standard deviation function
        df_stock['Vol_1w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=5).std() * np.sqrt(5)
        df_stock['Vol_2w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=10).std() * np.sqrt(10)
        df_stock['Vol_3w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=15).std() * np.sqrt(15)
        df_stock['Vol_4w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=20).std() * np.sqrt(20)
        df_stock['Vol_8w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=40).std() * np.sqrt(40)
        df_stock['Vol_12w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=60).std() * np.sqrt(60)
        df_stock['Vol_16w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=80).std() * np.sqrt(80)
        df_stock['Vol_20w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=100).std() * np.sqrt(100)
        df_stock['Vol_24w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=120).std() * np.sqrt(120)
        df_stock['Vol_28w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=140).std() * np.sqrt(140)
        df_stock['Vol_32w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=160).std() * np.sqrt(160)
        df_stock['Vol_36w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=180).std() * np.sqrt(180)
        df_stock['Vol_40w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=200).std() * np.sqrt(200)
        df_stock['Vol_44w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=220).std() * np.sqrt(220)
        df_stock['Vol_48w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=240).std() * np.sqrt(240)
        df_stock['Vol_52w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=260).std() * np.sqrt(260)
        df_stock['Vol_56w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=280).std() * np.sqrt(280)
        df_stock['Vol_60w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=300).std() * np.sqrt(300)
        df_stock['Vol_64w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=320).std() * np.sqrt(320)
        df_stock['Vol_68w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=340).std() * np.sqrt(340)
        df_stock['Vol_72w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=360).std() * np.sqrt(360)
        df_stock['Vol_76w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=380).std() * np.sqrt(380)
        df_stock['Vol_80w'] = pd.Series(df_stock['Log_Ret_1d']).rolling(window=400).std() * np.sqrt(400)

        # Compute Volumes using the pandas rolling mean function
        df_stock['Volume_1w'] = pd.Series(df_stock['volume']).rolling(window=5).mean()
        df_stock['Volume_2w'] = pd.Series(df_stock['volume']).rolling(window=10).mean()
        df_stock['Volume_3w'] = pd.Series(df_stock['volume']).rolling(window=15).mean()
        df_stock['Volume_4w'] = pd.Series(df_stock['volume']).rolling(window=20).mean()
        df_stock['Volume_8w'] = pd.Series(df_stock['volume']).rolling(window=40).mean()
        df_stock['Volume_12w'] = pd.Series(df_stock['volume']).rolling(window=60).mean()
        df_stock['Volume_16w'] = pd.Series(df_stock['volume']).rolling(window=80).mean()
        df_stock['Volume_20w'] = pd.Series(df_stock['volume']).rolling(window=100).mean()
        df_stock['Volume_24w'] = pd.Series(df_stock['volume']).rolling(window=120).mean()
        df_stock['Volume_28w'] = pd.Series(df_stock['volume']).rolling(window=140).mean()
        df_stock['Volume_32w'] = pd.Series(df_stock['volume']).rolling(window=160).mean()
        df_stock['Volume_36w'] = pd.Series(df_stock['volume']).rolling(window=180).mean()
        df_stock['Volume_40w'] = pd.Series(df_stock['volume']).rolling(window=200).mean()
        df_stock['Volume_44w'] = pd.Series(df_stock['volume']).rolling(window=220).mean()
        df_stock['Volume_48w'] = pd.Series(df_stock['volume']).rolling(window=240).mean()
        df_stock['Volume_52w'] = pd.Series(df_stock['volume']).rolling(window=260).mean()
        df_stock['Volume_56w'] = pd.Series(df_stock['volume']).rolling(window=280).mean()
        df_stock['Volume_60w'] = pd.Series(df_stock['volume']).rolling(window=300).mean()
        df_stock['Volume_64w'] = pd.Series(df_stock['volume']).rolling(window=320).mean()
        df_stock['Volume_68w'] = pd.Series(df_stock['volume']).rolling(window=340).mean()
        df_stock['Volume_72w'] = pd.Series(df_stock['volume']).rolling(window=360).mean()
        df_stock['Volume_76w'] = pd.Series(df_stock['volume']).rolling(window=380).mean()
        df_stock['Volume_80w'] = pd.Series(df_stock['volume']).rolling(window=400).mean()

        df_stock = df_stock.drop(['open', 'high', 'low', 'close'], axis=1)
        df_stock = df_stock.dropna()

        return df_stock

    @staticmethod
    def normalize_df(df_stock):
        """fill na for input and do transformation """
        if df_stock.columns.intersection(['ASIN', 'ACOS']).any():
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
        self.file_process(DF_STOCK, topk=None)

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

        train_feature, train_label = [], []
        test_feature, test_label = {}, {}

        for ticker, sframe in tqdm(df_stock.groupby('ticker')):
            sframe = sframe.sort_values('datetime', ascending=True)
            if len(sframe) > 10:
                sframe = sframe.reset_index()
                sframe = Helpers.gen_label(sframe)
                # sframe = Helpers.gen_rollings_labels(sframe)
                sframe = Helpers.gen_features(sframe)
                # sframe = Helpers.gen_rolling_features(sframe)
                sframe = Helpers.normalize_df(sframe)
                features = []
                labels = []

                for i in range(WINDOW_SIZE, len(sframe) - 1):
                    feature = sframe.iloc[i - WINDOW_SIZE:i, :]
                    x_feature = feature.drop(labels=['datetime', 'ticker', 'is_rise'], axis=1).to_numpy()
                    y_label = sframe['is_rise'].iloc[i].astype(int)
                    features.append(x_feature)
                    labels.append(y_label)

                mark = math.floor(len(features) * 0.95)
                train_feature.extend(features[:mark])
                train_label.extend(labels[:mark])

                if ticker in PREDICTED_TICKERS:
                    test_feature[ticker] = features[mark:]
                    test_label[ticker] = labels[mark:]

        with h5py.File(STOCK_H5, 'w') as h5_file:
            train_group = h5_file.create_group('train')
            train_group.create_dataset('x_feature', data=np.array(train_feature))
            train_group.create_dataset('y_feature', data=np.array(train_label))

            test_group = h5_file.create_group('test')
            for key in PREDICTED_TICKERS:
                test_group.create_dataset('%s_x_feature' % key, data=np.array(test_feature[key]))
                test_group.create_dataset('%s_y_feature' % key, data=np.array(test_label[key]))


class StockDataset(Dataset):
    def __init__(self, input_type, ticker=None):
        assert input_type in ['train', 'test']
        self.stock_h5 = h5py.File(STOCK_H5, 'r')
        group_feature = self.stock_h5.get(input_type)
        if input_type == 'train':
            self.X_feature = group_feature.get('x_feature')
            self.y_feature = group_feature.get('y_feature')
        elif input_type == 'test' and ticker:
            self.X_feature = group_feature.get('%s_x_feature' % ticker)
            self.y_feature = group_feature.get('%s_y_feature' % ticker)

        self.data_size = self.X_feature.shape[0]

    def __getitem__(self, index):
        return self.X_feature[index], self.y_feature[index]

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    utils = DataUtils()
