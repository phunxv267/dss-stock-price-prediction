from os.path import join

import pandas as pd

WINDOW_SIZE = 15
DATA_PATH = '../resources/'
STOCK_TEST = join(DATA_PATH, 'stock_test.csv')
STOCK_TRAIN = join(DATA_PATH, 'stock_train.csv')
TRAIN_TOY = join(DATA_PATH, 'train_toy.csv')
TEST_TOY = join(DATA_PATH, 'test_toy.csv')
DF_STOCK = join(DATA_PATH, 'b3_stocks_1994_2020.csv')

STOCK_MODEL = join(DATA_PATH, 'stock_model.pt')

STOCK_H5 = join(DATA_PATH, 'stock.h5')

SPAN = 5
SMOOTH_ALPHA = 0.3

PRICE_COLS = ['open', 'close', 'high', 'low']

NON_PRICE_COLS = ['volume', '']

PREDICTED_TICKERS = ['BOVA11', 'BBDC4', 'CIEL3', 'ITUB4', 'PETR4']

func_groups = ['Cycle Indicators', 'Math Operators', 'Math Transform', 'Momentum Indicators', 'Overlap Studies',
               'Pattern Recognition', 'Price Transform', 'Statistic Functions', 'Volatility Indicators',
               'Volume Indicators']
