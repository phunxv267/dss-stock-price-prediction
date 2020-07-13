import torch
from torch import nn
from torch.nn import LSTM, Linear, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sources.data_utils import StockDataset, DataUtils
from sources.configs import *


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, is_directional):
        super(StockLSTM, self).__init__()
        self.lstm_model = LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=is_directional)

        self.out_layer = Linear(hidden_size * 1, 1)  # regression

    def forward(self, in_feature):
        out, (h_state, c_state) = self.lstm_model(in_feature, None)
        output = self.out_layer(out[:, -1, :])

        return output


class Trainer:
    def __init__(self, input_size, hidden_size, num_layers,
                 is_directional, batch_size, learn_rate, num_epochs):
        self.num_epochs = num_epochs
        X_train, y_train, X_test, y_test = self.load_stock_data()
        train_stock = StockDataset(X_train, y_train)
        test_stock = StockDataset(X_test, y_test)

        self.g_truth = y_test.cpu().numpy()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_dloader = DataLoader(dataset=train_stock,
                                        batch_size=batch_size,
                                        shuffle=True)
        self.test_dloader = DataLoader(dataset=test_stock,
                                       batch_size=batch_size,
                                       shuffle=False)
        self.stock_model = StockLSTM(input_size, hidden_size, num_layers, is_directional)
        self.stock_model = self.stock_model.to(self.device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.stock_model.paramters(), lr=learn_rate)

    def train(self):
        for ep in range(self.num_epochs):
            self.stock_model.train()
            total_loss = 0
            with tqdm(ncols=200, total=len(self.train_dloader))as p_bar:
                for i, (x_train, y_train) in self.train_dloader:
                    x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                    self.optimizer.zero_grad()
                    model_out = self.stock_model(x_train)
                    loss = self.criterion(y_train, model_out)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()

                    p_bar.set_description(
                        'Train step %s/%s - epoch %s - loss %.3f' % (i + 1, len(self.train_dloader), ep, loss.item()))

            print('*** End epoch. Evaluating ... ***')
            eval_loss = self.evalute()
            print('**********************************************')

    def evalute(self):
        self.stock_model.eval()
        with torch.no_grad():
            total_loss = 0
            all_pred = []
            for i, (x_train, y_train) in self.test_dloader:
                model_out = self.stock_model(x_train)
                pred = model_out.topk(1)[1].cpu().numpy().to_list()
                all_pred.extend(pred)

            accuracy = accuracy_score(self.g_truth, np.array(all_pred))
            print('Accuracy %.3f ' % accuracy)
            print(classification_report(self.g_truth, np.array(all_pred)))

    def load_stock_data(self):
        X_train, y_train = DataUtils.load_feature(STOCK_TRAIN)
        X_test, y_test = DataUtils.load_feature(STOCK_TRAIN)

        return X_train, y_train, X_test, y_test
