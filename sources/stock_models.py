import torch
from torch import nn
from torch.nn import LSTM, Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from sources.data_utils import StockDataset


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

        train_stock = StockDataset()
        test_stock = StockDataset()
        self.train_dloader = DataLoader(dataset=train_stock,
                                        batch_size=batch_size,
                                        shuffle=True)
        self.test_dloader = DataLoader(dataset=test_stock,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.stock_model = StockLSTM(input_size, hidden_size, num_layers, is_directional)
        self.criterion = MSELoss()
        self.optimizer = Adam(self.stock_model.paramters(), lr=learn_rate)

    def train(self):
        for ep in range(self.num_epochs):
            self.stock_model.train()
            total_loss = 0
            with tqdm(ncols=200, total=len(self.train_dloader))as p_bar:
                for i, (x_train, y_train) in self.train_dloader:
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
            print('Done evaluation. Eval loss: %.3f ' % eval_loss)
            print('**********************************************')

    def evalute(self):
        self.stock_model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (x_train, y_train) in self.test_dloader:
                model_out = self.stock_model(x_train)
                loss = self.criterion(y_train, model_out)
                total_loss += loss.item()

            eval_loss = total_loss / len(self.test_dloader)

        return eval_loss
