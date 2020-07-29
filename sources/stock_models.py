import torch
from torch import nn
from torch.nn import LSTM, Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sources.data_utils import StockDataset, Helpers
from sources.configs import *


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, is_directional):
        super(StockLSTM, self).__init__()
        self.lstm_model = LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=is_directional)

        self.out_layer = Linear(hidden_size * 1, 2)  # doing classification
        self.drop_out = nn.Dropout(0.3)

    def forward(self, in_feature):
        out, (h_state, c_state) = self.lstm_model(in_feature, None)
        output = self.out_layer(self.drop_out(out[:, -1, :]))

        return output


class Trainer:
    def __init__(self, input_size, hidden_size, num_layers,
                 is_directional, batch_size, learn_rate, num_epochs):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        train_stock = StockDataset('train')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Done loading dataset')
        print(len(train_stock))
        self.train_dloader = DataLoader(dataset=train_stock,
                                        batch_size=batch_size,
                                        shuffle=True)
        self.stock_model = StockLSTM(input_size, hidden_size, num_layers, is_directional)
        self.stock_model = self.stock_model.double().to(self.device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.stock_model.parameters(), lr=learn_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)

    def train(self):
        for ep in range(self.num_epochs):
            self.stock_model.train()
            total_loss = 0
            step_mark = 0
            highest_score = 0.0
            with tqdm(total=len(self.train_dloader), ncols=100) as pbar:
                for i, (x_train, y_train) in enumerate(self.train_dloader):
                    x_train, y_train = Helpers.convert_feature(x_train, torch.DoubleTensor), \
                                       Helpers.convert_feature(y_train, torch.LongTensor)
                    self.optimizer.zero_grad()
                    model_out = self.stock_model(x_train)
                    loss = self.criterion(model_out, y_train)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()

                    pbar.set_description('Epoch %s - Iteration %s - loss %.3f ' % (ep + 1, i + 1, loss.item()))
                    pbar.update(1)

            print('*** End epoch. Evaluating ... ***')

            accuracy = self.evaluate()
            if accuracy > highest_score:
                highest_score = accuracy
            else:
                step_mark += 1
            if step_mark == 3:
                self.scheduler.step()
            print('Epoch accuracy: %.3f - Best accuracy %.3f ' % (accuracy, highest_score))
            print('**********************************************')

    def evaluate(self):
        self.stock_model.eval()
        ticker_acc = []
        with torch.no_grad():
            for ticker in PREDICTED_TICKERS:
                test_stock = StockDataset('test', ticker)
                test_dloader = DataLoader(dataset=test_stock,
                                          batch_size=self.batch_size,
                                          shuffle=False)
                all_pred = []
                all_label = []
                for i, (x_feature, y_label) in tqdm(enumerate(test_dloader)):
                    x_feature, y_label = Helpers.convert_feature(x_feature, torch.DoubleTensor), \
                                         Helpers.convert_feature(y_label, torch.LongTensor)
                    model_out = self.stock_model(x_feature)
                    pred = model_out.topk(1)[1].cpu().numpy().tolist()
                    all_pred.extend(pred)
                    all_label.extend(y_label.cpu().numpy().tolist())

                accuracy = accuracy_score(np.array(all_label), np.array(all_pred))
                ticker_acc.append(accuracy)
                print('\n ##################### %s ################## ' % ticker)
                print(classification_report(np.array(all_label), np.array(all_pred)))
                print('Ticker: %s - accuracy %.3f ' % (ticker, accuracy))
        return np.mean(ticker_acc)

    def save_model(self, epoch, model_sdict, optim_sdict, loss):
        torch.save({
            'epoch': epoch,
            'model': model_sdict,
            'optimizer': optim_sdict,
            'loss': loss
        }, STOCK_H5)

        print('Model saved successfully.')

    def load_model(self, is_finetune=False):
        check_point = torch.load(STOCK_H5)
        self.stock_model.load_state_dict(check_point['model'])
        epoch = check_point['epoch']
        loss = check_point['loss']
        if is_finetune:
            self.optimizer.load_state_dict(check_point['optimizer'])
            print('Continue training. Previous epoch %s - previous loss %.3f ' % (epoch, loss))


kargs = {'input_size': 177,
         'hidden_size': 128,
         'num_layers': 1,
         'is_directional': False,
         'batch_size': 32,
         'learn_rate': 0.0005,
         'num_epochs': 10}

trainer = Trainer(**kargs)
trainer.train()
