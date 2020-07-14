import torch
from torch import nn
from torch.nn import LSTM, Linear, CrossEntropyLoss
from torch.optim import Adam
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

    def forward(self, in_feature):
        out, (h_state, c_state) = self.lstm_model(in_feature, None)
        output = self.out_layer(out[:, -1, :])

        return output


class Trainer:
    def __init__(self, input_size, hidden_size, num_layers,
                 is_directional, batch_size, learn_rate, num_epochs):
        self.num_epochs = num_epochs
        train_stock = StockDataset('train')
        test_stock = StockDataset('test')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        print('Done load dataset')
        self.train_dloader = DataLoader(dataset=train_stock,
                                        batch_size=batch_size,
                                        shuffle=True)
        self.test_dloader = DataLoader(dataset=test_stock,
                                       batch_size=batch_size,
                                       shuffle=False)
        self.stock_model = StockLSTM(input_size, hidden_size, num_layers, is_directional)
        self.stock_model = self.stock_model.double().to(self.device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.stock_model.parameters(), lr=learn_rate)

    def train(self):
        for ep in range(self.num_epochs):
            self.stock_model.train()
            total_loss = 0
            with tqdm(ncols=200, total=len(self.train_dloader))as p_bar:
                for i, (x_train, y_train) in enumerate(self.train_dloader):
                    x_train, y_train = Helpers.convert_feature(x_train, torch.DoubleTensor), \
                                       Helpers.convert_feature(y_train, torch.LongTensor)
                    self.optimizer.zero_grad()
                    model_out = self.stock_model(x_train)
                    loss = self.criterion(model_out, y_train)
                    loss.backward()
                    total_loss += loss.item()
                    self.optimizer.step()

                    p_bar.set_description(
                        'Train step %s/%s - epoch %s - loss %.3f' % (i + 1, len(self.train_dloader), ep, loss.item()))

            print('*** End epoch. Evaluating ... ***')
            self.evalute()
            print('**********************************************')

    def evalute(self):
        self.stock_model.eval()
        with torch.no_grad():
            all_pred = []
            all_label = []
            for i, (x_feature, y_label) in tqdm(enumerate(self.test_dloader)):
                model_out = self.stock_model(x_feature)
                pred = model_out.topk(1)[1].cpu().numpy().tolist()
                all_pred.extend(pred)
                all_label.extend(y_label)

            accuracy = accuracy_score(np.array(all_label), np.array(all_pred))
            print('Accuracy %.3f ' % accuracy)
            print(classification_report(np.array(all_label), np.array(all_pred)))

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
         'learn_rate': 0.001,
         'num_epochs': 10}

trainer = Trainer(**kargs)
trainer.evalute()