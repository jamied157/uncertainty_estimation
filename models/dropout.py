from util import ensemble_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from scipy.stats import gamma, norm
import logging

class DropoutModel(nn.Module):
    """Model using dropout uncertainty estimates"""

    def __init__(self, n_features, param_dict):
        super(DropoutModel, self).__init__()
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.dropout_rate = param_dict["dropout_rate"]
        self.n_epochs = param_dict["n_epoch"]
        self.tau = param_dict["tau"]

        self.y_mean, self.y_std = 0, 1
        self.x_mean, self.x_std = 0, 1

        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.layer_1 = nn.Linear(n_features, 50)
        self.relu_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.layer_2 = nn.Linear(50, 1)

    def forward(self, x, sample=False):
        x = self.dropout_1(x)
        o_1 = self.layer_1(x)
        h_1 = self.relu_1(o_1)
        h_1 = self.dropout_2(h_1)
        y_hat_arr = self.layer_2(h_1)

        if sample:
            t = 10000
            y_hat_arr = torch.empty((t, x.shape[0], 1))
            for i in range(t):
                y_hat_arr[i, :, :] = self.forward(x, sample=False)

        return y_hat_arr

    def fit(self, train_data):
        batch_size = self.batch_size
        n_epochs = self.n_epochs
        tau = self.tau

        self.y_mean, self.y_std = train_data.normalise_y()
        self.x_mean, self.x_std = train_data.normalise_x()

        data_generator = data.DataLoader(train_data, batch_size=batch_size)

        weight_decay = 1e-4 * (1 - 0.005) / (2 * len(train_data) * tau)
        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        for _ in torch.arange(n_epochs):
            for _, sample in enumerate(data_generator):
                x, y = sample
                y_hat_arr = self(x, sample=False)
                optimiser.zero_grad()
                loss = ((y - y_hat_arr.squeeze()) ** 2).mean()
                loss.backward()
                optimiser.step()

    def test_loss(self, test_data):
        x, y = test_data[:]
        if not test_data.x_normalised:
            constant_x = self.x_std == 0
            x = (x - self.x_mean) / self.x_std
            x[:, constant_x] = 0

        y_hat_arr = self(x, sample=True)
        y_hat_arr = y_hat_arr * self.y_std + self.y_mean
        test_nll = ensemble_loss(y, y_hat_arr, self.tau)
        test_rmse = ((y.squeeze() - y_hat_arr.mean(dim=0).squeeze()) ** 2).mean() ** 0.5

        return float(test_nll), float(test_rmse), np.zeros(10)