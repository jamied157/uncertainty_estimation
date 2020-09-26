from util import ensemble_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from scipy.stats import gamma, norm
import logging


class NormalModel(nn.Module):
    """Model outputting normal distribution"""

    def __init__(self, n_features, param_dict):
        super(NormalModel, self).__init__()
        self.dropout_rate = param_dict["dropout_rate"]
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.n_epochs = param_dict["n_epochs"]
        self.y_mean, self.y_std = 0, 1
        self.x_mean, self.x_std = 0, 1

        self.layer_1 = nn.Linear(n_features, 50)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu_1 = nn.ReLU()
        self.layer_2_1 = nn.Linear(50, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.dropout(x)
        o_1 = self.layer_1(x)
        h_1 = self.relu_1(o_1)
        h_1 = self.dropout(h_1)
        output = self.layer_2_1(h_1)
        # mean = self.layer_2_1(h_1[:, :50])
        # var = self.softplus(self.layer_2_2(h_1[:, :])) + 10**(-6)
        mean = output[:, 0]
        var = self.softplus(output[:, 1]) + 10 ** (-6)
        return mean, var

    def fit(self, train_data):
        self.train(True)
        self.y_mean, self.y_std = train_data.normalise_y()
        self.x_mean, self.x_std = train_data.normalise_x()

        data_generator = data.DataLoader(train_data, batch_size=self.batch_size)

        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in torch.arange(self.n_epochs):
            for _, sample in enumerate(data_generator):
                x, y = sample
                mean, var = self(x)
                norm_dist = Normal(mean, torch.sqrt(var))
                optimiser.zero_grad()
                loss = -norm_dist.log_prob(y).mean()
                loss.backward()
                optimiser.step()

    def test_loss(self, test_data):
        """
        outputs the losses the test data
        """
        x, y = test_data[:]
        if not test_data.x_normalised:
            constant_x = self.x_std == 0
            x = (x - self.x_mean) / self.x_std
            x[:, constant_x] = 0
        self.train(False)
        mean, var = self(x)
        mean = self.y_mean + mean * self.y_std
        var = var * self.y_std ** 2
        norm_dist = Normal(mean, torch.sqrt(var))

        test_nll = -norm_dist.log_prob(y).mean()
        test_rmse = (((y.squeeze() - mean.squeeze()) ** 2).mean()) ** 0.5
        calibration_arr = self.calibration_test(
            y.detach().numpy(), mean.detach().numpy(), var.detach().numpy()
        )

        return float(test_nll), float(test_rmse), calibration_arr

    @staticmethod
    def calibration_test(y, mean, var):

        confidence_values = np.expand_dims(np.arange(0.1, 1, 0.1), axis=1)
        lower_bounds, upper_bounds = norm.interval(confidence_values, loc=mean, scale=np.sqrt(var))
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T