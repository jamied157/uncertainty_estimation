from util import ensemble_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from scipy.stats import gamma, norm
import logging


class GammaModel(nn.Module):
    """Model outputting Gamma distribution"""

    def __init__(self, n_features, param_dict):
        super(GammaModel, self).__init__()
        self.dropout_rate = param_dict["dropout_rate"]
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.n_epochs = param_dict["n_epochs"]
        self.x_mean, self.x_std = 0, 1

        self.layer_1 = nn.Linear(n_features, 50)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(50, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.dropout(x)
        o_1 = self.layer_1(x)
        h_1 = self.relu_1(o_1)
        h_1 = self.dropout(h_1)
        output = self.softplus(self.layer_2(h_1)) + 10 ** (-6)

        shape = output[:, 0]
        rate = output[:, 1]
        return shape, rate

    def fit(self, train_data):
        self.train(True)
        self.x_mean, self.x_std = train_data.normalise_x()
        data_generator = data.DataLoader(train_data, batch_size=self.batch_size)

        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in torch.arange(self.n_epochs):
            for i, sample in enumerate(data_generator):
                x, y = sample
                shape, rate = self(x)
                gamma_dist = Gamma(shape, rate)
                optimiser.zero_grad()
                loss = -gamma_dist.log_prob(y.squeeze() + 10 ** (-8)).mean()
                logging.debug("NLLs: {:.3f}".format(float(loss)))
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
        shape, rate = self(x)

        y = y.squeeze()
        shape = shape.squeeze()
        rate = rate.squeeze()

        gamma_dist = Gamma(shape, rate)

        test_nll = -gamma_dist.log_prob(y + 10 ** (-8)).mean()
        test_rmse = (((y - gamma_dist.mean) ** 2).mean()) ** 0.5
        calibration_arr = self.calibration_test(
            y.detach().numpy(), shape.detach().numpy(), rate.detach().numpy()
        )

        return float(test_nll), float(test_rmse), calibration_arr

    @staticmethod
    def calibration_test(y, shape, rate):

        confidence_values = np.expand_dims(np.arange(0.1, 1, 0.1), axis=1)
        lower_bounds, upper_bounds = gamma.interval(confidence_values, shape, scale=1 / rate)
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T


class StableGammaModel(nn.Module):
    """Gamma model with stabilising constant for better results"""

    def __init__(self, n_features, param_dict):
        super(StableGammaModel, self).__init__()
        self.dropout_rate = param_dict["dropout_rate"]
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.n_epochs = param_dict["n_epochs"]

        self.rate_constant = nn.Parameter(torch.tensor(1.0))
        self.shape_constant = nn.Parameter(torch.tensor(1.0))

        self.layer_1 = nn.Linear(n_features, 50)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(50, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.dropout(x)
        o_1 = self.layer_1(x)
        h_1 = self.relu_1(o_1)
        h_1 = self.dropout(h_1)
        output = self.softplus(self.layer_2(h_1)) + 10 ** (-6)

        shape = output[:, 0] * self.shape_constant
        rate = output[:, 1] * self.rate_constant
        return shape, rate

    def fit(self, train_data):
        self.train(True)
        self.x_mean, self.x_std = train_data.normalise_x()
        data_generator = data.DataLoader(train_data, batch_size=self.batch_size)

        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in torch.arange(self.n_epochs):
            for i, sample in enumerate(data_generator):
                x, y = sample
                shape, rate = self(x)
                gamma_dist = Gamma(shape, rate)
                optimiser.zero_grad()
                loss = -gamma_dist.log_prob(y.squeeze() + 10 ** (-8)).mean()
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
        shape, rate = self(x)

        y = y.squeeze()
        shape = shape.squeeze()
        rate = rate.squeeze()

        gamma_dist = Gamma(shape, rate)

        test_nll = -gamma_dist.log_prob(y + 10 ** (-8)).mean()
        test_rmse = (((y - gamma_dist.mean) ** 2).mean()) ** 0.5
        calibration_arr = self.calibration_test(
            y.detach().numpy(), shape.detach().numpy(), rate.detach().numpy()
        )

        return float(test_nll), float(test_rmse), calibration_arr

    @staticmethod
    def calibration_test(y, shape, rate):

        confidence_values = np.expand_dims(np.arange(0.1, 1, 0.1), axis=1)
        lower_bounds, upper_bounds = gamma.interval(confidence_values, shape, scale=1 / rate)
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T