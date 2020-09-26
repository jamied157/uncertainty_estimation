from util import ensemble_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from scipy.stats import gamma, norm
import logging


class NormalMixtureModel(nn.Module):
    """Mixture model of normal and gamma distribution"""

    def __init__(self, n_features, param_dict):
        super(NormalMixtureModel, self).__init__()
        self.dropout_rate = param_dict["dropout_rate"]
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.n_epochs = param_dict["n_epochs"]
        self.y_mean, self.y_std = 0, 1
        self.x_mean, self.x_std = 0, 1

        self.layer_1 = nn.Linear(n_features, 50)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu_1 = nn.ReLU()
        self.layer_2_1 = nn.Linear(50, 5)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        o_1 = self.layer_1(x)
        h_1 = self.relu_1(o_1)
        h_1 = self.dropout(h_1)
        output = self.layer_2_1(h_1)
        mean_1 = output[:, 0]
        var_1 = self.softplus(output[:, 1])
        mean_2 = output[:, 2]
        var_2 = self.softplus(output[:, 3])
        mixture_var = self.sigmoid(output[:, 4])

        return mean_1, var_1, mean_2, var_2, mixture_var

    def log_likelihood(self, x_norm, y_norm):
        mean_1, var_1, mean_2, var_2, mixture_var = self(x_norm)
        norm1_dist = Normal(mean_1, torch.sqrt(var_1))
        norm2_dist = Normal(mean_2, torch.sqrt(var_2))

        only_one_bool = (torch.abs(1 - mixture_var) < 10 ** (-4)).type(torch.float)
        only_two_bool = (mixture_var < 10 ** (-4)).type(torch.float)

        normal_component = norm1_dist.log_prob(y_norm) + torch.log(mixture_var)
        gamma_component = norm2_dist.log_prob(y_norm) + torch.log(1 - mixture_var)

        combined_tensor = torch.stack((normal_component, gamma_component), dim=0)
        old_output = torch.logsumexp(combined_tensor, dim=0)

        output = torch.log(
            (1 - only_two_bool) * mixture_var * torch.exp(norm1_dist.log_prob(y_norm))
            + ((1 - only_one_bool) * (1 - mixture_var) * torch.exp(norm2_dist.log_prob(y_norm)))
        )
        logging.debug(
            "{:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
                float(normal_component.mean()),
                float(gamma_component.mean()),
                float(mixture_var.mean()),
                float(output.mean()),
            )
        )
        # if mixture_var < 0.9:
        #    logging.debug('Mixture var: {}'.format(float(mixture_var.mean())))
        #    logging.debug('NLLs: {:.3f}, {:.3f}'.format(-float(norm_dist.log_prob(y_norm).mean()), -float(gamma_dist.log_prob(y).mean())))
        #    logging.debug('Combined NLL: {:.3f} or {:.3f}'.format(-float(output.mean()), -float(old_output)))

        return output.mean()

    def rmse(self, x_norm, y_norm):
        mean_1, var_1, mean_2, var_2, mixture_var = self(x_norm)
        norm1_dist = Normal(mean_1, torch.sqrt(var_1))
        norm2_dist = Normal(mean_2, torch.sqrt(var_2))
        y = y_norm * self.y_std + self.y_mean

        y_pred = mixture_var * (norm1_dist.mean * self.y_std + self.y_mean) + (1 - mixture_var) * (
            norm2_dist.mean * self.y_std + self.y_mean
        )
        return torch.sqrt(((y - y_pred) ** 2).mean())

    def fit(self, train_data):
        self.train(True)
        self.y_mean, self.y_std = train_data.normalise_y()
        self.x_mean, self.x_std = train_data.normalise_x()

        data_generator = data.DataLoader(train_data, batch_size=self.batch_size)

        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in torch.arange(self.n_epochs):
            for _, sample in enumerate(data_generator):
                x, y = sample

                optimiser.zero_grad()
                loss = -self.log_likelihood(x, y)
                loss.backward()
                optimiser.step()

    def test_loss(self, test_data):
        """
        outputs the losses the test data
        """
        x, y = test_data[:]
        if not test_data.y_normalised:
            y = (y - self.y_mean) / self.y_std
        if not test_data.x_normalised:
            constant_x = self.x_std == 0
            x = (x - self.x_mean) / self.x_std
            x[:, constant_x] = 0

        self.train(False)
        test_nll = -self.log_likelihood(x, y)
        test_rmse = self.rmse(x, y)
        # calibration_arr = self.calibration_test(x, y)

        return float(test_nll), float(test_rmse), np.zeros(10)

    def calibration_test(self, x, y_norm):
        mean, var, shape, rate, mixture_var = self(x)
        y = y_norm * self.y_std + self.y_mean

        y_norm = y_norm.detach().numpy()
        y = y.detach().numpy()

        confidence_values = np.expand_dims(np.arange(0.1, 1, 0.1), axis=1)

        norm_lower, norm_upper = norm.interval(
            confidence_values,
            loc=mean.detach().numpy(),
            scale=np.sqrt(var.detach().numpy()),
        )
        gamma_lower, gamma_upper = gamma.interval(
            confidence_values, shape.detach().numpy(), scale=1 / rate.detach().numpy()
        )

        output = torch.zeros_like(norm_upper)
        normal_check = np.logical_and(
            norm_lower < y_norm[np.newaxis, :], y_norm[np.newaxis, :] < norm_upper
        )
        gamma_check = np.logical_and(gamma_lower < y[np.newaxis, :], y[np.newaxis, :] < gamma_upper)

        output[mixture_var < 0.5] = normal_check
        output[mixture_var > 0.5] = gamma_check

        return output
