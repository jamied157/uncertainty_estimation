from util import ensemble_loss
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from scipy.stats import gamma, norm
import logging


class MixtureModel(nn.Module):
    """Mixture model of normal and gamma distribution"""

    def __init__(self, n_features, param_dict):
        super(MixtureModel, self).__init__()
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
        mean = output[:, 0]
        var = self.softplus(output[:, 1])
        shape = self.softplus(output[:, 2])
        rate = self.softplus(output[:, 3])
        mixture_var = self.sigmoid(output[:, 4])

        return mean, var, shape, rate, mixture_var

    def log_likelihood(self, x_norm, y_norm):
        mean, var, shape, rate, mixture_var = self(x_norm)
        norm_dist = Normal(mean, torch.sqrt(var))
        gamma_dist = Gamma(shape, rate)
        y = y_norm * self.y_std + self.y_mean + 10 ** (-4)

        only_normal_bool = (torch.abs(1 - mixture_var) < 10 ** (-4)).type(torch.float)
        only_gamma_bool = (mixture_var < 10 ** (-4)).type(torch.float)

        normal_component = norm_dist.log_prob(y_norm) + torch.log(mixture_var)
        gamma_component = gamma_dist.log_prob(y) + torch.log(1 - mixture_var)

        # logging.debug('shape,rate: {:.3f}, {:.3f}'.format(float(shape.mean()), float(rate.mean())))

        combined_tensor = torch.stack((normal_component, gamma_component), dim=0)
        output = torch.logsumexp(combined_tensor, dim=0)

        # old_output = (
        #     torch.log(
        #         (1 - only_gamma_bool)
        #         * mixture_var
        #         * torch.exp(norm_dist.log_prob(y_norm))
        #         + (
        #             (1 - only_normal_bool)
        #             * (1 - mixture_var)
        #             * torch.exp(gamma_dist.log_prob(y))
        #         )
        #     )
        # ).mean()
        if mixture_var < 0.9:
            logging.debug("Mixture var: {}".format(float(mixture_var.mean())))
            logging.debug(
                "NLLs: {:.3f}, {:.3f}".format(
                    -float(norm_dist.log_prob(y_norm).mean()),
                    -float(gamma_dist.log_prob(y).mean()),
                )
            )
            logging.debug(
                "Combined NLL: {:.3f} or {:.3f}".format(
                    -float(output.mean()), -float(old_output)
                )
            )

        return output.mean()

    def rmse(self, x_norm, y_norm):
        mean, var, shape, rate, mixture_var = self(x_norm)
        norm_dist = Normal(mean, torch.sqrt(var))
        gamma_dist = Gamma(shape, rate)
        y = y_norm * self.y_std + self.y_mean

        y_pred = (
            mixture_var * (norm_dist.mean * self.y_std + self.y_mean)
            + (1 - mixture_var) * gamma_dist.mean
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
        calibration_arr = self.calibration_test(x, y)

        return float(test_nll), float(test_rmse), calibration_arr

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
        gamma_check = np.logical_and(
            gamma_lower < y[np.newaxis, :], y[np.newaxis, :] < gamma_upper
        )

        output[mixture_var < 0.5] = normal_check
        output[mixture_var > 0.5] = gamma_check

        return output


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
        lower_bounds, upper_bounds = norm.interval(
            confidence_values, loc=mean, scale=np.sqrt(var)
        )
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T


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
        lower_bounds, upper_bounds = gamma.interval(
            confidence_values, shape, scale=1 / rate
        )
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T


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
        optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay
        )

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
        lower_bounds, upper_bounds = gamma.interval(
            confidence_values, shape, scale=1 / rate
        )
        lower_check = lower_bounds < y[np.newaxis, :]
        upper_check = y[np.newaxis, :] < upper_bounds

        return np.logical_and(lower_check, upper_check).T


class StableMixtureModel(nn.Module):
    """Mixture model with stabilising constant for gamma component for better results"""

    def __init__(self, n_features, param_dict):
        super(StableMixtureModel, self).__init__()
        self.dropout_rate = param_dict["dropout_rate"]
        self.lr = param_dict["learning_rate"]
        self.batch_size = param_dict["batch_size"]
        self.n_epochs = param_dict["n_epochs"]

        self.rate_constant = nn.Parameter(torch.tensor(1.0))
        self.shape_constant = nn.Parameter(torch.tensor(1.0))

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
        mean = output[:, 0]
        var = self.softplus(output[:, 1])
        shape = self.softplus(output[:, 2]) * self.shape_constant
        rate = self.softplus(output[:, 3]) * self.rate_constant
        mixture_var = self.sigmoid(output[:, 4])

        return mean, var, shape, rate, mixture_var

    def log_likelihood(self, x_norm, y_norm):
        mean, var, shape, rate, mixture_var = self(x_norm)
        norm_dist = Normal(mean, torch.sqrt(var))
        gamma_dist = Gamma(shape, rate)
        y = y_norm * self.y_std + self.y_mean + 10 ** (-4)

        only_normal_bool = (torch.abs(1 - mixture_var) < 10 ** (-4)).type(torch.float)
        only_gamma_bool = (mixture_var < 10 ** (-4)).type(torch.float)

        normal_component = norm_dist.log_prob(y_norm) + torch.log(mixture_var)
        gamma_component = gamma_dist.log_prob(y) + torch.log(1 - mixture_var)

        logging.debug(
            "shape,rate: {:.3f}, {:.3f}".format(float(shape.mean()), float(rate.mean()))
        )

        combined_tensor = torch.stack((normal_component, gamma_component), dim=0)
        old_output = torch.logsumexp(combined_tensor, dim=0).mean()

        output = (
            torch.log(
                (1 - only_gamma_bool)
                * mixture_var
                * torch.exp(norm_dist.log_prob(y_norm))
                + (
                    (1 - only_normal_bool)
                    * (1 - mixture_var)
                    * torch.exp(gamma_dist.log_prob(y))
                )
            )
        ).mean()
        logging.debug("Mixture var: {}".format(float(mixture_var.mean())))
        logging.debug(
            "NLLs: {:.3f}, {:.3f}".format(
                -float(norm_dist.log_prob(y_norm).mean()),
                -float(gamma_dist.log_prob(y).mean()),
            )
        )
        logging.debug(
            "Combined NLL: {:.3f} or {:.3f}".format(-float(output), -float(old_output))
        )

        return output

    def rmse(self, x_norm, y_norm):
        mean, var, shape, rate, mixture_var = self(x_norm)
        norm_dist = Normal(mean, torch.sqrt(var))
        gamma_dist = Gamma(shape, rate)
        y = y_norm * self.y_std + self.y_mean

        y_pred = (
            mixture_var * (norm_dist.mean * self.y_std + self.y_mean)
            + (1 - mixture_var) * gamma_dist.mean
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
                logging.debug("Loss: {:.2f}".format(float(loss)))
                loss.backward()
                optimiser.step()

    def test_loss(self, test_data):
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

        output = np.zeros_like(norm_upper)
        normal_check = np.logical_and(
            norm_lower < y_norm[np.newaxis, :], y_norm[np.newaxis, :] < norm_upper
        )
        gamma_check = np.logical_and(
            gamma_lower < y[np.newaxis, :], y[np.newaxis, :] < gamma_upper
        )

        output[mixture_var < 0.5, :] = normal_check
        output[mixture_var > 0.5, :] = gamma_check

        return output


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
            + (
                (1 - only_one_bool)
                * (1 - mixture_var)
                * torch.exp(norm2_dist.log_prob(y_norm))
            )
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

        y_pred = mixture_var * (norm1_dist.mean * self.y_std + self.y_mean) + (
            1 - mixture_var
        ) * (norm2_dist.mean * self.y_std + self.y_mean)
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
        gamma_check = np.logical_and(
            gamma_lower < y[np.newaxis, :], y[np.newaxis, :] < gamma_upper
        )

        output[mixture_var < 0.5] = normal_check
        output[mixture_var > 0.5] = gamma_check

        return output
