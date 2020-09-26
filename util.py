import logging

import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import ParameterGrid


class UncertaintyDataset(data.Dataset):
    """
    Characterises a dataset for PyTorch
    """

    def __init__(self, X, Y):
        self.X, self.Y = torch.Tensor(X), torch.Tensor(Y)

        self.y_normalised = False
        self.x_normalised = False

        self.y_std = self.Y.std()
        self.y_mean = self.Y.mean()

        self.x_std = self.X.std(dim=0)
        self.x_mean = self.X.mean(dim=0)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.X.shape[0]

    def n_features(self):
        return self.X.shape[1]

    def normalise_y(self):
        """Normalise outputs and save parameters to instance"""
        if not self.y_normalised:
            self.Y = (self.Y - self.y_mean) / self.y_std
            self.y_normalised = True
        else:
            pass
        return self.y_mean, self.y_std

    def normalise_x(self):
        """Normalise inputs and save parameters to instance"""
        if not self.x_normalised:
            # Check for 0 standard deviation variables
            constant_x = self.x_std == 0
            self.X = (self.X - self.x_mean) / self.x_std
            self.X[:, constant_x] = 0
            self.x_normalised = True
        else:
            pass

        return self.x_mean, self.x_std

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.X[index, :], self.Y[index]


def ensemble_loss(y: torch.Tensor, y_hat_arr: torch.Tensor, tau: float):
    """Loss for MC Dropout Net"""
    t = y_hat_arr.shape[0]
    ll = (
        torch.logsumexp(-0.5 * tau * (y - y_hat_arr.squeeze()) ** 2, 0)
        - np.log(t)
        - 0.5 * np.log(2 * np.pi)
        + 0.5 * np.log(tau)
    )

    return -ll.mean()


def normal_nll(targets: torch.Tensor, predictions: torch.Tensor, var: torch.Tensor):
    """
    Outputs the mean log likelihood of a normal rv for all targets
    :param targets: output variables
    :param predictions: predictions made for targets
    :param var: array of predicted variances for each target
    :return mean log likelihood for a normal rv:
    """
    targets = targets.squeeze()
    predictions = predictions.squeeze()
    var = var.squeeze()

    reg_term = 0.5 * torch.log(2 * np.pi * var)
    err_term = 0.5 * (var ** (-1)) * (targets - predictions) ** 2
    return torch.mean(reg_term + err_term)


def cross_validate(train_data, ModelClass, param_grid):
    """
    Cross validation routine - splits once and loops over possible parameters, 
    reports parameters with least error
    """
    param_grid = ParameterGrid(param_grid)
    num_training_examples = int(0.8 * len(train_data))

    X_train, y_train = train_data[:num_training_examples]
    new_train_data = UncertaintyDataset(X_train, y_train)

    X_valid, y_valid = train_data[num_training_examples:]
    valid_data = UncertaintyDataset(X_valid, y_valid)

    best_nll = float("inf")
    best_param_dict = {}

    for param_dict in param_grid:
        # print('Trying ' + str(param_dict))
        # logging.info('INFO DICT: ' + str(param_dict))
        model = ModelClass(X_train.shape[1], param_dict)
        model.fit(new_train_data)
        model_nll, _, _ = model.test_loss(valid_data)
        # print('nll: {:.3f}'.format(model_nll))
        if model_nll < best_nll:
            best_nll = model_nll
            best_param_dict = param_dict

    return best_param_dict


def standard_training_setup(
    train_data: UncertaintyDataset,
    test_data: UncertaintyDataset,
    ModelClass,
    param_grid,
):
    """Standard setup for uncertainty experiments"""
    best_param_dict = cross_validate(train_data, ModelClass, param_grid)
    print("Parameters: " + str(best_param_dict))
    model_1 = ModelClass(train_data.n_features(), best_param_dict)
    model_1.fit(train_data)

    test_nll, test_rmse, calibration_arr = model_1.test_loss(test_data)

    return test_nll, test_rmse, calibration_arr, best_param_dict
