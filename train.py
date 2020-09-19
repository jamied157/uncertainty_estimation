#!/usr/bin/env python
import data_import as data
import util as util
import models as models
import logging
import numpy as np
import torch
import argparse
import time

# Keep track of time for saving timestamp
execution_time = time.time()
execution_time = time.ctime(execution_time)

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_name",
    "-d",
    required=True,
    help='Name of the dataset eg "bostonHousing"',
)
parser.add_argument(
    "--random_seed",
    "-rs",
    required=False,
    default=1,
    type=int,
    help="Random seed to use when splitting",
)
parser.add_argument(
    "--model", "-m", required=True, type=str, help="which model you want to test"
)

args = parser.parse_args()
data_name = args.data_name
random_seed = args.random_seed
model_name = args.model
# Logging for debugging nets
logging.basicConfig(filename="experiment.log", filemode="w", level=logging.WARNING)

np.random.seed(random_seed)
torch.random.manual_seed(random_seed)

# Select model from arguments
if model_name.lower() == "dropoutmodel":
    model = models.DropoutModel
    param_dict = {
        "batch_size": [32, 64, 128],
        "n_epoch": [64, 128, 256],
        "dropout_rate": [0.05, 0.005],
        "tau": [1, 0.1, 0.01],
        "learning_rate": [0.1, 0.01, 0.001],
    }
    print("Loading Dropout Model for Cross Validation")
elif model_name.lower() == "normalmodel":
    model = models.NormalModel
    param_dict = {
        "batch_size": [32, 100],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Normal Model for Cross Validation")
elif model_name.lower() == "gammamodel":
    model = models.GammaModel
    param_dict = {
        "batch_size": [32, 100],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Gamma Model for Cross Validation")
elif model_name.lower() == "stablegammamodel":
    model = models.StableGammaModel
    param_dict = {
        "batch_size": [32, 100],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Stable Gamma Model for Cross Validation")
elif model_name.lower() == "mixturemodel":
    model = models.MixtureModel
    param_dict = {
        "batch_size": [1],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Mixture Model for Cross Validation")
elif model_name.lower() == "stablemixturemodel":
    model = models.StableMixtureModel
    param_dict = {
        "batch_size": [32, 100],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Stable Mixture Model for Cross Validation")
elif model_name.lower() == "normalmixturemodel":
    model = models.NormalMixtureModel
    param_dict = {
        "batch_size": [1],
        "n_epochs": [40],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [0],
    }
    print("Loading Normal Mixture Model for Cross Validation")
else:
    raise Exception("Model Name not recognised")


def main():
    # Import data
    X, y = data.importUCI(data_name)
    n_splits = data.load_split_num(data_name)
    err_array = np.zeros((n_splits, 2))
    calibration_arr = np.zeros((n_splits, 9))
    best_param_list = []

    for split in np.arange(n_splits):
        # Loop through each split - train and record test error
        logging.info("Split: {}".format(split))
        print("Split " + str(split))
        index_train = data.get_split_index(data_name, split, train=True)
        index_test = data.get_split_index(data_name, split, train=False)

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]
        train_data = util.UncertaintyDataset(X_train, y_train)

        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]
        test_data = util.UncertaintyDataset(X_test, y_test)

        (
            test_nll,
            test_rmse,
            calibration,
            best_param_dict,
        ) = util.standard_training_setup(train_data, test_data, model, param_dict)

        err_array[split, 0] = test_nll
        err_array[split, 1] = test_rmse
        calibration_arr[split, :] = calibration.mean(axis=0)
        best_param_list.append(best_param_dict)

        print("test nll = {:.4f} , test rmse = {:.4f}".format(test_nll, test_rmse))
        print("calibration = " + str(calibration.mean(0)))

    print(
        "Negative Log Likelihood: "
        + str(err_array.mean(axis=0)[0])
        + " +- "
        + str(err_array.std(axis=0)[0] / np.sqrt(n_splits))
    )

    print(
        "Root mean squared error: "
        + str(err_array.mean(axis=0)[1])
        + " +- "
        + str(err_array.std(axis=0)[1] / np.sqrt(n_splits))
    )

    info_dict = {
        "execution time": execution_time,
        "parameters allowed": param_dict,
        "parameters used": best_param_list,
        "error array": err_array.tolist(),
        "calibration_arr": calibration_arr.tolist(),
    }

    data.save_results(data_name, model_name, err_array, calibration_arr, info_dict)


if __name__ == "__main__":
    main()
