#!/usr/bin/env python
import data
import util
from models import get_model
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
    "--dataset",
    "-d",
    required=True,
    help='Name of the dataset eg "bostonHousing"',
)
parser.add_argument(
    "--seed",
    "-s",
    required=False,
    default=1,
    type=int,
    help="Random seed to use when splitting",
)
parser.add_argument("--model", "-m", required=True, type=str, help="which model you want to test")

args = parser.parse_args()
dataset = args.dataset
seed = args.seed
model_name = args.model
# Logging for debugging nets
logging.basicConfig(filename="experiment.log", filemode="w", level=logging.WARNING)

np.random.seed(seed)
torch.random.manual_seed(seed)

# Select model from arguments
model, param_dict = get_model(model_name.lower())

def main():
    # Import data
    X, y = data.import_uci(dataset)
    n_splits = data.load_split_num(dataset)
    err_array = np.zeros((n_splits, 2))
    calibration_arr = np.zeros((n_splits, 9))
    best_param_list = []

    for split in np.arange(n_splits):
        # Loop through each split - train and record test error
        logging.info("Split: {}".format(split))
        print("Split " + str(split))
        index_train = data.get_split_index(dataset, split, train=True)
        index_test = data.get_split_index(dataset, split, train=False)

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

    data.save_results(dataset, model_name, err_array, calibration_arr, info_dict)


if __name__ == "__main__":
    main()
