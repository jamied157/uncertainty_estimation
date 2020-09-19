import numpy as np
import json


def importUCI(dataset_name):
    """
    Imports a dataset from the UCI respository
    :param dataset_name: the string file name of the repository
    :return X, y: the input and response variables
    """
    _DATA_DIRECTORY_PATH = "./datasets/" + dataset_name + "/data/"
    _DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
    _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
    _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"

    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)

    data = np.loadtxt(_DATA_FILE)
    X = data[:, [int(i) for i in index_features.tolist()]]
    y = data[:, int(index_target.tolist())]

    return X, y


def load_split_num(dataset_name):
    """
    returns the number of splits for a given dataset
    :param dataset_name:
    :return n_splits:
    """
    _DATA_DIRECTORY_PATH = "./datasets/" + dataset_name + "/data/"
    _N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"
    n_splits = np.loadtxt(_N_SPLITS_FILE)

    return int(n_splits)


def get_split_index(dataset_name, split, train):
    """
    Returns the indices for a given split in a given dataset
    :param dataset_name:
    :param split: the number split that is requested
    :param train: a boolean specifying train or test
    :return X, y:
    """
    _DATA_DIRECTORY_PATH = "./datasets/" + dataset_name + "/data/"

    if train:
        _SPLIT_FILE = _DATA_DIRECTORY_PATH + "index_train_" + str(split) + ".txt"
    else:
        _SPLIT_FILE = _DATA_DIRECTORY_PATH + "index_test_" + str(split) + ".txt"

    indices = np.loadtxt(_SPLIT_FILE)

    return indices


def save_results(dataset_name, modelname, err_array, calibration_array, info_dict):
    """Save results from experiment"""
    _RESULTS_DIRECTORY_PATH = "./datasets/" + dataset_name + "/results/" + modelname
    np.savetxt(_RESULTS_DIRECTORY_PATH + "/nll_array.txt", err_array[:, 0])
    np.savetxt(_RESULTS_DIRECTORY_PATH + "/rmse_array.txt", err_array[:, 1])
    np.savetxt(_RESULTS_DIRECTORY_PATH + "/calibration_array.txt", calibration_array)
    timestamp = info_dict["execution time"]

    with open(
        _RESULTS_DIRECTORY_PATH + "/info_dicts/" + str(timestamp) + ".txt", "w+"
    ) as outfile:
        json.dump(info_dict, outfile)

    with open(
        _RESULTS_DIRECTORY_PATH + "/info_dicts/" + "most_recent" + ".txt", "w+"
    ) as outfile:
        json.dump(info_dict, outfile)
