import pandas as pd
import argparse
import numpy as np

# Script to split tractable datasets

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_name",
    "-d",
    required=True,
    help='Name of the tractable dataset eg "fbumper_repair_labor"',
)
parser.add_argument(
    "--random_seed",
    "-rs",
    required=False,
    default=7355,
    type=int,
    help="Random seed to use when splitting",
)

args = parser.parse_args()
data_name = args.data_name
random_seed = args.random_seed

data = pd.read_csv("csv/" + data_name + ".csv").drop_duplicates()

response_mapping = {
    "blwing_REPAIR_LABOR_HRS": "response",
    "fbumper_REPAIR_LABOR_HRS": "response",
    "fbumper_STRIP_LABOR_HRS": "response",
}

data.rename(columns=response_mapping, inplace=True)
train = data.loc[data.set == 0, :]
test = data.loc[data.set == 1, :]
# train, dev = ms.train_test_split(train, test_size=0.2, random_state=random_seed)
# train, dev = ms.train_test_split(train, test_size=0.2, random_state=2)

gbm_mean_pred = np.array(test["pred"])
gbm_var_pred = np.array(test["pred variance"])

np.save("pred/" + data_name + "/gbm/test", gbm_mean_pred)
np.save("pred/" + data_name + "/gbm/test_var", gbm_mean_pred)
print("Saved GBM predictions into directory: " + "pred/" + data_name + "/gbm/")
# Train: 977, Test: 489, Dev: 489
file_path = "datasets/" + data_name + "/data"


train.to_csv(file_path + "/data", index=False)
# dev.to_csv(file_path + '/dev', index=False)
test.to_csv(file_path + "/test", index=False)

print(
    "Split dataset: "
    + data_name
    + " with random seed: "
    + str(random_seed)
    + " into directory: "
    + file_path
)
