from .dropout import DropoutModel
from .gamma import GammaModel, StableGammaModel
from .mixture import MixtureModel, StableMixtureModel
from .normal_mixture import NormalMixtureModel
from .normal import NormalModel

def get_model(model_name):
    """
    Returns required model with appropriate possible hyper parameters in param_dict.
    These are used for cross validation when experimenting. 
    """
    if model_name.lower() == "dropout":
        model = DropoutModel
        param_dict = {
            "batch_size": [32, 64, 128],
            "n_epoch": [64, 128, 256],
            "dropout_rate": [0.05, 0.005],
            "tau": [1, 0.1, 0.01],
            "learning_rate": [0.1, 0.01, 0.001],
        }
        print("Loading Dropout Model for Cross Validation")
    elif model_name.lower() == "normal":
        model = NormalModel
        param_dict = {
            "batch_size": [32, 100],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Normal Model for Cross Validation")
    elif model_name.lower() == "gamma":
        model = GammaModel
        param_dict = {
            "batch_size": [32, 100],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Gamma Model for Cross Validation")
    elif model_name.lower() == "stablegamma":
        model = StableGammaModel
        param_dict = {
            "batch_size": [32, 100],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Stable Gamma Model for Cross Validation")
    elif model_name.lower() == "mixture":
        model = MixtureModel
        param_dict = {
            "batch_size": [1],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Mixture Model for Cross Validation")
    elif model_name.lower() == "stablemixture":
        model = StableMixtureModel
        param_dict = {
            "batch_size": [32, 100],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Stable Mixture Model for Cross Validation")
    elif model_name.lower() == "normalmixture":
        model = NormalMixtureModel
        param_dict = {
            "batch_size": [1],
            "n_epochs": [40],
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0],
        }
        print("Loading Normal Mixture Model for Cross Validation")
    else:
        raise Exception("Model Name not recognised")

    return model, param_dict