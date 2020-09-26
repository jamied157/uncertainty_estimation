# Simple Deep Learning Uncertainty Estimation for Regression tasks.

This repo contains code that runs experiments to test different uncertainty estimation methods. In particular the focus is on bridging the gap between classification methods (where the network parameterises a distribution) and regression methods (which usually produce point estimates). Many models in the repo instead parameterise a distribution over the real line in their outputs and train using the negative log likelihood as a loss function.

## Datasets

We use the standard set of UCI datasets which you can download in the right format from: https://drive.google.com/file/d/1usNWZqUxTROiWwB3lQ0ZWlLMwGd_RJpO/view?usp=sharing

Usage is very simple, run `train.py `and specify a dataset and model.

Models can be from the set `{dropout, normal, gamma, stablegamma, mixture, stablemixture, normalmixture}`. Look in the models folder for the implementation details.
Datasets are taken from the google drive link. 