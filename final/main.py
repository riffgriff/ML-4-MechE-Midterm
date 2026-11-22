from dataclasses import dataclass
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from final.student_support import load_project


###   helper functions   ###

def pull_dataset():
    # pull dataset from the student support file
    # returns X,b,encoded,metadata
    """
    X: n-by-m quantitative inputs; b: m-by-1 response; encoded: m-by-(p-1)
    unscaled quantitative inputs followed by the prescribed category indicators.
    All metadata indices are zero-based NumPy indices. See README.
    """
    return load_project.load_project(2) # same data as Group 2 from the midterm


def standardize_training(X):
    # mean-center the data

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_mean_centered = X - X_mean

    # divide by feature-wise standard deviation

    X_std = np.std(X_mean_centered, axis=1, keepdims=True, ddof=1)
    Z = X_mean_centered / X_std

    return Z, X_mean, X_std


def standardize_testing(X, mean_training, std_training):
    # standardize validation data by the mean/std from training

    X_mean_centered = X - mean_training
    Z = X_mean_centered / std_training

    return Z


@dataclass
class Configuration:
    angle: float # 2nd data point
    chord_l: float # 3rd data point
    velocity: float # 4th data point
    displacement_t: float # 5th data point
    data_indices: list[int] # list of the indices of data that are a part of this configuration

    def is_member(self, numpy_col):
        if (self.angle == numpy_col[1] and
            self.chord_l == numpy_col[2] and
            self.velocity == numpy_col[3] and
            self.displacement_t == numpy_col[4]):
            return True
        else:
            return False

    def __len__(self):
        return len(self.data_indices)


def pull_configurations(X):
    # find which data points all belong to a single configuration
    # returns a list of Configuration dataclasses

    configurations = []

    for i, column in enumerate(X.T):
        # check if the configuration already exists
        config_found = False
        for config in configurations:
            if config.is_member(column):
                config.data_indices.append(i)
                config_found = True
                break

        if not config_found:
            new_config = Configuration(column[1], column[2], column[3], column[4], [i])
            configurations.append(new_config)

    return configurations


def train_test_split(X, b, training_ratio, rand_seed=8813):
    # randomly split data into a training set and a validation set, keeping configurations together
    # X: (n_features, n_samples), b: (n_samples,)

    rng = random.Random(rand_seed)

    configs = pull_configurations(X)
    n_train = round(training_ratio * len(configs))
    training_config_idx = set(rng.sample(range(len(configs)), k=n_train))

    train_idx, test_idx = [], []
    for i, config in enumerate(configs):
        if i in training_config_idx:
            train_idx.extend(config.data_indices)
        else:
            test_idx.extend(config.data_indices)

    X_train, X_test = X[:, train_idx], X[:, test_idx]
    b_train, b_test = b[train_idx], b[test_idx]

    return X_train, X_test, b_train, b_test


###   MAIN   ###

# split into training & testing sets
training_ratio = 0.8
X_full, b_full, encoded, metadata = pull_dataset()
X_train, X_test, b_train, b_test = train_test_split(X_full, b_full, training_ratio)

# standardize training data
Z_train, X_train_mean, X_train_std = standardize_training(X_train)
y_train, b_train_mean, b_train_std = standardize_training(b_train) # y is standardized output vector

# perform PCA (necessary? not sure)
U, S, Vt = np.linalg.svd(Z_train, full_matrices=False)
n, m = X_train.shape

# train MLP - using standardized data without PCA transformation








### validation
# for whoever does this, the standardize_testing function should be helpful - just use X_train_mean and X_train_std




