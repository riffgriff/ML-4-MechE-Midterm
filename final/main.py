from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

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


def standardize(X):
    # mean-center the data

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_mean_centered = X - X_mean

    # divide by feature-wise standard deviation

    X_std = np.std(X_mean_centered, axis=1, keepdims=True, ddof=1)
    Z = X_mean_centered / X_std

    return Z, X_mean, X_std


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


###   MAIN   ###

# perform PCA
X, b, encoded, metadata = pull_dataset()
Z, x_mean, x_std = standardize(X)
U, S, Vt = np.linalg.svd(Z, full_matrices=False) # technically PCA but whatever
n, m = X.shape




