from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from student_support import load_project

###   helper functions   ###

def pull_dataset():
    # pull dataset from the student support file
    # returns X,b,encoded,metadata
    """
    X: n-by-m quantitative inputs; b: m-by-1 response; encoded: m-by-(p-1)
    unscaled quantitative inputs followed by the prescribed category indicators.
    All metadata indices are zero-based NumPy indices. See README.
    """
    return load_project.load_project(2)


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

# do economy SVD
X, b, encoded, metadata = pull_dataset()
Z, x_mean, x_std = standardize(X)
U_X, S_X, Vt_X = np.linalg.svd(X, full_matrices=False)
U_Z, S_Z, Vt_Z = np.linalg.svd(Z, full_matrices=False) # technically PCA but whatever
m, n = X.shape

# find singularity cutoff, rank, and condition number
eps_mach = np.finfo(float).eps # machine tolerance for numpy operations
tau_X = eps_mach*max([m,n])*S_X[0]
tau_Z = eps_mach*max([m,n])*S_Z[0]
r_X = int((S_X > tau_X).sum())
r_Z = int((S_Z > tau_Z).sum())
c_X = S_X[0]/S_X[r_X-1]
c_Z = S_Z[0]/S_Z[r_Z-1]
print(f"\ntau_X: {tau_X}\ntau_Z: {tau_Z}\n\n"+
      f"Rank X: {r_X}\nRank Z: {r_Z}\n\n"+
      f"Cond X: {c_X}\nCond Z: {c_Z}\n")


