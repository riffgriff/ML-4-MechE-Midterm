from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ucimlrepo import fetch_ucirepo
from student_support import load_project

###   helper functions   ###

def pull_dataset_uci_repo():
    # fetch dataset from the UCI ML repo - unnecessary with local data
    airfoil_self_noise = fetch_ucirepo(id=291)

    # data (as pandas dataframes)
    X_pandas = airfoil_self_noise.data.features
    b_pandas = airfoil_self_noise.data.targets

    # data comes in with rows/columns swapped from expectations
    X = X_pandas.to_numpy().transpose()
    b = b_pandas.to_numpy().transpose()

    X_labels = X_pandas.columns

    return X, b, X_labels


def pull_dataset_local():
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
    data_indices: list[int]

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
X, b, encoded, metadata = pull_dataset_local()
# Z, x_mean, x_std = standardize(X)
U, S, Vt = np.linalg.svd(X, full_matrices=False) # switch to Z for PCA
m, n = X.shape

# find the energy
energy = S**2
total_energy = np.sum(energy)
cumulative_energy = np.cumsum(energy) / total_energy * 100

# have labels associated with the 2 largest SVD dims
U_2 = U[:, :2]
Y_2 = U_2.T @ X

# plot 2-D SVD
fig2, ax2 = plt.subplots()
fig2.canvas.manager.set_window_title("SVD Scatter")
sc = ax2.scatter(Y_2[0, :], Y_2[1, :], c=b.flatten(), cmap="viridis", s=15)
cbar = fig2.colorbar(sc, ax=ax2)
cbar.set_label("Scaled sound pressure level (dB)")
ax2.set_xlabel("Mode 1")
ax2.set_ylabel("Mode 2")
ax2.set_title("First Two SVD Modes")

# do it again for 3 SVD dims
U_3 = U[:, :3]
Y_3 = U_3.T @ X

fig3 = plt.figure()
fig3.canvas.manager.set_window_title("SVD Scatter 3D")
ax3 = fig3.add_subplot(projection="3d")
sc2 = ax3.scatter(Y_3[0, :], Y_3[1, :], Y_3[2, :], c=b.flatten(), cmap="viridis", s=15)
cbar3 = fig3.colorbar(sc2, ax=ax3, pad=0.1)
cbar3.set_label("Scaled sound pressure level (dB)")
ax3.set_xlabel("Mode 1")
ax3.set_ylabel("Mode 2")
ax3.set_zlabel("Mode 3")
ax3.set_title("First Three SVD Modes")

# plot cumulative energy
fig1, ax1 = plt.subplots()
fig1.canvas.manager.set_window_title("Scree Plot")
yrange = range(1, m+1)
ax1.plot(yrange, cumulative_energy)
ax1.set_xlabel("Component")
ax1.set_ylabel("Cumulative Energy (%)")
ax1.set_title("Cumulative Energy")
plt.xticks(yrange) # makes all x-axis ticks integers in this range - beware for other plots

plt.show()
