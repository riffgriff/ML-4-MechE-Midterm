import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo

def pull_dataset():
    # fetch dataset
    airfoil_self_noise = fetch_ucirepo(id=291)

    # data (as pandas dataframes)
    X_pandas = airfoil_self_noise.data.features
    y_pandas = airfoil_self_noise.data.targets

    X = X_pandas.to_numpy()
    y = y_pandas.to_numpy()

    X_labels = X_pandas.columns

    return X, y, X_labels


def standardize(X):
    # mean-center the data

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_mean_centered = X - X_mean

    # divide by feature-wise standard deviation

    X_std = np.std(X_mean_centered, axis=1, keepdims=True, ddof=1)
    Z = X_mean_centered / X_std

    return Z, X_mean, X_std



### MAIN ###

# do economy SVD
X, y, X_labels = pull_dataset()
Z, x_mean, x_std = standardize(X)
U, S, Vt = np.linalg.svd(Z)

# find the energy
energy = S**2
total_energy = np.sum(energy)
cumulative_energy = np.cumsum(energy) / total_energy * 100

fig1, ax1 = plt.subplots()
fig1.canvas.manager.set_window_title("Scree Plot")
ax1.plot(cumulative_energy)

plt.show()