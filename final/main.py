from dataclasses import dataclass
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from final.student_support import load_project


###   general helper functions   ###

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


###   MLP structures   ###

class MLP(nn.Module):
    # an instance of an MLP model

    def __init__(self, n_inputs=5, n_outputs=1, hidden=(32, 32), activation=nn.Tanh):
        super().__init__()

        # build hidden layers
        layers = []
        prev = n_inputs # previous layer size
        for h in hidden:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, n_outputs))

        # unpack layers into the net builder
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # see __init__


def train_MLP(model, Z_train, y_train, Z_test, y_test,
              learn_rate=1e-3, weight_decay=1e-4,
              max_epochs=2000, patience=100, batch_size=128):
    # train a provided MLP model
    # returns a trained model and its corresponding loss function result against the validation data

    # using AdamW instead of SGD, less finicky
    opt = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    loss_func = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0

    # actually do the training loop
    for epoch in range(max_epochs):
        model.train()

        # scramble the rows so batches are also random
        perm = torch.randperm(Z_train.shape[0])

        # train/backpropagate doing batch_size chunks of the model at a time
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_func(model(Z_train[idx]), y_train[idx])
            loss.backward()
            opt.step()

        # check how this epoch did
        model.eval()
        with torch.no_grad():
            epoch_loss = loss_func(model(Z_test), y_test).item()

        # give a printout every 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch {epoch}:\tLoss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss, best_state, wait = epoch_loss, copy.deepcopy(model.state_dict()), 0
        else:
            # if performance against validation goes down, likely overfitting - avoid with patience
            wait += 1
            if wait >= patience:
                break

    # reset the model back to its best performance to return
    model.load_state_dict(best_state)
    return model, best_loss


###   MAIN   ###

# split into training & testing sets
training_ratio = 0.8
random_seed = 8813
X_full, b_full, encoded, metadata = pull_dataset()
X_train, X_test, b_train, b_test = train_test_split(X_full, b_full, training_ratio, rand_seed=random_seed)

# standardize data
Z_train, X_train_mean, X_train_std = standardize_training(X_train)
Z_test = standardize_testing(X_test, X_train_mean, X_train_std)
b_train_mean = b_train.mean()
b_train_std = b_train.std(ddof=1)
y_train = (b_train - b_train_mean) / b_train_std
y_test = (b_test - b_train_mean) / b_train_std

# perform PCA (necessary? not sure)
U, S, Vt = np.linalg.svd(Z_train, full_matrices=False)
n, m = X_train.shape


### train MLP - using standardized data without PCA transformation
# convert to pytorch tensors
Z_train_torch = torch.tensor(Z_train.T, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
Z_test_torch  = torch.tensor(Z_test.T, dtype=torch.float32)
y_test_torch  = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# try out the default vals
hidden_layers = (32, 32)
first_MLP = MLP(hidden=hidden_layers, activation=nn.Tanh)
first_MLP, loss = train_MLP(first_MLP, Z_train_torch, y_train_torch, Z_test_torch, y_test_torch)
print(f"\nLoss with default vals: {loss}")
# parameters to play with - training ratio, activation function, training parameters (learn rate, weight decay, batch size, patience)




