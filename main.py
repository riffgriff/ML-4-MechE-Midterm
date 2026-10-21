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
n, m = X.shape

print(f"m: {m}, n: {n}")

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

# Cumulative retained variance calculations
eigenvalues = S_Z**2/(m - 1)
print(f"\nEigenvalues: {eigenvalues}")
cumulative_variance = 100*np.cumsum(eigenvalues)/np.sum(eigenvalues)

#Compute k=number of components needed to reach 90% energy
k = np.argmax(cumulative_variance >= 90) + 1
print(f"\nNumber of components needed to reach 90% energy: {k}")

#Cumulative variance plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='b',
linewidth=2)
plt.axhline(y=90, color='r', linestyle='--', label='90% Variance Threshold')
plt.title("Cumulative Variance Plot")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Retained (%)")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#Reconstruction Error Calculation
U_k= U_Z[:, :k]
Y_k = U_k.T @ Z
Z_k = U_k @ Y_k
reconstruction_error = np.sum(eigenvalues[k:])/np.sum(eigenvalues)
print(f"\nRelative squared reconstruction error: {reconstruction_error:.6f}")

### PROBLEM 3: ORDINARY LEAST SQUARES ###

X_ols, b_ols, encoded_ols, metadata_ols = pull_dataset()
b_ols = b_ols.ravel()

dev_idx = np.array(metadata_ols["development"], dtype=int)
test_idx = np.array(metadata_ols["test"], dtype=int)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def standardize_train_eval(X_train, X_eval):
    mu = X_train.mean(axis=0)
    scale = X_train.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    return (X_train - mu) / scale, (X_eval - mu) / scale, mu, scale


# 3A. Five-fold grouped cross-validation
ols_sse = baseline_sse = 0.0
validation_n = 0

print("\n--- Problem 3: Ordinary Least Squares ---")

for fold_number, fold in enumerate(metadata_ols["folds"], start=1):
    train_idx = np.array(fold["train"], dtype=int)
    val_idx = np.array(fold["validation"], dtype=int)

    X_train, X_val = encoded_ols[train_idx], encoded_ols[val_idx]
    b_train, b_val = b_ols[train_idx], b_ols[val_idx]

    Z_train, Z_val, _, _ = standardize_train_eval(X_train, X_val)

    A_train = np.column_stack((Z_train, np.ones(len(train_idx))))
    A_val = np.column_stack((Z_val, np.ones(len(val_idx))))

    beta_ols = np.linalg.pinv(A_train) @ b_train
    pred_val_ols = A_val @ beta_ols
    pred_val_baseline = np.full(len(val_idx), b_train.mean())

    ols_sse += np.sum((pred_val_ols - b_val)**2)
    baseline_sse += np.sum((pred_val_baseline - b_val)**2)
    validation_n += len(val_idx)

    print(
        f"Fold {fold_number}: "
        f"OLS RMSE = {rmse(b_val, pred_val_ols):.4f} dB | "
        f"Baseline RMSE = {rmse(b_val, pred_val_baseline):.4f} dB"
    )

ols_validation_rmse = np.sqrt(ols_sse / validation_n)
baseline_validation_rmse = np.sqrt(baseline_sse / validation_n)


# 3B. Final fit using all development data
X_dev, X_test = encoded_ols[dev_idx], encoded_ols[test_idx]
b_dev, b_test = b_ols[dev_idx], b_ols[test_idx]

Z_dev, Z_test, mu_dev, scale_dev = standardize_train_eval(X_dev, X_test)

A_dev = np.column_stack((Z_dev, np.ones(len(dev_idx))))
A_test = np.column_stack((Z_test, np.ones(len(test_idx))))

# Numerical rank
s_A = np.linalg.svd(A_dev, compute_uv=False)
tau_A = np.finfo(float).eps * max(A_dev.shape) * s_A[0]
rank_A = int(np.sum(s_A > tau_A))

# Final OLS model
beta_final_ols = np.linalg.pinv(A_dev) @ b_dev
pred_dev_ols = A_dev @ beta_final_ols
pred_test_ols = A_test @ beta_final_ols

# Baseline
baseline_value = b_dev.mean()
pred_dev_baseline = np.full(len(dev_idx), baseline_value)
pred_test_baseline = np.full(len(test_idx), baseline_value)

# RMSE
ols_training_rmse = rmse(b_dev, pred_dev_ols)
ols_test_rmse = rmse(b_test, pred_test_ols)
baseline_training_rmse = rmse(b_dev, pred_dev_baseline)
baseline_test_rmse = rmse(b_test, pred_test_baseline)


# Results
print(f"\nA dimensions: {A_dev.shape}")
print(f"Numerical rank of A: {rank_A}")
print("Feature order: [frequency, angle, chord, velocity, displacement thickness, intercept]")
print("OLS coefficients:", beta_final_ols)

print("\nRMSE comparison (dB):")
print(f"{'Model':<12}{'Training':>12}{'Validation':>15}{'Test':>12}")
print(f"{'Baseline':<12}{baseline_training_rmse:>12.4f}{baseline_validation_rmse:>15.4f}{baseline_test_rmse:>12.4f}")
print(f"{'OLS':<12}{ols_training_rmse:>12.4f}{ols_validation_rmse:>15.4f}{ols_test_rmse:>12.4f}")


# 3C. Diagnostic plots
residuals_test = pred_test_ols - b_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(b_test, pred_test_ols, alpha=0.6)
plot_min = min(b_test.min(), pred_test_ols.min())
plot_max = max(b_test.max(), pred_test_ols.max())
axes[0].plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--")
axes[0].set(
    xlabel="Observed sound pressure level (dB)",
    ylabel="Predicted sound pressure level (dB)",
    title="OLS: Observed vs Predicted"
)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(pred_test_ols, residuals_test, alpha=0.6)
axes[1].axhline(0, linestyle="--")
axes[1].set(
    xlabel="Predicted sound pressure level (dB)",
    ylabel="Residual: predicted - observed (dB)",
    title="OLS: Residual vs Predicted"
)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
