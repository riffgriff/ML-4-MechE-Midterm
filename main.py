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

def calc_rmse(b_true, b_pred):
    # compute root mean squared error (Eq. 2)
    return np.sqrt(np.mean((b_pred - b_true)**2))


def build_design_matrix(X_train_raw, X_eval_raw):
    # standardize using training subset statistics only (Eq. 1,observations in rows)
    mu = np.mean(X_train_raw, axis=0, keepdims=True)
    std = np.std(X_train_raw - mu, axis=0, keepdims=True, ddof=1)

    # handle constant columns if any exist in the training subset
    const_cols = np.where(std.ravel() == 0)[0]
    if len(const_cols) > 0:
        print(f"Constant column indices detected (scale set to 1.0): {const_cols}")
        std[std == 0] = 1.0

    Z_train_T = (X_train_raw - mu) / std
    Z_eval_T = (X_eval_raw - mu) / std

    # append intercept column of ones last: A = [Z^T, 1_m]
    A_train = np.column_stack((Z_train_T, np.ones(X_train_raw.shape[0])))
    A_eval = np.column_stack((Z_eval_T, np.ones(X_eval_raw.shape[0])))

    return A_train, A_eval, mu.ravel(), std.ravel()


print("\n--- Problem 3: Ordinary Least Squares ---")

# extract zero-based development and held-out test indices
b_flat = b.ravel()
dev_idx = np.array(metadata["development"], dtype=int)
test_idx = np.array(metadata["test"], dtype=int)

# count configurations and observations per partition using pull_configurations
all_configs = pull_configurations(X)
dev_set = set(dev_idx)
test_set = set(test_idx)
n_configs_dev = sum(1 for cfg in all_configs if set(cfg.data_indices).issubset(dev_set))
n_configs_test = sum(1 for cfg in all_configs if set(cfg.data_indices).issubset(test_set))

print(f"Development partition: {len(dev_idx)} observations across {n_configs_dev} configurations")
print(f"Held-out test partition: {len(test_idx)} observations across {n_configs_test} configurations\n")

# 3A. Five-fold grouped cross-validation on development data
ols_val_sse = 0.0
base_val_sse = 0.0
val_obs_count = 0

for fold_num, fold in enumerate(metadata["folds"], start=1):
    train_idx = np.array(fold["train"], dtype=int)
    val_idx = np.array(fold["validation"], dtype=int)

    X_train_fold, X_val_fold = encoded[train_idx], encoded[val_idx]
    b_train_fold, b_val_fold = b_flat[train_idx], b_flat[val_idx]

    A_train_fold, A_val_fold, _, _ = build_design_matrix(X_train_fold, X_val_fold)

    # compute OLS coefficients via Moore-Penrose pseudoinverse with prescribed tolerance
    rcond_fold = eps_mach * max(A_train_fold.shape)
    beta_fold = np.linalg.pinv(A_train_fold, rcond=rcond_fold) @ b_train_fold

    # predictions on validation fold
    b_hat_val_ols = A_val_fold @ beta_fold
    b_hat_val_base = np.full(len(val_idx), np.mean(b_train_fold))

    # accumulate squared residuals for pooled CV RMSE (Eq. 3)
    ols_val_sse += np.sum((b_hat_val_ols - b_val_fold)**2)
    base_val_sse += np.sum((b_hat_val_base - b_val_fold)**2)
    val_obs_count += len(val_idx)

    print(f"Fold {fold_num}: OLS RMSE = {calc_rmse(b_val_fold, b_hat_val_ols):.4f} dB | "
          f"Baseline RMSE = {calc_rmse(b_val_fold, b_hat_val_base):.4f} dB")

ols_cv_rmse = np.sqrt(ols_val_sse / val_obs_count)
base_cv_rmse = np.sqrt(base_val_sse / val_obs_count)

# 3B. Final refit on all development data and evaluation on held-out test set
X_dev, X_test = encoded[dev_idx], encoded[test_idx]
b_dev, b_test = b_flat[dev_idx], b_flat[test_idx]

A_dev, A_test, mu_dev, std_dev = build_design_matrix(X_dev, X_test)

# SVD of A_dev for numerical column rank and condition number
U_A, S_A, Vt_A = np.linalg.svd(A_dev, full_matrices=False)
tau_A = eps_mach * max(A_dev.shape) * S_A[0]
r_A = int((S_A > tau_A).sum())
c_A = S_A[0] / S_A[r_A - 1]

# compute final OLS fit: beta_LS = A^+ b
rcond_dev = eps_mach * max(A_dev.shape)
beta_ls = np.linalg.pinv(A_dev, rcond=rcond_dev) @ b_dev

b_hat_dev_ols = A_dev @ beta_ls
b_hat_test_ols = A_test @ beta_ls

# compute training-mean baseline fit
b_mean_dev = np.mean(b_dev)
b_hat_dev_base = np.full(len(dev_idx), b_mean_dev)
b_hat_test_base = np.full(len(test_idx), b_mean_dev)

# calculate final training and test RMSEs
ols_train_rmse = calc_rmse(b_dev, b_hat_dev_ols)
ols_test_rmse = calc_rmse(b_test, b_hat_test_ols)
base_train_rmse = calc_rmse(b_dev, b_hat_dev_base)
base_test_rmse = calc_rmse(b_test, b_hat_test_base)

# print design matrix diagnostics and fitted coefficients
feature_names = list(metadata.get("input_names", [
    "frequency", "angle", "chord_length", "velocity", "displacement_thickness"
])) + ["intercept"]

print(f"\nDesign Matrix A_dev dimensions (m x p): {A_dev.shape}")
print(f"tau_A: {tau_A:.6e}")
print(f"Numerical column rank of A_dev: {r_A} (full column rank = {r_A == A_dev.shape[1]})")
print(f"Condition number of A_dev: {c_A:.4f}")
print(f"Feature order: {feature_names}")
print("Fitted OLS standardized coefficients (beta_LS):")
for name, coef in zip(feature_names, beta_ls):
    print(f"  {name:<24}: {coef:>10.4f} dB")

print("\nShared Error Table - RMSE Comparison (dB):")
print(f"{'Model':<12}{'Training (Dev)':>16}{'Validation (CV)':>18}{'Test (Held-Out)':>18}")
print("-" * 64)
print(f"{'Baseline':<12}{base_train_rmse:>16.4f}{base_cv_rmse:>18.4f}{base_test_rmse:>18.4f}")
print(f"{'OLS':<12}{ols_train_rmse:>16.4f}{ols_cv_rmse:>18.4f}{ols_test_rmse:>18.4f}")

# 3C. Diagnostic plots on final assessment (held-out test) predictions
# residual definition: r = A * beta - b (predicted minus measured)
residuals_test = b_hat_test_ols - b_test

plt.figure(figsize=(12, 5))

# Panel 1: Observed vs. Predicted
plt.subplot(1, 2, 1)
plt.scatter(b_test, b_hat_test_ols, color='b', alpha=0.6, label='Test Observations')
plot_min = min(b_test.min(), b_hat_test_ols.min())
plot_max = max(b_test.max(), b_hat_test_ols.max())
plt.plot([plot_min, plot_max], [plot_min, plot_max], color='r', linestyle='--', linewidth=2, label='1:1 Ideal Fit')
plt.title("OLS: Observed vs. Predicted (Test Set)")
plt.xlabel("Observed Sound Pressure Level (dB)")
plt.ylabel("Predicted Sound Pressure Level (dB)")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Panel 2: Residual vs. Predicted
plt.subplot(1, 2, 2)
plt.scatter(b_hat_test_ols, residuals_test, color='b', alpha=0.6, label='Test Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.title("OLS: Residual vs. Predicted (Test Set)")
plt.xlabel("Predicted Sound Pressure Level (dB)")
plt.ylabel("Residual: Predicted - Observed (dB)")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
