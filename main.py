"""ME 8813 Midterm, Group 2: NASA Airfoil Self-Noise.

Problems 1-4: SVD, PCA, ordinary least squares (OLS) and ridge regression.
Run from the repository root:  python main.py
Numerical results are printed to the console. Figures (PNG) and the data
behind them (CSV) are written to figures/; the report plots use these CSV files.
"""
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
    # Standardize each input ROW of X (rows = variables, columns = observations):
    # Z_ji = (X_ji - mean_j) / s_j, where s_j is the sample standard deviation (ddof=1,
    # i.e. divide by m-1). After this step every input has mean 0 and variance 1,
    # so inputs measured in Hz, degrees, meters and m/s become comparable.

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

import os
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

X_full, b, encoded, metadata = pull_dataset()

# Optional source-data check (handout p. 5): compare the loader's X and b with
# the original UCI file airfoil_self_noise.dat (no header; columns = 5 inputs
# followed by the response). Place the file in data/ to run the comparison.
SOURCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "airfoil_self_noise.dat")
if os.path.exists(SOURCE_FILE):
    source = np.loadtxt(SOURCE_FILE)
    X_check, b_check = source[:, :5].T, source[:, 5].reshape(-1, 1)
    print(f"Source file shape: {source.shape}; nonfinite values: {(~np.isfinite(source)).sum()}")
    print(f"X shapes equal: {X_check.shape == X_full.shape}, "
          f"X match: {X_check.shape == X_full.shape and np.allclose(X_check, X_full, rtol=1e-10, atol=1e-10)}")
    print(f"b shapes equal: {b_check.shape == b.shape}, "
          f"b match: {b_check.shape == b.shape and np.allclose(b_check, b, rtol=1e-10, atol=1e-10)}")
else:
    print("Source-data check skipped: data/airfoil_self_noise.dat not found")

# Problems 1-2 use development observations only (handout p. 7 and Group 2 checklist item 5)
dev_idx = np.array(metadata["development"], dtype=int)
X = X_full[:, dev_idx]

### PROBLEM 1: SVD OF THE INPUT DATA ###

# do economy SVD
# X = U S V^T with U (n x n), S = the n singular values (largest first) and V^T (n x m).
# full_matrices=False gives the economy SVD (no unused columns of V).
Z, x_mean, x_std = standardize(X)
U_X, S_X, Vt_X = np.linalg.svd(X, full_matrices=False)
U_Z, S_Z, Vt_Z = np.linalg.svd(Z, full_matrices=False) # technically PCA but whatever
n, m = X.shape

print(f"Descriptive SVD/PCA on development data: X is {n} x {m} (n x m)")
print(f"m: {m}, n: {n}")

# find singularity cutoff, rank, and condition number
# (handout p. 8: singular values at or below tau = eps * max(m, n) * sigma_1 count as zero;
#  condition number = largest / smallest nonzero singular value)
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

input_names = list(metadata["names"])
print(f"Singular values of X: {S_X}")
print(f"Singular values of Z: {S_Z}")
print("First left singular vector of X (u_1): " +
      ", ".join(f"{nm}={v:+.4e}" for nm, v in zip(input_names, U_X[:, 0])))

# Engineering question 1: smallest-singular-value direction of Z (handout p. 8 and p. 11 hint)
# u_n^T z_i = sum_j u_jn (X_ji - mu_j)/s_j ; since u_n^T Z = sigma_n v_n^T,
# its sample standard deviation over observations is sigma_n / sqrt(m-1).
# A small value means this weighted sum of standardized inputs hardly varies.
# (The overall sign of a singular vector is arbitrary.)
u_n = U_Z[:, -1]
weak_sum = u_n @ Z
print(f"\nSmallest singular value of Z: sigma_n = {S_Z[-1]:.4f}")
print("Smallest-singular-value direction u_n: " +
      ", ".join(f"{nm}={v:+.4f}" for nm, v in zip(input_names, u_n)))
print(f"u_n^T z_i over development observations: mean = {weak_sum.mean():.2e}, "
      f"std = {weak_sum.std(ddof=1):.4f} (each standardized input has std 1), "
      f"sigma_n/sqrt(m-1) = {S_Z[-1]/np.sqrt(m-1):.4f}")
corr = (Z @ Z.T) / (m - 1)
print("Correlation matrix R of the development inputs:")
print(np.array2string(corr, precision=3, suppress_small=True))

# Figure for the report (Fig. 1): singular values relative to sigma_1 and components of u_n.
# The CSV file stores exactly the numbers that are plotted.
np.savetxt(os.path.join(FIG_DIR, "singular_values.csv"),
           np.column_stack((np.arange(1, n + 1), S_X, S_Z, S_X / S_X[0], S_Z / S_Z[0], u_n)),
           delimiter=",", header="i,sigma_X,sigma_Z,sigma_X/sigma_1,sigma_Z/sigma_1,u_n", comments="")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogy(range(1, n + 1), S_X / S_X[0], 'o-', label=f"X (physical units), cond = {c_X:.2e}")
plt.semilogy(range(1, n + 1), S_Z / S_Z[0], 's-', label=f"Z (standardized), cond = {c_Z:.2f}")
plt.xticks(range(1, n + 1))
plt.xlabel("Index i")
plt.ylabel(r"$\sigma_i / \sigma_1$")
plt.title("Relative singular values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.subplot(1, 2, 2)
plt.bar(["frequency", "angle", "chord", "velocity", r"$\delta^*$"], u_n)
plt.axhline(0, color='k', linewidth=0.8)
plt.ylabel(r"Component of $u_n$")
plt.title(rf"Smallest-singular-value direction of Z ($\sigma_n$ = {S_Z[-1]:.2f})")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "svd_spectrum_and_un.png"), dpi=200)
plt.show()


### PROBLEM 2: PCA ###

# Cumulative retained variance calculations
# Eigenvalues of the correlation matrix R = Z Z^T/(m-1) follow from the SVD of Z:
# lambda_j = sigma_j^2/(m-1) (derivation A in the report). Each lambda_j is the
# variance of the inputs along principal direction j (column j of U_Z).
eigenvalues = S_Z**2/(m - 1)
print(f"\nEigenvalues: {eigenvalues}")
explained_variance = 100*eigenvalues/np.sum(eigenvalues)   # % of variance of each component
cumulative_variance = 100*np.cumsum(eigenvalues)/np.sum(eigenvalues)

#Compute k=number of components needed to reach 90% energy
k = np.argmax(cumulative_variance >= 90) + 1
print(f"\nNumber of components needed to reach 90% energy: {k}")

#Cumulative variance plot
# (Fig. 2 of the report: bars = variance of each component, line = cumulative variance)
np.savetxt(os.path.join(FIG_DIR, "pca_variance.csv"),
           np.column_stack((np.arange(1, n + 1), eigenvalues, explained_variance, cumulative_variance)),
           delimiter=",", header="k,lambda,explained_percent,cumulative_percent", comments="")
plt.figure(figsize=(8, 4))
plt.bar(range(1, n + 1), explained_variance, color='lightsteelblue', label='Variance of component j')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='b',
linewidth=2, marker='o', label='Cumulative retained variance')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.axhline(y=90, color='r', linestyle='--', label='90% Variance Threshold')
plt.axvline(x=k, color='gray', linestyle=':', label=f'k = {k}')
plt.title("Cumulative retained variance (development data, standardized inputs)")
plt.xlabel("Number of Principal Components")
plt.ylabel("Variance (%)")
plt.legend(loc='center right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cumulative_variance.pdf"))
plt.savefig(os.path.join(FIG_DIR, "cumulative_variance.png"), dpi=200)
plt.show()

#Reconstruction Error Calculation
# Keep the first k principal directions: scores Y_k = U_k^T Z (k x m) and
# reconstruction Z_k = U_k Y_k. The relative squared error equals the fraction
# of variance in the discarded components (sum of the discarded eigenvalues).
U_k= U_Z[:, :k]
Y_k = U_k.T @ Z
Z_k = U_k @ Y_k
reconstruction_error = np.sum(eigenvalues[k:])/np.sum(eigenvalues)
print(f"\nCumulative retained variance (%): {np.round(cumulative_variance, 2)}")
print(f"\nRelative squared reconstruction error: {reconstruction_error:.6f}")
# direct check from the reconstructed matrix: ||Z - Z_k||_F^2 / ||Z||_F^2
direct_error = np.linalg.norm(Z - Z_k, 'fro')**2 / np.linalg.norm(Z, 'fro')**2
print(f"Direct check ||Z - Z_k||_F^2 / ||Z||_F^2: {direct_error:.6f}")

### PROBLEM 3: ORDINARY LEAST SQUARES ###

def calc_rmse(b_true, b_pred):
    # compute root mean squared error (Eq. 2)
    # RMSE = sqrt( mean( (predicted - measured)^2 ) ), in dB like the response
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
test_idx = np.array(metadata["test"], dtype=int)

# count configurations and observations per partition using pull_configurations
all_configs = pull_configurations(X_full)
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
    # beta = A^+ b minimizes ||A beta - b||^2. numpy's rcond is relative to the largest
    # singular value, so the cutoff equals the handout tolerance eps * max(m, p) * sigma_1.
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

# pooled CV RMSE: total squared error over ALL validation observations, then one square root
# (not the average of the five fold RMSEs)
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
feature_names = list(metadata["names"]) + ["intercept"]

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
plt.savefig(os.path.join(FIG_DIR, "ols_diagnostics.pdf"))
plt.savefig(os.path.join(FIG_DIR, "ols_diagnostics.png"), dpi=200)
plt.show()

# 3D. Residual trends with frequency and angle of attack (supports the Group 2 closing statement)
freq_test = X_full[0, test_idx]
angle_test = X_full[1, test_idx]
np.savetxt(os.path.join(FIG_DIR, "test_predictions.csv"),
           np.column_stack((b_test, b_hat_test_ols, residuals_test, freq_test, angle_test)),
           delimiter=",", fmt="%.6f",
           header="observed_dB,predicted_ols_dB,residual_dB,frequency_Hz,angle_deg", comments="")
print(f"\nTest residual summary: mean = {residuals_test.mean():.4f} dB, std = {residuals_test.std(ddof=1):.4f} dB")
print("Mean OLS test residual (predicted - observed) by frequency band:")
band_rows = []   # saved for the red steps in Fig. 4b of the report
for lo, hi in [(0, 1000), (1000, 3000), (3000, 8000), (8000, 25000)]:
    sel = (freq_test >= lo) & (freq_test < hi)
    if sel.any():
        band_rows.append([lo, hi, sel.sum(), residuals_test[sel].mean()])
        print(f"  {lo:>5}-{hi:<5} Hz: n = {sel.sum():>3}, mean = {residuals_test[sel].mean():+.3f} dB, "
              f"RMSE = {np.sqrt(np.mean(residuals_test[sel]**2)):.3f} dB")
np.savetxt(os.path.join(FIG_DIR, "residual_bands.csv"), np.array(band_rows), delimiter=",",
           header="f_low_Hz,f_high_Hz,n_obs,mean_residual_dB", comments="")
print("Mean OLS test residual by angle of attack band:")
for lo, hi in [(0, 5), (5, 10), (10, 25)]:
    sel = (angle_test >= lo) & (angle_test < hi)
    if sel.any():
        print(f"  {lo:>2}-{hi:<2} deg: n = {sel.sum():>3}, mean = {residuals_test[sel].mean():+.3f} dB, "
              f"RMSE = {np.sqrt(np.mean(residuals_test[sel]**2)):.3f} dB")

plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
plt.scatter(freq_test, residuals_test, color='b', alpha=0.6)
plt.xscale('log')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("Frequency (Hz, log scale)")
plt.ylabel("Residual: Predicted - Observed (dB)")
plt.title("OLS test residuals vs. frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.subplot(1, 2, 2)
plt.scatter(angle_test, residuals_test, color='b', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("Angle of attack (deg)")
plt.ylabel("Residual: Predicted - Observed (dB)")
plt.title("OLS test residuals vs. angle of attack")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "residuals_vs_inputs.png"), dpi=200)
plt.show()


### PROBLEM 4: RIDGE REGRESSION ###

def fit_ridge(A, b, lam):
    # Calculates the regression coefficients :
    # A: design matrix, contains input variables and intercept column.
    # b: vector of measured sound pressure levels
    # lam: regularization parameter (lambda)
    m, p = A.shape

    if lam == 0:    # Calculates the case for lambda = 0 so it's basically Ordinary Least Squares (OLS)
        # same pseudoinverse and tolerance as the OLS code in Problem 3
        rcond = np.finfo(float).eps * max(A.shape)
        return np.linalg.pinv(A, rcond=rcond) @ b

    gamma = np.eye(p)   #Id matrix
    gamma[-1, -1] = 0   # Removes bottom right "1", this way the intercept isn't penalized

    # Normal equation of (1/m)||b - A beta||^2 + lam ||Gamma beta||^2, with m = number of
    # rows of A (training observations): (A^T A + m lam Gamma^T Gamma) beta = A^T b
    M = A.T @ A + m * lam * (gamma.T @ gamma) #solves the linear system M*beta=A^t*b
    return np.linalg.solve(M, A.T @ b)


# 4A. Select lambda using grouped cross-validation
# Every candidate is scored with the same five development folds and the same pooled RMSE
# as in Problem 3. The held-out test set is NOT used to choose lambda.

lambdas = np.concatenate(([0.0], np.logspace(-6, 6, 49))) #Test 50 values of lambda, logspace to cover larger span of lambda without too many values

best_lambda = None
best_cv_rmse = np.inf
cv_rmse_values = []   # pooled CV RMSE of every candidate (for the validation curve, Fig. 4a)

for lam in lambdas:
    total_sse = 0.0
    total_count = 0

    for fold in metadata["folds"]:      #Goes over all folds within the datasets
        train_idx = np.array(fold["train"], dtype=int)
        val_idx = np.array(fold["validation"], dtype=int)

        X_train = encoded[train_idx]
        X_val = encoded[val_idx]
        b_train = b_flat[train_idx]
        b_val = b_flat[val_idx]

        # encoded: all encoded input features (all 1503 rows; train_idx/val_idx select development rows).
        # b_flat: corresponding measured sound pressure levels.
        # train_idx: indices of training observations.
        # val_idx: indices of validation observations.

        # Training-only standardization
        A_train, A_val, _, _ = build_design_matrix(X_train, X_val)

        #standardizes the features using the training dataset's mean and standard deviation:

        beta = fit_ridge(A_train, b_train, lam)
        predictions = A_val @ beta

        # calculates Ridge coefficients using training observations.
        # Predicts the validation observations:
        # b_val_hat = A_val*beta_hat
        # Validation observations NOT USED to calculate Beta, ONLY model

        total_sse += np.sum((predictions - b_val)**2)
        total_count += len(b_val)

    cv_rmse = np.sqrt(total_sse / total_count)
    cv_rmse_values.append(cv_rmse)

    # Calculates Root Mean Square

    if cv_rmse < best_cv_rmse:
        best_cv_rmse = cv_rmse
        best_lambda = lam

    # If error from new lambda lower than previous highscorer lambda -> new lambda selected

# Validation curve: pooled grouped-CV RMSE versus lambda (lambda = 0 is drawn as the OLS line)
cv_rmse_values = np.array(cv_rmse_values)
np.savetxt(os.path.join(FIG_DIR, "validation_curve.csv"),
           np.column_stack((lambdas, cv_rmse_values)), delimiter=",",
           header="lambda,pooled_cv_rmse_dB", comments="")
plt.figure(figsize=(8, 4))
plt.semilogx(lambdas[1:], cv_rmse_values[1:], 'o-', markersize=4, label=r'Ridge, $\lambda > 0$')
plt.axhline(cv_rmse_values[0], color='g', linestyle='--', label=r'OLS ($\lambda$ = 0)')
plt.axhline(base_cv_rmse, color='gray', linestyle=':', label='Baseline (training mean)')
plt.plot(best_lambda, best_cv_rmse, 'r*', markersize=14, label=rf'Selected $\lambda$ = {best_lambda:.3g}')
plt.xlabel(r"$\lambda$ (log scale)")
plt.ylabel("Pooled grouped-CV RMSE (dB)")
plt.title("Ridge validation curve")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ridge_validation_curve.png"), dpi=200)
plt.show()

# 4B. Refit on all development data

beta_ridge = fit_ridge(A_dev, b_dev, best_lambda) # Use whole development dataset as training now that best lambda was found (test set still unused)

ridge_train_pred = A_dev @ beta_ridge # Run ridge reg on training sets
ridge_test_pred = A_test @ beta_ridge # Run ridge reg on test sets

ridge_train_rmse = calc_rmse(b_dev, ridge_train_pred) # RMSE value for training (development) data
ridge_test_rmse = calc_rmse(b_test, ridge_test_pred)  # Test RMSE value for testing data

# Effect of the penalty on the standardized coefficients (development fit only, Fig. 5).
# lambda = 1 is NOT a selected model; it only illustrates stronger shrinkage.
beta_ridge_1 = fit_ridge(A_dev, b_dev, 1.0)
print("\nStandardized coefficients (dB): OLS | ridge (selected lambda) | ridge (lambda = 1, illustration)")
for name, c0, c1, c2 in zip(feature_names, beta_ls, beta_ridge, beta_ridge_1):
    print(f"  {name:<24}: {c0:>9.4f} | {c1:>9.4f} | {c2:>9.4f}")
np.savetxt(os.path.join(FIG_DIR, "coefficients.csv"),
           np.column_stack((beta_ls, beta_ridge, beta_ridge_1)), delimiter=",",
           header="ols,ridge_selected,ridge_lambda_1 (rows: " + ", ".join(feature_names) + ")", comments="")
x_pos = np.arange(len(input_names))
plt.figure(figsize=(8, 4))
plt.bar(x_pos - 0.27, beta_ls[:-1], width=0.27, label=r'OLS ($\lambda$ = 0)')
plt.bar(x_pos, beta_ridge[:-1], width=0.27, label=rf'Ridge, selected $\lambda$ = {best_lambda:.3g}')
plt.bar(x_pos + 0.27, beta_ridge_1[:-1], width=0.27, label=r'Ridge, $\lambda$ = 1 (illustration)')
plt.xticks(x_pos, ["frequency", "angle", "chord", "velocity", r"$\delta^*$"])
plt.axhline(0, color='k', linewidth=0.8)
plt.ylabel("Standardized coefficient (dB)")
plt.title("Slope coefficients fitted on the development data")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ridge_coefficients.png"), dpi=200)
plt.show()


# 4C. Compare with OLS and baseline

print("Part 4: Ridge Regression")
print(f"Selected lambda: {best_lambda:.6g}") # prints lambdas with six significant digits

print(f"\n{'Model':<12}{'Training':>12}{'Validation':>14}{'Test':>12}")
print("-" * 50)

for name, train, val, test in [
    ("Baseline", base_train_rmse, base_cv_rmse, base_test_rmse),
    ("OLS", ols_train_rmse, ols_cv_rmse, ols_test_rmse),
    ("Ridge", ridge_train_rmse, best_cv_rmse, ridge_test_rmse)
]:
    print(f"{name:<12}{train:>12.4f}{val:>14.4f}{test:>12.4f}")

difference = ols_test_rmse - ridge_test_rmse #CALCULATES THE DIFFERENCE IN PERFORMANCE

print(f"\nOLS test RMSE - Ridge test RMSE: {difference:.4f} dB")
print(f"Pooled CV RMSE at lambda = 1e6 (largest candidate): {cv_rmse_values[-1]:.4f} dB")

