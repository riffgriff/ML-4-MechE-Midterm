# ME 8813 Midterm - Group 2 (NASA Airfoil Self-Noise)

Python implementation of Problems 1-4 (SVD, PCA, OLS, ridge regression) using the
instructor-provided data and splits in `student_support/`.

## Run

```
pip install -r requirements.txt
python main.py                    # interactive figure windows
MPLBACKEND=Agg python main.py     # headless; figures are still saved
```

Run from the repository root. Numerical results are printed to the console. Figures
(PNG) and the data behind them (CSV) are written to `figures/`:

| File | Content |
|---|---|
| `svd_spectrum_and_un.png`, `singular_values.csv` | Problem 1: relative singular values of X and Z, direction `u_n` |
| `cumulative_variance.png`, `pca_variance.csv` | Problem 2: variance per component and cumulative retained variance |
| `ols_diagnostics.png`, `residuals_vs_inputs.png`, `test_predictions.csv`, `residual_bands.csv` | Problem 3: test-set predictions, residuals and mean residual per frequency band |
| `ridge_validation_curve.png`, `validation_curve.csv` | Problem 4: pooled grouped-CV RMSE for all 50 lambda candidates |
| `ridge_coefficients.png`, `coefficients.csv` | Problem 4: standardized coefficients of OLS, selected ridge and ridge with lambda = 1 |

`run_output.txt` holds the console output of the verified run. No random numbers are generated: all partitions come from
`student_support/shared/project_2_metadata.json`.

Optional source-data check: download `airfoil_self_noise.dat` from
https://doi.org/10.24432/C5VW2C and place it in `data/`; `main.py` then compares it
with the loader's `X` and `b` using `np.allclose(rtol=1e-10, atol=1e-10)`.

## Functions used
`numpy.linalg.svd` (SVD/PCA, rank, conditioning), `numpy.linalg.pinv` with
`rcond = eps * max(m, p)` (OLS, ridge at lambda = 0), `numpy.linalg.solve`
(ridge normal equations for lambda > 0).
