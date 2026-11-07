# ME 8813 Midterm - Group 2 (NASA Airfoil Self-Noise)

Python implementation of Problems 1-4 (SVD, PCA, OLS, ridge regression) using the
instructor-provided data and splits in `student_support/`.

## Run

```
pip install -r requirements.txt
python main.py                    # interactive figure windows
MPLBACKEND=Agg python main.py     # headless; figures are still saved
```

Run from the repository root. Numerical results are printed to the console; figures
are written to `figures/`. No random numbers are generated: all partitions come from
`student_support/shared/project_2_metadata.json`.

Optional source-data check: download `airfoil_self_noise.dat` from
https://doi.org/10.24432/C5VW2C and place it in `data/`; `main.py` then compares it
with the loader's `X` and `b` using `np.allclose(rtol=1e-10, atol=1e-10)`.

## Functions used
`numpy.linalg.svd` (SVD/PCA, rank, conditioning), `numpy.linalg.pinv` with
`rcond = eps * max(m, p)` (OLS, ridge at lambda = 0), `numpy.linalg.solve`
(ridge normal equations for lambda > 0).
