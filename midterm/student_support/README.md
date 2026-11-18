# Student support for the ME8813 midterm

These files provide data import, categorical indicators and fixed split assignments. They contain no fitted coefficients, chosen penalties, plots or reference errors. You must implement and interpret SVD, PCA, OLS and ridge as specified in the handout.

Python: install NumPy, then call `X,b,encoded,meta=load_project(1)` from `load_project.py`. MATLAB: call `[X,b,encoded,meta]=load_project(1)`. Replace 1 by your project number (1–5). Examples of training-only preprocessing are included as comments in each loader.

`X` has n quantitative input rows and m observation columns. `b` is m-by-1. `encoded` has observations in rows, with the original quantitative inputs followed by the named zero/one indicator columns. Its columns have NOT been standardized. Add the intercept last after fitting training-only standardization. Names and units are recorded in the metadata.

Use `meta.folds` for validation in projects 2–5. Project 1 uses `meta.outer`: tune inside each outer split using its `inner` folds, then assess on its `test` indices. Only afterwards use `meta.folds` for final all-data tuning. Project 3's folds preserve chronological order. The `development` and `test` lists specify the final refit and assessment partitions for projects 2–5.

JSON indices and Python indices are zero-based. The MATLAB loader converts observation indices to one-based values. Group identifiers are labels, not indices. Do not reorder observations without updating every split.

The numeric inputs are extracted from the exact source files specified in the handout. Still inspect the source documentation, record units and report exclusions. You may use these files to avoid spending assessment time on spreadsheet import or split construction. Compare them with your source import if you download the data yourself. The provided categorical encoding is a supplied convenience, not a regression solution.

For PCA use the Chapter 2 notation: correlation matrix R, PC scores Y_k and eigenvalues lambda_j. The unsubscripted lambda denotes ridge strength. Do not treat high input variance as evidence of predictive or causal importance.
