# Concrete Carbonation Depth: Symbolic Regression via Operon

Supporting code for a manuscript submitted to the Asian Journal of Civil
Engineering (Springer), which discovers a closed-form equation (Equation 1)
for carbonation depth in concrete using genetic programming (Operon C++, via
[pyoperon](https://github.com/heal-research/pyoperon)), and benchmarks it
against Random Forest, XGBoost, CatBoost and simple regression baselines on a
study-grouped train/test split.

The training data itself is not redistributed here; it is drawn from the
database compiled in Liu et al. (2025), cited in the paper's Data Availability
statement. Every script below expects `concrete carbonation data.xls` (same
column names used in the paper: w/c, w/b, FA, CO2, RH, T, exposure time,
curing time, `References`) to sit next to it.

## Layout

- `tree_operon_model.py`: Equation (1) itself (`EQ1`) and the input envelope
  (`ENVELOPE`) used for extrapolation warnings. Imported by everything else.
- `search1_global_parallel.py`: the Operon search that discovered Equation (1)
  (Pareto search over the full dataset, parallelized across processes).
- `build_study_grouped_split.py`: builds the study-grouped 80/20 train/test
  split (grouping by source study via the `References` column) used
  everywhere else, so near-duplicate specimens from the same study never land
  on both sides.
- `compute_single_coef_baseline.py`, `compute_ols_baseline.py`: the two
  simple baselines in Table 4 (single-coefficient `X = k*sqrt(t)` fit, and
  multivariable OLS on the 8 raw features).
- `thicken_eq1_task1_blackbox_baselines.py`,
  `thicken_eq1_task1b_tuned_blackbox.py`: self-trained Random Forest,
  XGBoost and CatBoost baselines (untuned and tuned) on the same split.
- `thicken_eq1_task2_robustness.py`: the 50-independent-split robustness
  check on Equation (1) (no re-fitting, just re-evaluation).
- `thicken_eq1_task3_sobol.py`: Sobol global sensitivity analysis of
  Equation (1) (requires [SALib](https://github.com/SALib/SALib)).
- `compute_shap_eq1.py`: exact Shapley attribution for Equation (1).
- `eq1_only_model.py`, `eq1_only_app.py`: the desktop prediction app (PySide6)
  described in the paper, wired to Equation (1) directly.
- `CarbonationPredictorEq1.exe`: standalone Windows build of the app above
  (PyInstaller, `--onefile --windowed`), no Python install required.
- `results/`: the exact JSON/CSV outputs these scripts produced, as reported
  in the paper's tables and figures.

## Running

Each script hardcodes `ROOT = r'd:\Research\TienThe'` near the top (the
original development path). Either edit that line to point at wherever you
place this repo and the `.xls` file, or place this repo at that same path.

Dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `catboost`,
`pyoperon`, `SALib`, `PySide6`.

## License

MIT, see `LICENSE`.
