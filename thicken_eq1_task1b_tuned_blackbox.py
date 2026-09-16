"""
Address reviewer M2: Table 4b's RF/XGBoost/CatBoost used untuned library
defaults, an asymmetric comparison against Eq.1's 10,000-trial search.
Runs a randomized hyperparameter search (150 configs per model) on a
study-grouped internal train/validation split (never touching the real
473-sample test set for selection), then refits the winning config on the
full train partition and reports its real test performance -- same
selection discipline used for Eq.1 itself (search on held-out validation,
report on untouched test).

Full-feature (8 raw inputs) scenario only, since that is the scenario cited
in the Abstract/Conclusion's "beats RF/XGBoost/CatBoost" claim.
"""
import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

ROOT = r'd:\Research\TienThe'
N_CONFIGS = 150

with open(f'{ROOT}\\study_grouped_split.json', encoding='utf-8') as f:
    split = json.load(f)
test_idx = set(split['test_row_indices'])

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
df['ref_group'] = df['References'].ffill()
rename_dict = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
               'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
               'Curing time (d)': 't_cure'}
target_col = [c for c in df.columns if 'Carbonation depth' in c][0]
df2 = df.rename(columns=rename_dict).copy()
df2['Target'] = df2[target_col]
FULL_FEATURES = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
df2[FULL_FEATURES] = df2[FULL_FEATURES].fillna(df2[FULL_FEATURES].median())
df2['Target'] = df2['Target'].fillna(df2['Target'].median())
df2['ref_group'] = df['ref_group']

is_test = df2.index.isin(test_idx)
train, test = df2[~is_test], df2[is_test]

# Study-grouped internal validation split (greedy smallest-study-first, ~18% of train)
counts = train['ref_group'].value_counts().sort_values(ascending=True)
total = len(train)
val_studies, cum = [], 0
for study, n in counts.items():
    val_studies.append(study)
    cum += n
    if cum / total >= 0.18:
        break
val_mask = train['ref_group'].isin(val_studies)
tune_fit, tune_val = train[~val_mask], train[val_mask]
print(f'Tune-fit n={len(tune_fit)}, tune-val n={len(tune_val)} ({len(tune_val)/total*100:.1f}% of train)')

X_tune_fit, y_tune_fit = tune_fit[FULL_FEATURES].values, tune_fit['Target'].values
X_tune_val, y_tune_val = tune_val[FULL_FEATURES].values, tune_val['Target'].values
X_train_full, y_train_full = train[FULL_FEATURES].values, train['Target'].values
X_test, y_test = test[FULL_FEATURES].values, test['Target'].values

rng = random.Random(2026)


def sample_rf():
    return dict(
        n_estimators=rng.choice([200, 300, 400, 500, 700]),
        max_depth=rng.choice([None, 5, 8, 12, 16, 20]),
        min_samples_leaf=rng.choice([1, 2, 4, 8]),
        max_features=rng.choice(['sqrt', 'log2', 0.5, 0.8, 1.0]),
        random_state=42, n_jobs=-1,
    )


def sample_xgb():
    return dict(
        n_estimators=rng.choice([200, 300, 400, 600, 800]),
        max_depth=rng.choice([3, 4, 5, 6, 8]),
        learning_rate=rng.choice([0.01, 0.03, 0.05, 0.08, 0.1]),
        subsample=rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        colsample_bytree=rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        min_child_weight=rng.choice([1, 2, 4, 8]),
        reg_lambda=rng.choice([0.1, 1.0, 3.0, 10.0]),
        random_state=42,
    )


def sample_cat():
    return dict(
        iterations=rng.choice([300, 500, 700, 900]),
        depth=rng.choice([4, 5, 6, 7, 8]),
        learning_rate=rng.choice([0.01, 0.03, 0.05, 0.08, 0.1]),
        l2_leaf_reg=rng.choice([1, 3, 5, 10]),
        random_state=42, verbose=False,
    )


MODELS = {
    'RandomForest': (RandomForestRegressor, sample_rf),
    'XGBoost': (XGBRegressor, sample_xgb),
    'CatBoost': (CatBoostRegressor, sample_cat),
}

results = {}
for name, (cls, sampler) in MODELS.items():
    best_val_rmse, best_params = np.inf, None
    for _ in range(N_CONFIGS):
        params = sampler()
        model = cls(**params)
        model.fit(X_tune_fit, y_tune_fit)
        pred_val = model.predict(X_tune_val)
        rmse = float(np.sqrt(mean_squared_error(y_tune_val, pred_val)))
        if rmse < best_val_rmse:
            best_val_rmse, best_params = rmse, params
    print(f'{name}: best tune-val RMSE={best_val_rmse:.4f}, params={best_params}')

    final_model = cls(**best_params)
    final_model.fit(X_train_full, y_train_full)
    pred_train = final_model.predict(X_train_full)
    pred_test = final_model.predict(X_test)
    results[name] = {
        'best_params': {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                         for k, v in best_params.items()},
        'tune_val_rmse': best_val_rmse,
        'train_rmse': float(np.sqrt(mean_squared_error(y_train_full, pred_train))),
        'train_mae': float(mean_absolute_error(y_train_full, pred_train)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, pred_test))),
        'test_mae': float(mean_absolute_error(y_test, pred_test)),
        'test_r2': float(r2_score(y_test, pred_test)),
        'n_configs_tried': N_CONFIGS,
    }
    r = results[name]
    print(f'  -> Train RMSE={r["train_rmse"]:.4f} MAE={r["train_mae"]:.4f} | '
          f'Test RMSE={r["test_rmse"]:.4f} MAE={r["test_mae"]:.4f} R2={r["test_r2"]:.4f}')

with open(f'{ROOT}\\task1b_tuned_blackbox.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('\nWrote task1b_tuned_blackbox.json')
