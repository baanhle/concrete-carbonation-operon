"""
Task 2: robustness of the FIXED Eq.1 formula (no re-search, no re-selection)
across 50 independently-drawn study-grouped 80/20 splits. Mirrors the
accepted UHPC paper's 50-random-seed Table 4: same equation, many
partitions, report mean +/- std. Answers "is 6.8658mm a stable number or a
lucky draw of which studies landed in test."
"""
import json
import random
import numpy as np
import pandas as pd

from tree_operon_model import EQ1

ROOT = r'd:\Research\TienThe'
N_REPEATS = 50
TEST_FRAC = 0.20

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
df['ref_group'] = df['References'].ffill()
rename_dict = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
               'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
               'Curing time (d)': 't_cure'}
target_col = [c for c in df.columns if 'Carbonation depth' in c][0]
df2 = df.rename(columns=rename_dict).copy()
df2['Target'] = df2[target_col]
feature_cols = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
df2[feature_cols] = df2[feature_cols].fillna(df2[feature_cols].median())
df2['Target'] = df2['Target'].fillna(df2['Target'].median())
df2['ref_group'] = df['ref_group']
df2['X1'] = df2['w/c']; df2['X2'] = df2['w/b']; df2['X5'] = df2['RH']
df2['X6'] = df2['T']; df2['X7'] = df2['sqrt_t']; df2['X10'] = df2['w/b'] * df2['CO2']

studies = df2['ref_group'].unique().tolist()
env_funcs = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, '__builtins__': {}}


def eval_eq1(frame):
    env = dict(env_funcs)
    for col in ['X1', 'X2', 'X5', 'X6', 'X7', 'X10']:
        env[col] = frame[col].values
    with np.errstate(all='ignore'):
        return np.asarray(eval(EQ1, env), dtype=float)  # noqa: S307 -- fixed formula string


rmses, maes, ns = [], [], []
for rep in range(N_REPEATS):
    rng = random.Random(1000 + rep)
    shuffled = studies.copy()
    rng.shuffle(shuffled)
    counts = df2['ref_group'].value_counts()
    chosen, cum, total = [], 0, len(df2)
    for study in shuffled:
        chosen.append(study)
        cum += counts[study]
        if cum / total >= TEST_FRAC:
            break
    test_mask = df2['ref_group'].isin(chosen)
    test = df2[test_mask]
    pred = eval_eq1(test)
    y = test['Target'].values
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    rmses.append(rmse); maes.append(mae); ns.append(len(test))

rmses, maes, ns = np.array(rmses), np.array(maes), np.array(ns)
print(f'Eq.1 across {N_REPEATS} independent study-grouped splits (test n range {ns.min()}-{ns.max()}):')
print(f'  RMSE = {rmses.mean():.4f} +/- {rmses.std():.4f} mm  (min {rmses.min():.4f}, max {rmses.max():.4f})')
print(f'  MAE  = {maes.mean():.4f} +/- {maes.std():.4f} mm  (min {maes.min():.4f}, max {maes.max():.4f})')
print(f'  Original reported (single split): RMSE=6.8658 MAE=4.9067 (n=473)')

result = {
    'n_repeats': N_REPEATS,
    'rmse_mean': float(rmses.mean()), 'rmse_std': float(rmses.std()),
    'rmse_min': float(rmses.min()), 'rmse_max': float(rmses.max()),
    'mae_mean': float(maes.mean()), 'mae_std': float(maes.std()),
    'mae_min': float(maes.min()), 'mae_max': float(maes.max()),
    'test_n_min': int(ns.min()), 'test_n_max': int(ns.max()),
    'original_single_split': {'rmse': 6.8658, 'mae': 4.9067, 'n': 473},
}
with open(f'{ROOT}\\task2_eq1_robustness.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('\nWrote task2_eq1_robustness.json')
