"""
Task 1: train OUR OWN black-box baselines (XGBoost, CatBoost, Random Forest)
on the exact same study-grouped split used for Eq.1 (study_grouped_split.json),
under two input scenarios:
  (F)  full 8 raw features (w/c, w/b, FA, CO2, RH, T, sqrt_t, t_cure)
  (R)  the same 6 variables Eq.1 actually uses (X1,X2,X5,X6,X7,X10)
This gives an apples-to-apples black-box reference on Eq.1's own test
partition, mirroring the accepted UHPC paper's Table 3 structure, instead of
relying only on Liu et al.'s numbers from a different split.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

ROOT = r'd:\Research\TienThe'

with open(f'{ROOT}\\study_grouped_split.json', encoding='utf-8') as f:
    split = json.load(f)
test_idx = set(split['test_row_indices'])

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
rename_dict = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
               'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
               'Curing time (d)': 't_cure'}
target_col = [c for c in df.columns if 'Carbonation depth' in c][0]
df2 = df.rename(columns=rename_dict).copy()
df2['Target'] = df2[target_col]
feature_cols = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
df2[feature_cols] = df2[feature_cols].fillna(df2[feature_cols].median())
df2['Target'] = df2['Target'].fillna(df2['Target'].median())
df2['X10'] = df2['w/b'] * df2['CO2']

FULL_FEATURES = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
REDUCED_FEATURES = ['w/c', 'w/b', 'RH', 'T', 'sqrt_t', 'X10']  # Eq.1's X1,X2,X5,X6,X7,X10

is_test = df2.index.isin(test_idx)
train, test = df2[~is_test], df2[is_test]
print(f'Train n={len(train)}, Test n={len(test)}')

y_train, y_test = train['Target'].values, test['Target'].values

MODELS = {
    'RandomForest': lambda: RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1),
    'XGBoost': lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42),
    'CatBoost': lambda: CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                           random_state=42, verbose=False),
}

results = {}
for scenario, feats in [('F', FULL_FEATURES), ('R', REDUCED_FEATURES)]:
    X_train, X_test = train[feats].values, test[feats].values
    for name, ctor in MODELS.items():
        model = ctor()
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        key = f'{name} ({scenario})'
        results[key] = {
            'scenario': scenario, 'features': feats,
            'train_r2': float(r2_score(y_train, pred_train)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, pred_train))),
            'train_mae': float(mean_absolute_error(y_train, pred_train)),
            'test_r2': float(r2_score(y_test, pred_test)),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, pred_test))),
            'test_mae': float(mean_absolute_error(y_test, pred_test)),
            'test_n': int(len(y_test)),
        }
        r = results[key]
        print(f'{key:20s} train R2={r["train_r2"]:.4f} RMSE={r["train_rmse"]:.4f} MAE={r["train_mae"]:.4f} | '
              f'test R2={r["test_r2"]:.4f} RMSE={r["test_rmse"]:.4f} MAE={r["test_mae"]:.4f}')

with open(f'{ROOT}\\task1_blackbox_baselines.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('\nWrote task1_blackbox_baselines.json')
