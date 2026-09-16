"""
A safe, honestly-labeled multivariable baseline for Table 4: ordinary least
squares on the 8 raw physical features (no engineered interaction terms), to
sit between the single-coefficient X=k*sqrt(t) fit and the proposed symbolic
models.

This is NOT a reproduction of Papadakis / fib Model Code / CECS 220:2007 --
those standards' full multivariable forms require cement content (kg/m3),
aggregate/cement ratio, and CaO content, none of which exist in this open
dataset (only w/c, w/b, FA, CO2%, RH, T, exposure time, curing time are
available). Fabricating plausible-looking coefficients for a named standard
this dataset cannot actually support is exactly the kind of citation/number
fabrication this project has been burned by before; an honestly-labeled OLS
fit on the same inputs the proposed models use is the safe alternative.

Writes ols_baseline_verified.json.
"""
import sys
import io
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = r'd:\Research\TienThe'

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
rename_dict = {
    'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
    'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
    'Curing time (d)': 't_cure',
}
target_raw_col = [c for c in df.columns if 'Carbonation depth' in c][0]
df_clean = df.rename(columns=rename_dict).copy()
df_clean['Target'] = df_clean[target_raw_col]

feature_cols = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].median())
df_clean['Target'] = df_clean['Target'].fillna(df_clean['Target'].median())

train, test = train_test_split(df_clean, test_size=0.2, random_state=42)

ols = LinearRegression()
ols.fit(train[feature_cols], train['Target'])

pred_test = ols.predict(test[feature_cols])
pred_train = ols.predict(train[feature_cols])

result = {
    'feature_cols': feature_cols,
    'coefficients': {f: float(c) for f, c in zip(feature_cols, ols.coef_)},
    'intercept': float(ols.intercept_),
    'test_n': int(len(test)),
    'test_rmse_mm': float(np.sqrt(mean_squared_error(test['Target'], pred_test))),
    'test_mae_mm': float(mean_absolute_error(test['Target'], pred_test)),
    'train_n': int(len(train)),
    'train_rmse_mm': float(np.sqrt(mean_squared_error(train['Target'], pred_train))),
    'train_mae_mm': float(mean_absolute_error(train['Target'], pred_train)),
}

with open(f'{ROOT}\\ols_baseline_verified.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f"OLS test  RMSE={result['test_rmse_mm']:.4f} MAE={result['test_mae_mm']:.4f} n={result['test_n']}")
print(f"OLS train RMSE={result['train_rmse_mm']:.4f} MAE={result['train_mae_mm']:.4f} n={result['train_n']}")
print('✅ Wrote ols_baseline_verified.json')
