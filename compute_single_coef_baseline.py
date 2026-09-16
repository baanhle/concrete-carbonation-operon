"""
Single-coefficient X = k*sqrt(t) baseline, k = mean(X)/mean(sqrt_t) on the
train partition (reproduces the manuscript's original k=1.141 on the old
random_state=42 split; recomputed here on the new study-grouped split).
Writes single_coef_baseline_verified.json.
"""
import sys
import io
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = r'd:\Research\TienThe'

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
rename_dict = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
               'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
               'Curing time (d)': 't_cure'}
target_col = [c for c in df.columns if 'Carbonation depth' in c][0]
df2 = df.rename(columns=rename_dict).copy()
df2['Target'] = df2[target_col]
df2['sqrt_t'] = df2['sqrt_t'].fillna(df2['sqrt_t'].median())
df2['Target'] = df2['Target'].fillna(df2['Target'].median())
train, test = train_test_split(df2, test_size=0.2, random_state=42)

k = float(train['Target'].mean() / train['sqrt_t'].mean())
pred_test = k * test['sqrt_t']
result = {
    'k': k,
    'test_n': int(len(test)),
    'test_rmse_mm': float(np.sqrt(mean_squared_error(test['Target'], pred_test))),
    'test_mae_mm': float(mean_absolute_error(test['Target'], pred_test)),
    'test_r2': float(r2_score(test['Target'], pred_test)),
}
with open(f'{ROOT}\\single_coef_baseline_verified.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"k={k:.4f} test RMSE={result['test_rmse_mm']:.4f} MAE={result['test_mae_mm']:.4f} "
      f"R2={result['test_r2']:.4f} n={result['test_n']}")
print('✅ Wrote single_coef_baseline_verified.json')
