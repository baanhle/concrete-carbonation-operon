"""
Exact Shapley (SHAP) analysis for Equation (1) ALONE (the Eq.1-only version
of the paper, Tree-Operon removed). Same interventional, brute-force-coalition
methodology as compute_shap_tree_operon.py, but much simpler here since
Eq.1 is one fixed closed-form function of six active variables (X1,X2,X5,
X6,X7,X10) with no CART routing to re-evaluate per coalition.

For each of the 473 study-grouped test samples, all 2^6=64 coalitions of the
6 active features are evaluated with non-coalition features fixed at their
TRAINING-set mean; the exact Shapley formula gives each feature's
contribution, satisfying Sum_i SHAP_i(x) = f(x) - f(x_bar) by construction.
"""
import json
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tree_operon_model import EQ1

ROOT = r'd:\Research\TienThe'
ACTIVE = ['X1', 'X2', 'X5', 'X6', 'X7', 'X10']
LABELS = {
    'X1': 'X1 (w/c)', 'X2': 'X2 (w/b)', 'X5': 'X5 (RH)',
    'X6': 'X6 (T)', 'X7': 'X7 (√t)', 'X10': 'X10 (w/b·CO2)',
}

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
df2['X1'] = df2['w/c']; df2['X2'] = df2['w/b']; df2['X5'] = df2['RH']
df2['X6'] = df2['T']; df2['X7'] = df2['sqrt_t']; df2['X10'] = df2['w/b'] * df2['CO2']

is_test = df2.index.isin(test_idx)
train, test = df2[~is_test], df2[is_test]
x_bar = {f: float(train[f].mean()) for f in ACTIVE}
print('Training-set means (baseline):', {k: round(v, 4) for k, v in x_bar.items()})

env_funcs = {'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt, '__builtins__': {}}


def eval_eq1_scalar(values):
    env = dict(env_funcs, **values)
    return eval(EQ1, env)  # noqa: S307 -- fixed formula string, trusted


M = len(ACTIVE)
all_subsets = []
for r in range(M + 1):
    all_subsets.extend(itertools.combinations(range(M), r))

# Precompute Shapley weights for each coalition size (of the OTHER M-1 features)
fact = math.factorial
weight = {s: fact(s) * fact(M - s - 1) / fact(M) for s in range(M)}

shap_rows = []
f_x_list, f_xbar = [], None
for _, row in test.iterrows():
    x = {f: row[f] for f in ACTIVE}
    f_full = eval_eq1_scalar(x)
    if f_xbar is None:
        f_xbar = eval_eq1_scalar(x_bar)
    f_x_list.append(f_full)

    # cache f(S) for every subset S (as a set of feature names)
    cache = {}
    for subset in all_subsets:
        names = frozenset(ACTIVE[i] for i in subset)
        vals = {f: (x[f] if f in names else x_bar[f]) for f in ACTIVE}
        cache[names] = eval_eq1_scalar(vals)

    phi = {}
    full_set = frozenset(ACTIVE)
    for feat in ACTIVE:
        total = 0.0
        others = [f for f in ACTIVE if f != feat]
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                S_plus = S | {feat}
                total += weight[r] * (cache[S_plus] - cache[S])
        phi[feat] = total
    shap_rows.append(phi)

f_x_list = np.array(f_x_list)
shap_df = pd.DataFrame(shap_rows)
efficiency_err = np.abs(shap_df.sum(axis=1).values + f_xbar - f_x_list).max()
print(f'Baseline f(x_bar) = {f_xbar:.4f} mm')
print(f'Max |efficiency violation| across {len(test)} samples: {efficiency_err:.2e} mm')

mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print('\nMean |SHAP| per feature:')
for feat, val in mean_abs_shap.items():
    print(f'  {LABELS[feat]}: {val:.4f} mm')

# --- Fig 7 (Eq.1 SHAP): (a) mean|SHAP| bar chart, (b) beeswarm-style summary ---
order = mean_abs_shap.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.barh([LABELS[f] for f in order][::-1], mean_abs_shap[order].values[::-1], color='#2E5C8A')
ax.set_xlabel('Mean |SHAP value| (mm)')
ax.set_title('(a) Feature importance')

ax2 = axes[1]
rng = np.random.RandomState(0)
for yi, feat in enumerate(order[::-1]):
    vals = shap_df[feat].values
    fvals = test[feat].values
    norm = (fvals - fvals.min()) / (fvals.max() - fvals.min() + 1e-12)
    jitter = rng.uniform(-0.3, 0.3, size=len(vals))
    sc = ax2.scatter(vals, yi + jitter, c=norm, cmap='coolwarm', s=10, alpha=0.7)
ax2.set_yticks(range(len(order)))
ax2.set_yticklabels([LABELS[f] for f in order[::-1]])
ax2.axvline(0, color='gray', lw=0.8)
ax2.set_xlabel('SHAP value (mm)')
ax2.set_title('(b) Direction of effect (blue=low, red=high feature value)')
cbar = fig.colorbar(sc, ax=ax2, fraction=0.05)
cbar.set_label('Feature value (normalized)')

plt.tight_layout()
plt.savefig(f'{ROOT}\\fig7_shap_eq1_only.png', dpi=220)
print('\nWrote fig7_shap_eq1_only.png')

result = {
    'active_features': ACTIVE,
    'x_bar': x_bar,
    'f_xbar': float(f_xbar),
    'max_efficiency_violation_mm': float(efficiency_err),
    'mean_abs_shap': {k: float(v) for k, v in mean_abs_shap.items()},
    'test_n': int(len(test)),
}
with open(f'{ROOT}\\shap_eq1_verified.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('Wrote shap_eq1_verified.json')
