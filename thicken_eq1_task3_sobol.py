"""
Task 3: Sobol global sensitivity analysis for Eq.1, complementing the
existing SHAP analysis (Section 4.3, currently written for the Tree-Operon
framework). Sampled directly over the 6 raw/candidate variables from Table 1
that Eq.1 actually depends on: X1 (w/c), X2 (w/b), X4 (CO2, enters only via
X10=X2*X4), X5 (RH), X6 (T), X7 (sqrt_t) -- using each variable's own
Table 1 [min, max] envelope, consistent with the manuscript's own variable
definitions (no back-derivation of raw exposure days needed since X7 itself
is one of Table 1's eleven listed variables).
"""
import json
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from tree_operon_model import EQ1

problem = {
    'num_vars': 6,
    'names': ['X1', 'X2', 'X4', 'X5', 'X6', 'X7'],
    'bounds': [
        [0.31, 2.03],   # w/c
        [0.26, 0.77],   # w/b
        [0.03, 100.0],  # CO2
        [40.0, 90.0],   # RH
        [6.0, 45.0],    # T
        [1.73, 68.7],   # sqrt_t
    ],
}

N = 4096  # -> N*(2*d+2) = 4096*14 = 57,344 evaluations, cheap for a closed-form formula
param_values = saltelli.sample(problem, N, calc_second_order=True)

env_funcs = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, '__builtins__': {}}


def eval_eq1(X1, X2, X4, X5, X6, X7):
    X10 = X2 * X4
    env = dict(env_funcs, X1=X1, X2=X2, X5=X5, X6=X6, X7=X7, X10=X10)
    with np.errstate(all='ignore'):
        return eval(EQ1, env)  # noqa: S307 -- fixed formula string


Y = eval_eq1(param_values[:, 0], param_values[:, 1], param_values[:, 2],
             param_values[:, 3], param_values[:, 4], param_values[:, 5])
Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)

print('Sobol first-order (S1) and total-order (ST) indices for Eq.1:')
result = {'names': problem['names'], 'S1': [], 'S1_conf': [], 'ST': [], 'ST_conf': []}
for i, name in enumerate(problem['names']):
    print(f'  {name}: S1={Si["S1"][i]:.4f} (+/-{Si["S1_conf"][i]:.4f})  '
          f'ST={Si["ST"][i]:.4f} (+/-{Si["ST_conf"][i]:.4f})')
    result['S1'].append(float(Si['S1'][i]))
    result['S1_conf'].append(float(Si['S1_conf'][i]))
    result['ST'].append(float(Si['ST'][i]))
    result['ST_conf'].append(float(Si['ST_conf'][i]))

print('\nDominant second-order interactions (|S2| > 0.02):')
result['S2_pairs'] = []
names = problem['names']
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        s2 = Si['S2'][i][j]
        if np.isfinite(s2) and abs(s2) > 0.02:
            print(f'  {names[i]} x {names[j]}: S2={s2:.4f}')
            result['S2_pairs'].append({'pair': [names[i], names[j]], 'S2': float(s2)})

with open(r'd:\Research\TienThe\task3_eq1_sobol.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print('\nWrote task3_eq1_sobol.json')
