"""
Locks the OUTER train/test split. Domains 1-2 (41, 22 independent studies)
get a clean study-grouped test allocation (no near-duplicate-specimen leakage
possible). Domains 3-4 (6 studies total, 2 for Domain 4 alone) cannot be
study-grouped without starving train or test entirely -- confirmed by trying
it (Domain 4 collapsed to 100% test, i.e. zero train data). For those two,
test rows are a plain random 20% row-level sample instead; this residual
leakage risk for Domains 3-4 is disclosed in the manuscript's Limitations,
not hidden.

Writes study_grouped_split.json: the actual test-row DataFrame indices (not
just study ids, since Domains 3-4 aren't study-grouped) so every other
script rebuilds the identical split by filtering on it.
"""
import sys
import io
import json

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = r'd:\Research\TienThe'
RNG_SEED = 42  # only used for the Domain 3-4 row-level fallback sampling

df = pd.read_excel(f'{ROOT}\\concrete carbonation data.xls')
df['ref_group'] = df['References'].ffill()
rename_dict = {'w/c (%)': 'w/c', 'w/b (%)': 'w/b', 'FA (kg/m³)': 'FA', 'CO₂ (%)': 'CO2',
               'RH (%)': 'RH', 'T (℃)': 'T', 'Carbonation time (√d)': 'sqrt_t',
               'Curing time (d)': 't_cure'}
df2 = df.rename(columns=rename_dict).copy()
feature_cols = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure']
df2[feature_cols] = df2[feature_cols].fillna(df2[feature_cols].median())
df2['ref_group'] = df['ref_group']
df2['T_RH'] = df2['T'] * df2['RH']

T_RH_THRESH, WC_THRESH, SQRT_T_THRESH = 1837.5, 0.75, 9.875
low = df2['T_RH'] <= T_RH_THRESH
domains = {
    1: df2[low & (df2['w/c'] <= WC_THRESH)],
    2: df2[low & (df2['w/c'] > WC_THRESH)],
    3: df2[(~low) & (df2['sqrt_t'] <= SQRT_T_THRESH)],
    4: df2[(~low) & (df2['sqrt_t'] > SQRT_T_THRESH)],
}

test_idx = []
for d, sub in domains.items():
    if d in (1, 2):
        counts = sub['ref_group'].value_counts().sort_values(ascending=True)
        total = len(sub)
        cum = 0
        held_studies = []
        for study, n in counts.items():
            held_studies.append(study)
            cum += n
            if cum / total >= 0.20:
                break
        idx = sub[sub['ref_group'].isin(held_studies)].index.tolist()
        print(f'Domain {d}: study-grouped, {len(held_studies)} studies, '
              f'{len(idx)}/{total} = {len(idx)/total:.3f} test')
    else:
        rng = np.random.RandomState(RNG_SEED + d)
        n_test = round(len(sub) * 0.20)
        idx = rng.choice(sub.index.values, size=n_test, replace=False).tolist()
        print(f'Domain {d}: row-level fallback (too few independent studies), '
              f'{len(idx)}/{len(sub)} = {len(idx)/len(sub):.3f} test')
    test_idx.extend(idx)

test_idx = sorted(set(test_idx))
print(f'\nOverall: {len(test_idx)}/{len(df2)} = {len(test_idx)/len(df2):.4f} as TEST')

with open(f'{ROOT}\\study_grouped_split.json', 'w', encoding='utf-8') as f:
    json.dump({'test_row_indices': test_idx, 'test_n': len(test_idx), 'total_n': len(df2),
               'domain1_2_method': 'study-grouped', 'domain3_4_method': 'row-level (data-scarce)'},
              f, indent=2, ensure_ascii=False)
print('✅ Wrote study_grouped_split.json')
