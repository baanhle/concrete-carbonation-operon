"""
Search #1: the GLOBAL equation search (produces Eq.1), scaled from 3,188 to
10,000 trials and parallelized across worker processes (each single-threaded
via n_threads=1, so N_WORKERS processes saturate the machine's cores instead
of each trial internally fighting over all cores). Runs on the NEW
study-grouped train partition (study_grouped_split.json), not the old
random_state=42 split.

Same config-sampling scheme as the original 3,188-trial search (Table 2):
population in {1000, 2000}, generations in {200, 500}, max_length in
{15, 25, 35, 45}, each trial an independent random seed. Selection rule is
unchanged from the manuscript's own stated methodology: NOT the trial with
literally the lowest RMSE, but the best RMSE among candidates whose PRINTED
(6-decimal-rounded) form reproduces that same RMSE -- avoids picking a
formula that looks good only at full float64 precision internally.

Run under the microgen conda env (has pyoperon):
    C:\\Users\\Admin\\.conda\\envs\\microgen\\python.exe search1_global_parallel.py [N_TRIALS]

Writes search1_global_results.csv (same schema as the old
Carbonation_Operon_Results.csv) incrementally, and search1_best_equation.json
once done.
"""
import sys
import io
import os
import json
import time
import random

import multiprocessing

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = r'd:\Research\TienThe'
N_WORKERS = 20
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

# i7-14700KF: 8 P-cores (16 threads, logical 0-15, 2 threads/core) + 12
# E-cores (12 threads, logical 16-27, 1 thread/core) = 20 physical cores, 28
# threads. Windows was piling every worker onto the (slower) E-core threads
# instead of spreading across all 20 physical cores -- pin one worker per
# physical core instead: one thread per P-core (skip the SMT sibling, which
# shares the same execution resources) + all 12 E-core threads.
CORE_LIST = [0, 2, 4, 6, 8, 10, 12, 14] + list(range(16, 28))
_worker_idx = multiprocessing.Value('i', 0)


def _pin_worker(idx_counter, core_list):
    import psutil
    with idx_counter.get_lock():
        i = idx_counter.value
        idx_counter.value += 1
    core = core_list[i % len(core_list)]
    try:
        psutil.Process().cpu_affinity([core])
    except Exception as e:
        print(f'⚠️  Could not pin worker to core {core}: {e}')


def load_split():
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
    df2['T_RH'] = df2['T'] * df2['RH']
    df2['wb_CO2'] = df2['w/b'] * df2['CO2']
    df2['sqrt_CO2'] = np.sqrt(df2['CO2'])

    is_test = df2.index.isin(test_idx)
    return df2[~is_test], df2[is_test]


EQ_FEATURES = ['w/c', 'w/b', 'FA', 'CO2', 'RH', 'T', 'sqrt_t', 't_cure', 'T_RH', 'wb_CO2', 'sqrt_CO2']


def run_trial(args):
    pop, gens, maxlen, seed, X_train, y_train, X_test, y_test = args
    from pyoperon.sklearn import SymbolicRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    t0 = time.time()
    reg = SymbolicRegressor(allowed_symbols='add,sub,mul,div,sqrt,exp,log,constant,variable',
                             max_length=maxlen, population_size=pop, generations=gens,
                             n_threads=1, random_state=seed)
    reg.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred_test = np.nan_to_num(reg.predict(X_test), nan=0.0, posinf=0.0, neginf=0.0)
    pred_train = np.nan_to_num(reg.predict(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        model_str = reg.get_model_string(reg.model_, 6)
    except Exception:
        model_str = str(reg.model_)
    nodes = len(reg.model_) if hasattr(reg.model_, '__len__') else np.nan

    return {
        'POP_SIZE': pop, 'GENS': gens, 'MAX_LEN': maxlen, 'RD_STATE': seed,
        'NODES': nodes,
        'R2_test': float(r2_score(y_test, pred_test)),
        'RMSE_test': float(np.sqrt(mean_squared_error(y_test, pred_test))),
        'MAE_test': float(mean_absolute_error(y_test, pred_test)),
        'R2_train': float(r2_score(y_train, pred_train)),
        'RMSE_train': float(np.sqrt(mean_squared_error(y_train, pred_train))),
        'MAE_train': float(mean_absolute_error(y_train, pred_train)),
        'TIME': elapsed, 'MODEL': model_str,
    }


def reproduces_printed_form(model_str, X_test, y_test):
    """The manuscript's own selection rule: does the 6-decimal-ROUNDED
    printed formula, re-evaluated from scratch, reproduce ~the same test
    RMSE as the internal float64 model? (guards against a formula that only
    looks good at full precision, which nobody could actually copy into a
    spreadsheet)."""
    env = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, '__builtins__': {}}
    for i, col in enumerate(EQ_FEATURES, start=1):
        env[f'X{i}'] = X_test[:, i - 1]
    try:
        with np.errstate(all='ignore'):
            pred = eval(model_str, env)  # noqa: S307
        pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
        return rmse
    except Exception:
        return None


def main():
    train, test = load_split()
    print(f'train n={len(train)}, test n={len(test)}')
    X_train = train[EQ_FEATURES].values
    y_train = train['Target'].values
    X_test = test[EQ_FEATURES].values
    y_test = test['Target'].values

    rng = random.Random(42)
    jobs = []
    for _ in range(N_TRIALS):
        pop = rng.choice([1000, 2000])
        gens = rng.choice([200, 500])
        maxlen = rng.choice([15, 25, 35, 45])
        seed = rng.randint(0, 10_000_000)
        jobs.append((pop, gens, maxlen, seed, X_train, y_train, X_test, y_test))

    out_path = f'{ROOT}\\search1_global_results.csv'
    first_write = not os.path.exists(out_path)
    t_start = time.time()
    n_done = 0
    rows_buffer = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_pin_worker,
                              initargs=(_worker_idx, CORE_LIST)) as ex:
        futures = [ex.submit(run_trial, j) for j in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            rows_buffer.append(row)
            n_done += 1
            if len(rows_buffer) >= 100 or n_done == N_TRIALS:
                dfb = pd.DataFrame(rows_buffer)
                dfb.to_csv(out_path, mode='a', header=first_write, index=False)
                first_write = False
                rows_buffer = []
            if n_done % 500 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta = (N_TRIALS - n_done) / rate
                print(f'{n_done}/{N_TRIALS} done, {elapsed:.0f}s elapsed, '
                      f'{rate:.2f} trials/s, ETA {eta/60:.1f} min')

    print(f'✅ All {N_TRIALS} trials done in {(time.time()-t_start)/60:.1f} min. Wrote {out_path}')

    # Selection: best RMSE_test among candidates whose printed form reproduces it.
    df_res = pd.read_csv(out_path)
    df_res = df_res.sort_values('RMSE_test')
    best = None
    for _, row in df_res.iterrows():
        repro_rmse = reproduces_printed_form(row['MODEL'], X_test, y_test)
        if repro_rmse is not None and abs(repro_rmse - row['RMSE_test']) < 0.01:
            best = row
            best_repro_rmse = repro_rmse
            break
    if best is None:
        print('❌ No candidate reproduced its own printed form within tolerance!')
        sys.exit(1)

    print(f"\n🏆 Best reproducible Eq.1: RMSE_test={best['RMSE_test']:.4f} "
          f"(reproduced {best_repro_rmse:.4f}), MAE_test={best['MAE_test']:.4f}, "
          f"nodes={best['NODES']}")
    print(f"  Formula: {best['MODEL']}")

    with open(f'{ROOT}\\search1_best_equation.json', 'w', encoding='utf-8') as f:
        json.dump({
            'formula': best['MODEL'],
            'rmse_test': float(best['RMSE_test']),
            'mae_test': float(best['MAE_test']),
            'rmse_train': float(best['RMSE_train']),
            'mae_train': float(best['MAE_train']),
            'nodes': float(best['NODES']),
            'pop_size': int(best['POP_SIZE']), 'gens': int(best['GENS']), 'max_len': int(best['MAX_LEN']),
            'seed': int(best['RD_STATE']), 'time_s': float(best['TIME']),
            'n_trials': N_TRIALS, 'test_n': len(test), 'train_n': len(train),
        }, f, indent=2, ensure_ascii=False)
    print('✅ Wrote search1_best_equation.json')


if __name__ == '__main__':
    main()
