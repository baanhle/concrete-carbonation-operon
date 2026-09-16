"""
Pure prediction logic for the Eq.(1)-only version of the paper (Tree-Operon
removed). A single closed-form expression, no CART routing, no fallback --
Equation (1) is well-defined across the full training envelope (Section 4.1).
"""
import sys
import math

from tree_operon_model import EQ1, ENVELOPE  # noqa: F401 -- EQ1/ENVELOPE reused verbatim, single source of truth

if sys.stdout is not None and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def predict(w_c, w_b, co2, rh, t, exposure_days):
    """Predict carbonation depth (mm) using Equation (1) alone."""
    x1, x2, x5, x6 = w_c, w_b, rh, t
    x7 = math.sqrt(exposure_days)
    x10 = x2 * co2

    env = {'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
           'X1': x1, 'X2': x2, 'X5': x5, 'X6': x6, 'X7': x7, 'X10': x10,
           '__builtins__': {}}
    depth_mm = eval(EQ1, env)  # noqa: S307 -- fixed formula string, trusted

    raw_inputs = {'w_c': w_c, 'w_b': w_b, 'co2': co2, 'rh': rh, 't': t,
                  'exposure_days': exposure_days}
    out_of_range = []
    for key, val in raw_inputs.items():
        if key not in ENVELOPE:
            continue
        lo, hi, label = ENVELOPE[key]
        if val < lo or val > hi:
            out_of_range.append(f'{label} = {val:g} is outside the training range [{lo:g}, {hi:g}]')

    return {
        'depth_mm': depth_mm,
        'x_values': {'X1': x1, 'X2': x2, 'X5': x5, 'X6': x6, 'X7': x7, 'X10': x10},
        'out_of_range': out_of_range,
    }


if __name__ == '__main__':
    print(predict(w_c=0.5, w_b=0.45, co2=10.0, rh=65.0, t=20.0, exposure_days=365))
