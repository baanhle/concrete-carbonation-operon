"""
Pure prediction logic for the Compact Tree-Operon Framework (Eq. 2/3/4/5 of the
manuscript), with no GUI dependency. This is the single source of truth the GUI
app imports from, so the formula set is never re-typed by hand a second time.

Final decision (Sếp, 2026-09-11): the Compact Tree-Operon Framework uses the
ORIGINAL formula set (per-domain candidate selected by highest R2 on the test
partition), reproducing RMSE=5.7970mm/MAE=4.2443mm -- verified bit-exact
against the raw dataset (see verify_tree_operon_model.py). A later attempt
this session to reselect these formulas via study-grouped validation (to
avoid test-partition leakage) produced consistently WORSE held-out accuracy
across repeated large-scale reruns (10,000+ combined trials), so the original
formulas are kept as the reported result; Eq.(1), the Single Operon Global
Equation, is the one component that WAS re-run at larger scale (10,000
trials vs the original 3,188) and did improve, so it uses the new formula
(see build_geomate_paper.py / the manuscript for that string). The two
therefore come from different search rounds -- this is disclosed in the
manuscript.

CART routing thresholds are fit once on the training partition and are
independently verified to reproduce Table 4's exact test-set domain counts
(259/108/31/35).
"""
import sys
import math


def _safe_log(x):
    """log(0) raises in the stdlib math module but is a well-defined -inf in
    IEEE-754 (and in numpy, which is what actually produced this paper's
    published Table 4 numbers). For Eq.(3), log(0.124*X3)=-inf at X3=0
    (fly-ash-free mixtures) divides a finite numerator by -inf, which
    correctly resolves to 0 -- the formula collapses to a constant 10.514mm
    for X3=0, NOT an undefined/crashing result. Only X3 values near the true
    pole (0.124*X3≈1, i.e. X3≈8.06) make the denominator approach zero
    (not infinity) and blow the output up -- that is the genuine danger
    zone, not X3=0 itself."""
    if x == 0.0:
        return float('-inf')
    return math.log(x)

# In a PyInstaller --windowed build there is no console attached, so
# sys.stdout/sys.stderr are None outright (not just non-UTF-8) -- guard
# against that first, or the frozen .exe crashes on import before the GUI
# ever shows a window.
if sys.stdout is not None and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Domain 1 -> Eq.(2), Domain 2 -> Eq.(3), Domain 3 -> Eq.(4), Domain 4 -> Eq.(5),
# matching the equation numbers PRINTED in the manuscript's Table 3.
EQ_DOMAIN = {
    2: '-4.930 - 3.905*( (2.718*X11 - 0.318*X10)*(-0.629*X1)'
       ' + (1.531*X2 + 0.732*X6 - 11.484)*(exp(-0.198*X7) - 1.531*X2) )',
    3: '10.514 - 0.020*( (log(1.443*X10) - exp(0.098*X6))'
       '*(log(0.124*X4) + 0.468*X7 - 0.240*X2)*2.262*X11'
       '/( exp(-2.266*X1)*log(0.124*X3) ) )',
    4: '2.249 + 56.604*( exp(exp(-0.272*X4))'
       '*( 0.218757*X2*X7 + exp(-0.516*X4)*(-1.355976*X2*X7 + 0.012*X3) )'
       '*(1.865025*X2**2) )',
    5: '47.626 - 18.193*( (0.693*X4 - exp(-0.029*X3) + 0.221*X10 + 0.537*X7'
       ' - exp(1.844*X11 - 0.646*X10))*(log(0.961*X10) + 1.242*X2 - 0.630*X10) )',
}

# Exact float64 CART thresholds (not the manuscript's rounded "0.75") --
# see compute_shap_tree_operon.py for why the exact w/c threshold matters.
T_RH_THRESH = 1837.5
WC_THRESH = 0.7498991191387177
SQRT_T_THRESH = 9.875

# 'eq_number' is the equation number as PRINTED IN THE MANUSCRIPT's Table 3.
DOMAIN_INFO = {
    1: {'eq_number': 2, 'condition': 'T·RH ≤ 1837.5 and w/c ≤ 0.75 (low temperature-humidity, low water-cement ratio)'},
    2: {'eq_number': 3, 'condition': 'T·RH ≤ 1837.5 and w/c > 0.75 (low temperature-humidity, high water-cement ratio)'},
    3: {'eq_number': 4, 'condition': 'T·RH > 1837.5 and √t ≤ 9.875 (high temperature-humidity, short exposure time)'},
    4: {'eq_number': 5, 'condition': 'T·RH > 1837.5 and √t > 9.875 (high temperature-humidity, long exposure time)'},
}

# Eq.(1), the Single Operon Global Equation. This is a sympy-simplified,
# 2-decimal-place-rounded rewrite of the original 10,000-trial-search
# formula (see simplify_eq1_sympy.py): ln(a*b) was split into ln(a)+ln(b)
# to pull every purely-numeric factor (including sqrt(2) and 1/ln(10),
# which the raw search's 0.707107 and 0.434294 coefficients turned out to
# equal almost exactly) out of the logs and sqrt, then folded into single
# decimal constants -- verified bit-identical to the sympy expression it
# was printed from (max diff 0.0 across 200 random points), and its own
# test-set RMSE/MAE were recomputed fresh rather than assumed unchanged
# (Sếp, 2026-09-12: chose the rounded form over the exact-coefficient one
# for readability, accepting the small RMSE/MAE increase this causes).
# Depends only on X1, X2, X5, X6, X7, X10 (never X3), so it is well-defined
# for every mix in the Table 1 envelope with the usual physical bounds.
EQ1 = ('-0.034*sqrt(X10)*X2*sqrt(X5)*X7*(2.02 - 0.17*X6)*(log(X5) + log(X7) - 3.61)'
       ' - 0.0042*X5*X7 - 0.19*(0.038*X5 - 0.17*X6)*(0.08*X5 - 0.17*X6)*(2.0*log(X6)'
       ' + log(X7) - 5.47) + 15.4 - 0.07*X5/X1')

# Training-data envelope from Table 1 of the manuscript (2,163 samples) --
# used only to flag extrapolation, never to clip/reject an input.
ENVELOPE = {
    'w_c':  (0.31, 2.03, 'Water-cement ratio (X1)'),
    'w_b':  (0.26, 0.77, 'Water-binder ratio (X2)'),
    'fa':   (0.00, 373.0, 'Fly ash content (X3, kg/m³)'),
    'co2':  (0.03, 100.0, 'CO2 concentration (X4, %)'),
    'rh':   (40.0, 90.0, 'Relative humidity (X5, %)'),
    't':    (6.0, 45.0, 'Temperature (X6, °C)'),
    # exposure_days envelope derived from the manuscript's √days bounds
    # (1.73, 68.7): days = (√days)^2.
    'exposure_days': (1.73 ** 2, 68.7 ** 2, 'Exposure duration (X7, days)'),
}


def predict(w_c, w_b, fa, co2, rh, t, exposure_days):
    """Predict carbonation depth (mm) for one sample using the Compact
    Tree-Operon Framework, routing through the same depth-2 CART split used
    throughout the paper. Returns a dict with the prediction, which domain/
    equation was used, the derived X-symbols, and any out-of-envelope inputs.
    """
    x1, x2, x3, x4, x5, x6 = w_c, w_b, fa, co2, rh, t
    x7 = math.sqrt(exposure_days)
    x9 = x6 * x5
    x10 = x2 * x4
    x11 = math.sqrt(x4)

    low_trh = x9 <= T_RH_THRESH
    if low_trh and x1 <= WC_THRESH:
        domain = 1
    elif low_trh and x1 > WC_THRESH:
        domain = 2
    elif (not low_trh) and x7 <= SQRT_T_THRESH:
        domain = 3
    else:
        domain = 4

    eq_number = DOMAIN_INFO[domain]['eq_number']
    env = {
        'exp': math.exp, 'log': _safe_log, 'sqrt': math.sqrt,
        'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X6': x6, 'X7': x7,
        'X10': x10, 'X11': x11,
        '__builtins__': {},
    }
    error = None
    fallback_depth_mm = None
    fallback_note = None
    try:
        depth_mm = eval(EQ_DOMAIN[eq_number], env)  # noqa: S307 -- fixed formula strings above, no user input in the expression itself
        if not math.isfinite(depth_mm):
            raise ValueError('non-finite result')
    except (ValueError, ZeroDivisionError, OverflowError):
        # Eq.(3) contains ln(0.124*X3) in a denominator with a genuine pole
        # at X3≈8.06 kg/m³ (0.124*X3=1). X3=0 itself is fine (log(0)=-inf,
        # dividing by -inf resolves to 0, giving a defined constant output --
        # see _safe_log above); this branch only fires for X3 landing
        # extremely close to 8.06, where the denominator approaches zero
        # (not infinity) and the output genuinely blows up.
        depth_mm = None
        error = ('Equation (3) is undefined for this mix: ln(0.124·X3) has a pole '
                 'at X3≈8.06 kg/m³, and this input lands on it.')
        try:
            fb_env = dict(env)
            fb_env['X5'], fb_env['X9'] = x5, x9
            fallback_depth_mm = eval(EQ1, fb_env)  # noqa: S307
            fallback_note = ('Falling back to the Single Operon Global Equation, Eq. (1), '
                              'which does not reference X3 and stays well-defined here.')
        except (ValueError, ZeroDivisionError, OverflowError):
            fallback_note = 'Eq. (1) fallback is also undefined for this input.'

    raw_inputs = {'w_c': w_c, 'w_b': w_b, 'fa': fa, 'co2': co2, 'rh': rh,
                  't': t, 'exposure_days': exposure_days}
    out_of_range = []
    for key, val in raw_inputs.items():
        lo, hi, label = ENVELOPE[key]
        if val < lo or val > hi:
            out_of_range.append(f'{label} = {val:g} is outside the training range [{lo:g}, {hi:g}]')

    return {
        'depth_mm': depth_mm,
        'domain': domain,
        'eq_number': eq_number,
        'condition': DOMAIN_INFO[domain]['condition'],
        'x_values': {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5, 'X6': x6,
                     'X7': x7, 'X9': x9, 'X10': x10, 'X11': x11},
        'out_of_range': out_of_range,
        'error': error,
        'fallback_depth_mm': fallback_depth_mm,
        'fallback_note': fallback_note,
    }


if __name__ == '__main__':
    example = predict(w_c=0.5, w_b=0.45, fa=50.0, co2=10.0, rh=65.0, t=20.0, exposure_days=365)
    print(example)
