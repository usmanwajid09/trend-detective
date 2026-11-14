"""Mann-Kendall trend tests and Sen's slope, matching pymannkendall exactly.

This module re-implements the reference algorithms from pymannkendall
(Hussain et al. 2019, pyMannKendall 1.4.3) so the app does not depend on
that package at runtime, and adds a vectorised grid version that runs the
same test on every cell of a global grid at once.

Conventions (mirroring pymannkendall):
- S is the Mann-Kendall score; Var(S) uses the standard tie correction.
- z uses the continuity correction: (S-1)/sqrt(Var) for S>0, (S+1)/sqrt(Var)
  for S<0, 0 for S==0.  p is two-sided from the normal approximation.
- Theil-Sen slope is the median of all pairwise slopes; the intercept is
  median(x) - median(t) * slope, i.e. the fitted line passes through the
  median point.  Times are 0..n-1 unless explicit times are given.
- Hamed-Rao (1998) autocorrelation correction: the detrended series is
  ranked, the autocorrelation of the ranks at each lag is computed, and
  only lags that are significant at the alpha level widen Var(S).
  Yue-Wang (2004) is also available: same idea, but on the detrended
  values (not ranks) and using all lags.
- The 95% confidence interval for Sen's slope follows Gilbert (1987):
  ranks (N-C)/2 and (N+C)/2 of the sorted pairwise slopes, where
  N = n(n-1)/2 and C = z_{1-alpha/2} * sqrt(Var(S)).

Differences from pymannkendall, all deliberate and documented:
- Missing values: pymannkendall drops NaN rows for S/Var but keeps the
  original index gaps when computing Sen's slope.  Here a series is
  cleaned to its finite values first and pairwise slopes use the actual
  time gaps, so a series with gaps in it is treated honestly.  For a
  complete series the results are identical to pymannkendall.
- A series whose values are all tied has Var(S) = 0; pymannkendall would
  divide by zero.  Here z = 0 and p = 1 (no trend), which is the correct
  answer for a flat series.
- If an autocorrelation correction would make Var(S) <= 0, the
  uncorrected variance is used instead.
"""

import numpy as np
from scipy.stats import norm, rankdata

__all__ = [
    "sens_slope",
    "mann_kendall",
    "mann_kendall_grid",
    "sen_slope_ci",
]


def _clean(values, times=None):
    """Return (x, t) restricted to finite pairs, sorted by time."""
    x = np.asarray(values, dtype=float).ravel()
    if times is None:
        t = np.arange(len(x), dtype=float)
    else:
        t = np.asarray(times, dtype=float).ravel()
        if t.shape != x.shape:
            raise ValueError("values and times must have the same length")
    order = np.argsort(t, kind="stable")
    x, t = x[order], t[order]
    mask = np.isfinite(x) & np.isfinite(t)
    return x[mask], t[mask]


def _mk_score(x):
    """Mann-Kendall score S for a clean 1-D array."""
    n = len(x)
    s = 0
    for k in range(n - 1):
        s += int(np.sum(x[k + 1:] > x[k]) - np.sum(x[k + 1:] < x[k]))
    return s


def _var_s(x, n):
    """Variance of S with the standard tie correction."""
    unique, counts = np.unique(x, return_counts=True)
    tie = counts[counts > 1]
    return (n * (n - 1) * (2 * n + 5)
            - np.sum(tie * (tie - 1) * (2 * tie + 5))) / 18.0


def _z_score(s, var_s):
    if var_s <= 0:
        return 0.0
    if s > 0:
        return (s - 1) / np.sqrt(var_s)
    if s < 0:
        return (s + 1) / np.sqrt(var_s)
    return 0.0


def _p_value(z, alpha):
    p = float(2 * (1 - norm.cdf(abs(z))))
    h = bool(abs(z) > norm.ppf(1 - alpha / 2))
    if z < 0 and h:
        trend = "decreasing"
    elif z > 0 and h:
        trend = "increasing"
    else:
        trend = "no trend"
    return p, h, trend


def _acf(x, nlags):
    """Lag-k autocorrelations, k = 0..nlags (biased estimator, /n).

    Mirrors pymannkendall's __acf: acf[k] = sum_t y_t*y_{t+k} / sum_t y_t^2.
    """
    y = x - x.mean()
    n = len(x)
    denom = float(np.dot(y, y))
    if denom == 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = float(np.dot(y[:n - k], y[k:]) / denom) if k < n else 0.0
    return out


def _pairwise_slopes(x, t):
    """All pairwise slopes (x[j]-x[i])/(t[j]-t[i]), j > i, as a 1-D array."""
    n = len(x)
    iu = np.triu_indices(n, k=1)
    dt = t[iu[1]] - t[iu[0]]
    return (x[iu[1]] - x[iu[0]]) / dt


def sens_slope(values, times=None):
    """Theil-Sen slope and intercept, mirroring pymannkendall.sens_slope.

    Returns (slope, intercept); the fitted line is value = slope*t +
    intercept, and it passes through (median(t), median(x)).
    """
    x, t = _clean(values, times)
    if len(x) < 2:
        raise ValueError("need at least 2 finite points")
    slopes = _pairwise_slopes(x, t)
    slope = float(np.median(slopes))
    intercept = float(np.median(x) - np.median(t) * slope)
    return slope, intercept


def sen_slope_ci(values, times=None, var_s=None, alpha=0.05):
    """95% confidence interval for Sen's slope (Gilbert 1987).

    var_s: variance of S to use; defaults to the tie-corrected Mann-Kendall
    variance of the series.  Pass the autocorrelation-corrected variance
    to get the wider interval that goes with a corrected test.
    Returns (lower, upper) in slope-per-time-unit.
    """
    x, t = _clean(values, times)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 finite points")
    slopes = np.sort(_pairwise_slopes(x, t))
    n_pairs = len(slopes)
    if var_s is None:
        var_s = _var_s(x, n)
    c = norm.ppf(1 - alpha / 2) * np.sqrt(var_s)
    lo_rank = (n_pairs - c) / 2.0   # 1-based ranks into the sorted slopes
    hi_rank = (n_pairs + c) / 2.0
    return (_rank_value(slopes, lo_rank), _rank_value(slopes, hi_rank))


def _rank_value(sorted_slopes, rank_1based):
    """Value at a (possibly fractional) 1-based rank, clamped to the range."""
    n = len(sorted_slopes)
    pos = min(max(rank_1based - 1.0, 0.0), n - 1.0)  # 0-based position
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_slopes[lo] + frac * (sorted_slopes[hi] - sorted_slopes[lo]))


def mann_kendall(values, times=None, alpha=0.05, correction=None, lag=None):
    """Mann-Kendall test for one series, matching pymannkendall.

    correction: None (original test), "hamed-rao" or "yue-wang".
    lag: number of lags the autocorrelation correction looks at
    (pymannkendall's ``lag``; Hamed & Rao propose 3).  None means all
    lags, which is pymannkendall's default.
    Returns a dict with trend, significant, p_value, z, tau, s, var_s,
    slope (per time unit), slope_per_decade, intercept, ci_lo, ci_hi
    (95% CI for the slope, per time unit) and n.
    """
    x, t = _clean(values, times)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 finite points")

    s = _mk_score(x)
    var_s = _var_s(x, n)
    tau = s / (0.5 * n * (n - 1))
    slope, intercept = sens_slope(x, t)

    if correction in ("hamed-rao", "yue-wang"):
        # Detrend with 1-based index, exactly as pymannkendall does.
        x_detrend = x - np.arange(1, n + 1) * slope
        lag_n = n if lag is None else lag + 1
        if correction == "hamed-rao":
            ranks = rankdata(x_detrend)
            acf = _acf(ranks, lag_n - 1)
            interval = norm.ppf(1 - alpha / 2) / np.sqrt(n)
            sni = 0.0
            for k in range(1, lag_n):
                if abs(acf[k]) > interval:
                    sni += (n - k) * (n - k - 1) * (n - k - 2) * acf[k]
            factor = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * sni if n > 2 else 1.0
        else:
            acf = _acf(x_detrend, lag_n - 1)
            sni = float(np.sum((1 - np.arange(1, lag_n) / n) * acf[1:lag_n]))
            factor = 1.0 + 2.0 * sni
        corrected = var_s * factor
        if corrected > 0:
            var_s = corrected

    z = _z_score(s, var_s)
    p, h, trend = _p_value(z, alpha)
    ci_lo, ci_hi = sen_slope_ci(x, t, var_s=var_s, alpha=alpha)
    return {
        "trend": trend,
        "significant": h,
        "p_value": p,
        "z": float(z),
        "tau": float(tau),
        "s": int(s),
        "var_s": float(var_s),
        "slope": float(slope),
        "slope_per_decade": float(slope * 10),
        "intercept": float(intercept),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": int(n),
    }


def mann_kendall_grid(values, times, alpha=0.05, correction="hamed-rao",
                      min_n=10, lag=3):
    """Mann-Kendall test for every cell of a grid, vectorised.

    values: (n_time, n_cells) float array, NaN where a cell has no value.
    times: (n_time,) time coordinates (e.g. years); must be increasing.
    Cells with fewer than min_n finite values get NaN results.
    lag: lags the autocorrelation correction looks at (Hamed & Rao propose
    3; None means all lags).  Returns a dict of (n_cells,) arrays: n, s,
    tau, var_s, z, p_value, slope (per time unit), slope_per_decade,
    intercept, ci_lo, ci_hi (95% CI, per time unit), significant, rho1
    (lag-1 autocorrelation of the ranked detrended series).
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    n_time, n_cells = values.shape
    if len(times) != n_time:
        raise ValueError("times must match the first axis of values")
    if n_time < 3:
        raise ValueError("need at least 3 time steps")

    finite = np.isfinite(values)
    m = finite.sum(axis=0)                       # finite count per cell
    ok = m >= min_n
    z975 = float(norm.ppf(1 - alpha / 2))
    block = 256

    iu = np.triu_indices(n_time, k=1)
    i_idx, j_idx = iu
    dt = times[j_idx] - times[i_idx]            # (n_pairs,)

    def block_slopes(cols):
        """Sorted pairwise slopes (NaN last) + valid pair counts per cell."""
        xb = values[:, cols]
        diffs = xb[j_idx] - xb[i_idx]           # (P, B), NaN if either end missing
        slopes = diffs / dt[:, None]
        return np.sort(slopes, axis=0), np.isfinite(slopes).sum(axis=0)

    # ---- pass 1: MK score and Sen's slope, in blocks of cells -------------
    s_arr = np.full(n_cells, np.nan)
    slope_arr = np.full(n_cells, np.nan)
    for start in range(0, n_cells, block):
        cols = slice(start, min(start + block, n_cells))
        xb = values[:, cols]
        diffs = xb[j_idx] - xb[i_idx]
        s_arr[cols] = np.nansum(np.sign(diffs), axis=0)
        sorted_slopes, q = block_slopes(cols)
        slope_arr[cols] = _median_of_sorted(sorted_slopes, q)

    # ---- pass 2: autocorrelation correction ---------------------------------
    var_s = _grid_var_s(values, finite, m)
    rho1 = np.full(n_cells, np.nan)
    if correction in ("hamed-rao", "yue-wang"):
        ok_cols = np.flatnonzero(ok)
        if len(ok_cols):
            factor_ok, rho1_ok = _grid_correction(
                values[:, ok_cols], finite[:, ok_cols], m[ok_cols],
                slope_arr[ok_cols], times, alpha, correction, lag)
            corrected = var_s[ok_cols] * factor_ok
            use_corrected = np.isfinite(corrected) & (corrected > 0)
            var_s[ok_cols] = np.where(use_corrected, corrected, var_s[ok_cols])
            rho1[ok_cols] = rho1_ok

    # ---- pass 3: z, p, tau, intercept and CI -------------------------------
    z_arr = np.zeros(n_cells)
    safe_var = np.where((var_s > 0) & np.isfinite(var_s), var_s, np.nan)
    pos = s_arr > 0
    neg = s_arr < 0
    z_arr[pos] = (s_arr[pos] - 1) / np.sqrt(safe_var[pos])
    z_arr[neg] = (s_arr[neg] + 1) / np.sqrt(safe_var[neg])
    z_arr[~np.isfinite(z_arr)] = 0.0
    p_arr = 2 * (1 - norm.cdf(np.abs(z_arr)))
    tau_arr = np.where(m >= 2, s_arr / np.where(m >= 2, 0.5 * m * (m - 1), np.nan), np.nan)
    significant = ok & np.isfinite(p_arr) & (p_arr < alpha)

    med_x = _median_of_sorted(np.sort(np.where(finite, values, np.nan), axis=0), m)
    med_t = _median_finite_time(times, finite, m)
    intercept_arr = med_x - med_t * slope_arr

    ci_lo = np.full(n_cells, np.nan)
    ci_hi = np.full(n_cells, np.nan)
    for start in range(0, n_cells, block):
        cols = slice(start, min(start + block, n_cells))
        sorted_slopes, q = block_slopes(cols)
        c = z975 * np.sqrt(np.where(var_s[cols] > 0, var_s[cols], np.nan))
        ci_lo[cols] = _rank_values_sorted(sorted_slopes, (q - c) / 2.0, q)
        ci_hi[cols] = _rank_values_sorted(sorted_slopes, (q + c) / 2.0, q)

    bad = ~ok
    for arr in (s_arr, slope_arr, ci_lo, ci_hi, var_s, z_arr, p_arr, tau_arr,
                intercept_arr, rho1):
        arr[bad] = np.nan
    significant[bad] = False

    return {
        "n": m.astype(int),
        "s": s_arr,
        "tau": tau_arr,
        "var_s": var_s,
        "z": z_arr,
        "p_value": p_arr,
        "slope": slope_arr,
        "slope_per_decade": slope_arr * 10,
        "intercept": intercept_arr,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": significant,
        "rho1": rho1,
    }


def _median_of_sorted(sorted_slopes, q):
    """Median of each column's first q (finite) sorted values."""
    p, b = sorted_slopes.shape
    col = np.arange(b)
    mid_hi = np.minimum(q // 2, p - 1)
    mid_lo = np.minimum((q - 1) // 2, p - 1)
    hi = sorted_slopes[mid_hi, col]
    lo = sorted_slopes[mid_lo, col]
    return np.where(q % 2 == 1, np.where(q > 0, lo, np.nan),
                    np.where(q > 0, 0.5 * (lo + hi), np.nan))


def _rank_values_sorted(sorted_slopes, rank_1based, q):
    """Interpolated value at fractional 1-based ranks, per column."""
    p, b = sorted_slopes.shape
    col = np.arange(b)
    pos = np.clip(np.nan_to_num(rank_1based - 1.0, nan=0.0),
                  0.0, np.maximum(q - 1, 0))
    lo_i = np.floor(pos).astype(int)
    hi_i = np.minimum(lo_i + 1, np.maximum(q - 1, 0))
    frac = pos - lo_i
    lo_v = sorted_slopes[lo_i, col]
    hi_v = sorted_slopes[hi_i, col]
    out = lo_v + frac * (hi_v - lo_v)
    return np.where(q > 0, out, np.nan)


def _grid_var_s(values, finite, m):
    """Tie-corrected Var(S) for every cell, via grouped value counts."""
    n_time, n_cells = values.shape
    n = m.astype(float)
    base = n * (n - 1) * (2 * n + 5) / 18.0
    # (cell, value) pairs for finite entries, sorted to find tie groups.
    cell_ids = np.repeat(np.arange(n_cells), m)
    vals = values.T[finite.T]
    order = np.lexsort((vals, cell_ids))
    cell_s = cell_ids[order]
    val_s = vals[order]
    change = np.ones(len(val_s), dtype=bool)
    change[1:] = (cell_s[1:] != cell_s[:-1]) | (val_s[1:] != val_s[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(val_s))
    run_len = ends - starts
    tie = run_len > 1
    if np.any(tie):
        tp = run_len[tie].astype(float)
        corr = tp * (tp - 1) * (2 * tp + 5)
        var = base - np.bincount(cell_s[starts[tie]], weights=corr,
                                 minlength=n_cells) / 18.0
    else:
        var = base
    return np.where(m >= 3, var, np.nan)


def _grid_correction(values, finite, m, slope_arr, times, alpha, correction,
                     lag=3):
    """Hamed-Rao / Yue-Wang variance inflation factor per cell.

    Returns (factor, rho1).  The detrended series is ranked (Hamed-Rao)
    or used as-is (Yue-Wang); autocorrelations are computed for lags
    1..lag (Hamed & Rao propose 3), vectorised over cells.  Only cells
    passed in need a correction; callers pass the well-sampled ones.
    """
    n_time, n_cells = values.shape
    lag_n = n_time if lag is None else min(lag + 1, n_time)
    idx = np.arange(1, n_time + 1, dtype=float)[:, None]
    detrended = np.where(finite, values - idx * slope_arr[None, :], np.nan)
    if correction == "hamed-rao":
        series = rankdata(detrended, axis=0, nan_policy="omit")
    else:
        series = detrended
    y = series - np.nanmean(series, axis=0, keepdims=True)
    denom = np.nansum(y * y, axis=0)
    rho1 = np.full(n_cells, np.nan)
    interval = norm.ppf(1 - alpha / 2) / np.sqrt(np.where(m > 0, m, np.nan))
    sni = np.zeros(n_cells)
    for k in range(1, lag_n):
        num = np.nansum(y[:n_time - k] * y[k:], axis=0)
        acf_k = np.where(denom > 0, num / np.where(denom == 0, np.nan, denom), 0.0)
        if k == 1:
            rho1 = acf_k
        if correction == "hamed-rao":
            weight = (m - k) * (m - k - 1) * (m - k - 2)
            sig = np.abs(acf_k) > interval
            sni += np.where(sig & (weight > 0), weight * acf_k, 0.0)
        else:
            sni += (1 - k / np.where(m > 0, m, np.nan)) * acf_k
    if correction == "hamed-rao":
        factor = 1.0 + (2.0 / (m * (m - 1) * (m - 2))) * sni
        factor = np.where(m > 2, factor, 1.0)
    else:
        factor = 1.0 + 2.0 * sni
    return factor, rho1


def _median_finite_time(times, finite, m):
    """Median of the finite time coordinates, per cell."""
    n_time, n_cells = times.shape[0], finite.shape[1]
    masked = np.where(finite, times[:, None], np.nan)
    srt = np.sort(masked, axis=0)  # NaN sorts last; finite values first
    col = np.arange(n_cells)
    mid_hi = np.minimum(m // 2, n_time - 1)
    mid_lo = np.minimum((m - 1) // 2, n_time - 1)
    hi = srt[mid_hi, col]
    lo = srt[mid_lo, col]
    return np.where(m % 2 == 1, np.where(m > 0, lo, np.nan),
                    np.where(m > 0, 0.5 * (lo + hi), np.nan))
