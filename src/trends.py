"""Trend analysis for the Trend Detective app.

Honest, small statistics: ordinary least-squares regression of an annual
series against year, giving the slope per decade and a two-sided p-value
for the null hypothesis "no trend".  Significance is judged at p < 0.05
and the verdict is always shown next to the numbers, never instead of
them.
"""

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover - scipy is in requirements.txt
    _stats = None


def analyse_trend(years, values):
    """Fit y ~ year and return a dict of trend statistics.

    years, values: array-like of equal length, at least 3 points.
    Returns slope per decade, p-value, r, n, and a plain-language verdict.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("need at least 3 paired points")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        raise ValueError("need at least 3 finite paired points")

    if _stats is not None:
        r = _stats.linregress(x, y)
        slope, intercept, rvalue, pvalue = r.slope, r.intercept, r.rvalue, r.pvalue
    else:  # pure-numpy fallback: slope/intercept/r, p-value unavailable
        slope, intercept = np.polyfit(x, y, 1)
        rvalue = float(np.corrcoef(x, y)[0, 1])
        pvalue = None

    slope_per_decade = float(slope * 10)
    significant = bool(pvalue is not None and pvalue < 0.05)
    return {
        "n": int(len(x)),
        "start_year": int(x.min()),
        "end_year": int(x.max()),
        "slope_per_decade": slope_per_decade,
        "intercept": float(intercept),
        "r": float(rvalue),
        "p_value": None if pvalue is None else float(pvalue),
        "significant": significant,
    }


def record_years(years, values):
    """Warmest/coldest (or wettest/driest) year of a series.

    years, values: array-like of equal length; values may hold None/NaN
    for missing years, which are skipped.  Returns
    {"max": {"year": int, "value": float},
     "min": {"year": int, "value": float}}, or {"max": None, "min": None}
    when no year has data.  A tie goes to the earliest year.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("years and values must have equal length")
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return {"max": None, "min": None}
    xf, yf = x[mask], y[mask]
    i_max = int(np.argmax(yf))
    i_min = int(np.argmin(yf))
    return {"max": {"year": int(xf[i_max]), "value": float(yf[i_max])},
            "min": {"year": int(xf[i_min]), "value": float(yf[i_min])}}


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_p(p):
    """Render a p-value the way the page does: never a bare 0 or 4.2e-11."""
    if p is None:
        return "p-value unavailable"
    if p <= 0:
        return "p < 0.001"
    if p < 0.001:
        mantissa, _, exp = f"{p:.1e}".partition("e")
        return f"p ≈ {mantissa}×10{str(int(exp)).translate(_SUPERSCRIPTS)}"
    return f"p = {p:.3g}"


def verdict(trend, unit, label):
    """Plain-language verdict from a trend dict (never hides the numbers)."""
    slope = trend["slope_per_decade"]
    p = trend["p_value"]
    direction = "warming" if slope > 0 else "cooling" if slope < 0 else "flat"
    if label == "precipitation":
        direction = "wetter" if slope > 0 else "drier" if slope < 0 else "flat"
    if p is None:
        return f"{direction} at {slope:+.3f} {unit} per decade (p-value unavailable)"
    if trend["significant"]:
        return (
            f"Significant {direction}: {slope:+.3f} {unit} per decade "
            f"({fmt_p(p)}, n = {trend['n']})"
        )
    return (
        f"No significant trend detected: {slope:+.3f} {unit} per decade "
        f"({fmt_p(p)}, n = {trend['n']})"
    )
