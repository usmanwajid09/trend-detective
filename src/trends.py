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
            f"(p = {p:.3g}, n = {trend['n']})"
        )
    return (
        f"No significant trend detected: {slope:+.3f} {unit} per decade "
        f"(p = {p:.3g}, n = {trend['n']})"
    )
