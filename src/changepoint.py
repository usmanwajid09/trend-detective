"""Change-point detection for a global mean series (Trend Detective).

Method, in plain words
----------------------
Take an annual series (e.g. the GISTEMP global mean temperature anomaly,
1880-2025) and fit a single straight line through it by ordinary least
squares.  If the warming rate really changed at some point, one straight
line cannot describe the whole series: the residuals (what is left after
subtracting the line) will sit mostly on one side of zero before the
change and mostly on the other side after it.

So we add the residuals up as we move through the years — the running
total is called a CUSUM (cumulative sum).  Under "no change, one straight
line is right", the running total wobbles around zero and drifts back to
zero at the end (residuals always sum to zero).  Under "the rate changed",
the running total marches away from zero and only comes back at the very
end, so it peaks near the change year.  The candidate break is the year
where |CUSUM| is largest.

A big peak by itself is not proof: any noisy series has some largest
peak.  So we test it against a null hypothesis — "no change, the residuals
are just noise in any order" — by shuffling the residuals 2,000 times
(a permutation test) and recording the largest |CUSUM| peak of each
shuffled series.  If fewer than 5% of the shuffles reach the observed
peak, the peak is unlikely to be chance, and we call the break supported.

One guard on top of that: a supported break is only reported as an
acceleration ("warming accelerated around YEAR") when the Mann-Kendall +
Sen's-slope trend after the break is steeper than before it.  A break
that fails either check is reported as unsupported, with no break year.

A limitation worth stating plainly: the CUSUM peak marks where the
series crosses its single straight-line fit, not the exact year the
slope changed.  For a change in the *level* (a step in the mean) the
peak lands on the break year.  For a change in *slope* — an
acceleration, like global warming — the residuals of the single-line
fit cross zero only after the kink, so the peak lags the kink (in our
synthetic tests, by roughly 15-20 years for a mid-series kink).  The
reported break year is therefore best read as "around this year the
warming rate had clearly shifted", not as the exact year it changed.

The candidate is restricted to the interior of the series (at least 10
years on each side), because a "break" at the very start or end is just
the series' overall slope in disguise.
"""

import numpy as np

from stats.mannkendall import mann_kendall

METHOD = (
    "CUSUM scan on the residuals of a straight-line (OLS) fit to the "
    "global mean series: the break candidate is the year where the "
    "cumulative sum of residuals is largest. Support is judged by a "
    "permutation test (2,000 shuffles of the residuals) — the break is "
    "supported only if fewer than 5% of the shuffles produce a larger "
    "peak. It is reported as an acceleration only when the Mann-Kendall "
    "+ Sen's-slope trend after the break is steeper than before it."
)

DEFAULT_N_PERM = 2000
DEFAULT_SEED = 20261114
_MIN_SIDE = 10   # years required on each side of a candidate break
_MIN_N_MK = 5    # years required in a segment to run Mann-Kendall on it


def _cusum_peak(residuals):
    """Largest |CUSUM| of a residual series, standardised as in the
    classic CUSUM test: cumulative sums divided by sigma * sqrt(n), so
    the peak is comparable across series.  Returns (peak, index)."""
    n = len(residuals)
    sigma = float(np.std(residuals, ddof=1))
    if sigma == 0 or n < 3:
        return 0.0, 0
    csum = np.cumsum(residuals) / (sigma * np.sqrt(n))
    # Endpoints are always ~0 and the sides must hold enough years:
    # only the interior can be a break.
    lo, hi = _MIN_SIDE - 1, n - _MIN_SIDE
    if hi <= lo:
        return 0.0, 0
    interior = np.abs(csum[lo:hi + 1])
    k = int(np.argmax(interior)) + lo
    return float(abs(csum[k])), k


def _segment_trend(years, values):
    """Mann-Kendall + Sen's slope for one segment, or null slopes when
    the segment is too short to test."""
    n = int(np.isfinite(values).sum())
    out = {"start_year": int(years[0]) if len(years) else None,
           "end_year": int(years[-1]) if len(years) else None,
           "slope_per_decade": None, "p_value": None,
           "significant": False, "n": n}
    if n < _MIN_N_MK:
        return out
    mk = mann_kendall(values, years, correction="hamed-rao", lag=3)
    out.update({"slope_per_decade": round(mk["slope_per_decade"], 4),
                "p_value": mk["p_value"],
                "significant": bool(mk["significant"]),
                "n": mk["n"]})
    return out


def detect_changepoint(years, values, n_perm=DEFAULT_N_PERM,
                       seed=DEFAULT_SEED, alpha=0.05):
    """Find a single change point in an annual series.

    Returns a dict with break_year (int or None), statistic (the CUSUM
    peak), p_value (permutation-test p), supported (bool), and
    before/after segment trends.  ``supported`` is True only when the
    permutation test clears alpha AND the after-segment slope exceeds
    the before-segment slope; otherwise break_year is None.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(years) & np.isfinite(values)
    years, values = years[finite], values[finite]
    order = np.argsort(years, kind="stable")
    years, values = years[order], values[order]
    n = len(values)

    empty_segments = {"before": _segment_trend(years[:0], values[:0]),
                      "after": _segment_trend(years[:0], values[:0])}
    if n < 2 * _MIN_SIDE + 1:
        return {"method": METHOD, "supported": False, "break_year": None,
                "statistic": 0.0, "p_value": None, "n": n, **empty_segments}

    # OLS line, residuals, CUSUM peak.
    slope, intercept = np.polyfit(years, values, 1)
    residuals = values - (slope * years + intercept)
    peak, k = _cusum_peak(residuals)

    # Permutation null: shuffle the residuals, re-peak each time.
    rng = np.random.default_rng(seed)
    peaks = np.empty(n_perm)
    for b in range(n_perm):
        peaks[b], _ = _cusum_peak(rng.permutation(residuals))
    p_value = float((1 + int((peaks >= peak).sum())) / (1 + n_perm))

    supported = bool(p_value < alpha)
    break_year = int(years[k]) if supported else None
    before = _segment_trend(years[:k], values[:k])
    after = _segment_trend(years[k:], values[k:])
    if supported:
        b_slope = before["slope_per_decade"]
        a_slope = after["slope_per_decade"]
        if b_slope is None or a_slope is None or a_slope <= b_slope:
            supported = False
            break_year = None
    return {"method": METHOD, "supported": supported,
            "break_year": break_year, "statistic": peak,
            "p_value": p_value, "n": n,
            "before": before, "after": after}
