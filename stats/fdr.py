"""Benjamini-Hochberg false-discovery control for map cells.

Testing thousands of grid cells at p < 0.05 flags hundreds by luck alone.
The Benjamini-Hochberg procedure (Benjamini & Hochberg 1995) converts the
raw p-values into adjusted p-values (q-values): a cell with q <= alpha is
a discovery you can trust at that false-discovery rate.  NaN p-values
(cells with too little data) pass through as NaN and are never discoveries.
"""

import numpy as np

__all__ = ["benjamini_hochberg", "bh_significant"]


def benjamini_hochberg(pvalues):
    """BH adjusted p-values (q-values), same shape as the input.

    Standard step-up procedure: sort the m finite p-values, set
    q_(i) = p_(i) * m / i, then enforce monotonicity from the largest
    p-value downwards and clip to [0, 1].
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    flat = p.ravel()
    mask = np.isfinite(flat)
    pv = flat[mask]
    n = len(pv)
    if n == 0:
        return out
    pv = np.clip(pv, 0.0, 1.0)
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    adjusted = np.empty(n)
    adjusted[order] = q
    out.ravel()[mask] = adjusted
    return out


def bh_significant(pvalues, alpha=0.05):
    """Boolean mask of BH discoveries at false-discovery rate alpha."""
    q = benjamini_hochberg(pvalues)
    return np.isfinite(q) & (q <= alpha)
