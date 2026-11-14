"""Public interface for the stats package."""

from stats.fdr import benjamini_hochberg, bh_significant
from stats.mannkendall import mann_kendall, mann_kendall_grid, sen_slope_ci, sens_slope

__all__ = [
    "mann_kendall",
    "mann_kendall_grid",
    "sens_slope",
    "sen_slope_ci",
    "benjamini_hochberg",
    "bh_significant",
]
