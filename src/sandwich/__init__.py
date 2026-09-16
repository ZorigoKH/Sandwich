"""sandwich: an econometrics engine in NumPy.

OLS, sandwich (HC / cluster / HAC) standard errors, IV/2SLS, panel fixed effects,
first differences and Wald tests, verified against statsmodels and linearmodels.
"""

from .covariance import COV_TYPES, CovarianceResult, compute_covariance
from .iv import IV2SLS, IVDiagnostics
from .ols import OLS, add_constant
from .panel import FirstDifferenceOLS, PanelOLS
from .results import RegressionResults, WaldTestResult

__version__ = "0.2.0"
__all__ = [
    "COV_TYPES",
    "IV2SLS",
    "IVDiagnostics",
    "OLS",
    "CovarianceResult",
    "FirstDifferenceOLS",
    "PanelOLS",
    "RegressionResults",
    "WaldTestResult",
    "add_constant",
    "compute_covariance",
]
