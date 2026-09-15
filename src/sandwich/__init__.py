"""sandwich: an econometrics engine in NumPy.

OLS, sandwich (HC / cluster / HAC) standard errors, IV/2SLS and Wald tests,
verified against statsmodels and linearmodels.
"""

from .covariance import COV_TYPES, CovarianceResult, compute_covariance
from .iv import IV2SLS, IVDiagnostics
from .ols import OLS, add_constant
from .results import RegressionResults, WaldTestResult

__version__ = "0.1.0"
__all__ = [
    "COV_TYPES",
    "IV2SLS",
    "IVDiagnostics",
    "OLS",
    "CovarianceResult",
    "RegressionResults",
    "WaldTestResult",
    "add_constant",
    "compute_covariance",
]
