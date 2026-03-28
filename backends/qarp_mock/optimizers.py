"""
qarp_mock.optimizers — ScipyOptimizer and RotosolveOptimizer
============================================================
Mirrors ``qarp.optimizers`` from QARP v0.4.3.
"""

from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from scipy.optimize import minimize


class ScipyOptimizer:
    """
    Mock of qarp.optimizers.ScipyOptimizer.

    Wraps scipy.optimize.minimize with QARP's interface.

    Parameters
    ----------
    method  : scipy optimizer method (e.g. 'COBYLA', 'L-BFGS-B', 'Nelder-Mead')
    options : dict passed to scipy.optimize.minimize options
    """

    def __init__(self, method: str = "COBYLA",
                 options: Optional[dict] = None):
        self.method = method
        self.options = dict(options) if options else {}

    def minimize(self, objective_function: Callable,
                 initial_parameters=None,
                 callback: Callable = None,
                 gradient: Callable = None,
                 tol: float = None,
                 bounds=None):
        return minimize(
            fun=objective_function,
            x0=np.array(initial_parameters),
            method=self.method,
            options=self.options,
            callback=callback,
            jac=gradient,
            tol=tol,
            bounds=bounds,
        )


class RotosolveOptimizer:
    """
    Mock of qarp.optimizers.RotosolveOptimizer.
    Falls back to COBYLA in mock mode.
    """

    def __init__(self, maxiter: int = 100, tol: float = 1e-7,
                 lr: float = 1, schedule: str = None,
                 gamma: float = 0.99, power: int = 2,
                 verbose: bool = False):
        self.maxiter = maxiter
        self.tol = tol

    def minimize(self, objective_function, initial_parameters=None,
                 callback=None, gradient=None):
        return minimize(
            fun=objective_function,
            x0=np.array(initial_parameters),
            method="COBYLA",
            options={"maxiter": self.maxiter},
        )
