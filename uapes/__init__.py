"""
UAPES: Uncertainty-Aware Path Exploration Sampling
Enhanced Sampling Method

This package provides an implementation of the UAPES algorithm.

Modules:
--------
- bnn_committor: Bayesian neural network for committor prediction
- uncertainty_bias: Uncertainty-based bias computation
- muller_brown: Müller-Brown potential and dynamics
- sampling: Main UAPES sampling algorithm

Author: Jaafar Mehrez (2026)
Email: jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org
"""

__version__ = "0.1.0"
__author__ = "Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn>"

from .muller_brown import MullerBrownPotential, LangevinDynamics
from .sampling import UAPESConfig, UAPESSampler, SimpleCommittorModel

__all__ = [
    "MullerBrownPotential",
    "LangevinDynamics",
    "UAPESConfig",
    "UAPESSampler",
    "SimpleCommittorModel",
]
