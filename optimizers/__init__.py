"""
AMCAS Optimizer Package
"""

from .amcas import AMCAS
from .base import BaseOptimizer
from .ultron import ULTRON
from .ultron_v2 import ULTRON_V2
from .nexus import NEXUS
from .utils import gradient_consistency, compute_curvature_update, compute_trust_ratio, compute_adaptive_beta1, compute_predicted_reduction, compute_actual_reduction, initialize_optimizer_state, get_optimizer_statistics, check_gradient_stats, clip_gradients

__all__ = [
    'AMCAS',
    'BaseOptimizer',
    'ULTRON',
    'ULTRON_V2',
    'NEXUS',
    'gradient_consistency',
    'compute_curvature_update',
    'compute_trust_ratio',
    'compute_adaptive_beta1',
    'compute_predicted_reduction',
    'compute_actual_reduction',
    'initialize_optimizer_state',
    'get_optimizer_statistics',
    'check_gradient_stats',
    'clip_gradients',
]

__version__ = '0.2.0'