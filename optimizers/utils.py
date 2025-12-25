import torch
import numpy as np
from typing import Tuple, Dict, Any, List
import math


def gradient_consistency(grad: torch.Tensor, prev_grad: torch.Tensor, 
                         lambda_param: float = 0.01) -> float:
    """
    Calculate gradient consistency between current and previous gradient.
    
    Args:
        grad: Current gradient tensor
        prev_grad: Previous gradient tensor
        lambda_param: Consistency sensitivity parameter
        
    Returns:
        Consistency value between 0 and 1 (higher means more consistent)
    """
    if prev_grad is None or torch.all(prev_grad == 0):
        return 1.0  # First iteration, assume perfect consistency
    
    grad_change = torch.norm(grad - prev_grad).item()
    consistency = math.exp(-lambda_param * grad_change ** 2)
    return consistency


def compute_curvature_update(grad: torch.Tensor, exp_avg_sq: torch.Tensor, 
                            gamma: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute curvature update using BFGS-inspired diagonal approximation.
    
    Args:
        grad: Current gradient tensor
        exp_avg_sq: Exponential moving average of squared gradients
        gamma: Curvature update rate
        eps: Small constant for numerical stability
        
    Returns:
        Curvature update tensor
    """
    safe_denom = exp_avg_sq + eps
    curvature_update = gamma * (grad.pow(2) / safe_denom)
    return curvature_update


def compute_trust_ratio(predicted_reduction: float, actual_reduction: float,
                       eta_low: float = 0.8, eta_high: float = 1.2,
                       tau_increase: float = 1.5, tau_decrease: float = 0.5) -> float:
    """
    Compute trust region ratio based on predicted vs actual reduction.
    
    Args:
        predicted_reduction: Predicted reduction from quadratic model
        actual_reduction: Actual reduction observed
        eta_low: Lower threshold for ratio (decrease step if ratio < eta_low)
        eta_high: Upper threshold for ratio (increase step if ratio > eta_high)
        tau_increase: Factor to increase step size
        tau_decrease: Factor to decrease step size
        
    Returns:
        Trust ratio to multiply step size by
    """
    if predicted_reduction == 0:
        return 1.0
    
    ratio = actual_reduction / predicted_reduction
    
    if ratio < eta_low:
        # Poor model accuracy, decrease step size
        return tau_decrease
    elif ratio > eta_high:
        # Good model accuracy, increase step size
        return tau_increase
    else:
        # Acceptable model accuracy, keep step size
        return 1.0


def compute_adaptive_beta1(grad: torch.Tensor, prev_grad: torch.Tensor,
                          base_beta1: float, lambda_consistency: float) -> float:
    """
    Compute adaptive beta1 based on gradient consistency.
    
    Args:
        grad: Current gradient tensor
        prev_grad: Previous gradient tensor
        base_beta1: Base beta1 value
        lambda_consistency: Consistency sensitivity parameter
        
    Returns:
        Adaptive beta1 value
    """
    if prev_grad is None or torch.all(prev_grad == 0):
        return base_beta1
    
    consistency = gradient_consistency(grad, prev_grad, lambda_consistency)
    adaptive_beta1 = base_beta1 * consistency
    return adaptive_beta1


def compute_predicted_reduction(step: torch.Tensor, grad: torch.Tensor,
                               curvature: torch.Tensor) -> float:
    """
    Compute predicted reduction from quadratic model.
    
    Args:
        step: Proposed step (negative learning rate * direction)
        grad: Current gradient
        curvature: Curvature estimate (diagonal Hessian approximation)
        
    Returns:
        Predicted reduction value
    """
    # Quadratic model: Δ_pred = -g^T s + 0.5 * s^T H s
    linear_term = -torch.dot(grad.flatten(), step.flatten())
    quadratic_term = 0.5 * torch.dot(step.flatten() * curvature.flatten(), step.flatten())
    predicted_reduction = linear_term.item() + quadratic_term.item()
    return max(predicted_reduction, 0.0)  # Ensure non-negative


def compute_actual_reduction(current_loss: float, new_loss: float) -> float:
    """
    Compute actual reduction in loss.
    
    Args:
        current_loss: Loss before step
        new_loss: Loss after step
        
    Returns:
        Actual reduction value
    """
    return current_loss - new_loss


def initialize_optimizer_state(param: torch.Tensor, amsgrad: bool = False) -> Dict[str, Any]:
    """
    Initialize optimizer state for a parameter.
    
    Args:
        param: Parameter tensor
        amsgrad: Whether to use AMSGrad variant
        
    Returns:
        Initialized state dictionary
    """
    state = {
        'step': torch.tensor(0.0),
        'exp_avg': torch.zeros_like(param, memory_format=torch.preserve_format),
        'exp_avg_sq': torch.zeros_like(param, memory_format=torch.preserve_format),
        'curvature': torch.ones_like(param, memory_format=torch.preserve_format),
        'prev_grad': torch.zeros_like(param, memory_format=torch.preserve_format),
        'trust_ratio': torch.tensor(1.0),
        'predicted_reduction': torch.tensor(0.0),
        'actual_reduction': torch.tensor(0.0),
    }
    
    if amsgrad:
        state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    
    return state


def get_optimizer_statistics(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Get statistics from optimizer state.
    
    Args:
        optimizer: Optimizer instance
        
    Returns:
        Dictionary containing optimizer statistics
    """
    stats = {
        'total_parameters': 0,
        'state_size_bytes': 0,
        'learning_rates': [],
        'momentum_values': [],
        'trust_ratios': [],
    }
    
    for group in optimizer.param_groups:
        stats['learning_rates'].append(group.get('lr', 0.0))
        
        # Get momentum or beta1
        if 'momentum' in group:
            stats['momentum_values'].append(group['momentum'])
        elif 'betas' in group:
            stats['momentum_values'].append(group['betas'][0])
        
        for param in group['params']:
            if param.requires_grad:
                stats['total_parameters'] += param.numel()
                
                if param in optimizer.state:
                    param_state = optimizer.state[param]
                    for key, value in param_state.items():
                        if isinstance(value, torch.Tensor):
                            stats['state_size_bytes'] += value.numel() * value.element_size()
                        elif isinstance(value, (int, float)):
                            stats['state_size_bytes'] += 8
                        
                        if key == 'trust_ratio':
                            stats['trust_ratios'].append(value.item() if hasattr(value, 'item') else value)
    
    # Compute averages
    if stats['learning_rates']:
        stats['avg_learning_rate'] = np.mean(stats['learning_rates'])
    if stats['momentum_values']:
        stats['avg_momentum'] = np.mean(stats['momentum_values'])
    if stats['trust_ratios']:
        stats['avg_trust_ratio'] = np.mean(stats['trust_ratios'])
        stats['min_trust_ratio'] = np.min(stats['trust_ratios'])
        stats['max_trust_ratio'] = np.max(stats['trust_ratios'])
    
    return stats


def clip_gradients(parameters, max_norm: float = 1.0, norm_type: float = 2.0):
    """
    Clip gradients by norm.
    
    Args:
        parameters: Iterable of parameters
        max_norm: Max norm of the gradients
        norm_type: Type of the used p-norm
        
    Returns:
        Total norm of the parameters (before clipping)
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


def check_gradient_stats(grad: torch.Tensor) -> Dict[str, float]:
    """
    Check gradient statistics.
    
    Args:
        grad: Gradient tensor
        
    Returns:
        Dictionary containing gradient statistics
    """
    if grad is None:
        return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0, 'norm': 0.0}
    
    grad_np = grad.detach().cpu().numpy().flatten()
    
    stats = {
        'mean': float(np.mean(grad_np)),
        'std': float(np.std(grad_np)),
        'max': float(np.max(grad_np)),
        'min': float(np.min(grad_np)),
        'norm': float(torch.norm(grad).item()),
    }
    
    return stats