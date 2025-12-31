"""
NEXUS-Stable: Neural EXploration with Unified Scaling (Stable Version)

A simplified, more stable version of NEXUS optimizer that combines
multiple advanced techniques for faster convergence without instability issues.

Key Features:
1. Multi-Scale Momentum (MSM): Multiple momentum buffers with different time scales
2. Gradient Direction Consistency (GDC): Track gradient direction stability
3. Layer-wise Adaptation (LWA): Different adaptation strategies for different layers
4. Adaptive Step Size with Lookahead (ASSL): Combine current step with lookahead
5. Dynamic Weight Decay (DWD): Adapt weight decay based on parameter importance
6. Curvature-Aware Scaling (CAS): Estimate local curvature and adjust step sizes
7. Gradient Noise Injection (GNI): Controlled noise injection to escape local minima

This version removes the problematic meta-learning adaptive learning rate that was causing instability.
"""

import torch
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from .base import BaseOptimizer


class NEXUS_Stable(BaseOptimizer):
    """
    Neural EXploration with Unified Scaling (Stable Version).
    
    A simplified, more stable version of NEXUS optimizer.
    
    Key Features:
    1. Multi-Scale Momentum (MSM): Multiple momentum buffers with different time scales
    2. Gradient Direction Consistency (GDC): Track gradient direction stability
    3. Layer-wise Adaptation (LWA): Different adaptation strategies for different layers
    4. Adaptive Step Size with Lookahead (ASSL): Combine current step with lookahead
    5. Dynamic Weight Decay (DWD): Adapt weight decay based on parameter importance
    6. Curvature-Aware Scaling (CAS): Estimate local curvature and adjust step sizes
    7. Gradient Noise Injection (GNI): Controlled noise injection to escape local minima
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for multi-scale momentum (default: (0.9, 0.99, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum_scales: Number of momentum scales (default: 3)
        direction_consistency_alpha: Alpha for gradient direction consistency (default: 0.1)
        layer_adaptation: Whether to use layer-wise adaptation (default: True)
        lookahead_steps: Number of lookahead steps (default: 5)
        lookahead_alpha: Lookahead mixing coefficient (default: 0.5)
        noise_scale: Scale for gradient noise injection (default: 0.0)
        noise_decay: Decay rate for noise injection (default: 0.99)
        curvature_window: Window size for curvature estimation (default: 10)
        adaptive_weight_decay: Whether to use adaptive weight decay (default: True)
        weight_decay_alpha: Alpha for adaptive weight decay (default: 0.01)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        amsgrad: Whether to use the AMSGrad variant (default: False)
        state_precision: Precision for state buffers: 'fp32', 'fp16', or 'bf16' (default: 'fp32')
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float, float] = (0.9, 0.99, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, momentum_scales: int = 3,
                 direction_consistency_alpha: float = 0.1, layer_adaptation: bool = True,
                 lookahead_steps: int = 5, lookahead_alpha: float = 0.5,
                 noise_scale: float = 0.0, noise_decay: float = 0.99,
                 curvature_window: int = 10, adaptive_weight_decay: bool = True,
                 weight_decay_alpha: float = 0.01, max_grad_norm: float = 1.0,
                 amsgrad: bool = False, state_precision: str = 'fp32'):
        
        # Validate parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != momentum_scales:
            raise ValueError(f"betas must have {momentum_scales} elements, got {len(betas)}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 1 <= momentum_scales <= 5:
            raise ValueError(f"momentum_scales must be between 1 and 5, got {momentum_scales}")
        if not 0.0 <= direction_consistency_alpha <= 1.0:
            raise ValueError(f"Invalid direction_consistency_alpha: {direction_consistency_alpha}")
        if not 1 <= lookahead_steps:
            raise ValueError(f"lookahead_steps must be >= 1, got {lookahead_steps}")
        if not 0.0 <= lookahead_alpha <= 1.0:
            raise ValueError(f"Invalid lookahead_alpha: {lookahead_alpha}")
        if not 0.0 <= noise_scale:
            raise ValueError(f"Invalid noise_scale: {noise_scale}")
        if not 0.0 <= noise_decay <= 1.0:
            raise ValueError(f"Invalid noise_decay: {noise_decay}")
        if not 1 <= curvature_window:
            raise ValueError(f"curvature_window must be >= 1, got {curvature_window}")
        if not 0.0 <= weight_decay_alpha <= 1.0:
            raise ValueError(f"Invalid weight_decay_alpha: {weight_decay_alpha}")
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if state_precision not in ['fp32', 'fp16', 'bf16']:
            raise ValueError(f"Invalid state_precision: {state_precision}. "
                           f"Must be 'fp32', 'fp16', or 'bf16'")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            momentum_scales=momentum_scales, direction_consistency_alpha=direction_consistency_alpha,
            layer_adaptation=layer_adaptation, lookahead_steps=lookahead_steps,
            lookahead_alpha=lookahead_alpha, noise_scale=noise_scale,
            noise_decay=noise_decay, curvature_window=curvature_window,
            adaptive_weight_decay=adaptive_weight_decay, weight_decay_alpha=weight_decay_alpha,
            max_grad_norm=max_grad_norm, amsgrad=amsgrad, state_precision=state_precision
        )
        super().__init__(params, defaults)
        
        # Initialize lookahead parameters
        self._lookahead_step = 0
        self._slow_params = {}
        
        # Initialize noise scale tracking
        self._current_noise_scale = noise_scale
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum_scales', 3)
            group.setdefault('direction_consistency_alpha', 0.1)
            group.setdefault('layer_adaptation', True)
            group.setdefault('lookahead_steps', 5)
            group.setdefault('lookahead_alpha', 0.5)
            group.setdefault('noise_scale', 0.0)
            group.setdefault('noise_decay', 0.99)
            group.setdefault('curvature_window', 10)
            group.setdefault('adaptive_weight_decay', True)
            group.setdefault('weight_decay_alpha', 0.01)
            group.setdefault('max_grad_norm', 1.0)
            group.setdefault('amsgrad', False)
            group.setdefault('state_precision', 'fp32')
        
        # Initialize lookahead parameters if not present
        if not hasattr(self, '_lookahead_step'):
            self._lookahead_step = 0
        if not hasattr(self, '_slow_params'):
            self._slow_params = {}
        if not hasattr(self, '_current_noise_scale'):
            self._current_noise_scale = noise_scale
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            
        Returns:
            loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            # Extract parameters
            params_with_grad = []
            grads = []
            states = []
            state_steps = []
            
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum_scales = group['momentum_scales']
            direction_consistency_alpha = group['direction_consistency_alpha']
            layer_adaptation = group['layer_adaptation']
            lookahead_steps = group['lookahead_steps']
            lookahead_alpha = group['lookahead_alpha']
            noise_scale = group['noise_scale']
            noise_decay = group['noise_decay']
            curvature_window = group['curvature_window']
            adaptive_weight_decay = group['adaptive_weight_decay']
            weight_decay_alpha = group['weight_decay_alpha']
            max_grad_norm = group['max_grad_norm']
            amsgrad = group['amsgrad']
            state_precision = group['state_precision']
            
            # Collect parameters with gradients
            for p in group['params']:
                if p.grad is None:
                    continue
                
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    
                    # Initialize state buffer based on precision
                    if state_precision == 'fp16':
                        dtype = torch.float16
                    elif state_precision == 'bf16':
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float32
                    
                    # Multi-scale momentum buffers
                    state['momentums'] = [
                        torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
                        for _ in range(momentum_scales)
                    ]
                    
                    # Second moment buffer (for adaptive learning rates)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=dtype,
                                                          memory_format=torch.preserve_format)
                    
                    # Gradient direction consistency buffer
                    state['direction_consistency'] = torch.ones_like(p, dtype=dtype,
                                                                    memory_format=torch.preserve_format)
                    
                    # Curvature estimation buffer
                    state['curvature'] = torch.ones_like(p, dtype=dtype,
                                                        memory_format=torch.preserve_format)
                    
                    # Gradient history for curvature estimation
                    state['grad_history'] = torch.zeros(curvature_window, *p.shape,
                                                       dtype=dtype, device=p.device)
                    
                    # Parameter importance for adaptive weight decay
                    state['param_importance'] = torch.ones_like(p, dtype=dtype,
                                                              memory_format=torch.preserve_format)
                    
                    # For AMSGrad variant
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=dtype,
                                                                   memory_format=torch.preserve_format)
                
                states.append(state)
                state['step'] += 1
                state_steps.append(state['step'])
            
            if not params_with_grad:
                continue
            
            # Apply gradient clipping if specified
            if max_grad_norm is not None:
                self.clip_grad_norm_(params_with_grad, max_grad_norm)
            
            # Apply gradient noise injection
            if self._current_noise_scale > 0:
                for grad in grads:
                    noise = torch.randn_like(grad) * self._current_noise_scale
                    grad.add_(noise)
                # Decay noise scale
                self._current_noise_scale *= noise_decay
            
            # Compute layer-wise adaptation factors
            layer_factors = self._compute_layer_factors(params_with_grad, layer_adaptation)
            
            # Perform NEXUS-Stable update
            self._nexus_stable_update(
                params_with_grad,
                grads,
                states,
                state_steps,
                lr,
                betas,
                eps,
                weight_decay,
                momentum_scales,
                direction_consistency_alpha,
                layer_factors,
                adaptive_weight_decay,
                weight_decay_alpha,
                amsgrad
            )
            
            # Apply lookahead mechanism
            self._apply_lookahead(params_with_grad, lookahead_steps, lookahead_alpha)
        
        return loss
    
    def _compute_layer_factors(self, params: List[torch.Tensor], layer_adaptation: bool) -> List[float]:
        """
        Compute layer-wise adaptation factors based on parameter depth and size.
        """
        if not layer_adaptation:
            return [1.0] * len(params)
        
        factors = []
        total_params = sum(p.numel() for p in params)
        
        for i, p in enumerate(params):
            # Factor based on layer index (earlier layers get higher factor)
            depth_factor = 1.0 + 0.3 * (1.0 - i / len(params))
            
            # Factor based on parameter size (smaller layers get higher factor)
            size_factor = 1.0 + 0.3 * (1.0 - p.numel() / total_params)
            
            # Combined factor
            factor = depth_factor * size_factor
            factors.append(factor)
        
        return factors
    
    def _nexus_stable_update(self,
                           params: List[torch.Tensor],
                           grads: List[torch.Tensor],
                           states: List[Dict],
                           state_steps: List[torch.Tensor],
                           lr: float,
                           betas: Tuple[float, ...],
                           eps: float,
                           weight_decay: float,
                           momentum_scales: int,
                           direction_consistency_alpha: float,
                           layer_factors: List[float],
                           adaptive_weight_decay: bool,
                           weight_decay_alpha: float,
                           amsgrad: bool):
        """
        Perform the NEXUS-Stable parameter update.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            state = states[i]
            step = state_steps[i]
            layer_factor = layer_factors[i]
            
            # Apply adaptive weight decay
            if weight_decay != 0:
                if adaptive_weight_decay:
                    # Update parameter importance
                    param_importance = state['param_importance']
                    grad_norm = torch.norm(grad)
                    param_norm = torch.norm(param)
                    
                    # Importance based on gradient and parameter norms
                    importance_update = weight_decay_alpha * (grad_norm / (param_norm + eps))
                    param_importance.mul_(1 - weight_decay_alpha).add_(importance_update)
                    
                    # Apply adaptive weight decay using addcmul for element-wise operation
                    adaptive_wd = weight_decay * param_importance
                    grad = grad.addcmul_(param, adaptive_wd)
                else:
                    grad = grad.add(param, alpha=weight_decay)
            
            # Update multi-scale momentum buffers
            momentums = state['momentums']
            for j, (momentum, beta) in enumerate(zip(momentums, betas)):
                # Standard momentum update
                momentum.mul_(beta).add_(grad, alpha=1 - beta)
            
            # Update second moment
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul_(betas[-1]).addcmul_(grad, grad, value=1 - betas[-1])
            
            # For AMSGrad variant
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            # Update gradient direction consistency
            direction_consistency = state['direction_consistency']
            if step > 1:
                # Compute cosine similarity between current and previous gradient
                prev_grad = state['grad_history'][(int(step) - 2) % len(state['grad_history'])]
                grad_flat = grad.flatten()
                prev_grad_flat = prev_grad.flatten()
                
                # Cosine similarity
                similarity = torch.sum(grad_flat * prev_grad_flat) / (
                    torch.norm(grad_flat) * torch.norm(prev_grad_flat) + eps
                )
                
                # Update direction consistency with smoothing
                direction_consistency.mul_(1 - direction_consistency_alpha).add_(
                    torch.clamp(similarity, 0.0, 1.0), alpha=direction_consistency_alpha
                )
            
            # Update curvature estimation
            curvature = state['curvature']
            grad_history = state['grad_history']
            
            # Update gradient history
            grad_history[int(step) % len(grad_history)] = grad.clone()
            
            if step > 1:
                # Estimate curvature using gradient changes
                prev_grad = grad_history[(int(step) - 2) % len(grad_history)]
                grad_change = grad - prev_grad
                
                # Curvature estimate (simplified diagonal Hessian)
                curvature_update = torch.abs(grad_change) / (torch.abs(grad) + eps)
                curvature.mul_(0.9).add_(curvature_update * 0.1)
            
            # Combine multi-scale momentum with adaptive weights
            # Use exponential weighting for different scales
            combined_momentum = torch.zeros_like(momentums[0])
            total_weight = 0.0
            
            for j, momentum in enumerate(momentums):
                # Weight decreases with scale (short-term momentum gets higher weight)
                weight = math.exp(-j)
                combined_momentum.add_(momentum * weight)
                total_weight += weight
            
            combined_momentum.div_(total_weight)
            
            # Apply curvature-aware scaling
            curvature_sqrt = curvature.sqrt().add_(eps)
            scaled_momentum = combined_momentum / curvature_sqrt
            
            # Apply direction consistency scaling
            scaled_momentum.mul_(direction_consistency)
            
            # Apply layer-wise adaptation
            scaled_momentum.mul_(layer_factor)
            
            # Bias correction
            bias_correction1 = 1 - betas[-1] ** step
            bias_correction2 = 1 - betas[0] ** step
            step_size = lr / bias_correction1
            
            # Update parameter
            param.addcdiv_(scaled_momentum, denom, value=-step_size)
    
    def _apply_lookahead(self, params: List[torch.Tensor], lookahead_steps: int, lookahead_alpha: float):
        """
        Apply lookahead mechanism for better convergence.
        """
        self._lookahead_step += 1
        
        if self._lookahead_step % lookahead_steps == 0:
            # Update slow parameters
            for p in params:
                if p not in self._slow_params:
                    self._slow_params[p] = p.data.clone()
                
                if p in self._slow_params:
                    slow_param = self._slow_params[p]
                    # Mix fast and slow parameters
                    slow_param.mul_(1 - lookahead_alpha).add_(p.data * lookahead_alpha)
                    # Copy slow parameters back to fast parameters
                    p.data.copy_(slow_param)
    
    def get_adaptive_lr_stats(self) -> Dict[str, float]:
        """
        Get statistics about gradient direction consistency (replaces adaptive LR stats).
        
        Returns:
            Dictionary containing direction consistency statistics
        """
        stats = {
            'mean_consistency': 0.0,
            'max_consistency': float('-inf'),
            'min_consistency': float('inf'),
            'std_consistency': 0.0,
        }
        
        consistency_values = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'direction_consistency' in state:
                        consistency_flat = state['direction_consistency'].flatten()
                        consistency_values.extend(consistency_flat.tolist())
        
        if consistency_values:
            import numpy as np
            stats['mean_consistency'] = np.mean(consistency_values)
            stats['max_consistency'] = np.max(consistency_values)
            stats['min_consistency'] = np.min(consistency_values)
            stats['std_consistency'] = np.std(consistency_values)
        
        return stats
    
    def get_direction_consistency_stats(self) -> Dict[str, float]:
        """
        Get statistics about gradient direction consistency.
        
        Returns:
            Dictionary containing direction consistency statistics
        """
        stats = {
            'mean_consistency': 0.0,
            'max_consistency': float('-inf'),
            'min_consistency': float('inf'),
            'std_consistency': 0.0,
        }
        
        consistency_values = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'direction_consistency' in state:
                        consistency_flat = state['direction_consistency'].flatten()
                        consistency_values.extend(consistency_flat.tolist())
        
        if consistency_values:
            import numpy as np
            stats['mean_consistency'] = np.mean(consistency_values)
            stats['max_consistency'] = np.max(consistency_values)
            stats['min_consistency'] = np.min(consistency_values)
            stats['std_consistency'] = np.std(consistency_values)
        
        return stats
    
    def get_curvature_stats(self) -> Dict[str, float]:
        """
        Get statistics about curvature estimates.
        
        Returns:
            Dictionary containing curvature statistics
        """
        stats = {
            'mean_curvature': 0.0,
            'max_curvature': float('-inf'),
            'min_curvature': float('inf'),
            'std_curvature': 0.0,
        }
        
        curvature_values = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'curvature' in state:
                        curvature_flat = state['curvature'].flatten()
                        curvature_values.extend(curvature_flat.tolist())
        
        if curvature_values:
            import numpy as np
            stats['mean_curvature'] = np.mean(curvature_values)
            stats['max_curvature'] = np.max(curvature_values)
            stats['min_curvature'] = np.min(curvature_values)
            stats['std_curvature'] = np.std(curvature_values)
        
        return stats
    
    def get_momentum_stats(self) -> Dict[str, List[float]]:
        """
        Get statistics about multi-scale momentum buffers.
        
        Returns:
            Dictionary containing momentum statistics for each scale
        """
        stats = {}
        
        for scale_idx in range(3):
            momentum_values = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = self.state[p]
                        if 'momentums' in state and scale_idx < len(state['momentums']):
                            momentum_flat = state['momentums'][scale_idx].flatten()
                            momentum_values.extend(momentum_flat.tolist())
            
            if momentum_values:
                import numpy as np
                stats[f'scale_{scale_idx}'] = {
                    'mean': np.mean(momentum_values),
                    'std': np.std(momentum_values)
                }
            else:
                stats[f'scale_{scale_idx}'] = {
                    'mean': 0.0,
                    'std': 0.0
                }
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for the optimizer.
        
        Returns:
            Dictionary containing memory usage in bytes and MB
        """
        total_bytes = 0
        state_buffers = 0
        param_buffers = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_buffers += p.numel() * p.element_size()
                    
                    state = self.state[p]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state_buffers += value.numel() * value.element_size()
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, torch.Tensor):
                                    state_buffers += item.numel() * item.element_size()
        
        # Add slow parameters memory
        for p in self._slow_params:
            param_buffers += self._slow_params[p].numel() * self._slow_params[p].element_size()
        
        total_bytes = param_buffers + state_buffers
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'state_bytes': state_buffers,
            'state_mb': state_buffers / (1024 * 1024),
            'param_bytes': param_buffers,
            'param_mb': param_buffers / (1024 * 1024),
        }
    
    def reset_state(self):
        """
        Reset all state buffers to zero.
        Useful for fine-tuning or changing datasets.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'momentums' in state:
                        for momentum in state['momentums']:
                            momentum.zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                    if 'direction_consistency' in state:
                        state['direction_consistency'].fill_(1.0)
                    if 'curvature' in state:
                        state['curvature'].fill_(1.0)
                    if 'grad_history' in state:
                        state['grad_history'].zero_()
                    if 'param_importance' in state:
                        state['param_importance'].fill_(1.0)
                    if 'max_exp_avg_sq' in state:
                        state['max_exp_avg_sq'].zero_()
        
        # Reset lookahead parameters
        self._lookahead_step = 0
        self._slow_params = {}
        
        # Reset noise scale
        self._current_noise_scale = 0.0
