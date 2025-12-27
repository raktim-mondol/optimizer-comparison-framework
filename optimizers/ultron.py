import torch
import math
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseOptimizer


class ULTRON(BaseOptimizer):
    """
    Ultra-Light Trust-Region Optimizer with Normalization (ULTRON).
    
    A computationally inexpensive optimizer designed for maximum efficiency
    while maintaining competitive performance with state-of-the-art methods.
    
    Key Features:
    1. Sign-based updates for extreme computational efficiency
    2. Adaptive gradient normalization for stable training
    3. Minimal state (only momentum buffer)
    4. Built-in gradient clipping for robustness
    5. Learning rate warmup and decay support
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        clip_threshold: Maximum absolute value for updates (default: 1.0)
        normalize_gradients: Whether to normalize gradients by their RMS (default: True)
        warmup_steps: Number of warmup steps for learning rate (default: 0)
        decay_steps: Number of steps for learning rate decay (default: 0)
        decay_rate: Learning rate decay rate (default: 0.95)
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, clip_threshold: float = 1.0,
                 normalize_gradients: bool = True, warmup_steps: int = 0,
                 decay_steps: int = 0, decay_rate: float = 0.95):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < clip_threshold:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            clip_threshold=clip_threshold, normalize_gradients=normalize_gradients,
            warmup_steps=warmup_steps, decay_steps=decay_steps, decay_rate=decay_rate
        )
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('normalize_gradients', True)
            group.setdefault('warmup_steps', 0)
            group.setdefault('decay_steps', 0)
            group.setdefault('decay_rate', 0.95)
    
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
            params_with_grad = []
            grads = []
            momentums = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            clip_threshold = group['clip_threshold']
            normalize_gradients = group['normalize_gradients']
            warmup_steps = group['warmup_steps']
            decay_steps = group['decay_steps']
            decay_rate = group['decay_rate']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    # Exponential moving average of gradient values (momentum)
                    state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (for normalization)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                momentums.append(state['momentum'])
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Apply weight decay and compute effective learning rate
            effective_lr = lr
            if warmup_steps > 0:
                for step in state_steps:
                    if step <= warmup_steps:
                        effective_lr = lr * (step / warmup_steps)
            
            if decay_steps > 0:
                for step in state_steps:
                    if step > warmup_steps:
                        decay_factor = decay_rate ** ((step - warmup_steps) / decay_steps)
                        effective_lr = lr * decay_factor
            
            # Perform ULTRON update
            self._ultron_update(
                params_with_grad,
                grads,
                momentums,
                state_steps,
                beta1,
                beta2,
                effective_lr,
                weight_decay,
                eps,
                clip_threshold,
                normalize_gradients,
            )
        
        return loss
    
    def _ultron_update(self,
                      params: List[torch.Tensor],
                      grads: List[torch.Tensor],
                      momentums: List[torch.Tensor],
                      state_steps: List[torch.Tensor],
                      beta1: float,
                      beta2: float,
                      lr: float,
                      weight_decay: float,
                      eps: float,
                      clip_threshold: float,
                      normalize_gradients: bool):
        """
        Perform the ULTRON parameter update.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            momentum = momentums[i]
            step = state_steps[i]
            
            # Apply weight decay
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            
            # Update momentum (first moment)
            momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Normalize gradients if enabled
            if normalize_gradients:
                # Update second moment for normalization
                state = self.state[param]
                if 'exp_avg_sq' in state:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction2 = 1 - beta2 ** step
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                    
                    # Normalize momentum
                    normalized_momentum = momentum / denom
                else:
                    normalized_momentum = momentum
            else:
                normalized_momentum = momentum
            
            # Apply sign-based update with clipping
            update = torch.sign(normalized_momentum) * torch.clamp(
                torch.abs(normalized_momentum), max=clip_threshold
            )
            
            # Update parameter
            param.add_(update, alpha=-lr)
    
    def get_clip_threshold(self) -> float:
        """
        Get the current clip threshold.
        
        Returns:
            Clip threshold value
        """
        return self.param_groups[0]['clip_threshold'] if self.param_groups else 1.0
    
    def set_clip_threshold(self, clip_threshold: float):
        """
        Set clip threshold for all parameter groups.
        
        Args:
            clip_threshold: Clip threshold to set
        """
        if not 0.0 < clip_threshold:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        
        for param_group in self.param_groups:
            param_group['clip_threshold'] = clip_threshold
    
    def get_normalization_stats(self) -> Dict[str, float]:
        """
        Get statistics about gradient normalization.
        
        Returns:
            Dictionary containing normalization statistics
        """
        stats = {
            'mean_gradient_norm': 0.0,
            'max_gradient_norm': float('-inf'),
            'min_gradient_norm': float('inf'),
            'num_parameters': 0,
        }
        
        gradient_norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad).item()
                    gradient_norms.append(grad_norm)
                    stats['num_parameters'] += p.numel()
        
        if gradient_norms:
            stats['mean_gradient_norm'] = sum(gradient_norms) / len(gradient_norms)
            stats['max_gradient_norm'] = max(gradient_norms)
            stats['min_gradient_norm'] = min(gradient_norms)
        
        return stats
    
    def get_update_stats(self) -> Dict[str, float]:
        """
        Get statistics about parameter updates.
        
        Returns:
            Dictionary containing update statistics
        """
        stats = {
            'mean_update_magnitude': 0.0,
            'max_update_magnitude': float('-inf'),
            'min_update_magnitude': float('inf'),
            'clipped_updates_ratio': 0.0,
        }
        
        update_magnitudes = []
        clipped_count = 0
        total_count = 0
        
        for group in self.param_groups:
            clip_threshold = group['clip_threshold']
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'momentum' in state:
                        momentum = state['momentum']
                        update_mag = torch.abs(momentum).mean().item()
                        update_magnitudes.append(update_mag)
                        
                        # Count clipped updates
                        clipped = torch.sum(torch.abs(momentum) > clip_threshold).item()
                        total = momentum.numel()
                        clipped_count += clipped
                        total_count += total
        
        if update_magnitudes:
            stats['mean_update_magnitude'] = sum(update_magnitudes) / len(update_magnitudes)
            stats['max_update_magnitude'] = max(update_magnitudes)
            stats['min_update_magnitude'] = min(update_magnitudes)
        
        if total_count > 0:
            stats['clipped_updates_ratio'] = clipped_count / total_count
        
        return stats
    
    def reset_momentum(self):
        """
        Reset momentum buffers to zero.
        Useful for fine-tuning or changing datasets.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'momentum' in state:
                        state['momentum'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
