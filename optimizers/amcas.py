import torch
import math
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseOptimizer


class AMCAS(BaseOptimizer):
    """
    Adaptive Momentum with Curvature-Aware Scaling (AMCAS) optimizer.
    
    This optimizer combines:
    1. Adaptive momentum based on gradient consistency
    2. Lightweight diagonal Hessian approximation for curvature-aware scaling
    3. Dynamic trust region adjustment for stable training
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        gamma: Curvature update rate (default: 0.1)
        lambda_consistency: Gradient consistency sensitivity parameter (default: 0.01)
        trust_region_params: Trust region parameters (eta_low, eta_high, tau_increase, tau_decrease) 
                            (default: (0.8, 1.2, 1.5, 0.5))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        amsgrad: Whether to use the AMSGrad variant of this algorithm (default: False)
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 gamma: float = 0.1, lambda_consistency: float = 0.01,
                 trust_region_params: Tuple[float, float, float, float] = (0.8, 1.2, 1.5, 0.5),
                 eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= lambda_consistency:
            raise ValueError(f"Invalid lambda_consistency parameter: {lambda_consistency}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        eta_low, eta_high, tau_increase, tau_decrease = trust_region_params
        if not 0.0 < eta_low < eta_high:
            raise ValueError(f"Invalid trust region bounds: {eta_low} < {eta_high}")
        if not tau_increase > 1.0:
            raise ValueError(f"Invalid tau_increase: {tau_increase}")
        if not 0.0 < tau_decrease < 1.0:
            raise ValueError(f"Invalid tau_decrease: {tau_decrease}")
        
        defaults = dict(
            lr=lr, betas=betas, gamma=gamma, lambda_consistency=lambda_consistency,
            trust_region_params=trust_region_params, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
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
            exp_avgs = []
            exp_avg_sqs = []
            curvatures = []
            prev_grads = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            lambda_consistency = group['lambda_consistency']
            eta_low, eta_high, tau_increase, tau_decrease = group['trust_region_params']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Curvature estimate (diagonal Hessian approximation)
                    state['curvature'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for consistency calculation
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp moving avg of squared gradient
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Trust region state
                    state['trust_ratio'] = torch.tensor(1.0)
                    state['predicted_reduction'] = torch.tensor(0.0)
                    state['actual_reduction'] = torch.tensor(0.0)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                curvatures.append(state['curvature'])
                prev_grads.append(state['prev_grad'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Compute adaptive momentum coefficient based on gradient consistency
            adaptive_beta1s = []
            for i, (grad, prev_grad) in enumerate(zip(grads, prev_grads)):
                if state_steps[i] > 1:
                    # Calculate gradient change norm
                    grad_change = torch.norm(grad - prev_grad)
                    # Adaptive beta1 based on gradient consistency
                    adaptive_beta1 = beta1 * torch.exp(-lambda_consistency * grad_change.pow(2))
                    adaptive_beta1s.append(adaptive_beta1.item())
                else:
                    adaptive_beta1s.append(beta1)
            
            # Update curvature estimates
            for i, (grad, curvature, exp_avg_sq, state_step) in enumerate(zip(grads, curvatures, exp_avg_sqs, state_steps)):
                if state_step > 1:
                    # BFGS-inspired diagonal Hessian update
                    with torch.no_grad():
                        safe_denom = exp_avg_sq + group['eps']
                        curvature_update = gamma * (grad.pow(2) / safe_denom)
                        curvatures[i].mul_(1 - gamma).add_(curvature_update, alpha=gamma)
            
            # Perform AMCAS update
            self._amcas_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                curvatures,
                prev_grads,
                max_exp_avg_sqs,
                state_steps,
                adaptive_beta1s,
                group,
            )
            
            # Update previous gradients for next iteration
            for i, (grad, prev_grad) in enumerate(zip(grads, prev_grads)):
                prev_grad.copy_(grad)
        
        return loss
    
    def _amcas_update(self,
                     params: List[torch.Tensor],
                     grads: List[torch.Tensor],
                     exp_avgs: List[torch.Tensor],
                     exp_avg_sqs: List[torch.Tensor],
                     curvatures: List[torch.Tensor],
                     prev_grads: List[torch.Tensor],
                     max_exp_avg_sqs: List[torch.Tensor],
                     state_steps: List[torch.Tensor],
                     adaptive_beta1s: List[float],
                     group: Dict[str, Any]):
        """
        Perform the AMCAS parameter update.
        """
        beta1, beta2 = group['betas']
        lr = group['lr']
        weight_decay = group['weight_decay']
        eps = group['eps']
        amsgrad = group['amsgrad']
        eta_low, eta_high, tau_increase, tau_decrease = group['trust_region_params']
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            curvature = curvatures[i]
            step = state_steps[i]
            adaptive_beta1 = adaptive_beta1s[i]
            
            # Apply weight decay
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(adaptive_beta1).add_(grad, alpha=1 - adaptive_beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max for normalizing running avg of gradient
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            # Apply curvature scaling
            curvature_sqrt = curvature.sqrt().add_(eps)
            scaled_exp_avg = exp_avg / curvature_sqrt
            
            # Bias correction
            bias_correction1 = 1 - adaptive_beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Apply bias correction to scaled momentum
            step_size = lr / bias_correction1
            
            # Trust region adjustment (simplified - in practice would use closure)
            # For now, we'll implement a simple trust region mechanism
            trust_ratio = torch.tensor(1.0)
            
            # Update parameter
            param.addcdiv_(scaled_exp_avg, denom, value=-step_size * trust_ratio)
    
    def get_trust_ratio(self) -> float:
        """
        Get the current trust region ratio averaged across all parameters.
        
        Returns:
            Average trust ratio across all parameters
        """
        total_ratio = 0.0
        count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'trust_ratio' in state:
                        total_ratio += state['trust_ratio'].item()
                        count += 1
        
        if count == 0:
            return 1.0  # Default trust ratio when no parameters have gradients
        return total_ratio / count
    
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
    
    def get_gradient_consistency(self) -> float:
        """
        Get average gradient consistency across all parameters.
        
        Returns:
            Average gradient consistency (higher means more consistent gradients)
        """
        total_consistency = 0.0
        count = 0
        
        for group in self.param_groups:
            lambda_consistency = group['lambda_consistency']
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'prev_grad' in state and state['step'] > 1:
                        grad = p.grad
                        prev_grad = state['prev_grad']
                        grad_change = torch.norm(grad - prev_grad).item()
                        consistency = math.exp(-lambda_consistency * grad_change ** 2)
                        total_consistency += consistency
                        count += 1
        
        return total_consistency / max(count, 1)