import torch
from typing import List, Dict, Any, Optional, Tuple
import math


class BaseOptimizer(torch.optim.Optimizer):
    """
    Base class for optimizers with common functionality.
    """
    
    def __init__(self, params, defaults: Dict[str, Any]):
        """
        Initialize the base optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            defaults: Default values for optimizer hyperparameters
        """
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            
        Returns:
            loss if closure is provided, otherwise None
        """
        raise NotImplementedError("Subclasses must implement step()")
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Sets the gradients of all optimized parameters to zero.
        
        Args:
            set_to_none: Instead of setting to zero, set the grads to None.
        """
        super().zero_grad(set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a dict.
        """
        return super().state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the optimizer state.
        
        Args:
            state_dict: Optimizer state.
        """
        super().load_state_dict(state_dict)
    
    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a param group to the optimizer's param_groups.
        
        Args:
            param_group: Specifies what parameters should be optimized along with group specific options.
        """
        super().add_param_group(param_group)
    
    @staticmethod
    def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
        """
        Clips gradient norm of an iterable of parameters.
        
        Args:
            parameters: Iterable of parameters
            max_norm: Max norm of the gradients
            norm_type: Type of the used p-norm
            
        Returns:
            Total norm of the parameters (viewed as a single vector)
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def clip_grad_value_(parameters, clip_value: float):
        """
        Clips gradient of an iterable of parameters at specified value.
        
        Args:
            parameters: Iterable of parameters
            clip_value: Maximum allowed value of the gradients
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    
    def get_lr(self) -> List[float]:
        """
        Get current learning rates for all parameter groups.
        
        Returns:
            List of learning rates for each parameter group
        """
        return [group['lr'] for group in self.param_groups]
    
    def set_lr(self, lr: float):
        """
        Set learning rate for all parameter groups.
        
        Args:
            lr: Learning rate to set
        """
        for param_group in self.param_groups:
            param_group['lr'] = lr
    
    def get_momentum(self) -> List[float]:
        """
        Get momentum values for all parameter groups if available.
        
        Returns:
            List of momentum values for each parameter group
        """
        momenta = []
        for group in self.param_groups:
            if 'momentum' in group:
                momenta.append(group['momentum'])
            elif 'betas' in group:
                momenta.append(group['betas'][0])
            else:
                momenta.append(None)
        return momenta
    
    def get_state_stats(self) -> Dict[str, Any]:
        """
        Get statistics about optimizer state.
        
        Returns:
            Dictionary containing statistics about optimizer state
        """
        stats = {
            'num_param_groups': len(self.param_groups),
            'total_parameters': 0,
            'state_size': 0,
        }
        
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    stats['total_parameters'] += param.numel()
        
        if hasattr(self, 'state'):
            for param_id, param_state in self.state.items():
                for key, value in param_state.items():
                    if isinstance(value, torch.Tensor):
                        stats['state_size'] += value.numel() * value.element_size()
                    elif isinstance(value, (int, float)):
                        stats['state_size'] += 8  # Approximate size
        
        return stats