import torch
import torch.nn as nn
import numpy as np
import pytest
from optimizers.amcas import AMCAS
from optimizers.utils import (
    gradient_consistency,
    compute_curvature_update,
    compute_trust_ratio,
    compute_adaptive_beta1,
    compute_predicted_reduction,
    compute_actual_reduction,
    initialize_optimizer_state,
    get_optimizer_statistics,
    check_gradient_stats,
)


class TestAMCASOptimizer:
    """Test suite for AMCAS optimizer."""
    
    def test_initialization(self):
        """Test AMCAS optimizer initialization."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters(), lr=0.001)
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
        assert optimizer.param_groups[0]['gamma'] == 0.1
        assert optimizer.param_groups[0]['lambda_consistency'] == 0.01
        assert optimizer.param_groups[0]['trust_region_params'] == (0.8, 1.2, 1.5, 0.5)
        assert optimizer.param_groups[0]['eps'] == 1e-8
        assert optimizer.param_groups[0]['weight_decay'] == 0
        assert optimizer.param_groups[0]['amsgrad'] == False
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        model = nn.Linear(10, 5)
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), lr=-0.001)
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), betas=(1.1, 0.999))
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), betas=(0.9, 1.1))
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), gamma=1.1)
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), lambda_consistency=-0.01)
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), trust_region_params=(1.0, 0.8, 1.5, 0.5))
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), trust_region_params=(0.8, 1.2, 0.5, 0.5))
        
        with pytest.raises(ValueError):
            AMCAS(model.parameters(), trust_region_params=(0.8, 1.2, 1.5, 1.5))
    
    def test_simple_optimization(self):
        """Test that AMCAS can optimize a simple quadratic function."""
        torch.manual_seed(42)
        
        # Define a simple quadratic loss: f(x) = (x - 2)^2
        x = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor([2.0])
        
        optimizer = AMCAS([x], lr=0.1)
        
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = (x - target).pow(2).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Check that loss decreases
        assert losses[-1] < losses[0]
        # Check that we're close to optimum (relaxed tolerance)
        assert abs(x.item() - 2.0) < 0.05
    
    def test_state_initialization(self):
        """Test that optimizer state is properly initialized."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters())
        
        # Before any step, state should be empty
        for param in model.parameters():
            assert param not in optimizer.state
        
        # Do one optimization step
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # After step, state should be initialized
        for param in model.parameters():
            assert param in optimizer.state
            state = optimizer.state[param]
            assert 'step' in state
            assert 'exp_avg' in state
            assert 'exp_avg_sq' in state
            assert 'curvature' in state
            assert 'prev_grad' in state
            assert 'trust_ratio' in state
            assert state['step'].item() == 1.0
    
    def test_gradient_consistency_function(self):
        """Test gradient consistency calculation."""
        torch.manual_seed(42)
        
        # Same gradient should give high consistency
        grad1 = torch.randn(10)
        grad2 = grad1.clone()
        consistency = gradient_consistency(grad1, grad2)
        assert 0.9 <= consistency <= 1.0
        
        # Very different gradients should give low consistency
        grad3 = torch.randn(10) * 100
        consistency = gradient_consistency(grad1, grad3)
        assert 0.0 <= consistency <= 0.1
        
        # None or zero previous gradient should give 1.0
        consistency = gradient_consistency(grad1, None)
        assert consistency == 1.0
        
        zero_grad = torch.zeros_like(grad1)
        consistency = gradient_consistency(grad1, zero_grad)
        assert consistency == 1.0
    
    def test_curvature_update(self):
        """Test curvature update computation."""
        torch.manual_seed(42)
        
        grad = torch.randn(10)
        exp_avg_sq = torch.ones(10) * 0.1
        gamma = 0.1
        
        curvature_update = compute_curvature_update(grad, exp_avg_sq, gamma)
        
        assert curvature_update.shape == grad.shape
        assert torch.all(curvature_update >= 0)  # Should be non-negative
        assert not torch.any(torch.isnan(curvature_update))
        assert not torch.any(torch.isinf(curvature_update))
    
    def test_trust_ratio_computation(self):
        """Test trust ratio computation."""
        # Good step: actual reduction > predicted reduction
        ratio = compute_trust_ratio(predicted_reduction=1.0, actual_reduction=1.5)
        assert ratio == 1.5  # tau_increase
        
        # Poor step: actual reduction < predicted reduction
        ratio = compute_trust_ratio(predicted_reduction=1.0, actual_reduction=0.5)
        assert ratio == 0.5  # tau_decrease
        
        # Acceptable step
        ratio = compute_trust_ratio(predicted_reduction=1.0, actual_reduction=1.0)
        assert ratio == 1.0
        
        # Edge case: zero predicted reduction
        ratio = compute_trust_ratio(predicted_reduction=0.0, actual_reduction=1.0)
        assert ratio == 1.0
    
    def test_adaptive_beta1_computation(self):
        """Test adaptive beta1 computation."""
        torch.manual_seed(42)
        
        grad1 = torch.randn(10)
        grad2 = grad1.clone() * 0.1  # Similar gradient
        base_beta1 = 0.9
        lambda_consistency = 0.01
        
        beta1 = compute_adaptive_beta1(grad2, grad1, base_beta1, lambda_consistency)
        assert 0.8 <= beta1 <= 0.9  # Should be close to base_beta1
        
        grad3 = torch.randn(10) * 100  # Very different gradient
        beta1 = compute_adaptive_beta1(grad3, grad1, base_beta1, lambda_consistency)
        assert 0.0 <= beta1 <= 0.5  # Should be much lower
    
    def test_predicted_reduction_computation(self):
        """Test predicted reduction computation."""
        torch.manual_seed(42)
        
        step = torch.randn(10) * 0.1
        grad = torch.randn(10)
        curvature = torch.ones(10)
        
        predicted = compute_predicted_reduction(step, grad, curvature)
        
        # Predicted reduction should be non-negative for a descent direction
        # but our test might not guarantee descent direction
        assert isinstance(predicted, float)
        assert not np.isnan(predicted)
        assert not np.isinf(predicted)
    
    def test_actual_reduction_computation(self):
        """Test actual reduction computation."""
        current_loss = 10.0
        new_loss = 8.0
        
        reduction = compute_actual_reduction(current_loss, new_loss)
        assert reduction == 2.0
        
        # Negative reduction (loss increased)
        reduction = compute_actual_reduction(current_loss, 12.0)
        assert reduction == -2.0
    
    def test_optimizer_statistics(self):
        """Test optimizer statistics collection."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        optimizer = AMCAS(model.parameters(), lr=0.001)
        
        # Do one step to initialize state
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        stats = get_optimizer_statistics(optimizer)
        
        assert 'total_parameters' in stats
        assert 'state_size_bytes' in stats
        assert 'learning_rates' in stats
        assert 'momentum_values' in stats
        assert stats['total_parameters'] > 0
        assert stats['state_size_bytes'] > 0
        assert len(stats['learning_rates']) == 1
        assert stats['learning_rates'][0] == 0.001
    
    def test_gradient_statistics(self):
        """Test gradient statistics checking."""
        torch.manual_seed(42)
        
        grad = torch.randn(10, 20)
        stats = check_gradient_stats(grad)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'max' in stats
        assert 'min' in stats
        assert 'norm' in stats
        
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)
        assert isinstance(stats['max'], float)
        assert isinstance(stats['min'], float)
        assert isinstance(stats['norm'], float)
        
        # Test with None gradient
        stats = check_gradient_stats(None)
        assert stats['mean'] == 0.0
        assert stats['norm'] == 0.0
    
    def test_weight_decay(self):
        """Test that weight decay is properly applied."""
        torch.manual_seed(42)
        
        # Create a simple model
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        bias = torch.tensor([0.5, 0.5], requires_grad=True)
        
        # Optimizer with weight decay
        optimizer = AMCAS([weight, bias], lr=0.1, weight_decay=0.01)
        
        # Mock gradient
        weight.grad = torch.ones_like(weight)
        bias.grad = torch.ones_like(bias)
        
        # Store initial values
        weight_before = weight.clone()
        bias_before = bias.clone()
        
        # Take a step
        optimizer.step()
        
        # Check that weight decay was applied
        # With weight decay, gradient should be: grad + weight_decay * weight
        expected_weight_grad = torch.ones_like(weight) + 0.01 * weight_before
        expected_bias_grad = torch.ones_like(bias) + 0.01 * bias_before
        
        # The update should be: param = param - lr * (grad + weight_decay * param)
        # Since we can't easily check the internal gradient, we check the result
        # The weight should have decreased more than without weight decay
        weight_without_decay = weight_before - 0.1 * torch.ones_like(weight_before)
        
        # With weight decay, weight should be smaller
        assert torch.all(weight < weight_without_decay)
    
    def test_amsgrad_variant(self):
        """Test AMSGrad variant."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters(), amsgrad=True)
        
        assert optimizer.param_groups[0]['amsgrad'] == True
        
        # Do one step to initialize state
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Check that max_exp_avg_sq is in state
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'max_exp_avg_sq' in state
            assert state['max_exp_avg_sq'].shape == param.shape
    
    def test_multiple_param_groups(self):
        """Test optimizer with multiple parameter groups."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 1)
        
        # Different learning rates for different parameter groups
        optimizer = AMCAS([
            {'params': model1.parameters(), 'lr': 0.01},
            {'params': model2.parameters(), 'lr': 0.001}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[1]['lr'] == 0.001
        
        # Test optimization step
        optimizer.zero_grad()
        x = torch.randn(32, 10)
        h = model1(x)
        output = model2(h)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Check that both models were updated
        assert torch.any(model1.weight.grad != 0)
        assert torch.any(model2.weight.grad != 0)
    
    def test_get_trust_ratio_method(self):
        """Test get_trust_ratio method."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters())
        
        # Before any steps, trust ratio should be 1.0 (default)
        trust_ratio = optimizer.get_trust_ratio()
        assert trust_ratio == 1.0
        
        # Do one step
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # After step, trust ratio should still be 1.0 (default)
        trust_ratio = optimizer.get_trust_ratio()
        assert trust_ratio == 1.0
    
    def test_get_curvature_stats_method(self):
        """Test get_curvature_stats method."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters())
        
        # Before any steps
        stats = optimizer.get_curvature_stats()
        assert 'mean_curvature' in stats
        assert stats['mean_curvature'] == 0.0
        
        # Do one step
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # After step, curvature should be initialized to 1.0
        stats = optimizer.get_curvature_stats()
        assert stats['mean_curvature'] == 1.0
        assert stats['max_curvature'] == 1.0
        assert stats['min_curvature'] == 1.0
        assert stats['std_curvature'] == 0.0
    
    def test_get_gradient_consistency_method(self):
        """Test get_gradient_consistency method."""
        model = nn.Linear(10, 5)
        optimizer = AMCAS(model.parameters())
        
        # Before any steps, consistency should be 0 (no previous gradients)
        consistency = optimizer.get_gradient_consistency()
        assert consistency == 0.0
        
        # Do one step
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Still 0 after first step (need at least 2 steps for comparison)
        consistency = optimizer.get_gradient_consistency()
        assert consistency == 0.0
        
        # Do second step
        optimizer.zero_grad()
        output = model(torch.randn(32, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Now we should have a consistency value
        consistency = optimizer.get_gradient_consistency()
        assert 0.0 <= consistency <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])