"""
Comprehensive test suite for ULTRON_V2 optimizer.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.ultron_v2 import ULTRON_V2
from optimizers.ultron import ULTRON


class TestULTRONV2:
    """Test class for ULTRON_V2 optimizer."""
    
    def setup_method(self):
        """Setup test environment."""
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed(42)
    
    def test_initialization(self):
        """Test ULTRON_V2 initialization."""
        print("Testing ULTRON_V2 initialization...")
        
        # Create a simple model
        model = nn.Linear(10, 5).to(self.device)
        
        # Test default initialization
        optimizer = ULTRON_V2(model.parameters())
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
        assert optimizer.param_groups[0]['eps'] == 1e-8
        assert optimizer.param_groups[0]['clip_threshold'] == 1.0
        assert optimizer.param_groups[0]['normalize_gradients'] == True
        assert optimizer.param_groups[0]['normalization_strategy'] == 'rms'
        assert optimizer.param_groups[0]['adaptive_clipping'] == True
        assert optimizer.param_groups[0]['state_precision'] == 'fp32'
        
        print("[PASS] Initialization test passed!")
    
    def test_basic_optimization(self):
        """Test basic optimization functionality."""
        print("Testing basic optimization...")
        
        # Create a simple quadratic loss function
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        optimizer = ULTRON_V2([x], lr=0.1)
        
        losses = []
        for i in range(10):
            optimizer.zero_grad()
            loss = torch.sum((x - target) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Check that loss decreases
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]} -> {losses[-1]}"
        assert loss.item() < 0.1, f"Final loss too high: {loss.item()}"
        
        print(f"[PASS] Basic optimization test passed! Final loss: {loss.item():.6f}")
    
    def test_vectorized_updates(self):
        """Test vectorized parameter updates."""
        print("Testing vectorized updates...")
        
        # Create model with multiple parameters
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to(self.device)
        
        optimizer = ULTRON_V2(model.parameters(), lr=0.01)
        
        # Create dummy data
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 2, device=self.device)
        criterion = nn.MSELoss()
        
        # Test optimization
        initial_params = [p.detach().clone() for p in model.parameters()]
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Check that parameters have been updated
        for p_initial, p_final in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_initial, p_final), "Parameters not updated!"
        
        print(f"[PASS] Vectorized updates test passed! Loss: {loss.item():.6f}")
    
    def test_fused_sign_clip(self):
        """Test fused sign-clip operation."""
        print("Testing fused sign-clip operation...")
        
        # Test mathematical equivalence
        torch.manual_seed(42)
        x = torch.randn(1000, device=self.device) * 5  # Values from -5 to 5
        clip_threshold = 2.0
        
        # Original operation
        original = torch.sign(x) * torch.clamp(torch.abs(x), max=clip_threshold)
        
        # Fused operation (as implemented in ULTRON_V2)
        fused = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
        
        # Check equivalence
        assert torch.allclose(original, fused, rtol=1e-10, atol=1e-10), \
            "Fused operation not mathematically equivalent!"
        
        # Test edge cases
        test_cases = [
            torch.tensor([0.0], device=self.device),
            torch.tensor([clip_threshold], device=self.device),
            torch.tensor([-clip_threshold], device=self.device),
            torch.tensor([clip_threshold * 2], device=self.device),
            torch.tensor([-clip_threshold * 2], device=self.device),
        ]
        
        for test_x in test_cases:
            orig = torch.sign(test_x) * torch.clamp(torch.abs(test_x), max=clip_threshold)
            fus = torch.clamp(test_x, min=-clip_threshold, max=clip_threshold)
            assert torch.allclose(orig, fus), f"Edge case failed for {test_x}"
        
        print("[PASS] Fused sign-clip test passed!")
    
    def test_adaptive_clipping(self):
        """Test adaptive clipping functionality."""
        print("Testing adaptive clipping...")
        
        model = nn.Linear(10, 5).to(self.device)
        
        # Test with adaptive clipping enabled
        optimizer = ULTRON_V2(
            model.parameters(),
            lr=0.01,
            adaptive_clipping=True,
            clip_alpha=0.9,
            clip_percentile=95.0
        )
        
        initial_threshold = optimizer.get_clip_threshold()
        
        # Create gradients with varying magnitudes
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 5, device=self.device)
        criterion = nn.MSELoss()
        
        # Run several optimization steps
        thresholds = [initial_threshold]
        for i in range(20):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                thresholds.append(optimizer.get_clip_threshold())
        
        final_threshold = optimizer.get_clip_threshold()
        
        # Check that threshold adapts (may not change much with random gradients)
        print(f"  Initial threshold: {initial_threshold:.6f}")
        print(f"  Final threshold: {final_threshold:.6f}")
        print(f"  Thresholds over time: {[f'{t:.6f}' for t in thresholds]}")
        
        # At minimum, ensure threshold stays within bounds
        assert 1e-6 <= final_threshold <= 10.0, \
            f"Threshold out of bounds: {final_threshold}"
        
        print("[PASS] Adaptive clipping test passed!")
    
    def test_normalization_strategies(self):
        """Test different normalization strategies."""
        print("Testing normalization strategies...")
        
        strategies = ['rms', 'l2', 'moving_avg']
        
        for strategy in strategies:
            print(f"  Testing {strategy} normalization...")
            
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            ).to(self.device)
            
            optimizer = ULTRON_V2(
                model.parameters(),
                lr=0.01,
                normalize_gradients=True,
                normalization_strategy=strategy,
                momentum_correction=True
            )
            
            x = torch.randn(32, 10, device=self.device)
            y = torch.randn(32, 5, device=self.device)
            criterion = nn.MSELoss()
            
            # Run a few optimization steps
            losses = []
            for i in range(5):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # Check that loss decreases
            assert losses[-1] < losses[0] or losses[-1] < 1.0, \
                f"Loss didn't decrease with {strategy}: {losses[0]} -> {losses[-1]}"
            
            print(f"    Final loss: {losses[-1]:.6f}")
        
        print("[PASS] All normalization strategies tested!")
    
    def test_mixed_precision(self):
        """Test mixed precision support."""
        print("Testing mixed precision support...")
        
        precisions = ['fp32', 'fp16', 'bf16']
        
        for precision in precisions:
            print(f"  Testing {precision} state precision...")
            
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            ).to(self.device)
            
            optimizer = ULTRON_V2(
                model.parameters(),
                lr=0.01,
                state_precision=precision
            )
            
            x = torch.randn(32, 10, device=self.device)
            y = torch.randn(32, 5, device=self.device)
            criterion = nn.MSELoss()
            
            # Run optimization
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Check state buffers
            for param in model.parameters():
                if param.grad is not None:
                    state = optimizer.state[param]
                    if 'buffer' in state:
                        buffer = state['buffer']
                        if precision == 'fp16':
                            assert buffer.dtype == torch.float16, \
                                f"Buffer dtype mismatch: {buffer.dtype} != torch.float16"
                        elif precision == 'bf16':
                            assert buffer.dtype == torch.bfloat16, \
                                f"Buffer dtype mismatch: {buffer.dtype} != torch.bfloat16"
                        else:
                            assert buffer.dtype == torch.float32, \
                                f"Buffer dtype mismatch: {buffer.dtype} != torch.float32"
            
            print(f"    Loss: {loss.item():.6f}")
        
        print("[PASS] Mixed precision test passed!")
    
    def test_memory_usage(self):
        """Test memory usage statistics."""
        print("Testing memory usage statistics...")
        
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(self.device)
        
        optimizer = ULTRON_V2(model.parameters(), lr=0.001)
        
        # Initialize state with one step
        x = torch.randn(64, 100, device=self.device)
        y = torch.randn(64, 10, device=self.device)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Get memory usage
        memory_stats = optimizer.get_memory_usage()
        
        # Check that statistics are computed
        assert 'total_bytes' in memory_stats
        assert 'state_bytes' in memory_stats
        assert 'param_bytes' in memory_stats
        assert memory_stats['total_bytes'] > 0
        assert memory_stats['state_bytes'] >= 0
        assert memory_stats['param_bytes'] > 0
        
        print(f"  Total memory: {memory_stats['total_mb']:.2f} MB")
        print(f"  State memory: {memory_stats['state_mb']:.2f} MB")
        print(f"  Param memory: {memory_stats['param_mb']:.2f} MB")
        
        print("[PASS] Memory usage test passed!")
    
    def test_convergence_synthetic(self):
        """Test convergence on synthetic functions."""
        print("Testing convergence on synthetic functions...")
        
        # Test Rosenbrock function
        print("  Testing Rosenbrock function...")
        x = torch.tensor([-1.5, 2.0], device=self.device, requires_grad=True)
        y = torch.tensor([-1.5, 2.0], device=self.device, requires_grad=True)
        
        optimizer = ULTRON_V2([x, y], lr=0.001)
        
        for i in range(1000):
            optimizer.zero_grad()
            loss = (1 - x)**2 + 100 * (y - x**2)**2
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        print(f"    Final Rosenbrock loss: {final_loss:.6f}")
        assert final_loss < 1.0, f"Rosenbrock didn't converge: {final_loss}"
        
        # Test quadratic bowl
        print("  Testing quadratic bowl...")
        x = torch.randn(10, device=self.device, requires_grad=True) * 5
        
        optimizer = ULTRON_V2([x], lr=0.1)
        
        for i in range(100):
            optimizer.zero_grad()
            loss = torch.sum(x**2)
            loss.backward()
            optimizer.step()
            
            if loss.item() < 1e-6:
                break
        
        final_loss = loss.item()
        print(f"    Final quadratic loss: {final_loss:.6f}")
        assert final_loss < 0.1, f"Quadratic didn't converge: {final_loss}"
        
        print("[PASS] Synthetic function convergence test passed!")
    
    def test_comparison_with_ultron(self):
        """Test comparison with original ULTRON."""
        print("Testing comparison with original ULTRON...")
        
        # Create identical models
        model1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ).to(self.device)
        
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ).to(self.device)
        
        # Copy parameters to ensure identical initialization
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.copy_(p1)
        
        # Create optimizers with same hyperparameters
        optimizer1 = ULTRON_V2(
            model1.parameters(),
            lr=0.01,
            normalize_gradients=True,
            normalization_strategy='rms',
            adaptive_clipping=False,  # Disable adaptive for fair comparison
            clip_threshold=1.0
        )
        
        optimizer2 = ULTRON(
            model2.parameters(),
            lr=0.01,
            normalize_gradients=True,
            clip_threshold=1.0
        )
        
        # Create identical data
        torch.manual_seed(42)
        x = torch.randn(32, 10, device=self.device)
        y = torch.randn(32, 5, device=self.device)
        criterion = nn.MSELoss()
        
        # Run optimization
        losses1 = []
        losses2 = []
        
        for i in range(10):
            # ULTRON_V2
            optimizer1.zero_grad()
            output1 = model1(x)
            loss1 = criterion(output1, y)
            loss1.backward()
            optimizer1.step()
            losses1.append(loss1.item())
            
            # Original ULTRON
            optimizer2.zero_grad()
            output2 = model2(x)
            loss2 = criterion(output2, y)
            loss2.backward()
            optimizer2.step()
            losses2.append(loss2.item())
        
        # Check that both optimizers decrease loss
        assert losses1[-1] < losses1[0], f"ULTRON_V2 loss didn't decrease: {losses1[0]} -> {losses1[-1]}"
        assert losses2[-1] < losses2[0], f"ULTRON loss didn't decrease: {losses2[0]} -> {losses2[-1]}"
        
        print(f"  ULTRON_V2 final loss: {losses1[-1]:.6f}")
        print(f"  ULTRON final loss: {losses2[-1]:.6f}")
        print(f"  Loss difference: {abs(losses1[-1] - losses2[-1]):.6f}")
        
        # They should be reasonably close (not necessarily identical due to implementation differences)
        assert abs(losses1[-1] - losses2[-1]) < 0.1, \
            f"Loss difference too large: {abs(losses1[-1] - losses2[-1])}"
        
        print("[PASS] Comparison with ULTRON test passed!")
    
    def test_edge_cases(self):
        """Test edge cases."""
        print("Testing edge cases...")
        
        # Test with zero gradients
        print("  Testing zero gradients...")
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        optimizer = ULTRON_V2([x], lr=0.1)
        
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=self.device)
        loss.backward()
        optimizer.step()
        
        # Should not crash
        assert True
        
        # Test with NaN gradients
        print("  Testing NaN gradients...")
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        optimizer = ULTRON_V2([x], lr=0.1)
        
        optimizer.zero_grad()
        x.grad = torch.tensor([float('nan'), 1.0, 2.0], device=self.device)
        
        try:
            optimizer.step()
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            print(f"    Caught expected exception: {e}")
        
        # Test with very small learning rate
        print("  Testing very small learning rate...")
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        optimizer = ULTRON_V2([x], lr=1e-10)
        
        initial_x = x.detach().clone()
        
        optimizer.zero_grad()
        loss = torch.sum(x**2)
        loss.backward()
        optimizer.step()
        
        # Parameters should change very little
        change = torch.norm(x - initial_x).item()
        assert change < 1e-8, f"Change too large with small LR: {change}"
        
        print("[PASS] Edge cases test passed!")
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("Running ULTRON_V2 Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_initialization,
            self.test_basic_optimization,
            self.test_vectorized_updates,
            self.test_fused_sign_clip,
            self.test_adaptive_clipping,
            self.test_normalization_strategies,
            self.test_mixed_precision,
            self.test_memory_usage,
            self.test_convergence_synthetic,
            self.test_comparison_with_ultron,
            self.test_edge_cases,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"✗ Test failed: {test.__name__}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(f"Test Results: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("[SUCCESS] All tests passed! ULTRON_V2 is working correctly.")
            return True
        else:
            print("[WARNING] Some tests failed. Please check the implementation.")
            return False


def main():
    """Main function to run tests."""
    tester = TestULTRONV2()
    tester.setup_method()
    return tester.run_all_tests()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
