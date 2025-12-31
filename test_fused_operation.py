#!/usr/bin/env python3
"""
Test the mathematical equivalence of fused sign-clip operation.
"""

import torch

def test_equivalence():
    """Test that torch.clamp is equivalent to sign * clamp(abs)."""
    print("Testing mathematical equivalence of fused operation...")
    
    # Test with random values
    torch.manual_seed(42)
    x = torch.randn(1000) * 10  # Values from -10 to 10
    
    clip_threshold = 2.0
    
    # Original operation
    original = torch.sign(x) * torch.clamp(torch.abs(x), max=clip_threshold)
    
    # Fused operation
    fused = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
    
    # Check equivalence
    diff = torch.abs(original - fused).max().item()
    
    print(f"  Maximum difference: {diff:.10f}")
    print(f"  Are they equal? {torch.allclose(original, fused, rtol=1e-10, atol=1e-10)}")
    
    # Test edge cases
    print("\nTesting edge cases:")
    
    # Zero values
    x_zero = torch.tensor([0.0])
    orig_zero = torch.sign(x_zero) * torch.clamp(torch.abs(x_zero), max=clip_threshold)
    fused_zero = torch.clamp(x_zero, min=-clip_threshold, max=clip_threshold)
    print(f"  Zero: original={orig_zero.item()}, fused={fused_zero.item()}, equal={torch.allclose(orig_zero, fused_zero)}")
    
    # Exactly at threshold
    x_at_thresh = torch.tensor([clip_threshold, -clip_threshold])
    orig_at = torch.sign(x_at_thresh) * torch.clamp(torch.abs(x_at_thresh), max=clip_threshold)
    fused_at = torch.clamp(x_at_thresh, min=-clip_threshold, max=clip_threshold)
    print(f"  At threshold: original={orig_at}, fused={fused_at}, equal={torch.allclose(orig_at, fused_at)}")
    
    # Beyond threshold
    x_beyond = torch.tensor([clip_threshold * 2, -clip_threshold * 2])
    orig_beyond = torch.sign(x_beyond) * torch.clamp(torch.abs(x_beyond), max=clip_threshold)
    fused_beyond = torch.clamp(x_beyond, min=-clip_threshold, max=clip_threshold)
    print(f"  Beyond threshold: original={orig_beyond}, fused={fused_beyond}, equal={torch.allclose(orig_beyond, fused_beyond)}")
    
    # Performance comparison
    print("\nPerformance comparison:")
    import time
    
    # Time original operation
    start = time.time()
    for _ in range(1000):
        _ = torch.sign(x) * torch.clamp(torch.abs(x), max=clip_threshold)
    original_time = time.time() - start
    
    # Time fused operation
    start = time.time()
    for _ in range(1000):
        _ = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
    fused_time = time.time() - start
    
    print(f"  Original operation time: {original_time:.6f}s")
    print(f"  Fused operation time: {fused_time:.6f}s")
    print(f"  Speedup: {original_time/fused_time:.2f}x")
    
    return torch.allclose(original, fused, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    if test_equivalence():
        print("\n[PASS] Mathematical equivalence verified!")
    else:
        print("\n[FAIL] Mathematical equivalence failed!")
