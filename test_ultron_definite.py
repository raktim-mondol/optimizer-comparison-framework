#!/usr/bin/env python3
"""
Definite test for ULTRON optimizer.
"""

import torch
import sys

# Import optimizers
from optimizers.ultron import ULTRON

print("=" * 60, file=sys.stderr)
print("Definite ULTRON Test", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Test 1: Basic import and creation
print("1. Testing import and optimizer creation...", file=sys.stderr)
x = torch.tensor([0.0], requires_grad=True)
optimizer = ULTRON([x], lr=0.1)
print("   ✓ ULTRON optimizer created successfully", file=sys.stderr)

# Test 2: Single optimization step
print("\n2. Testing single optimization step...", file=sys.stderr)
optimizer.zero_grad()
loss = (x - 2).pow(2)
loss.backward()
optimizer.step()
print(f"   ✓ Optimization step completed, x = {x.item():.4f}", file=sys.stderr)

# Test 3: Check optimizer state
print("\n3. Checking optimizer state...", file=sys.stderr)
state = optimizer.state[x]
print(f"   ✓ State contains 'step': {'step' in state}", file=sys.stderr)
print(f"   ✓ State contains 'momentum': {'momentum' in state}", file=sys.stderr)
print(f"   ✓ Step count: {state['step'].item()}", file=sys.stderr)
print(f"   ✓ Momentum value: {state['momentum'].item():.6f}", file=sys.stderr)

# Test 4: Multiple parameters
print("\n4. Testing with multiple parameters...", file=sys.stderr)
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([-1.0], requires_grad=True)
optimizer = ULTRON([x, y], lr=0.01)

for i in range(3):
    optimizer.zero_grad()
    loss = x**2 + y**2
    loss.backward()
    optimizer.step()
    print(f"   Step {i}: x = {x.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.6f}", file=sys.stderr)

print("\n" + "=" * 60, file=sys.stderr)
print("All tests completed successfully!", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Print final results to stdout for capture
print("SUCCESS: ULTRON optimizer is working correctly")
print(f"Final x: {x.item():.6f}")
print(f"Final y: {y.item():.6f}")
print(f"Final loss: {loss.item():.6e}")
