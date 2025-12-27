#!/usr/bin/env python3
"""
Simple test script for ULTRON optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import optimizers
from optimizers.ultron import ULTRON

print("=" * 60)
print("Testing ULTRON Optimizer")
print("=" * 60)

# Test 1: Basic functionality
print("\n1. Testing basic functionality...")
x = torch.tensor([0.0], requires_grad=True)
optimizer = ULTRON([x], lr=0.1)

for i in range(5):
    optimizer.zero_grad()
    loss = (x - 2).pow(2)
    loss.backward()
    optimizer.step()
    print(f"  Iteration {i}: x = {x.item():.4f}, loss = {loss.item():.6f}")

# Test 2: Multiple parameters
print("\n2. Testing multiple parameters...")
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([-1.0], requires_grad=True)
optimizer = ULTRON([x, y], lr=0.01)

for i in range(5):
    optimizer.zero_grad()
    loss = x**2 + y**2
    loss.backward()
    optimizer.step()
    print(f"  Iteration {i}: x = {x.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.6f}")

# Test 3: Check optimizer state
print("\n3. Checking optimizer state...")
x = torch.tensor([0.0], requires_grad=True)
optimizer = ULTRON([x], lr=0.1)

# Do one step
optimizer.zero_grad()
loss = (x - 2).pow(2)
loss.backward()
optimizer.step()

# Check state
state = optimizer.state[x]
print(f"  Step count: {state['step'].item()}")
print(f"  Momentum shape: {state['momentum'].shape}")
print(f"  Momentum value: {state['momentum'].item():.6f}")

# Test 4: Test with a simple neural network
print("\n4. Testing with simple neural network...")
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
optimizer = ULTRON(model.parameters(), lr=0.01)

# Create dummy data
dummy_input = torch.randn(4, 10)
dummy_target = torch.randn(4, 1)

for i in range(3):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.mse_loss(output, dummy_target)
    loss.backward()
    optimizer.step()
    print(f"  Iteration {i}: loss = {loss.item():.6f}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
