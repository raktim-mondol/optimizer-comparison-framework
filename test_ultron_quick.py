#!/usr/bin/env python3
"""
Quick test for ULTRON optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Import optimizers
from optimizers.ultron import ULTRON

print("=" * 60)
print("Quick ULTRON Test")
print("=" * 60)

# Test 1: Basic functionality
print("\n1. Basic functionality test...")
x = torch.tensor([0.0], requires_grad=True)
optimizer = ULTRON([x], lr=0.1)

print(f"Initial x: {x.item():.4f}")
for i in range(5):
    optimizer.zero_grad()
    loss = (x - 2).pow(2)
    loss.backward()
    optimizer.step()
    print(f"  Step {i}: x = {x.item():.4f}, loss = {loss.item():.6f}")

# Test 2: Memory check
print("\n2. Memory efficiency check...")
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

optimizer = ULTRON(model.parameters(), lr=0.001)

# Count state parameters
state_size = 0
for param in model.parameters():
    if param.requires_grad:
        # Initialize state with one step
        optimizer.zero_grad()
        dummy_input = torch.randn(1, 100)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        state = optimizer.state[param]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state_size += value.numel()

print(f"ULTRON state parameters: {state_size:,}")

# Compare with Adam
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
adam_state_size = 0
for param in model.parameters():
    if param.requires_grad:
        adam_optimizer.zero_grad()
        dummy_input = torch.randn(1, 100)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        adam_optimizer.step()
        
        state = adam_optimizer.state[param]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                adam_state_size += value.numel()

print(f"Adam state parameters: {adam_state_size:,}")
print(f"ULTRON uses {100 * (1 - state_size / adam_state_size):.1f}% less memory than Adam")

# Test 3: Speed test
print("\n3. Speed test...")
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
dummy_input = torch.randn(32, 100)
dummy_target = torch.randint(0, 10, (32,))

# Test ULTRON
ultron_optimizer = ULTRON(model.parameters(), lr=0.001)
start_time = time.time()
for i in range(100):
    ultron_optimizer.zero_grad()
    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    ultron_optimizer.step()
ultron_time = time.time() - start_time

# Test Adam
model2 = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
adam_optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
start_time = time.time()
for i in range(100):
    adam_optimizer.zero_grad()
    output = model2(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    adam_optimizer.step()
adam_time = time.time() - start_time

print(f"ULTRON time for 100 iterations: {ultron_time:.3f}s")
print(f"Adam time for 100 iterations: {adam_time:.3f}s")
print(f"ULTRON is {100 * (1 - ultron_time / adam_time):.1f}% faster than Adam")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
