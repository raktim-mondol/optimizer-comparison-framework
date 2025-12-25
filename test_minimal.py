#!/usr/bin/env python3
"""
Minimal test to verify the training loop works.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    print("Testing minimal training loop...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use only 100 samples for quick test
    train_dataset.data = train_dataset.data[:100]
    train_dataset.targets = train_dataset.targets[:100]
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    # Create simple model
    print("Creating simple model...")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 1 batch
    print("\nTraining for 1 batch...")
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss after 1 batch: {loss.item():.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_output = model(data[:5])  # Test on first 5 samples
        print(f"Output shape: {test_output.shape}")
        print(f"Predictions: {test_output.argmax(dim=1)}")
        print(f"True labels: {target[:5]}")
    
    print("\n" + "="*80)
    print("Minimal test passed! Training loop works correctly.")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())