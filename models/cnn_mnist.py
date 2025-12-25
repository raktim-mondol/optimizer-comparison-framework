"""
Enhanced CNN models for MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification (from existing example).
    
    Architecture:
    - Conv2d(1, 32, 3x3) -> ReLU -> MaxPool2d(2x2)
    - Conv2d(32, 64, 3x3) -> ReLU -> MaxPool2d(2x2)
    - Dropout2d(0.25)
    - Linear(64*7*7, 128) -> ReLU -> Dropout(0.5)
    - Linear(128, 10)
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 1, 28, 28)):
        """
        Estimate FLOPs for forward pass.
        Note: This is a rough estimate.
        """
        flops = 0
        # conv1: (28*28)*(3*3)*1*32 = 28*28*9*32
        flops += input_size[2] * input_size[3] * 3 * 3 * input_size[1] * 32
        # conv2: (14*14)*(3*3)*32*64 = 14*14*9*32*64
        flops += (input_size[2]//2) * (input_size[3]//2) * 3 * 3 * 32 * 64
        # fc1: (64*7*7)*128 = 64*49*128
        flops += 64 * 7 * 7 * 128
        # fc2: 128*10
        flops += 128 * 10
        return flops


class MNISTCNNV2(nn.Module):
    """
    Enhanced CNN for MNIST with more layers and batch normalization.
    
    Architecture:
    - Conv2d(1, 32, 3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
    - Conv2d(32, 64, 3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
    - Conv2d(64, 128, 3x3) -> BatchNorm -> ReLU
    - Dropout2d(0.25)
    - Linear(128*7*7, 256) -> BatchNorm -> ReLU -> Dropout(0.5)
    - Linear(256, 128) -> BatchNorm -> ReLU -> Dropout(0.5)
    - Linear(128, 10)
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 1, 28, 28)):
        """
        Estimate FLOPs for forward pass.
        """
        flops = 0
        # conv1
        flops += input_size[2] * input_size[3] * 3 * 3 * input_size[1] * 32
        # conv2
        flops += (input_size[2]//2) * (input_size[3]//2) * 3 * 3 * 32 * 64
        # conv3
        flops += (input_size[2]//4) * (input_size[3]//4) * 3 * 3 * 64 * 128
        # fc1
        flops += 128 * 7 * 7 * 256
        # fc2
        flops += 256 * 128
        # fc3
        flops += 128 * 10
        return flops


class MNISTCNNV3(nn.Module):
    """
    Deeper CNN for MNIST with residual connections.
    
    Architecture:
    - Initial conv block
    - 3 residual blocks
    - Global average pooling
    - Fully connected layer
    """
    
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.res_block3 = ResidualBlock(128, 256, stride=1)
        
        # Global average pooling and final layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 1, 28, 28)):
        """
        Estimate FLOPs for forward pass.
        """
        flops = 0
        # conv1
        flops += input_size[2] * input_size[3] * 3 * 3 * input_size[1] * 32
        # Residual blocks (simplified calculation)
        # Each residual block has 2 conv layers
        # res_block1: 28x28 -> 14x14
        flops += 28*28*3*3*32*64*2
        # res_block2: 14x14 -> 7x7
        flops += 14*14*3*3*64*128*2
        # res_block3: 7x7 -> 7x7
        flops += 7*7*3*3*128*256*2
        # fc layer
        flops += 256 * 10
        return flops


class ResidualBlock(nn.Module):
    """Residual block for MNISTCNNV3."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_mnist_model(model_name='simple_cnn'):
    """
    Factory function to get MNIST model by name.
    
    Args:
        model_name: Name of the model to create.
                    Options: 'simple_cnn', 'cnn_v2', 'cnn_v3', 'vit_small', 'vit_medium', 'vit_large'
    
    Returns:
        PyTorch model instance
    """
    from .vit_mnist import (
        VisionTransformerMNISTSmall,
        VisionTransformerMNISTMedium,
        VisionTransformerMNISTLarge
    )
    
    model_registry = {
        'simple_cnn': SimpleCNN,
        'cnn_v2': MNISTCNNV2,
        'cnn_v3': MNISTCNNV3,
        'vit_small': VisionTransformerMNISTSmall,
        'vit_medium': VisionTransformerMNISTMedium,
        'vit_large': VisionTransformerMNISTLarge,
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_registry.keys())}")
    
    return model_registry[model_name]()