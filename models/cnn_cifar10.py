"""
Enhanced CNN models for CIFAR10 dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CIFAR10ResNet(nn.Module):
    """
    ResNet-18 adapted for CIFAR10 (from existing example).
    
    Modifies first conv layer for CIFAR-10 (3 channels instead of RGB images).
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Use a pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify first conv layer for CIFAR-10 (3 channels instead of RGB images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the original fc layer and add a new one for CIFAR-10
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 3, 32, 32)):
        """
        Estimate FLOPs for forward pass.
        Note: This is a rough estimate for ResNet-18.
        """
        # Simplified FLOPs calculation for ResNet-18
        # Each residual block has ~2 conv layers
        # Total ~18 conv layers in ResNet-18
        flops = 0
        # First conv: 32x32x3x64x3x3
        flops += input_size[2] * input_size[3] * 3 * 64 * 3 * 3
        # Remaining layers (simplified)
        flops += 17 * (16*16 * 64*64 * 3*3)  # Approximate
        # FC layer
        flops += 512 * 10
        return flops


class CIFAR10CNN(nn.Module):
    """
    Custom CNN for CIFAR10 classification.
    
    Architecture:
    - Conv2d(3, 64, 3x3) -> BatchNorm -> ReLU -> Conv2d(64, 64, 3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
    - Conv2d(64, 128, 3x3) -> BatchNorm -> ReLU -> Conv2d(128, 128, 3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
    - Conv2d(128, 256, 3x3) -> BatchNorm -> ReLU -> Conv2d(256, 256, 3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
    - Dropout(0.25)
    - Linear(256*4*4, 1024) -> BatchNorm -> ReLU -> Dropout(0.5)
    - Linear(1024, 512) -> BatchNorm -> ReLU -> Dropout(0.5)
    - Linear(512, 10)
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 3, 32, 32)):
        """
        Estimate FLOPs for forward pass.
        """
        flops = 0
        
        # Block 1: conv1 + conv2
        flops += input_size[2] * input_size[3] * 3 * 64 * 3 * 3  # conv1
        flops += input_size[2] * input_size[3] * 64 * 64 * 3 * 3  # conv2
        
        # After pool: 16x16
        # Block 2: conv3 + conv4
        flops += 16 * 16 * 64 * 128 * 3 * 3  # conv3
        flops += 16 * 16 * 128 * 128 * 3 * 3  # conv4
        
        # After pool: 8x8
        # Block 3: conv5 + conv6
        flops += 8 * 8 * 128 * 256 * 3 * 3  # conv5
        flops += 8 * 8 * 256 * 256 * 3 * 3  # conv6
        
        # After pool: 4x4
        # FC layers
        flops += 256 * 4 * 4 * 1024  # fc1
        flops += 1024 * 512  # fc2
        flops += 512 * 10  # fc3
        
        return flops


class CIFAR10VGG(nn.Module):
    """
    VGG-style network for CIFAR10.
    
    Architecture:
    - 2x[Conv2d(3, 64, 3x3) -> BatchNorm -> ReLU]
    - MaxPool2d(2x2)
    - 2x[Conv2d(64, 128, 3x3) -> BatchNorm -> ReLU]
    - MaxPool2d(2x2)
    - 3x[Conv2d(128, 256, 3x3) -> BatchNorm -> ReLU]
    - MaxPool2d(2x2)
    - Linear(256*4*4, 512) -> ReLU -> Dropout(0.5)
    - Linear(512, 512) -> ReLU -> Dropout(0.5)
    - Linear(512, 10)
    """
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 3, 32, 32)):
        """
        Estimate FLOPs for forward pass.
        """
        flops = 0
        
        # Block 1
        flops += 32*32*3*64*3*3  # conv1
        flops += 32*32*64*64*3*3  # conv2
        
        # Block 2 (after pool: 16x16)
        flops += 16*16*64*128*3*3  # conv3
        flops += 16*16*128*128*3*3  # conv4
        
        # Block 3 (after pool: 8x8)
        flops += 8*8*128*256*3*3  # conv5
        flops += 8*8*256*256*3*3  # conv6
        flops += 8*8*256*256*3*3  # conv7
        
        # FC layers
        flops += 256*4*4*512  # fc1
        flops += 512*512  # fc2
        flops += 512*10  # fc3
        
        return flops


def get_cifar10_model(model_name='resnet'):
    """
    Factory function to get CIFAR10 model by name.
    
    Args:
        model_name: Name of the model to create.
                    Options: 'resnet', 'cnn', 'vgg', 'vit_small', 'vit_medium', 'vit_large', 'hybrid'
    
    Returns:
        PyTorch model instance
    """
    from .vit_cifar10 import (
        VisionTransformerCIFAR10Small,
        VisionTransformerCIFAR10Medium,
        VisionTransformerCIFAR10Large,
        HybridViTCNN
    )
    
    model_registry = {
        'resnet': CIFAR10ResNet,
        'cnn': CIFAR10CNN,
        'vgg': CIFAR10VGG,
        'vit_small': VisionTransformerCIFAR10Small,
        'vit_medium': VisionTransformerCIFAR10Medium,
        'vit_large': VisionTransformerCIFAR10Large,
        'hybrid': HybridViTCNN,
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_registry.keys())}")
    
    return model_registry[model_name]()