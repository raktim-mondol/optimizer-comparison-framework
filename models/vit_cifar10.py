"""
Vision Transformer (ViT) implementation for CIFAR10 dataset.
Standard ViT adapted for 32x32 RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .vit_mnist import PatchEmbedding, MultiHeadSelfAttention, MLP, TransformerBlock


class VisionTransformerCIFAR10(nn.Module):
    """
    Vision Transformer for CIFAR10 dataset.
    
    Args:
        img_size: Input image size (default: 32 for CIFAR10)
        patch_size: Size of patches (default: 4)
        in_channels: Input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 10)
        embed_dim: Embedding dimension (default: 128)
        depth: Number of transformer blocks (default: 8)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        drop_path_rate: Stochastic depth rate (default: 0.0)
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=8, num_heads=8, mlp_ratio=4.0, 
                 dropout=0.1, drop_path_rate=0.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_module)
        
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Layer norm
        x = self.norm(x)
        
        # Return CLS token
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, img_size=32):
        """
        Estimate FLOPs for forward pass.
        Note: This is a rough estimate.
        """
        # Patch embedding FLOPs
        patch_size = self.patch_embed.patch_size
        num_patches = (img_size // patch_size) ** 2
        embed_dim = self.patch_embed.proj.out_channels
        patch_flops = img_size * img_size * 3 * embed_dim * patch_size * patch_size
        
        # Attention FLOPs
        attn_flops_per_block = 4 * num_patches * embed_dim * embed_dim
        attn_flops = attn_flops_per_block * len(self.blocks)
        
        # MLP FLOPs
        mlp_hidden_dim = int(embed_dim * 4.0)  # Assuming mlp_ratio=4.0
        mlp_flops_per_block = 2 * num_patches * embed_dim * mlp_hidden_dim
        mlp_flops = mlp_flops_per_block * len(self.blocks)
        
        # Classification head FLOPs
        head_flops = embed_dim * self.head.out_features
        
        total_flops = patch_flops + attn_flops + mlp_flops + head_flops
        return total_flops


class VisionTransformerCIFAR10Small(VisionTransformerCIFAR10):
    """Smaller ViT for CIFAR10."""
    def __init__(self):
        super().__init__(embed_dim=96, depth=6, num_heads=6)


class VisionTransformerCIFAR10Medium(VisionTransformerCIFAR10):
    """Medium ViT for CIFAR10."""
    def __init__(self):
        super().__init__(embed_dim=192, depth=12, num_heads=12)


class VisionTransformerCIFAR10Large(VisionTransformerCIFAR10):
    """Large ViT for CIFAR10."""
    def __init__(self):
        super().__init__(embed_dim=256, depth=16, num_heads=16)


class HybridViTCNN(nn.Module):
    """
    Hybrid CNN-ViT model for CIFAR10.
    Uses CNN for feature extraction and ViT for classification.
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # CNN backbone (lightweight)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # ViT head
        self.vit_head = VisionTransformerCIFAR10Small()
        
        # Final classification layer
        self.fc = nn.Linear(256 * 4 * 4 + 96, num_classes)  # CNN features + ViT features
        
    def forward(self, x):
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.flatten(1)
        
        # Extract ViT features
        vit_features = self.vit_head.forward_features(x)
        
        # Concatenate and classify
        combined_features = torch.cat([cnn_features, vit_features], dim=1)
        output = self.fc(combined_features)
        return output