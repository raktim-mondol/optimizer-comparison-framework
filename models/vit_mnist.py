"""
Vision Transformer (ViT) implementation for MNIST dataset.
Lightweight ViT adapted for 28x28 grayscale images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention module."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron block."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerMNIST(nn.Module):
    """
    Vision Transformer for MNIST dataset.
    
    Args:
        img_size: Input image size (default: 28 for MNIST)
        patch_size: Size of patches (default: 4)
        in_channels: Input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 10)
        embed_dim: Embedding dimension (default: 64)
        depth: Number of transformer blocks (default: 6)
        num_heads: Number of attention heads (default: 4)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
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
    
    def get_flops(self, img_size=28):
        """
        Estimate FLOPs for forward pass.
        Note: This is a rough estimate.
        """
        # Patch embedding FLOPs
        patch_size = self.patch_embed.patch_size
        num_patches = (img_size // patch_size) ** 2
        embed_dim = self.patch_embed.proj.out_channels
        patch_flops = img_size * img_size * embed_dim * patch_size * patch_size
        
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


class VisionTransformerMNISTSmall(VisionTransformerMNIST):
    """Smaller ViT for MNIST."""
    def __init__(self):
        super().__init__(embed_dim=48, depth=4, num_heads=3)


class VisionTransformerMNISTMedium(VisionTransformerMNIST):
    """Medium ViT for MNIST."""
    def __init__(self):
        super().__init__(embed_dim=96, depth=8, num_heads=6)


class VisionTransformerMNISTLarge(VisionTransformerMNIST):
    """Large ViT for MNIST."""
    def __init__(self):
        super().__init__(embed_dim=128, depth=12, num_heads=8)