# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Domain-adaptive modules for cross-domain object detection.

This module provides domain normalization and other domain adaptation components
to improve YOLOv11's performance across different domains (e.g., different lighting,
weather conditions, sensors).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import dist2bbox

__all__ = (
    "DomainNormalization",
    "DomainNormalizedConv",
    "DAConvAtt",
    "DAConvAttLite",
    "FeatureMixer",
    "AdaptiveFeatureCalibration",
    "FrequencyDomainSeparator",
    "DomainAdaptiveBottleneck",
    "C3k2_DN",
    "SPPF_MS",
    "PAN_DA",
    "Detect_DR",
)


class DomainNormalization(nn.Module):
    """
    Domain Normalization Layer for cross-domain feature adaptation.
    
    This layer maintains multiple sets of batch normalization parameters (one for each domain)
    and dynamically weights them based on the input features. This allows the model to adapt
    to different domains without explicit domain labels during inference.
    
    The domain weights are predicted by a lightweight network that analyzes the global
    statistics of the input feature map.
    
    Args:
        num_features (int): Number of input channels (C).
        num_domains (int): Number of domains to model. Default: 3.
        eps (float): Epsilon for numerical stability in BN. Default: 1e-5.
        momentum (float): Momentum for running mean/variance in BN. Default: 0.1.
        
    Attributes:
        domain_bns (nn.ModuleList): List of BatchNorm2d layers, one per domain.
        weight (nn.Parameter): Shared affine scale parameter.
        bias (nn.Parameter): Shared affine shift parameter.
        domain_predictor (nn.Sequential): Network to predict domain weights.
        
    Shape:
        - Input: (B, C, H, W) where B is batch size, C is channels, H is height, W is width
        - Output: (B, C, H, W) with the same shape as input
        
    Examples:
        >>> dn = DomainNormalization(num_features=256, num_domains=3)
        >>> x = torch.randn(2, 256, 40, 40)
        >>> y = dn(x)
        >>> assert y.shape == x.shape
        
    Reference:
        Inspired by "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation"
    """
    
    def __init__(
        self,
        num_features: int,
        num_domains: int = 3,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """Initialize DomainNormalization layer with specified parameters."""
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.eps = eps
        self.momentum = momentum
        
        # Create separate BN layers for each domain (without affine parameters)
        # Each domain has its own running_mean and running_var
        self.domain_bns = nn.ModuleList([
            nn.BatchNorm2d(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=False,  # We use shared affine parameters
                track_running_stats=True,
            )
            for _ in range(num_domains)
        ])
        
        # Shared learnable affine parameters (applied after domain-specific normalization)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Lightweight domain predictor network
        # Uses global average pooling + small MLP to predict domain weights
        reduction = max(num_features // 16, 8)  # Ensure minimum width of 8
        self.domain_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(num_features, reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, num_domains, kernel_size=1, bias=True),
            nn.Softmax(dim=1),  # Normalize weights to sum to 1
        )
        
        # Initialize domain predictor with small weights for stability
        self._init_domain_predictor()
        
    def _init_domain_predictor(self):
        """Initialize domain predictor with small weights for training stability."""
        for m in self.domain_predictor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Initialize bias to give roughly equal weights to all domains
                    nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with domain-adaptive normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Domain-normalized output of shape (B, C, H, W).
            
        Algorithm:
            1. Predict domain weights from input statistics
            2. Apply each domain's batch normalization
            3. Weight and aggregate normalized outputs
            4. Apply shared affine transformation
        """
        # Step 1: Predict domain weights based on input features
        # domain_weights shape: (B, num_domains, 1, 1)
        domain_weights = self.domain_predictor(x)
        
        # Step 2 & 3: Apply each domain's BN and weight the results
        normalized_outputs = []
        for i, bn in enumerate(self.domain_bns):
            # Normalize with domain-specific statistics
            norm_out = bn(x)  # Shape: (B, C, H, W)
            
            # Weight by predicted domain probability
            # domain_weights[:, i:i+1, :, :] shape: (B, 1, 1, 1)
            weighted_out = norm_out * domain_weights[:, i:i+1, :, :]
            normalized_outputs.append(weighted_out)
        
        # Aggregate all domain-specific normalized outputs
        out = sum(normalized_outputs)  # Element-wise sum
        
        # Step 4: Apply shared affine transformation
        # weight shape: (C,) -> (1, C, 1, 1)
        # bias shape: (C,) -> (1, C, 1, 1)
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation of the layer."""
        return (
            f"num_features={self.num_features}, "
            f"num_domains={self.num_domains}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}"
        )


class DomainNormalizedConv(nn.Module):
    """
    Convolutional layer with Domain Normalization and activation.
    
    This is a drop-in replacement for the standard Conv module in Ultralytics,
    but uses DomainNormalization instead of BatchNorm2d for better cross-domain
    generalization.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size. Default: 1.
        s (int): Stride. Default: 1.
        p (int | None): Padding. If None, uses k//2. Default: None.
        g (int): Number of groups for grouped convolution. Default: 1.
        num_domains (int): Number of domains for DomainNormalization. Default: 3.
        act (bool): Whether to use activation function (SiLU). Default: True.
        
    Shape:
        - Input: (B, c1, H, W)
        - Output: (B, c2, H', W') where H' = H/s, W' = W/s
        
    Examples:
        >>> conv = DomainNormalizedConv(64, 128, k=3, s=2)
        >>> x = torch.randn(2, 64, 40, 40)
        >>> y = conv(x)
        >>> assert y.shape == (2, 128, 20, 20)
        
    Note:
        This module is compatible with Ultralytics YAML configuration format.
        You can use it in model config like: [c1, c2, k, s, p, g, num_domains]
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        num_domains: int = 3,
        act: bool = True,
    ):
        """Initialize DomainNormalizedConv with specified parameters."""
        super().__init__()
        
        # Validate inputs
        if not isinstance(c1, int) or not isinstance(c2, int):
            raise TypeError(f"c1 and c2 must be integers, got c1={type(c1)} c2={type(c2)}")
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"c1 and c2 must be positive, got c1={c1} c2={c2}")
        
        # Convolution layer (no bias, as DN has bias)
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p if p is not None else k // 2,
            groups=g,
            bias=False,
        )
        
        # Domain Normalization
        self.dn = DomainNormalization(c2, num_domains=num_domains)
        
        # Activation function (SiLU/Swish is default in YOLOv11)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Conv -> DomainNorm -> Activation."""
        return self.act(self.dn(self.conv(x)))
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for fused model (used during deployment, not yet implemented)."""
        # For now, just use standard forward
        # TODO: Implement fusion of conv and DN for faster inference
        return self.act(self.conv(x))


class DAConvAtt(nn.Module):
    """
    Domain-Adaptive Convolution with Attention (DA-ConvAtt).
    
    This module combines domain normalization, convolution, and attention mechanisms
    to create a powerful feature extraction layer that adapts to different domains while
    maintaining strong representational capacity.
    
    The module consists of:
    1. Domain-Normalized Convolution: Base feature extraction with domain adaptation
    2. Domain-Aware Channel Attention: Multiple domain-specific channel attention modules
    3. Spatial Attention: Shared spatial attention for all domains
    4. Domain Weight Prediction: Automatic domain weight learning
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size. Default: 3.
        s (int): Stride. Default: 1.
        p (int | None): Padding. If None, uses k//2. Default: None.
        g (int): Number of groups. Default: 1.
        num_domains (int): Number of domains. Default: 3.
        reduction (int): Reduction ratio for channel attention. Default: 16.
        act (bool): Whether to use activation. Default: True.
        
    Shape:
        - Input: (B, c1, H, W)
        - Output: (B, c2, H', W')
        
    Examples:
        >>> conv_att = DAConvAtt(256, 256, k=3)
        >>> x = torch.randn(4, 256, 40, 40)
        >>> y = conv_att(x)
        >>> assert y.shape == (4, 256, 40, 40)
        
    Benefits:
        - Combines domain adaptation with attention mechanisms
        - Automatic domain weight learning (no domain labels needed)
        - Better cross-domain generalization than standard Conv
        - Moderate computational overhead
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        num_domains: int = 3,
        reduction: int = 16,
        act: bool = True,
    ):
        """Initialize DAConvAtt with specified parameters."""
        super().__init__()
        
        self.num_domains = num_domains
        self.c2 = c2
        
        # 1. Domain-Normalized Convolution (base feature extraction)
        self.dn_conv = DomainNormalizedConv(c1, c2, k, s, p, g, num_domains, act)
        
        # 2. Domain-Aware Channel Attention
        # Create separate channel attention modules for each domain
        self.domain_channel_att = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, max(c2 // reduction, 1), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(c2 // reduction, 1), c2, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_domains)
        ])
        
        # 3. Domain Weight Predictor
        # Predicts which domain the input is most similar to
        self.domain_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, max(num_domains * 2, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(num_domains * 2, 16), num_domains),
            nn.Softmax(dim=1)
        )
        
        # 4. Spatial Attention (shared across all domains)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with domain adaptation and attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)
            
        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H', W')
        """
        # Step 1: Domain-Normalized Convolution
        features = self.dn_conv(x)
        
        # Step 2: Predict domain weights
        domain_weights = self.domain_predictor(x)  # (B, num_domains)
        
        # Step 3: Domain-Aware Channel Attention
        # Weighted combination of domain-specific channel attention
        channel_att = torch.zeros_like(features[:, :, :1, :1])  # (B, c2, 1, 1)
        
        for i, att_module in enumerate(self.domain_channel_att):
            domain_weight = domain_weights[:, i:i+1, None, None]  # (B, 1, 1, 1)
            att = att_module(features)  # (B, c2, 1, 1)
            channel_att = channel_att + att * domain_weight
        
        # Apply channel attention
        features = features * channel_att
        
        # Step 4: Spatial Attention
        # Compute channel-wise statistics
        avg_out = torch.mean(features, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(features, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and compute spatial attention
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))  # (B, 1, H, W)
        
        # Apply spatial attention
        features = features * spatial_att
        
        return features


class DAConvAttLite(nn.Module):
    """
    Lightweight version of DA-ConvAtt (Domain-Adaptive Convolution with Attention).
    
    This is a simplified version optimized for speed with minimal performance drop.
    It uses:
    1. Domain-Normalized Convolution
    2. ECA (Efficient Channel Attention) instead of domain-specific attention
    3. No spatial attention (to save computation)
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size. Default: 3.
        s (int): Stride. Default: 1.
        p (int | None): Padding. If None, uses k//2. Default: None.
        g (int): Number of groups. Default: 1.
        num_domains (int): Number of domains. Default: 3.
        act (bool): Whether to use activation. Default: True.
        
    Shape:
        - Input: (B, c1, H, W)
        - Output: (B, c2, H', W')
        
    Examples:
        >>> conv_att = DAConvAttLite(256, 256, k=3)
        >>> x = torch.randn(4, 256, 40, 40)
        >>> y = conv_att(x)
        >>> assert y.shape == (4, 256, 40, 40)
        
    Benefits:
        - ~2x faster than full version
        - Minimal parameters overhead
        - Good for real-time applications
        - Still benefits from domain normalization
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        num_domains: int = 3,
        act: bool = True,
    ):
        """Initialize DAConvAttLite with specified parameters."""
        super().__init__()
        
        # 1. Domain-Normalized Convolution
        self.dn_conv = DomainNormalizedConv(c1, c2, k, s, p, g, num_domains, act)
        
        # 2. ECA (Efficient Channel Attention)
        # Adaptive kernel size based on channel number
        k_size = int(abs((math.log(c2, 2) + 1) / 2))
        k_size = k_size if k_size % 2 else k_size + 1
        
        self.eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with lightweight attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W)
            
        Returns:
            (torch.Tensor): Output tensor of shape (B, c2, H', W')
        """
        # Domain-Normalized Convolution
        x = self.dn_conv(x)
        
        # ECA attention
        b, c, _, _ = x.size()
        # Global average pooling: (B, C, H, W) -> (B, C)
        y = x.view(b, c, -1).mean(dim=2)  # (B, C)
        # 1D convolution: (B, C) -> (B, 1, C) -> (B, 1, C) -> (B, C)
        y = self.eca(y.unsqueeze(1)).squeeze(1)  # (B, C)
        # Apply sigmoid
        y = self.sigmoid(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        return x * y.expand_as(x)


class FeatureMixer(nn.Module):
    """
    Feature Mixer for cross-domain feature enhancement.
    
    Enhances domain generalization by mixing features across channels and spatial dimensions
    using lightweight operations. This encourages the model to learn domain-invariant
    representations.
    
    The mixer uses:
    1. Channel mixing via grouped convolutions
    2. Spatial mixing via depthwise convolutions
    3. Adaptive gating to balance original and mixed features
    
    Args:
        channels (int): Number of input/output channels.
        reduction (int): Reduction factor for gating network. Default: 4.
        
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)
        
    Examples:
        >>> mixer = FeatureMixer(channels=256)
        >>> x = torch.randn(2, 256, 40, 40)
        >>> y = mixer(x)
        >>> assert y.shape == x.shape
        
    Reference:
        Inspired by MLP-Mixer and feature shuffling techniques for domain adaptation.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """Initialize FeatureMixer with specified channels."""
        super().__init__()
        self.channels = channels
        
        # Channel mixing (group convolution for efficient mixing)
        # Split channels into groups and mix within groups
        num_groups = max(channels // 16, 4)  # Ensure reasonable number of groups
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
        # Spatial mixing (depthwise convolution)
        # Each channel independently processes spatial information
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
        # Adaptive gating mechanism
        # Learns how much to mix vs keep original features
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Feature fusion weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # Learnable fusion weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel and spatial mixing.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Mixed features of shape (B, C, H, W).
        """
        # Channel mixing
        channel_mixed = self.channel_mix(x)
        
        # Spatial mixing
        spatial_mixed = self.spatial_mix(x)
        
        # Combine channel and spatial mixing
        # Use learnable weight to balance both types of mixing
        alpha = torch.sigmoid(self.alpha)  # Ensure 0-1 range
        mixed = alpha * channel_mixed + (1 - alpha) * spatial_mixed
        
        # Adaptive gating
        # Decide how much to use mixed features vs original features
        gate_weight = self.gate(x)
        
        # Final output: gated combination
        out = x * gate_weight + mixed * (1 - gate_weight)
        
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"channels={self.channels}"


class AdaptiveFeatureCalibration(nn.Module):
    """
    Adaptive Feature Calibration for domain-robust feature adjustment.
    
    Dynamically calibrates features based on their statistical properties to reduce
    domain shift effects. Uses both channel-wise and spatial attention mechanisms
    to recalibrate feature responses.
    
    This module:
    1. Extracts global statistics (mean, max) from features
    2. Predicts calibration weights via a small network
    3. Applies scale and shift transformations
    
    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for squeeze network. Default: 16.
        
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)
        
    Examples:
        >>> calibration = AdaptiveFeatureCalibration(channels=256)
        >>> x = torch.randn(2, 256, 40, 40)
        >>> y = calibration(x)
        >>> assert y.shape == x.shape
        
    Reference:
        Inspired by Squeeze-and-Excitation and feature calibration techniques.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize AdaptiveFeatureCalibration with specified channels."""
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global pooling for feature statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Calibration weight prediction network
        # Uses both avg and max pooled features
        hidden_channels = max(channels // reduction, 8)
        self.scale_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),  # Scale factors in (0, 1) range, will be adjusted to (0.5, 1.5)
        )
        
        # Offset prediction network (shift/bias term)
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Tanh(),  # Offset in (-1, 1) range
        )
        
        # Learnable temperature for soft calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive calibration.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Calibrated features of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        
        # Extract global statistics
        avg_feat = self.avg_pool(x)  # (B, C, 1, 1)
        max_feat = self.max_pool(x)  # (B, C, 1, 1)
        
        # Concatenate statistics for scale prediction
        combined_stats = torch.cat([avg_feat, max_feat], dim=1)  # (B, 2C, 1, 1)
        
        # Predict scale factors
        scale_weight = self.scale_net(combined_stats)  # (B, C, 1, 1) in (0, 1)
        # Adjust to (0.5, 1.5) range for reasonable scaling
        scale_weight = scale_weight + 0.5
        
        # Apply scaling with temperature
        temperature = F.softplus(self.temperature)  # Ensure positive
        scaled = x * (scale_weight ** (1.0 / temperature))
        
        # Predict offset (shift term)
        offset = self.offset_net(avg_feat)  # (B, C, 1, 1) in (-1, 1)
        # Scale offset based on input statistics
        offset_scale = torch.std(x.view(B, C, -1), dim=2, keepdim=True).unsqueeze(-1)
        offset = offset * offset_scale * 0.1  # Small offset relative to std
        
        # Apply calibration: scale and shift
        calibrated = scaled + offset
        
        return calibrated
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"channels={self.channels}, reduction={self.reduction}"


class FrequencyDomainSeparator(nn.Module):
    """
    Frequency Domain Feature Separator for domain-invariant representation.
    
    Separates features into low-frequency and high-frequency components:
    - Low-frequency: Contains global structure, more domain-invariant
    - High-frequency: Contains details, more domain-specific
    
    The module learns to weight these components adaptively for better
    cross-domain generalization.
    
    Args:
        channels (int): Number of input channels.
        kernel_size (int): Kernel size for filtering operations. Default: 3.
        reduction (int): Channel reduction ratio. Default: 4.
        
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)
        
    Examples:
        >>> freq_sep = FrequencyDomainSeparator(channels=256)
        >>> x = torch.randn(2, 256, 40, 40)
        >>> y = freq_sep(x)
        >>> assert y.shape == x.shape
        
    Reference:
        Inspired by "FDA: Fourier Domain Adaptation" and frequency-based
        domain adaptation techniques.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 4):
        """Initialize FrequencyDomainSeparator with specified channels."""
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.reduction = reduction
        
        # Low-pass filter (preserves low-frequency components)
        # Uses average pooling + lightweight 1x1 conv
        # Reduce parameters by using channel reduction
        mid_channels = max(channels // reduction, 16)  # At least 16 channels
        self.low_pass = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # High-pass filter (preserves high-frequency components)
        # Uses depthwise-separable convolution for efficiency
        self.high_pass = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                     padding=kernel_size // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            # Pointwise convolution with reduction
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # Learnable fusion weights
        # Low-frequency should have higher weight for domain invariance
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]))  # [low_freq, high_freq]
        
        # Lightweight refinement with depthwise-separable conv
        self.refine = nn.Sequential(
            # Depthwise
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            # Pointwise
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with frequency separation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Frequency-separated features of shape (B, C, H, W).
        """
        # Extract low-frequency component (smooth/global structure)
        low_freq = self.low_pass(x)
        
        # Compute high-frequency component (details/textures)
        # Method: Original - Smoothed version
        smoothed = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        high_freq_input = x - smoothed
        high_freq = self.high_pass(high_freq_input)
        
        # Adaptive weighting using softmax to ensure weights sum to 1
        weights = F.softmax(self.fusion_weight, dim=0)
        
        # Weighted fusion
        # Emphasize low-frequency for domain invariance
        fused = low_freq * weights[0] + high_freq * weights[1]
        
        # Refine the fused features
        out = self.refine(fused)
        
        # Residual connection with original input
        out = out + x
        
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"channels={self.channels}, kernel_size={self.kernel_size}, reduction={self.reduction}"


class DomainAdaptiveBottleneck(nn.Module):
    """
    Domain-Adaptive Bottleneck for cross-domain detection.
    
    An enhanced version of the standard Bottleneck that integrates domain adaptation
    components for better cross-domain generalization.
    
    Components:
    1. DomainNormalizedConv for domain-aware normalization
    2. Optional FeatureMixer for feature enhancement
    3. Optional AdaptiveFeatureCalibration for feature adjustment
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Whether to use shortcut connection. Default: True.
        g (int): Groups for convolutions. Default: 1.
        k (tuple): Kernel sizes for two convolutions. Default: (3, 3).
        e (float): Expansion ratio for hidden channels. Default: 0.5.
        num_domains (int): Number of domains for domain normalization. Default: 3.
        use_mixer (bool): Whether to use FeatureMixer. Default: True.
        use_calibration (bool): Whether to use AdaptiveFeatureCalibration. Default: False.
        
    Shape:
        - Input: (B, C1, H, W)
        - Output: (B, C2, H, W)
        
    Examples:
        >>> # Basic usage
        >>> bottleneck = DomainAdaptiveBottleneck(256, 256)
        >>> x = torch.randn(4, 256, 40, 40)
        >>> y = bottleneck(x)
        
        >>> # With all enhancements
        >>> bottleneck = DomainAdaptiveBottleneck(
        ...     256, 256, use_mixer=True, use_calibration=True
        ... )
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        num_domains: int = 3,
        use_mixer: bool = True,
        use_calibration: bool = False,
    ):
        """Initialize Domain-Adaptive Bottleneck."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        # First convolution with domain normalization
        self.cv1 = DomainNormalizedConv(c1, c_, k=k[0], s=1, num_domains=num_domains)
        
        # Second convolution with domain normalization
        self.cv2 = DomainNormalizedConv(c_, c2, k=k[1], s=1, g=g, num_domains=num_domains)
        
        # Optional feature mixer
        self.mixer = FeatureMixer(c_) if use_mixer else None
        
        # Optional adaptive calibration
        self.calibration = AdaptiveFeatureCalibration(c2) if use_calibration else None
        
        # Shortcut connection
        self.add = shortcut and c1 == c2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through domain-adaptive bottleneck.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        # First conv
        out = self.cv1(x)
        
        # Optional feature mixing
        if self.mixer is not None:
            out = self.mixer(out)
        
        # Second conv
        out = self.cv2(out)
        
        # Optional calibration
        if self.calibration is not None:
            out = self.calibration(out)
        
        # Shortcut
        if self.add:
            out = out + x
            
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"use_mixer={self.mixer is not None}, use_calibration={self.calibration is not None}"


class C3k2_DN(nn.Module):
    """
    Domain-Normalized C3k2 for cross-domain object detection.
    
    This is the main architectural innovation that integrates domain adaptation
    into YOLOv11's C3k2 module. It replaces standard Bottlenecks with
    DomainAdaptiveBottlenecks and adds optional frequency domain processing.
    
    Key Features:
    1. Domain-adaptive bottleneck blocks
    2. Optional frequency domain separation
    3. Maintains C3k2 interface for drop-in replacement
    4. Compatible with YOLOv11 YAML configuration
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of bottleneck blocks. Default: 1.
        c3k (bool): Whether to use C3k-style blocks. Default: False.
        shortcut (bool): Whether to use shortcut connections. Default: True.
        g (int): Groups for convolutions. Default: 1.
        e (float): Expansion ratio. Default: 0.5.
        num_domains (int): Number of domains for normalization. Default: 3.
        use_mixer (bool): Whether to use FeatureMixer in bottlenecks. Default: True.
        use_freq_sep (bool): Whether to use FrequencyDomainSeparator. Default: False.
        
    Shape:
        - Input: (B, C1, H, W)
        - Output: (B, C2, H, W)
        
    Examples:
        >>> # Drop-in replacement for C3k2
        >>> c3k2_dn = C3k2_DN(256, 512, n=3)
        >>> x = torch.randn(4, 256, 40, 40)
        >>> y = c3k2_dn(x)
        >>> assert y.shape == (4, 512, 40, 40)
        
        >>> # With all enhancements
        >>> c3k2_dn = C3k2_DN(
        ...     256, 512, n=3, use_mixer=True, use_freq_sep=True
        ... )
        
    Reference:
        Based on YOLOv11's C3k2 module with domain adaptation enhancements.
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        num_domains: int = 3,
        use_mixer: bool = True,
        use_freq_sep: bool = False,
    ):
        """Initialize C3k2_DN module."""
        super().__init__()
        # Store parameters
        self.n = n
        self.use_freq_sep = use_freq_sep
        self.c = int(c2 * e)  # hidden channels
        
        # Input projection with domain normalization
        self.cv1 = DomainNormalizedConv(c1, 2 * self.c, k=1, s=1, num_domains=num_domains)
        
        # Output projection with domain normalization
        self.cv2 = DomainNormalizedConv((2 + n) * self.c, c2, k=1, s=1, num_domains=num_domains)
        
        # Domain-adaptive bottleneck blocks
        # Note: c3k parameter is kept for compatibility but not used in DN version
        self.m = nn.ModuleList(
            DomainAdaptiveBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=(3, 3),
                e=1.0,
                num_domains=num_domains,
                use_mixer=use_mixer,
                use_calibration=False,  # Keep lightweight
            )
            for _ in range(n)
        )
        
        # Optional frequency domain separator
        self.freq_sep = FrequencyDomainSeparator(c2) if use_freq_sep else None
    
    def __repr__(self) -> str:
        """String representation of C3k2_DN module."""
        return f"C3k2_DN(n={self.n}, use_freq_sep={self.use_freq_sep})"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C3k2_DN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        # Split input
        y = list(self.cv1(x).chunk(2, 1))
        
        # Process through bottleneck blocks
        y.extend(m(y[-1]) for m in self.m)
        
        # Concatenate and project
        out = self.cv2(torch.cat(y, 1))
        
        # Optional frequency domain processing
        if self.freq_sep is not None:
            out = self.freq_sep(out)
            
        return out
    
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().
        
        This is an alternative implementation that may be faster on some hardware.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        
        if self.freq_sep is not None:
            out = self.freq_sep(out)
            
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"n={len(self.m)}, use_freq_sep={self.freq_sep is not None}"


class SPPF_MS(nn.Module):
    """
    Multi-Scale Spatial Pyramid Pooling Fast (SPPF-MS) for cross-domain detection.
    
    An enhanced version of SPPF that uses multiple pooling kernel sizes and
    adaptive weighting for better cross-domain generalization.
    
    Key Features:
    1. Multi-scale pooling branches (kernel sizes: 3, 5, 7, 9)
    2. Adaptive pooling weights learned per-domain
    3. Cross-scale attention mechanism
    4. Domain normalization for robustness
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (tuple): Pooling kernel sizes. Default: (3, 5, 7, 9).
        num_domains (int): Number of domains for normalization. Default: 3.
        use_attention (bool): Whether to use cross-scale attention. Default: True.
        
    Shape:
        - Input: (B, C1, H, W)
        - Output: (B, C2, H, W)
        
    Examples:
        >>> sppf_ms = SPPF_MS(256, 256)
        >>> x = torch.randn(4, 256, 20, 20)
        >>> y = sppf_ms(x)
        >>> assert y.shape == (4, 256, 20, 20)
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int | tuple[int, ...] = 5,
        num_domains: int = 3,
        use_attention: bool = True,
    ):
        """Initialize SPPF-MS module."""
        super().__init__()
        
        # Convert single k to tuple of multi-scale sizes
        if isinstance(k, int):
            k = (3, 5, 7) if k == 5 else (k, k+2, k+4)
        c_ = c1 // 2  # hidden channels
        
        # Input projection with domain normalization
        self.cv1 = DomainNormalizedConv(c1, c_, k=1, s=1, num_domains=num_domains)
        
        # Multi-scale max pooling branches
        self.pooling_sizes = k
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in k
        ])
        
        # Cross-scale attention (optional)
        self.use_attention = use_attention
        if use_attention:
            # Attention over different pooling scales
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c_ * (1 + len(k)), c_ // 4, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_ // 4, 1 + len(k), 1),
                nn.Softmax(dim=1),
            )
        
        # Output projection with domain normalization
        out_channels = c_ * (1 + len(k))
        self.cv2 = DomainNormalizedConv(out_channels, c2, k=1, s=1, num_domains=num_domains)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SPPF-MS.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        # Input projection
        x = self.cv1(x)
        
        # Multi-scale pooling
        features = [x]  # Original features
        
        # Apply different pooling sizes
        for pool in self.pools:
            features.append(pool(x))
        
        # Concatenate all scales
        concat_features = torch.cat(features, 1)
        
        # Optional cross-scale attention
        if self.use_attention:
            # Compute attention weights
            att_weights = self.attention(concat_features)  # (B, 1+len(k), 1, 1)
            
            # Apply attention to each scale
            weighted_features = []
            for i, feat in enumerate(features):
                weighted_features.append(feat * att_weights[:, i:i+1, :, :])
            
            # Re-concatenate weighted features
            concat_features = torch.cat(weighted_features, 1)
        
        # Output projection
        out = self.cv2(concat_features)
        
        return out
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"pooling_sizes={self.pooling_sizes}, use_attention={self.use_attention}"


class PAN_DA(nn.Module):
    """
    Domain-Adaptive Path Aggregation Network (PAN-DA) for cross-domain detection.
    
    An enhanced version of PAN that integrates domain adaptation for better
    feature fusion across different domains.
    
    Key Features:
    1. Domain-adaptive feature fusion
    2. Top-down and bottom-up pathways with domain normalization
    3. Cross-layer domain consistency
    4. Adaptive fusion weights
    
    Args:
        in_channels (list): List of input channel sizes for each level.
        out_channels (int): Output channels for all levels.
        num_domains (int): Number of domains for normalization. Default: 3.
        use_fusion_attention (bool): Whether to use attention for fusion. Default: True.
        
    Shape:
        - Input: List of tensors [(B, C_i, H_i, W_i), ...] for each FPN level
        - Output: List of tensors [(B, out_channels, H_i, W_i), ...] for each level
        
    Examples:
        >>> # 3-level FPN: P3, P4, P5
        >>> pan_da = PAN_DA([128, 256, 512], 256)
        >>> p3 = torch.randn(4, 128, 80, 80)
        >>> p4 = torch.randn(4, 256, 40, 40)
        >>> p5 = torch.randn(4, 512, 20, 20)
        >>> outputs = pan_da([p3, p4, p5])
        >>> assert len(outputs) == 3
    """
    
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_domains: int = 3,
        use_fusion_attention: bool = True,
    ):
        """Initialize PAN-DA module."""
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channels
        self.use_fusion_attention = use_fusion_attention
        
        # Lateral connections (input projections with domain normalization)
        self.lateral_convs = nn.ModuleList([
            DomainNormalizedConv(in_c, out_channels, k=1, s=1, num_domains=num_domains)
            for in_c in in_channels
        ])
        
        # Top-down pathway (upsampling + fusion)
        self.td_convs = nn.ModuleList([
            DomainNormalizedConv(out_channels, out_channels, k=3, s=1, num_domains=num_domains)
            for _ in range(self.num_levels - 1)
        ])
        
        # Bottom-up pathway (downsampling + fusion)
        self.bu_convs = nn.ModuleList([
            DomainNormalizedConv(out_channels, out_channels, k=3, s=2, num_domains=num_domains)
            for _ in range(self.num_levels - 1)
        ])
        
        # Fusion layers
        self.fusion_convs = nn.ModuleList([
            DomainNormalizedConv(out_channels * 2, out_channels, k=1, s=1, num_domains=num_domains)
            for _ in range(self.num_levels)
        ])
        
        # Optional fusion attention
        if use_fusion_attention:
            self.fusion_attention = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels * 2, out_channels // 4, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels // 4, 2, 1),
                    nn.Softmax(dim=1),
                )
                for _ in range(self.num_levels)
            ])
        
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass through PAN-DA.
        
        Args:
            features (list[torch.Tensor]): List of input features from FPN levels.
                Expected order: [P3, P4, P5] (low to high level).
                
        Returns:
            list[torch.Tensor]: List of output features for each level.
        """
        assert len(features) == self.num_levels, \
            f"Expected {self.num_levels} input features, got {len(features)}"
        
        # Apply lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway (high-res <- low-res)
        td_features = [laterals[-1]]  # Start from highest level
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample higher level
            upsampled = F.interpolate(
                td_features[0],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
            
            # Fuse with lateral
            fused = self.td_convs[i](upsampled + laterals[i])
            td_features.insert(0, fused)
        
        # Bottom-up pathway (low-res <- high-res)
        bu_features = [td_features[0]]  # Start from lowest level
        for i in range(1, self.num_levels):
            # Downsample lower level
            downsampled = self.bu_convs[i-1](bu_features[-1])
            
            # Fuse with top-down feature
            fused = downsampled + td_features[i]
            bu_features.append(fused)
        
        # Final fusion of top-down and bottom-up
        outputs = []
        for i in range(self.num_levels):
            # Concatenate td and bu features
            combined = torch.cat([td_features[i], bu_features[i]], dim=1)
            
            # Optional attention-based fusion
            if self.use_fusion_attention:
                att_weights = self.fusion_attention[i](combined)  # (B, 2, 1, 1)
                td_weight = att_weights[:, 0:1, :, :]
                bu_weight = att_weights[:, 1:2, :, :]
                combined = torch.cat([
                    td_features[i] * td_weight,
                    bu_features[i] * bu_weight
                ], dim=1)
            
            # Final projection
            output = self.fusion_convs[i](combined)
            outputs.append(output)
        
        return outputs
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return f"num_levels={self.num_levels}, out_channels={self.out_channels}"


class Detect_DR(nn.Module):
    """
    Domain-Robust Detection Head (Detect-DR) for cross-domain object detection.
    
    An enhanced detection head that maintains robustness across different domains by:
    1. Domain-invariant classification branch
    2. Domain-adaptive regression branch  
    3. Shared feature extractor with domain normalization
    4. Optional domain discriminator for adversarial training
    
    Key Features:
    - Domain normalization in shared layers
    - Separate domain-invariant and domain-adaptive paths
    - Compatible with standard YOLO training pipeline
    - Maintains detection accuracy while improving cross-domain generalization
    
    Args:
        nc (int): Number of classes. Default: 80.
        ch (tuple): Tuple of input channel sizes from PAN. Default: ().
        num_domains (int): Number of domains for normalization. Default: 3.
        reg_max (int): DFL channels for box regression. Default: 16.
        use_domain_classifier (bool): Whether to use domain classifier. Default: False.
        
    Shape:
        - Input: List of tensors [(B, C_i, H_i, W_i), ...] from PAN levels
        - Output (training): List of tensors [(B, no, H_i, W_i), ...] 
        - Output (inference): Processed detections
        
    Examples:
        >>> # Standard detection head
        >>> detect = Detect_DR(nc=80, ch=(256, 256, 256))
        >>> p3 = torch.randn(4, 256, 80, 80)
        >>> p4 = torch.randn(4, 256, 40, 40)
        >>> p5 = torch.randn(4, 256, 20, 20)
        >>> outputs = detect([p3, p4, p5])
        
        >>> # With domain classifier for adversarial training
        >>> detect = Detect_DR(nc=80, ch=(256, 256, 256), use_domain_classifier=True)
        >>> outputs, domain_preds = detect([p3, p4, p5], return_domain_pred=True)
    """
    
    # Class attributes for compatibility with ultralytics
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    
    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (),
        num_domains: int = 3,
        reg_max: int = 16,
        use_domain_classifier: bool = False,
    ):
        """Initialize Detect_DR head."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.num_domains = num_domains
        self.use_domain_classifier = use_domain_classifier
        
        # Shared feature extractors with domain normalization
        c_shared = max(ch[0] // 4, 64)
        self.shared_convs = nn.ModuleList([
            nn.Sequential(
                DomainNormalizedConv(x, c_shared, k=3, s=1, num_domains=num_domains),
                DomainNormalizedConv(c_shared, c_shared, k=3, s=1, num_domains=num_domains),
            )
            for x in ch
        ])
        
        # Domain-adaptive regression branch (bbox prediction)
        # Uses domain normalization to adapt to different domains
        c_reg = max(16, ch[0] // 4, self.reg_max * 4)
        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                DomainNormalizedConv(c_shared, c_reg, k=3, s=1, num_domains=num_domains),
                DomainNormalizedConv(c_reg, c_reg, k=3, s=1, num_domains=num_domains),
                nn.Conv2d(c_reg, 4 * self.reg_max, 1),
            )
            for _ in ch
        ])
        
        # Domain-invariant classification branch
        # Uses standard convolutions to learn domain-invariant features
        c_cls = max(ch[0] // 4, min(self.nc, 100))
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                # First layer: domain-normalized for initial adaptation
                DomainNormalizedConv(c_shared, c_cls, k=3, s=1, num_domains=num_domains),
                # Second layer: standard conv for domain-invariant features
                nn.Conv2d(c_cls, c_cls, 3, 1, 1),
                nn.BatchNorm2d(c_cls),
                nn.SiLU(inplace=True),
                # Final prediction (explicitly enable bias for initialization)
                nn.Conv2d(c_cls, self.nc, 1, bias=True),
            )
            for _ in ch
        ])
        
        # DFL (Distribution Focal Loss) layer
        if self.reg_max > 1:
            self.dfl = DFL(self.reg_max)
        else:
            self.dfl = nn.Identity()
        
        # Optional domain classifier for adversarial training
        if use_domain_classifier:
            self.domain_classifier = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c_shared, c_shared // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(c_shared // 4, num_domains),
                )
                for _ in ch
            ])
        
        # Initialize strides (will be set during model build)
        self.stride = torch.zeros(self.nl)
        
    def forward(
        self, 
        x: list[torch.Tensor],
        return_domain_pred: bool = False
    ) -> list[torch.Tensor] | tuple:
        """
        Forward pass through Detect_DR head.
        
        Args:
            x (list[torch.Tensor]): List of input feature maps from PAN.
            return_domain_pred (bool): Whether to return domain predictions. Default: False.
            
        Returns:
            list[torch.Tensor] | tuple: Detection outputs, optionally with domain predictions.
        """
        domain_preds = [] if (return_domain_pred and self.use_domain_classifier) else None
        
        for i in range(self.nl):
            # Shared feature extraction
            shared_feat = self.shared_convs[i](x[i])
            
            # Domain-adaptive bbox regression
            reg_out = self.reg_convs[i](shared_feat)
            
            # Domain-invariant classification
            cls_out = self.cls_convs[i](shared_feat)
            
            # Concatenate bbox and class predictions (in-place)
            x[i] = torch.cat((reg_out, cls_out), 1)
            
            # Optional domain prediction
            if domain_preds is not None:
                domain_preds.append(self.domain_classifier[i](shared_feat))
        
        if self.training:  # Training path
            if return_domain_pred and self.use_domain_classifier:
                return x, domain_preds
            return x
        
        # Inference path
        y = self._inference(x)
        if return_domain_pred and self.use_domain_classifier:
            return (y, x, domain_preds)
        return (y, x)
    
    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Perform inference with DFL decoding.
        
        Args:
            x (list[torch.Tensor]): List of detection tensors.
            
        Returns:
            torch.Tensor: Concatenated inference tensor.
        """
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self._make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # Split bbox and class predictions
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # DFL decoding for bbox
        if self.reg_max > 1:
            dbox = self.dfl(box)
        else:
            dbox = box
        
        # Decode boxes
        dbox = self._decode_bboxes(dbox)
        
        return torch.cat((dbox, cls.sigmoid()), 1)
    
    def _make_anchors(self, feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    
    def _decode_bboxes(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Decode bounding boxes."""
        return dist2bbox(bboxes, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    
    def bias_init(self):
        """Initialize detection head biases, WARNING: requires stride availability."""
        m = self  # Detect_DR module
        # Initialize regression head biases
        for a, s in zip(m.reg_convs, m.stride):
            a[-1].bias.data[:] = 1.0  # box bias initialization
        
        # Initialize classification head biases
        for b, s in zip(m.cls_convs, m.stride):
            # b[-1] is the final Conv2d layer
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls bias initialization
    
    def extra_repr(self) -> str:
        """Return extra string representation."""
        return (
            f"nc={self.nc}, nl={self.nl}, "
            f"num_domains={self.num_domains}, "
            f"use_domain_classifier={self.use_domain_classifier}"
        )


# DFL (Distribution Focal Loss) helper class
class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    
    Converts bbox predictions from distribution to precise values.
    Used in YOLO for more accurate bbox regression.
    
    Args:
        c1 (int): Number of input channels (reg_max).
    """
    
    def __init__(self, c1: int = 16):
        """Initialize DFL module."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DFL to input tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


if __name__ == "__main__":
    # Quick self-test
    print("Testing Domain Adaptation modules...")
    print("=" * 70)
    
    # Test DomainNormalization
    print("\n1. Testing DomainNormalization...")
    dn = DomainNormalization(num_features=256, num_domains=3)
    x = torch.randn(4, 256, 40, 40)
    y = dn(x)
    print(f"✓ DomainNormalization: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dn.parameters()):,}")
    
    # Test DomainNormalizedConv
    print("\n2. Testing DomainNormalizedConv...")
    dnconv = DomainNormalizedConv(c1=128, c2=256, k=3, s=2)
    x = torch.randn(4, 128, 80, 80)
    y = dnconv(x)
    print(f"✓ DomainNormalizedConv: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dnconv.parameters()):,}")
    
    # Test FeatureMixer
    print("\n3. Testing FeatureMixer...")
    mixer = FeatureMixer(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = mixer(x)
    print(f"✓ FeatureMixer: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mixer.parameters()):,}")
    
    # Test AdaptiveFeatureCalibration
    print("\n4. Testing AdaptiveFeatureCalibration...")
    calibration = AdaptiveFeatureCalibration(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = calibration(x)
    print(f"✓ AdaptiveFeatureCalibration: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in calibration.parameters()):,}")
    
    # Test FrequencyDomainSeparator
    print("\n5. Testing FrequencyDomainSeparator...")
    freq_sep = FrequencyDomainSeparator(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = freq_sep(x)
    print(f"✓ FrequencyDomainSeparator: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in freq_sep.parameters()):,}")
    
    # Test DomainAdaptiveBottleneck
    print("\n6. Testing DomainAdaptiveBottleneck...")
    bottleneck = DomainAdaptiveBottleneck(256, 256, use_mixer=True)
    x = torch.randn(4, 256, 40, 40)
    y = bottleneck(x)
    print(f"✓ DomainAdaptiveBottleneck: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in bottleneck.parameters()):,}")
    
    # Test C3k2_DN
    print("\n7. Testing C3k2_DN...")
    c3k2_dn = C3k2_DN(256, 512, n=3, use_mixer=True)
    x = torch.randn(4, 256, 40, 40)
    y = c3k2_dn(x)
    print(f"✓ C3k2_DN: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in c3k2_dn.parameters()):,}")
    print(f"  Number of blocks: {len(c3k2_dn.m)}")
    
    # Test SPPF_MS
    print("\n8. Testing SPPF_MS...")
    sppf_ms = SPPF_MS(256, 256)
    x = torch.randn(4, 256, 20, 20)
    y = sppf_ms(x)
    print(f"✓ SPPF_MS: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in sppf_ms.parameters()):,}")
    print(f"  Pooling sizes: {sppf_ms.pooling_sizes}")
    
    # Test PAN_DA
    print("\n9. Testing PAN_DA...")
    pan_da = PAN_DA([128, 256, 512], 256)
    p3 = torch.randn(4, 128, 80, 80)
    p4 = torch.randn(4, 256, 40, 40)
    p5 = torch.randn(4, 512, 20, 20)
    outputs = pan_da([p3, p4, p5])
    print(f"✓ PAN_DA:")
    for i, out in enumerate(outputs):
        print(f"    Level {i}: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in pan_da.parameters()):,}")
    print(f"  Number of levels: {pan_da.num_levels}")
    
    # Test Detect_DR
    print("\n10. Testing Detect_DR...")
    detect_dr = Detect_DR(nc=80, ch=(256, 256, 256))
    p3 = torch.randn(4, 256, 80, 80)
    p4 = torch.randn(4, 256, 40, 40)
    p5 = torch.randn(4, 256, 20, 20)
    outputs = detect_dr([p3, p4, p5])
    print(f"✓ Detect_DR:")
    for i, out in enumerate(outputs):
        print(f"    Level {i}: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in detect_dr.parameters()):,}")
    print(f"  Number of classes: {detect_dr.nc}")
    
    # Test Detect_DR with domain classifier
    print("\n11. Testing Detect_DR (with domain classifier)...")
    detect_dr_dc = Detect_DR(nc=80, ch=(256, 256, 256), use_domain_classifier=True)
    outputs, domain_preds = detect_dr_dc([p3, p4, p5], return_domain_pred=True)
    print(f"✓ Detect_DR with domain classifier:")
    print(f"  Detection outputs: {len(outputs)}")
    print(f"  Domain predictions: {len(domain_preds)}")
    for i, (det, dom) in enumerate(zip(outputs, domain_preds)):
        print(f"    Level {i}: Det {det.shape}, Domain {dom.shape}")
    print(f"  Parameters: {sum(p.numel() for p in detect_dr_dc.parameters()):,}")
    
    print("\n" + "=" * 70)
    print("✅ All basic tests passed!")
    print("=" * 70)

