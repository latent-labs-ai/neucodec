"""
Multi-Scale Discriminator (MSD) for NeuCodec training.
Based on HiFi-GAN: https://arxiv.org/abs/2010.05646
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Tuple


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator operating on raw or downsampled waveform."""
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        max_channels: int = 1024,
        kernel_sizes: List[int] = [15, 41, 41, 41, 41, 5, 3],
        strides: List[int] = [1, 2, 2, 4, 4, 1, 1],
        groups: List[int] = [1, 4, 16, 16, 16, 1, 1],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            norm_f(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    padding=kernel_sizes[0] // 2,
                )
            )
        )
        
        # Middle layers
        in_ch = channels
        for i in range(1, len(kernel_sizes) - 1):
            out_ch = min(channels * (2 ** i), max_channels)
            self.convs.append(
                norm_f(
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        groups=groups[i],
                        padding=kernel_sizes[i] // 2,
                    )
                )
            )
            in_ch = out_ch
        
        # Last conv before output
        self.convs.append(
            norm_f(
                nn.Conv1d(
                    in_ch,
                    in_ch,
                    kernel_size=kernel_sizes[-1],
                    stride=strides[-1],
                    padding=kernel_sizes[-1] // 2,
                )
            )
        )
        
        # Output projection
        self.conv_post = norm_f(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            output: Discriminator output
            features: List of intermediate features for feature matching loss
        """
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator combining multiple ScaleDiscriminators.
    Each operates at different temporal resolutions using average pooling.
    """
    
    def __init__(
        self,
        scales: int = 3,
        channels: int = 128,
        max_channels: int = 1024,
        pool_kernel_size: int = 4,
        pool_stride: int = 2,
        pool_padding: int = 2,
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        self.pooling = nn.ModuleList()
        
        for i in range(scales):
            # First discriminator uses spectral norm, others use weight norm
            self.discriminators.append(
                ScaleDiscriminator(
                    channels=channels,
                    max_channels=max_channels,
                    use_spectral_norm=(i == 0),
                )
            )
            
            # Add pooling for downsampling (except for last scale)
            if i < scales - 1:
                self.pooling.append(
                    nn.AvgPool1d(
                        kernel_size=pool_kernel_size,
                        stride=pool_stride,
                        padding=pool_padding,
                    )
                )
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            outputs: List of discriminator outputs
            features: List of feature lists for feature matching loss
        """
        outputs = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
            
            # Downsample for next scale
            if i < len(self.pooling):
                x = self.pooling[i](x)
        
        return outputs, features
