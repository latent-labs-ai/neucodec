"""
Multi-Period Discriminator (MPD) for NeuCodec training.
Based on HiFi-GAN: https://arxiv.org/abs/2010.05646
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Tuple


class PeriodDiscriminator(nn.Module):
    """Single period discriminator that operates on reshapsed input."""
    
    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        channels: int = 32,
        max_channels: int = 1024,
        kernel_size: int = 5,
        stride: int = 3,
        num_layers: int = 4,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            norm_f(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(kernel_size // 2, 0),
                )
            )
        )
        
        # Middle layers with channel growth
        in_ch = channels
        for i in range(1, num_layers):
            out_ch = min(channels * (2 ** i), max_channels)
            self.convs.append(
                norm_f(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
            )
            in_ch = out_ch
        
        # Final conv before output
        self.convs.append(
            norm_f(
                nn.Conv2d(
                    in_ch,
                    in_ch,
                    kernel_size=(kernel_size, 1),
                    stride=(1, 1),
                    padding=(kernel_size // 2, 0),
                )
            )
        )
        
        # Output layer
        self.conv_post = norm_f(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            output: Discriminator output
            features: List of intermediate features for feature matching loss
        """
        features = []
        
        # Reshape input based on period
        b, c, t = x.shape
        
        # Pad to make divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        
        # Reshape to [B, C, T/period, period]
        x = x.view(b, c, t // self.period, self.period)
        
        # Apply conv layers
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        # Output
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator combining multiple PeriodDiscriminators.
    Each operates on the input reshaped with different periods to capture
    diverse periodic patterns in speech.
    """
    
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        channels: int = 32,
        max_channels: int = 1024,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(
                period=p,
                channels=channels,
                max_channels=max_channels,
                use_spectral_norm=use_spectral_norm,
            )
            for p in periods
        ])
        
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
        
        for disc in self.discriminators:
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
            
        return outputs, features
