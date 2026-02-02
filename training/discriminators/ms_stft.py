"""
Multi-Scale STFT Discriminator for NeuCodec training.
Based on EnCodec/DAC discriminator design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Tuple, Optional


class ComplexSTFTDiscriminator(nn.Module):
    """
    Single-scale complex STFT discriminator.
    Operates on both magnitude and phase information.
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        filters: int = 32,
        max_filters: int = 1024,
        num_layers: int = 5,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Register window buffer
        self.register_buffer("window", torch.hann_window(win_length))
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Number of frequency bins (real + imag)
        in_channels = 2  # Real and imaginary parts
        
        self.convs = nn.ModuleList()
        
        # First conv layer
        self.convs.append(
            norm_f(
                nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=(7, 5),
                    stride=(2, 2),
                    padding=(3, 2),
                )
            )
        )
        
        # Middle layers
        in_ch = filters
        for i in range(1, num_layers):
            out_ch = min(filters * (2 ** i), max_filters)
            self.convs.append(
                norm_f(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                )
            )
            in_ch = out_ch
        
        # Output projection
        self.conv_post = norm_f(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            output: Discriminator output
            features: List of intermediate features
        """
        features = []
        
        # Squeeze channel dim for STFT
        x = x.squeeze(1)  # [B, T]
        
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        
        # Stack real and imaginary parts: [B, 2, F, T]
        x = torch.stack([stft.real, stft.imag], dim=1)
        
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


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator combining multiple ComplexSTFTDiscriminators
    with different STFT resolutions for multi-resolution frequency analysis.
    """
    
    def __init__(
        self,
        filters: int = 32,
        max_filters: int = 1024,
        n_ffts: List[int] = [1024, 2048, 512],
        hop_lengths: List[int] = [256, 512, 128],
        win_lengths: List[int] = [1024, 2048, 512],
    ):
        super().__init__()
        
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        
        self.discriminators = nn.ModuleList([
            ComplexSTFTDiscriminator(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                filters=filters,
                max_filters=max_filters,
                use_spectral_norm=(i == 0),
            )
            for i, (n_fft, hop_length, win_length) in enumerate(
                zip(n_ffts, hop_lengths, win_lengths)
            )
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
