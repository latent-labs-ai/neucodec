"""
Multi-Resolution Mel Spectrogram Loss for NeuCodec training.
Based on HiFi-GAN and EnCodec loss implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MelSpectrogramLoss(nn.Module):
    """
    Single-resolution Mel spectrogram loss.
    Computes L1 loss between mel spectrograms of real and generated audio.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        center: bool = True,
        power: float = 1.0,
        normalized: bool = False,
        norm: Optional[str] = None,
        log_scale: bool = True,
        log_offset: float = 1e-5,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.power = power
        self.log_scale = log_scale
        self.log_offset = log_offset
        
        # Create mel filterbank
        f_max = f_max or sample_rate // 2
        
        # Register window as buffer
        self.register_buffer("window", torch.hann_window(win_length))
        
        # Create mel filterbank using torchaudio-compatible approach
        mel_fb = self._create_mel_filterbank(
            sample_rate, n_fft, n_mels, f_min, f_max, norm
        )
        self.register_buffer("mel_fb", mel_fb)
        
    def _create_mel_filterbank(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        norm: Optional[str],
    ) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Frequency to mel conversion
        def hz_to_mel(f):
            return 2595.0 * torch.log10(1.0 + f / 700.0)
        
        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        
        # Compute mel points
        n_freqs = n_fft // 2 + 1
        all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
        
        m_min = hz_to_mel(torch.tensor(f_min))
        m_max = hz_to_mel(torch.tensor(f_max))
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        
        # Create filterbank
        fb = torch.zeros(n_freqs, n_mels)
        for i in range(n_mels):
            f_left = f_pts[i]
            f_center = f_pts[i + 1]
            f_right = f_pts[i + 2]
            
            # Rising slope
            up_slope = (all_freqs - f_left) / (f_center - f_left + 1e-8)
            # Falling slope
            down_slope = (f_right - all_freqs) / (f_right - f_center + 1e-8)
            
            fb[:, i] = torch.max(
                torch.zeros_like(all_freqs),
                torch.min(up_slope, down_slope)
            )
        
        if norm == "slaney":
            # Slaney-style normalization
            enorm = 2.0 / (f_pts[2:n_mels+2] - f_pts[:n_mels])
            fb *= enorm.unsqueeze(0)
        
        return fb.T  # [n_mels, n_freqs]
    
    def forward(
        self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mel spectrogram loss.
        
        Args:
            y_hat: Generated audio [B, 1, T] or [B, T]
            y: Target audio [B, 1, T] or [B, T]
            
        Returns:
            loss: L1 loss between mel spectrograms
        """
        # Ensure 2D input
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        
        # Compute spectrograms
        mel_hat = self._compute_mel(y_hat)
        mel_target = self._compute_mel(y)
        
        # L1 loss
        loss = F.l1_loss(mel_hat, mel_target)
        
        return loss
    
    def _compute_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram."""
        # STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        
        # Magnitude
        mag = torch.abs(stft) ** self.power
        
        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb, mag)
        
        # Log scale
        if self.log_scale:
            mel = torch.log(mel + self.log_offset)
        
        return mel


class MultiResolutionMelLoss(nn.Module):
    """
    Multi-resolution mel spectrogram loss.
    Combines losses at multiple STFT resolutions for better frequency coverage.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        
        self.losses = nn.ModuleList([
            MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            )
            for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths)
        ])
    
    def forward(
        self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-resolution mel loss.
        
        Args:
            y_hat: Generated audio [B, 1, T] or [B, T]
            y: Target audio [B, 1, T] or [B, T]
            
        Returns:
            loss: Sum of mel losses at all resolutions
        """
        total_loss = 0.0
        
        for mel_loss in self.losses:
            total_loss = total_loss + mel_loss(y_hat, y)
        
        return total_loss / len(self.losses)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss (spectral convergence + log magnitude).
    Alternative to mel loss for more direct spectral supervision.
    """
    
    def __init__(
        self,
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
    ):
        super().__init__()
        
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        
        # Register windows
        for i, win_length in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(win_length))
    
    def forward(
        self, 
        y_hat: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            y_hat: Generated audio [B, 1, T] or [B, T]
            y: Target audio [B, 1, T] or [B, T]
            
        Returns:
            loss: Sum of spectral convergence and log magnitude losses
        """
        if y_hat.dim() == 3:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        
        for i, (n_fft, hop_length, win_length) in enumerate(
            zip(self.n_ffts, self.hop_lengths, self.win_lengths)
        ):
            window = getattr(self, f"window_{i}")
            
            # Compute STFTs
            stft_hat = torch.stft(
                y_hat, n_fft, hop_length, win_length, window, return_complex=True
            )
            stft_target = torch.stft(
                y, n_fft, hop_length, win_length, window, return_complex=True
            )
            
            # Magnitudes
            mag_hat = torch.abs(stft_hat)
            mag_target = torch.abs(stft_target)
            
            # Spectral convergence loss
            sc_loss = torch.norm(mag_target - mag_hat, p="fro") / (
                torch.norm(mag_target, p="fro") + 1e-8
            )
            total_sc_loss = total_sc_loss + sc_loss
            
            # Log magnitude loss
            log_mag_hat = torch.log(mag_hat + 1e-8)
            log_mag_target = torch.log(mag_target + 1e-8)
            mag_loss = F.l1_loss(log_mag_hat, log_mag_target)
            total_mag_loss = total_mag_loss + mag_loss
        
        return (total_sc_loss + total_mag_loss) / len(self.n_ffts)
