"""
Audio Augmentation Pipeline for NeuCodec training.
Provides various audio augmentations to improve model robustness.
"""

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import random
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""
    enabled: bool = True
    
    # Noise augmentation
    noise_prob: float = 0.3
    noise_snr_range: Tuple[float, float] = (5.0, 30.0)  # dB
    
    # Reverb augmentation
    reverb_prob: float = 0.2
    
    # Pitch shift
    pitch_shift_prob: float = 0.1
    pitch_shift_range: Tuple[int, int] = (-2, 2)  # semitones
    
    # Time stretch
    time_stretch_prob: float = 0.1
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    # Volume perturbation
    volume_prob: float = 0.5
    volume_range: Tuple[float, float] = (-6.0, 3.0)  # dB
    
    # Low-pass filter (simulate bandwidth limitation)
    lowpass_prob: float = 0.1
    lowpass_cutoff_range: Tuple[int, int] = (4000, 7000)  # Hz
    
    # High-pass filter (remove low frequency noise)
    highpass_prob: float = 0.1
    highpass_cutoff_range: Tuple[int, int] = (50, 300)  # Hz
    
    # Codec simulation (simulate compression artifacts)
    codec_prob: float = 0.1
    codec_bitrates: List[int] = None  # Will default to [8000, 16000, 32000]


class AudioAugmentor:
    """
    Audio augmentation pipeline for training neural audio codecs.
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        sample_rate: int = 16000,
    ):
        self.config = config or AugmentationConfig()
        self.sample_rate = sample_rate
        
        if self.config.codec_bitrates is None:
            self.config.codec_bitrates = [8000, 16000, 32000]
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to the input waveform.
        
        Args:
            waveform: Input audio tensor [1, T] or [T]
            
        Returns:
            Augmented waveform tensor
        """
        if not self.config.enabled:
            return waveform
        
        # Ensure 2D tensor [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply augmentations with their respective probabilities
        
        # Volume perturbation (most common)
        if random.random() < self.config.volume_prob:
            waveform = self.apply_volume(waveform)
        
        # Add noise
        if random.random() < self.config.noise_prob:
            waveform = self.add_noise(waveform)
        
        # Apply reverb
        if random.random() < self.config.reverb_prob:
            waveform = self.add_reverb(waveform)
        
        # Pitch shift
        if random.random() < self.config.pitch_shift_prob:
            waveform = self.pitch_shift(waveform)
        
        # Time stretch
        if random.random() < self.config.time_stretch_prob:
            waveform = self.time_stretch(waveform)
        
        # Low-pass filter
        if random.random() < self.config.lowpass_prob:
            waveform = self.apply_lowpass(waveform)
        
        # High-pass filter
        if random.random() < self.config.highpass_prob:
            waveform = self.apply_highpass(waveform)
        
        # Codec simulation
        if random.random() < self.config.codec_prob:
            waveform = self.simulate_codec(waveform)
        
        # Ensure no clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform
    
    def apply_volume(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random volume change in dB."""
        db_change = random.uniform(*self.config.volume_range)
        gain = 10 ** (db_change / 20)
        return waveform * gain
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        snr_db = random.uniform(*self.config.noise_snr_range)
        
        # Calculate signal power
        signal_power = waveform.pow(2).mean()
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add synthetic reverb using simple convolution."""
        # Generate simple impulse response
        ir_length = int(self.sample_rate * random.uniform(0.1, 0.5))
        decay = random.uniform(0.3, 0.7)
        
        # Exponential decay IR
        t = torch.linspace(0, 1, ir_length)
        ir = torch.exp(-decay * t * 10) * torch.randn(ir_length)
        ir = ir / ir.abs().max()  # Normalize
        ir = ir.unsqueeze(0).unsqueeze(0)
        
        # Convolve with padding
        waveform_padded = F.pad(waveform.unsqueeze(0), (ir_length - 1, 0))
        reverbed = F.conv1d(waveform_padded, ir)
        reverbed = reverbed.squeeze(0)
        
        # Mix dry and wet
        wet_ratio = random.uniform(0.1, 0.4)
        return (1 - wet_ratio) * waveform + wet_ratio * reverbed[..., :waveform.shape[-1]]
    
    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shift using resampling."""
        n_steps = random.randint(*self.config.pitch_shift_range)
        if n_steps == 0:
            return waveform
        
        # Calculate stretch factor
        stretch_factor = 2 ** (-n_steps / 12)
        
        # Resample to shift pitch
        new_freq = int(self.sample_rate * stretch_factor)
        
        # Resample down/up then back to original rate
        resampler_down = T.Resample(self.sample_rate, new_freq)
        resampler_up = T.Resample(new_freq, self.sample_rate)
        
        shifted = resampler_up(resampler_down(waveform))
        
        # Match original length
        if shifted.shape[-1] > waveform.shape[-1]:
            shifted = shifted[..., :waveform.shape[-1]]
        elif shifted.shape[-1] < waveform.shape[-1]:
            shifted = F.pad(shifted, (0, waveform.shape[-1] - shifted.shape[-1]))
        
        return shifted
    
    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretch using phase vocoder."""
        rate = random.uniform(*self.config.time_stretch_range)
        if abs(rate - 1.0) < 0.01:
            return waveform
        
        # Use torchaudio's Stretch
        stretch = T.TimeStretch(
            hop_length=256,
            n_freq=513,
            fixed_rate=rate,
        )
        
        # Convert to spectrogram, stretch, convert back
        spec_transform = T.Spectrogram(n_fft=1024, hop_length=256, power=None)
        inv_spec = T.InverseSpectrogram(n_fft=1024, hop_length=256)
        
        spec = spec_transform(waveform)
        stretched_spec = stretch(spec)
        stretched = inv_spec(stretched_spec)
        
        # Match original length
        target_len = waveform.shape[-1]
        if stretched.shape[-1] > target_len:
            stretched = stretched[..., :target_len]
        elif stretched.shape[-1] < target_len:
            stretched = F.pad(stretched, (0, target_len - stretched.shape[-1]))
        
        return stretched
    
    def apply_lowpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter."""
        cutoff = random.randint(*self.config.lowpass_cutoff_range)
        
        # Simple FIR low-pass filter
        lowpass = T.Lowpass(self.sample_rate, cutoff)
        return lowpass(waveform)
    
    def apply_highpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply high-pass filter."""
        cutoff = random.randint(*self.config.highpass_cutoff_range)
        
        # Simple FIR high-pass filter
        highpass = T.Highpass(self.sample_rate, cutoff)
        return highpass(waveform)
    
    def simulate_codec(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate lossy codec compression artifacts."""
        # Simple quantization-based simulation
        bitrate = random.choice(self.config.codec_bitrates)
        
        # Simulate bit reduction based on bitrate
        bits = max(4, int(math.log2(bitrate / 1000)))
        levels = 2 ** bits
        
        # Quantize and dequantize
        quantized = torch.round(waveform * levels) / levels
        
        return quantized


class SpecAugment:
    """
    SpecAugment-style augmentation for spectrograms.
    Useful for semantic encoder training.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
    ):
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [B, F, T] or [F, T]
            
        Returns:
            Augmented spectrogram
        """
        for _ in range(self.num_freq_masks):
            spectrogram = self.freq_mask(spectrogram)
        
        for _ in range(self.num_time_masks):
            spectrogram = self.time_mask(spectrogram)
        
        return spectrogram
