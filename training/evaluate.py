"""
NeuCodec Evaluation Script

Evaluates trained NeuCodec models using standard audio quality metrics:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- MCD (Mel Cepstral Distortion)
- Optional: WER via ASR model

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.pt --data path/to/test/data
    python evaluate.py --model neuphonic/neucodec --data path/to/test/data
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neucodec import NeuCodec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    pesq_wb: Optional[float] = None  # Wideband PESQ
    pesq_nb: Optional[float] = None  # Narrowband PESQ
    stoi: Optional[float] = None
    estoi: Optional[float] = None  # Extended STOI
    si_sdr: Optional[float] = None
    mcd: Optional[float] = None  # Mel Cepstral Distortion
    snr: Optional[float] = None
    lsd: Optional[float] = None  # Log Spectral Distance


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Remove mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Compute SI-SDR
    dot = np.dot(reference, estimate)
    s_target = dot * reference / (np.dot(reference, reference) + 1e-8)
    e_noise = estimate - s_target
    
    si_sdr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8
    )
    
    return float(si_sdr)


def compute_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        
    Returns:
        SNR in dB
    """
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    noise = reference - estimate
    
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8) + 1e-8)
    
    return float(snr)


def compute_lsd(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """
    Compute Log Spectral Distance.
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        LSD in dB
    """
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Compute spectrograms
    ref_spec = np.abs(
        np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(reference, (n_fft // 2, n_fft // 2)), n_fft
            )[::hop_length] * np.hanning(n_fft),
            axis=-1
        )
    )
    est_spec = np.abs(
        np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(
                np.pad(estimate, (n_fft // 2, n_fft // 2)), n_fft
            )[::hop_length] * np.hanning(n_fft),
            axis=-1
        )
    )
    
    # Compute LSD
    log_ref = np.log10(ref_spec + 1e-8)
    log_est = np.log10(est_spec + 1e-8)
    
    lsd = np.sqrt(np.mean((log_ref - log_est) ** 2))
    
    return float(lsd)


def compute_mcd(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
) -> float:
    """
    Compute Mel Cepstral Distortion.
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MCD in dB
    """
    try:
        import librosa
        
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]
        
        # Compute MFCCs
        mfcc_ref = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_est = librosa.feature.mfcc(y=estimate, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Align lengths
        min_frames = min(mfcc_ref.shape[1], mfcc_est.shape[1])
        mfcc_ref = mfcc_ref[:, :min_frames]
        mfcc_est = mfcc_est[:, :min_frames]
        
        # Compute MCD (exclude c0)
        diff = mfcc_ref[1:] - mfcc_est[1:]
        mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))
        
        return float(mcd)
    
    except ImportError:
        logger.warning("librosa not installed, skipping MCD computation")
        return None


def compute_pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> Optional[float]:
    """
    Compute PESQ score.
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        sample_rate: Sample rate (16000 for wideband, 8000 for narrowband)
        mode: 'wb' for wideband, 'nb' for narrowband
        
    Returns:
        PESQ score (1.0 to 4.5)
    """
    try:
        from pesq import pesq
        
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]
        
        # PESQ requires specific sample rates
        if mode == "wb" and sample_rate != 16000:
            # Resample to 16kHz for wideband
            reference = torchaudio.functional.resample(
                torch.tensor(reference), sample_rate, 16000
            ).numpy()
            estimate = torchaudio.functional.resample(
                torch.tensor(estimate), sample_rate, 16000
            ).numpy()
            sample_rate = 16000
        elif mode == "nb" and sample_rate != 8000:
            # Resample to 8kHz for narrowband
            reference = torchaudio.functional.resample(
                torch.tensor(reference), sample_rate, 8000
            ).numpy()
            estimate = torchaudio.functional.resample(
                torch.tensor(estimate), sample_rate, 8000
            ).numpy()
            sample_rate = 8000
        
        score = pesq(sample_rate, reference, estimate, mode)
        return float(score)
    
    except ImportError:
        logger.warning("pesq not installed, skipping PESQ computation. Install with: pip install pesq")
        return None
    except Exception as e:
        logger.warning(f"PESQ computation failed: {e}")
        return None


def compute_stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False,
) -> Optional[float]:
    """
    Compute STOI (Short-Time Objective Intelligibility).
    
    Args:
        reference: Clean reference signal
        estimate: Estimated/reconstructed signal
        sample_rate: Sample rate
        extended: Whether to compute extended STOI
        
    Returns:
        STOI score (0.0 to 1.0)
    """
    try:
        from pystoi import stoi
        
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]
        
        score = stoi(reference, estimate, sample_rate, extended=extended)
        return float(score)
    
    except ImportError:
        logger.warning("pystoi not installed, skipping STOI computation. Install with: pip install pystoi")
        return None
    except Exception as e:
        logger.warning(f"STOI computation failed: {e}")
        return None


class NeuCodecEvaluator:
    """
    Evaluator for NeuCodec models.
    """
    
    def __init__(
        self,
        model: NeuCodec,
        device: torch.device,
        sample_rate: int = 16000,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = sample_rate
    
    @classmethod
    def from_pretrained(cls, model_id: str, device: torch.device):
        """Load model from HuggingFace."""
        model = NeuCodec.from_pretrained(model_id)
        return cls(model, device)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device):
        """Load model from training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = NeuCodec(24000, 480)
        model.load_state_dict(checkpoint["generator"], strict=False)
        
        return cls(model, device)
    
    @torch.no_grad()
    def encode_decode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode audio through the codec.
        
        Args:
            audio: Input audio tensor [1, T] or [B, 1, T]
            
        Returns:
            Reconstructed audio tensor
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        # Encode
        codes = self.model.encode_code(audio)
        
        # Decode
        recon = self.model.decode_code(codes)
        
        return recon.cpu()
    
    def evaluate_file(
        self,
        audio_path: str,
        compute_pesq_score: bool = True,
        compute_stoi_score: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate codec on single audio file.
        
        Args:
            audio_path: Path to audio file
            compute_pesq_score: Whether to compute PESQ
            compute_stoi_score: Whether to compute STOI
            
        Returns:
            EvaluationMetrics with all computed scores
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        
        # Encode and decode
        recon = self.encode_decode(waveform)
        
        # Resample reconstruction to match input rate for comparison
        if self.model.sample_rate != self.sample_rate:
            recon = T.Resample(self.model.sample_rate, self.sample_rate)(recon)
        
        # Convert to numpy
        ref = waveform.squeeze().numpy()
        est = recon.squeeze().numpy()
        
        # Align lengths
        min_len = min(len(ref), len(est))
        ref = ref[:min_len]
        est = est[:min_len]
        
        # Compute metrics
        metrics = EvaluationMetrics()
        
        # SI-SDR
        metrics.si_sdr = compute_si_sdr(ref, est)
        
        # SNR
        metrics.snr = compute_snr(ref, est)
        
        # LSD
        metrics.lsd = compute_lsd(ref, est, self.sample_rate)
        
        # MCD
        metrics.mcd = compute_mcd(ref, est, self.sample_rate)
        
        # PESQ
        if compute_pesq_score:
            metrics.pesq_wb = compute_pesq(ref, est, self.sample_rate, "wb")
            metrics.pesq_nb = compute_pesq(ref, est, 8000, "nb")  # Will resample
        
        # STOI
        if compute_stoi_score:
            metrics.stoi = compute_stoi(ref, est, self.sample_rate, extended=False)
            metrics.estoi = compute_stoi(ref, est, self.sample_rate, extended=True)
        
        return metrics
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        compute_pesq_score: bool = True,
        compute_stoi_score: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate codec on dataset.
        
        Args:
            data_path: Path to directory of audio files
            output_path: Path to save results JSON
            max_samples: Maximum number of samples to evaluate
            compute_pesq_score: Whether to compute PESQ
            compute_stoi_score: Whether to compute STOI
            
        Returns:
            Dictionary of average metrics
        """
        data_path = Path(data_path)
        
        # Find audio files
        audio_files = []
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            audio_files.extend(data_path.rglob(f"*{ext}"))
        
        if max_samples:
            audio_files = audio_files[:max_samples]
        
        logger.info(f"Evaluating on {len(audio_files)} files...")
        
        # Evaluate all files
        all_metrics = []
        results = []
        
        for audio_path in tqdm(audio_files, desc="Evaluating"):
            try:
                metrics = self.evaluate_file(
                    str(audio_path),
                    compute_pesq_score=compute_pesq_score,
                    compute_stoi_score=compute_stoi_score,
                )
                all_metrics.append(metrics)
                results.append({
                    "file": str(audio_path),
                    "metrics": asdict(metrics)
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate {audio_path}: {e}")
        
        # Compute averages
        avg_metrics = {}
        metric_names = ["pesq_wb", "pesq_nb", "stoi", "estoi", "si_sdr", "mcd", "snr", "lsd"]
        
        for name in metric_names:
            values = [getattr(m, name) for m in all_metrics if getattr(m, name) is not None]
            if values:
                avg_metrics[f"avg_{name}"] = float(np.mean(values))
                avg_metrics[f"std_{name}"] = float(np.std(values))
        
        # Save results
        if output_path:
            output = {
                "summary": avg_metrics,
                "num_samples": len(all_metrics),
                "per_file_results": results,
            }
            
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        for k, v in avg_metrics.items():
            if "avg_" in k:
                logger.info(f"{k}: {v:.4f}")
        logger.info("=" * 50)
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuCodec")
    parser.add_argument("--model", type=str, default="neuphonic/neucodec",
                        help="Model ID from HuggingFace")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to training checkpoint (overrides --model)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to evaluation data directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save results JSON")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--no-pesq", action="store_true",
                        help="Skip PESQ computation (faster)")
    parser.add_argument("--no-stoi", action="store_true",
                        help="Skip STOI computation (faster)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if args.checkpoint:
        logger.info(f"Loading from checkpoint: {args.checkpoint}")
        evaluator = NeuCodecEvaluator.from_checkpoint(args.checkpoint, device)
    else:
        logger.info(f"Loading from HuggingFace: {args.model}")
        evaluator = NeuCodecEvaluator.from_pretrained(args.model, device)
    
    # Evaluate
    evaluator.evaluate_dataset(
        data_path=args.data,
        output_path=args.output,
        max_samples=args.max_samples,
        compute_pesq_score=not args.no_pesq,
        compute_stoi_score=not args.no_stoi,
    )


if __name__ == "__main__":
    main()
