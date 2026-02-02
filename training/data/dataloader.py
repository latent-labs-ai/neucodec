"""
Audio Dataset and DataLoader for NeuCodec training.
Supports multiple data formats and efficient loading.
"""

import os
import random
import json
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import Optional, List, Dict, Union, Callable
from dataclasses import dataclass
import logging

from .augmentation import AudioAugmentor, AugmentationConfig

logger = logging.getLogger(__name__)


class ChatMLDataset(Dataset):
    """
    Dataset for ChatML format used in TTS training.
    Extracts audio files from ChatML JSON for codec training.
    
    ChatML format expected:
    [
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "audio", "audio_url": "/path/to/ref.wav"},
            {"type": "text", "text": "..."}
          ]},
          {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
            {"type": "audio", "audio_url": "/path/to/target.wav", "duration": 4.85}
          ]}
        ],
        "speaker": "speaker_id",
        "misc": {"duration": 4.85, ...}
      }
    ]
    """
    
    def __init__(
        self,
        json_paths: List[str],
        segment_length: int = 32000,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        augmentor: Optional[AudioAugmentor] = None,
        is_training: bool = True,
        use_reference_audio: bool = True,
        use_target_audio: bool = True,
        audio_base_path: Optional[str] = None,
    ):
        """
        Initialize ChatML dataset.
        
        Args:
            json_paths: List of paths to ChatML JSON files
            segment_length: Length of audio segments in samples
            sample_rate: Target sample rate
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            augmentor: Optional audio augmentor
            is_training: Whether this is training data
            use_reference_audio: Include reference audio from user messages
            use_target_audio: Include target audio from assistant messages
            audio_base_path: Base path to prepend to audio URLs if they're relative
        """
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.augmentor = augmentor
        self.is_training = is_training
        self.use_reference_audio = use_reference_audio
        self.use_target_audio = use_target_audio
        self.audio_base_path = audio_base_path
        
        # Collect all audio entries
        self.audio_entries = []
        
        for json_path in json_paths:
            self._load_chatml_json(json_path)
        
        logger.info(f"Loaded {len(self.audio_entries)} audio entries from ChatML")
        
        # Filter by duration
        self._filter_by_duration()
        
        logger.info(f"After filtering: {len(self.audio_entries)} audio entries")
    
    def _load_chatml_json(self, json_path: str):
        """Load and parse ChatML JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            logger.warning(f"JSON file not found: {json_path}")
            return
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            data = [data]
        
        for entry in data:
            self._parse_chatml_entry(entry)
    
    def _parse_chatml_entry(self, entry: Dict):
        """Parse a single ChatML entry and extract audio info."""
        messages = entry.get("messages", [])
        speaker = entry.get("speaker", "unknown")
        misc = entry.get("misc", {})
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])
            
            # Skip system messages
            if role == "system":
                continue
            
            # Handle content that could be string or list
            if isinstance(content, str):
                continue
            
            for item in content:
                if not isinstance(item, dict):
                    continue
                    
                if item.get("type") == "audio":
                    audio_url = item.get("audio_url", "")
                    
                    if not audio_url:
                        continue
                    
                    # Determine if this is reference or target audio
                    is_reference = (role == "user")
                    is_target = (role == "assistant")
                    
                    # Skip based on settings
                    if is_reference and not self.use_reference_audio:
                        continue
                    if is_target and not self.use_target_audio:
                        continue
                    
                    # Get duration
                    duration = item.get("duration") or misc.get("duration")
                    
                    # Resolve audio path
                    audio_path = self._resolve_audio_path(audio_url)
                    
                    if audio_path and os.path.exists(audio_path):
                        self.audio_entries.append({
                            "audio_path": audio_path,
                            "duration": duration,
                            "speaker": speaker,
                            "role": role,
                            "is_reference": is_reference,
                            "is_target": is_target,
                        })
                    else:
                        logger.debug(f"Audio file not found: {audio_path}")
    
    def _resolve_audio_path(self, audio_url: str) -> str:
        """Resolve audio URL to absolute path."""
        if os.path.isabs(audio_url):
            return audio_url
        
        if self.audio_base_path:
            return os.path.join(self.audio_base_path, audio_url)
        
        return audio_url
    
    def _filter_by_duration(self):
        """Filter entries by duration constraints."""
        filtered = []
        for entry in self.audio_entries:
            duration = entry.get("duration")
            if duration is None:
                filtered.append(entry)  # Keep entries without duration info
            elif self.min_duration <= duration <= self.max_duration:
                filtered.append(entry)
        
        self.audio_entries = filtered
    
    def __len__(self) -> int:
        return len(self.audio_entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process audio file.
        
        Returns:
            Dictionary with audio tensor and metadata
        """
        entry = self.audio_entries[idx]
        audio_path = entry["audio_path"]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Extract segment
            waveform = self._extract_segment(waveform)
            
            # Apply augmentation
            if self.is_training and self.augmentor is not None:
                waveform = self.augmentor(waveform)
            
            # Normalize
            waveform = self._normalize(waveform)
            
            return {
                "audio": waveform,
                "audio_path": audio_path,
                "speaker": entry.get("speaker", "unknown"),
                "is_reference": entry.get("is_reference", False),
                "is_target": entry.get("is_target", False),
            }
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return random other sample on error
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def _extract_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment from waveform."""
        length = waveform.shape[-1]
        
        if length < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif length > self.segment_length and self.is_training:
            # Random crop during training
            start = random.randint(0, length - self.segment_length)
            waveform = waveform[..., start:start + self.segment_length]
        elif length > self.segment_length:
            # Center crop during validation
            start = (length - self.segment_length) // 2
            waveform = waveform[..., start:start + self.segment_length]
        
        return waveform
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to [-1, 1]."""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        return waveform


@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_paths: List[str]
    val_paths: List[str]
    segment_length: int = 32000  # 2 seconds at 16kHz
    sample_rate: int = 16000
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    min_duration: float = 1.0
    max_duration: float = 30.0
    augmentation: Optional[AugmentationConfig] = None


class AudioDataset(Dataset):
    """
    Audio dataset for training neural audio codecs.
    
    Supports:
    - Directory of audio files (wav, flac, mp3, etc.)
    - Manifest files (JSON/JSONL with file paths)
    - HuggingFace datasets format
    """
    
    SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}
    
    def __init__(
        self,
        data_paths: List[str],
        segment_length: int = 32000,
        sample_rate: int = 16000,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        augmentor: Optional[AudioAugmentor] = None,
        is_training: bool = True,
    ):
        """
        Initialize audio dataset.
        
        Args:
            data_paths: List of paths to data directories or manifest files
            segment_length: Length of audio segments in samples
            sample_rate: Target sample rate
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            augmentor: Optional audio augmentor
            is_training: Whether this is training data (enables augmentation)
        """
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.augmentor = augmentor
        self.is_training = is_training
        
        # Collect all audio files
        self.audio_files = []
        self.metadata = {}  # Optional metadata per file
        
        for path in data_paths:
            self._load_path(path)
        
        logger.info(f"Loaded {len(self.audio_files)} audio files")
        
        # Filter by duration if metadata available
        self._filter_by_duration()
        
        logger.info(f"After filtering: {len(self.audio_files)} audio files")
    
    def _load_path(self, path: str):
        """Load audio files from a path (directory or manifest)."""
        path = Path(path)
        
        if path.is_dir():
            self._load_directory(path)
        elif path.suffix in {".json", ".jsonl"}:
            self._load_manifest(path)
        elif path.suffix == ".txt":
            self._load_filelist(path)
        else:
            logger.warning(f"Unknown path type: {path}")
    
    def _load_directory(self, directory: Path):
        """Recursively load audio files from directory."""
        for ext in self.SUPPORTED_EXTENSIONS:
            for audio_path in directory.rglob(f"*{ext}"):
                self.audio_files.append(str(audio_path))
    
    def _load_manifest(self, manifest_path: Path):
        """Load audio files from JSON/JSONL manifest."""
        with open(manifest_path, "r") as f:
            if manifest_path.suffix == ".jsonl":
                for line in f:
                    entry = json.loads(line.strip())
                    self._add_manifest_entry(entry, manifest_path.parent)
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        self._add_manifest_entry(entry, manifest_path.parent)
                elif isinstance(data, dict) and "data" in data:
                    for entry in data["data"]:
                        self._add_manifest_entry(entry, manifest_path.parent)
    
    def _add_manifest_entry(self, entry: Dict, base_path: Path):
        """Add entry from manifest."""
        # Support various manifest formats
        audio_path = entry.get("audio_path") or entry.get("path") or entry.get("audio")
        
        if audio_path:
            # Handle relative paths
            if not os.path.isabs(audio_path):
                audio_path = str(base_path / audio_path)
            
            if os.path.exists(audio_path):
                self.audio_files.append(audio_path)
                
                # Store metadata if available
                if "duration" in entry:
                    self.metadata[audio_path] = {"duration": entry["duration"]}
    
    def _load_filelist(self, filelist_path: Path):
        """Load audio files from text file (one path per line)."""
        with open(filelist_path, "r") as f:
            for line in f:
                audio_path = line.strip()
                if audio_path and os.path.exists(audio_path):
                    self.audio_files.append(audio_path)
    
    def _filter_by_duration(self):
        """Filter files by duration constraints."""
        if not self.metadata:
            return
        
        filtered = []
        for path in self.audio_files:
            if path in self.metadata:
                duration = self.metadata[path].get("duration", float("inf"))
                if self.min_duration <= duration <= self.max_duration:
                    filtered.append(path)
            else:
                filtered.append(path)  # Keep files without duration info
        
        self.audio_files = filtered
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process audio file.
        
        Returns:
            Dictionary with:
                - audio: Processed audio tensor [1, T]
                - audio_path: Path to source file
        """
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Extract segment
            waveform = self._extract_segment(waveform)
            
            # Apply augmentation
            if self.is_training and self.augmentor is not None:
                waveform = self.augmentor(waveform)
            
            # Normalize
            waveform = self._normalize(waveform)
            
            return {
                "audio": waveform,
                "audio_path": audio_path,
            }
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return random other sample on error
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def _extract_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment from waveform."""
        length = waveform.shape[-1]
        
        if length < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif length > self.segment_length and self.is_training:
            # Random crop during training
            start = random.randint(0, length - self.segment_length)
            waveform = waveform[..., start:start + self.segment_length]
        elif length > self.segment_length:
            # Center crop during validation
            start = (length - self.segment_length) // 2
            waveform = waveform[..., start:start + self.segment_length]
        
        return waveform
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to [-1, 1]."""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        return waveform


class AudioDataLoader:
    """
    Wrapper around DataLoader with distributed training support.
    """
    
    def __init__(
        self,
        dataset: AudioDataset,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        distributed: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distributed = distributed
        
        # Create sampler
        if distributed:
            self.sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling
        else:
            self.sampler = None
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if not distributed else False,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        audios = torch.stack([item["audio"] for item in batch])
        
        return {
            "audio": audios,
            "audio_paths": [item["audio_path"] for item in batch],
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_dataloader(
    config: DataConfig,
    is_training: bool = True,
    distributed: bool = False,
) -> AudioDataLoader:
    """
    Create dataloader from config.
    
    Args:
        config: Data configuration
        is_training: Whether this is training data
        distributed: Whether using distributed training
        
    Returns:
        AudioDataLoader instance
    """
    # Create augmentor for training
    augmentor = None
    if is_training and config.augmentation is not None:
        augmentor = AudioAugmentor(
            config=config.augmentation,
            sample_rate=config.sample_rate,
        )
    
    # Select paths
    paths = config.train_paths if is_training else config.val_paths
    
    # Create dataset
    dataset = AudioDataset(
        data_paths=paths,
        segment_length=config.segment_length,
        sample_rate=config.sample_rate,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        augmentor=augmentor,
        is_training=is_training,
    )
    
    # Create dataloader
    return AudioDataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        distributed=distributed,
        shuffle=is_training,
        drop_last=is_training,
    )


@dataclass
class ChatMLDataConfig:
    """Configuration for ChatML data loading."""
    train_json_paths: List[str]
    val_json_paths: List[str]
    segment_length: int = 32000  # 2 seconds at 16kHz
    sample_rate: int = 16000
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    min_duration: float = 0.5
    max_duration: float = 30.0
    use_reference_audio: bool = True
    use_target_audio: bool = True
    audio_base_path: Optional[str] = None
    augmentation: Optional[AugmentationConfig] = None


def create_chatml_dataloader(
    config: ChatMLDataConfig,
    is_training: bool = True,
    distributed: bool = False,
) -> AudioDataLoader:
    """
    Create dataloader for ChatML format dataset.
    
    Args:
        config: ChatML data configuration
        is_training: Whether this is training data
        distributed: Whether using distributed training
        
    Returns:
        AudioDataLoader instance
    """
    # Create augmentor for training
    augmentor = None
    if is_training and config.augmentation is not None:
        augmentor = AudioAugmentor(
            config=config.augmentation,
            sample_rate=config.sample_rate,
        )
    
    # Select paths
    json_paths = config.train_json_paths if is_training else config.val_json_paths
    
    # Create dataset
    dataset = ChatMLDataset(
        json_paths=json_paths,
        segment_length=config.segment_length,
        sample_rate=config.sample_rate,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        augmentor=augmentor,
        is_training=is_training,
        use_reference_audio=config.use_reference_audio,
        use_target_audio=config.use_target_audio,
        audio_base_path=config.audio_base_path,
    )
    
    # Create dataloader
    return AudioDataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        distributed=distributed,
        shuffle=is_training,
        drop_last=is_training,
    )
