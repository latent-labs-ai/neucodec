#!/usr/bin/env python3
"""
NeuCodec ChatML Dataset Validator and Training Estimator

Validates ChatML dataset and estimates training time for NeuCodec finetuning.

Usage:
    python validate_dataset.py --json /path/to/dataset.json
    python validate_dataset.py --json /path/to/train.json /path/to/val.json --gpus 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AudioEntry:
    """Single audio entry from ChatML."""
    path: str
    duration: Optional[float]
    speaker: str
    role: str  # user or assistant
    exists: bool = False
    

@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_entries: int = 0
    valid_entries: int = 0
    missing_files: int = 0
    total_duration_seconds: float = 0.0
    duration_unknown: int = 0
    
    speakers: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duration_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    reference_audio_count: int = 0
    target_audio_count: int = 0
    
    min_duration: float = float('inf')
    max_duration: float = 0.0
    
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_hours(self) -> float:
        return self.total_duration_seconds / 3600
    
    @property
    def avg_duration(self) -> float:
        if self.valid_entries == 0:
            return 0.0
        return self.total_duration_seconds / self.valid_entries


def parse_chatml_entry(entry: Dict, audio_base_path: Optional[str] = None) -> List[AudioEntry]:
    """Parse a single ChatML entry and extract audio info."""
    audio_entries = []
    
    messages = entry.get("messages", [])
    speaker = entry.get("speaker", "unknown")
    misc = entry.get("misc", {})
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", [])
        
        if role == "system":
            continue
        
        if isinstance(content, str):
            continue
        
        for item in content:
            if not isinstance(item, dict):
                continue
                
            if item.get("type") == "audio":
                audio_url = item.get("audio_url", "")
                
                if not audio_url:
                    continue
                
                # Get duration
                duration = item.get("duration") or misc.get("duration")
                
                # Resolve path
                if audio_base_path and not os.path.isabs(audio_url):
                    audio_path = os.path.join(audio_base_path, audio_url)
                else:
                    audio_path = audio_url
                
                audio_entries.append(AudioEntry(
                    path=audio_path,
                    duration=duration,
                    speaker=speaker,
                    role=role,
                    exists=os.path.exists(audio_path)
                ))
    
    return audio_entries


def validate_dataset(
    json_paths: List[str],
    audio_base_path: Optional[str] = None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> DatasetStats:
    """
    Validate ChatML dataset and collect statistics.
    
    Args:
        json_paths: List of paths to ChatML JSON files
        audio_base_path: Base path for relative audio URLs
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        
    Returns:
        DatasetStats with validation results
    """
    stats = DatasetStats()
    
    for json_path in json_paths:
        logger.info(f"Processing: {json_path}")
        
        if not os.path.exists(json_path):
            stats.errors.append(f"JSON file not found: {json_path}")
            continue
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            stats.errors.append(f"Invalid JSON in {json_path}: {e}")
            continue
        
        if isinstance(data, dict):
            data = [data]
        
        for entry in data:
            audio_entries = parse_chatml_entry(entry, audio_base_path)
            
            for audio in audio_entries:
                stats.total_entries += 1
                
                # Track role
                if audio.role == "user":
                    stats.reference_audio_count += 1
                elif audio.role == "assistant":
                    stats.target_audio_count += 1
                
                # Check existence
                if not audio.exists:
                    stats.missing_files += 1
                    continue
                
                # Track duration
                if audio.duration is not None:
                    # Apply duration filter
                    if min_duration <= audio.duration <= max_duration:
                        stats.valid_entries += 1
                        stats.total_duration_seconds += audio.duration
                        stats.min_duration = min(stats.min_duration, audio.duration)
                        stats.max_duration = max(stats.max_duration, audio.duration)
                        
                        # Duration buckets
                        if audio.duration < 2:
                            stats.duration_distribution["0-2s"] += 1
                        elif audio.duration < 5:
                            stats.duration_distribution["2-5s"] += 1
                        elif audio.duration < 10:
                            stats.duration_distribution["5-10s"] += 1
                        elif audio.duration < 20:
                            stats.duration_distribution["10-20s"] += 1
                        else:
                            stats.duration_distribution["20-30s"] += 1
                        
                        # Track speaker
                        stats.speakers[audio.speaker] += 1
                else:
                    stats.duration_unknown += 1
                    stats.valid_entries += 1  # Count as valid but unknown duration
    
    return stats


def estimate_training_time(
    stats: DatasetStats,
    batch_size: int = 16,
    num_gpus: int = 4,
    segment_length_seconds: float = 3.0,
    target_epochs: int = 10,
    steps_per_second: float = 2.5,  # Estimated training throughput
) -> Dict:
    """
    Estimate training time based on dataset statistics.
    
    Args:
        stats: Dataset statistics
        batch_size: Batch size per GPU
        num_gpus: Number of GPUs
        segment_length_seconds: Audio segment length in seconds
        target_epochs: Target number of training epochs
        steps_per_second: Estimated training steps per second
        
    Returns:
        Dictionary with training estimates
    """
    # Estimate number of segments
    if stats.avg_duration > 0:
        segments_per_sample = max(1, stats.avg_duration / segment_length_seconds)
    else:
        segments_per_sample = 1
    
    total_segments = int(stats.valid_entries * segments_per_sample)
    
    # Steps per epoch
    effective_batch_size = batch_size * num_gpus
    steps_per_epoch = total_segments // effective_batch_size
    
    # Total steps
    total_steps = steps_per_epoch * target_epochs
    
    # Estimated time
    total_seconds = total_steps / steps_per_second
    total_hours = total_seconds / 3600
    total_days = total_hours / 24
    
    return {
        "total_segments": total_segments,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_hours": total_hours,
        "estimated_days": total_days,
        "target_epochs": target_epochs,
    }


def print_report(stats: DatasetStats, estimates: Dict, json_paths: List[str]):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("                    NEUCODEC DATASET VALIDATION REPORT")
    print("=" * 70)
    
    print("\n--- INPUT FILES ---")
    for path in json_paths:
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{exists}] {path}")
    
    print("\n--- DATASET STATISTICS ---")
    print(f"  Total audio entries:     {stats.total_entries:,}")
    print(f"  Valid entries:           {stats.valid_entries:,}")
    print(f"  Missing audio files:     {stats.missing_files:,}")
    print(f"  Unknown duration:        {stats.duration_unknown:,}")
    print(f"  Reference audio (user):  {stats.reference_audio_count:,}")
    print(f"  Target audio (assistant):{stats.target_audio_count:,}")
    
    print("\n--- DURATION STATISTICS ---")
    print(f"  Total duration:          {stats.total_hours:.2f} hours ({stats.total_duration_seconds:,.0f} seconds)")
    print(f"  Average duration:        {stats.avg_duration:.2f} seconds")
    if stats.min_duration != float('inf'):
        print(f"  Min duration:            {stats.min_duration:.2f} seconds")
        print(f"  Max duration:            {stats.max_duration:.2f} seconds")
    
    print("\n--- DURATION DISTRIBUTION ---")
    for bucket, count in sorted(stats.duration_distribution.items()):
        pct = (count / stats.valid_entries * 100) if stats.valid_entries > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {bucket:>8}: {count:>8,} ({pct:>5.1f}%) {bar}")
    
    print("\n--- SPEAKER DISTRIBUTION ---")
    top_speakers = sorted(stats.speakers.items(), key=lambda x: -x[1])[:10]
    for speaker, count in top_speakers:
        pct = (count / stats.valid_entries * 100) if stats.valid_entries > 0 else 0
        print(f"  {speaker[:30]:30} {count:>8,} ({pct:>5.1f}%)")
    if len(stats.speakers) > 10:
        print(f"  ... and {len(stats.speakers) - 10} more speakers")
    
    print("\n--- TRAINING ESTIMATES ---")
    print(f"  Total segments:          {estimates['total_segments']:,}")
    print(f"  Effective batch size:    {estimates['effective_batch_size']}")
    print(f"  Steps per epoch:         {estimates['steps_per_epoch']:,}")
    print(f"  Target epochs:           {estimates['target_epochs']}")
    print(f"  Total training steps:    {estimates['total_steps']:,}")
    print(f"  Estimated training time: {estimates['estimated_hours']:.1f} hours ({estimates['estimated_days']:.1f} days)")
    
    if stats.errors:
        print("\n--- ERRORS ---")
        for error in stats.errors[:10]:
            print(f"  ERROR: {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")
    
    print("\n--- RECOMMENDATIONS ---")
    if stats.missing_files > 0:
        pct_missing = stats.missing_files / stats.total_entries * 100
        print(f"  WARNING: {pct_missing:.1f}% of audio files are missing")
        if pct_missing > 10:
            print(f"  ACTION: Check audio_base_path or file paths in your JSON")
    
    if stats.total_hours < 100:
        print(f"  INFO: Dataset has {stats.total_hours:.1f} hours - consider adding more data for best results")
    elif stats.total_hours > 1000:
        print(f"  INFO: Large dataset ({stats.total_hours:.1f} hours) - training may take several days")
    
    if estimates['estimated_days'] > 7:
        print(f"  TIP: Consider reducing epochs or increasing batch size to speed up training")
    
    print("\n--- SUGGESTED CONFIG ---")
    print(f"""
  data:
    format: "chatml"
    train_json_paths:
      - "{json_paths[0] if json_paths else '/path/to/train.json'}"
    segment_length: {int(min(stats.avg_duration * 16000, 48000))}  # {min(stats.avg_duration, 3.0):.1f}s at 16kHz
    batch_size: {max(8, min(32, 64 // len(stats.speakers) if stats.speakers else 16))}
    min_duration: {max(0.5, stats.min_duration):.1f}
    max_duration: {min(30.0, stats.max_duration):.1f}
  
  training:
    total_steps: {estimates['total_steps']}
""")
    
    print("=" * 70)
    
    # Return status code
    if stats.missing_files > stats.total_entries * 0.5:
        return 1  # More than 50% missing - error
    elif stats.valid_entries == 0:
        return 1  # No valid entries - error
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate ChatML dataset and estimate training time"
    )
    parser.add_argument(
        "--json", "-j",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to ChatML JSON file(s)"
    )
    parser.add_argument(
        "--audio-base-path", "-b",
        type=str,
        default=None,
        help="Base path for relative audio URLs"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per GPU (default: 16)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Number of GPUs (default: 4)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Target number of epochs (default: 10)"
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="Audio segment length in seconds (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    logger.info("Validating dataset...")
    stats = validate_dataset(
        json_paths=args.json,
        audio_base_path=args.audio_base_path,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    
    # Estimate training time
    estimates = estimate_training_time(
        stats=stats,
        batch_size=args.batch_size,
        num_gpus=args.gpus,
        segment_length_seconds=args.segment_length,
        target_epochs=args.epochs,
    )
    
    # Print report
    exit_code = print_report(stats, estimates, args.json)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
