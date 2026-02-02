"""
NeuCodec Training Infrastructure

This module provides everything needed to train NeuCodec neural audio codecs:
- Discriminators (MPD, MSD, MS-STFT)
- Loss functions (mel, adversarial, feature matching, semantic)
- Data loading with augmentation (standard and ChatML formats)
- Training and evaluation scripts

Usage:
    # Training with standard format
    python training/train.py --config training/configs/neucodec_train.yaml
    
    # Training with ChatML format
    python training/train.py --config training/configs/finetune_chatml.yaml
    
    # Multi-GPU training
    torchrun --nproc_per_node=4 training/train.py --config training/configs/finetune_chatml.yaml
    
    # Evaluation
    python training/evaluate.py --model neuphonic/neucodec --data /path/to/test/data
"""

from .discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from .losses import (
    MultiResolutionMelLoss,
    MelSpectrogramLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    FeatureMatchingLoss,
    SemanticReconstructionLoss,
)
from .data import (
    AudioDataset,
    AudioDataLoader,
    ChatMLDataset,
    DataConfig,
    ChatMLDataConfig,
    create_dataloader,
    create_chatml_dataloader,
    AudioAugmentor,
    AugmentationConfig,
)

__all__ = [
    # Discriminators
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "MultiScaleSTFTDiscriminator",
    # Losses
    "MultiResolutionMelLoss",
    "MelSpectrogramLoss",
    "DiscriminatorLoss",
    "GeneratorAdversarialLoss",
    "FeatureMatchingLoss",
    "SemanticReconstructionLoss",
    # Data
    "AudioDataset",
    "AudioDataLoader",
    "ChatMLDataset",
    "DataConfig",
    "ChatMLDataConfig",
    "create_dataloader",
    "create_chatml_dataloader",
    "AudioAugmentor",
    "AugmentationConfig",
]
