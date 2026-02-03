"""
NeuCodec Training Script

Usage:
    Single GPU:
        python train.py --config configs/neucodec_train.yaml
    
    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 train.py --config configs/neucodec_train.yaml
    
    Resume training:
        python train.py --config configs/neucodec_train.yaml --resume checkpoints/step_100000.pt
"""

import os
import sys
import argparse
import logging
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neucodec import NeuCodec
from training.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from training.losses import (
    MultiResolutionMelLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    FeatureMatchingLoss,
    SemanticReconstructionLoss,
)
from training.data import create_dataloader, create_chatml_dataloader, create_chatml_dataloaders, AudioAugmentor, AugmentationConfig
from training.data.dataloader import DataConfig, ChatMLDataConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NeuCodecTrainer:
    """
    Trainer class for NeuCodec neural audio codec.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.distributed = world_size > 1
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Setup logging
        if self.is_main_process:
            self._setup_logging()
        
        # Build models
        self._build_models()
        
        # Build optimizers
        self._build_optimizers()
        
        # Build dataloaders
        self._build_dataloaders()
        
        # Build losses
        self._build_losses()
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=config["training"]["mixed_precision"] != "fp32")
        
        # Resampler for 24kHz decoder output to 16kHz for loss computation
        self.output_resampler = torchaudio.transforms.Resample(24000, 16000)
        
    @property
    def is_main_process(self) -> bool:
        return self.local_rank == 0
    
    def _setup_logging(self):
        """Setup logging and output directories."""
        output_config = self.config["output"]
        
        # Create directories
        self.checkpoint_dir = Path(output_config["checkpoint_dir"])
        self.log_dir = Path(output_config["log_dir"])
        self.sample_dir = Path(output_config["sample_dir"])
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        if self.config["logging"]["wandb"]["enabled"]:
            try:
                import wandb
                run_name = self.config["logging"]["run_name"] or f"neucodec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project=self.config["logging"]["project_name"],
                    name=run_name,
                    config=self.config,
                    entity=self.config["logging"]["wandb"]["entity"],
                )
                self.use_wandb = True
            except ImportError:
                logger.warning("wandb not installed, skipping")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        # Setup tensorboard
        if self.config["logging"]["tensorboard"]["enabled"]:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config["logging"]["tensorboard"]["log_dir"])
        else:
            self.writer = None
    
    def _build_models(self):
        """Build generator and discriminator models."""
        model_config = self.config["model"]
        disc_config = self.config["discriminators"]
        
        # Generator (NeuCodec)
        logger.info("Building NeuCodec generator...")
        self.generator = NeuCodec.from_pretrained("neuphonic/neucodec")
        
        # Freeze semantic encoder
        if model_config["semantic"]["freeze"]:
            for param in self.generator.semantic_model.parameters():
                param.requires_grad = False
            # Note: feature_extractor is a HuggingFace preprocessing class, not a PyTorch module
            logger.info("Froze semantic encoder parameters")
        
        self.generator = self.generator.to(self.device)
        
        # Discriminators
        logger.info("Building discriminators...")
        self.discriminators = nn.ModuleDict()
        
        if disc_config["mpd"]["enabled"]:
            self.discriminators["mpd"] = MultiPeriodDiscriminator(
                periods=disc_config["mpd"]["periods"],
                channels=disc_config["mpd"]["channels"],
                max_channels=disc_config["mpd"]["max_channels"],
            )
        
        if disc_config["msd"]["enabled"]:
            self.discriminators["msd"] = MultiScaleDiscriminator(
                scales=disc_config["msd"]["scales"],
                channels=disc_config["msd"]["channels"],
                max_channels=disc_config["msd"]["max_channels"],
            )
        
        if disc_config["ms_stft"]["enabled"]:
            self.discriminators["ms_stft"] = MultiScaleSTFTDiscriminator(
                filters=disc_config["ms_stft"]["filters"],
                n_ffts=disc_config["ms_stft"]["n_ffts"],
                hop_lengths=disc_config["ms_stft"]["hop_lengths"],
                win_lengths=disc_config["ms_stft"]["win_lengths"],
            )
        
        self.discriminators = self.discriminators.to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            self.generator = DDP(
                self.generator,
                device_ids=[self.local_rank],
                find_unused_parameters=self.config["distributed"]["find_unused_parameters"],
            )
            self.discriminators = DDP(
                self.discriminators,
                device_ids=[self.local_rank],
                find_unused_parameters=self.config["distributed"]["find_unused_parameters"],
            )
        
        # Log model info
        gen_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in self.discriminators.parameters() if p.requires_grad)
        logger.info(f"Generator trainable params: {gen_params:,}")
        logger.info(f"Discriminator trainable params: {disc_params:,}")
    
    def _build_optimizers(self):
        """Build optimizers and schedulers."""
        opt_config = self.config["optimizer"]
        sched_config = self.config["scheduler"]
        
        # Generator optimizer
        gen_module = self.generator.module if self.distributed else self.generator
        gen_params = [p for p in gen_module.parameters() if p.requires_grad]
        
        self.optimizer_g = torch.optim.AdamW(
            gen_params,
            lr=opt_config["generator"]["lr"],
            betas=tuple(opt_config["generator"]["betas"]),
            weight_decay=opt_config["generator"]["weight_decay"],
        )
        
        # Discriminator optimizer
        disc_module = self.discriminators.module if self.distributed else self.discriminators
        disc_params = [p for p in disc_module.parameters() if p.requires_grad]
        
        self.optimizer_d = torch.optim.AdamW(
            disc_params,
            lr=opt_config["discriminator"]["lr"],
            betas=tuple(opt_config["discriminator"]["betas"]),
            weight_decay=opt_config["discriminator"]["weight_decay"],
        )
        
        # Schedulers
        if sched_config["type"] == "ExponentialLR":
            self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_g, gamma=sched_config["gamma"]
            )
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_d, gamma=sched_config["gamma"]
            )
        elif sched_config["type"] == "CosineAnnealingWarmRestarts":
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_g, T_0=sched_config["T_0"], T_mult=sched_config.get("T_mult", 1)
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_d, T_0=sched_config["T_0"], T_mult=sched_config.get("T_mult", 1)
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None
    
    def _build_dataloaders(self):
        """Build training and validation dataloaders."""
        data_config = self.config["data"]
        aug_config = data_config.get("augmentation", {})
        
        # Create augmentation config
        augmentation_config = AugmentationConfig(
            enabled=aug_config.get("enabled", True),
            noise_prob=aug_config.get("noise_prob", 0.3),
            noise_snr_range=tuple(aug_config.get("noise_snr_range", [5, 30])),
            reverb_prob=aug_config.get("reverb_prob", 0.2),
            pitch_shift_prob=aug_config.get("pitch_shift_prob", 0.1),
            pitch_shift_range=tuple(aug_config.get("pitch_shift_range", [-2, 2])),
            time_stretch_prob=aug_config.get("time_stretch_prob", 0.1),
            time_stretch_range=tuple(aug_config.get("time_stretch_range", [0.9, 1.1])),
            volume_range=tuple(aug_config.get("volume_range", [-6, 3])),
        )
        
        # Check data format
        data_format = data_config.get("format", "standard")
        
        if data_format == "chatml":
            # ChatML format dataset
            val_json_paths = data_config.get("val_json_paths")
            val_split_ratio = data_config.get("val_split_ratio", 0.05)
            
            config = ChatMLDataConfig(
                train_json_paths=data_config["train_json_paths"],
                val_json_paths=val_json_paths,
                val_split_ratio=val_split_ratio,
                segment_length=data_config["segment_length"],
                sample_rate=data_config["sample_rate"],
                batch_size=data_config["batch_size"],
                num_workers=data_config["num_workers"],
                pin_memory=data_config["pin_memory"],
                min_duration=data_config.get("min_duration", 0.5),
                max_duration=data_config.get("max_duration", 30.0),
                use_reference_audio=data_config.get("use_reference_audio", True),
                use_target_audio=data_config.get("use_target_audio", True),
                audio_base_path=data_config.get("audio_base_path"),
                augmentation=augmentation_config,
            )
            
            # Create dataloaders (handles auto-split if val_json_paths is None)
            self.train_loader, self.val_loader = create_chatml_dataloaders(
                config, distributed=self.distributed
            )
        else:
            # Standard format dataset
            config = DataConfig(
                train_paths=data_config["train_paths"],
                val_paths=data_config["val_paths"],
                segment_length=data_config["segment_length"],
                sample_rate=data_config["sample_rate"],
                batch_size=data_config["batch_size"],
                num_workers=data_config["num_workers"],
                pin_memory=data_config["pin_memory"],
                min_duration=data_config.get("min_duration", 1.0),
                max_duration=data_config.get("max_duration", 30.0),
                augmentation=augmentation_config,
            )
            
            # Create dataloaders
            self.train_loader = create_dataloader(
                config, is_training=True, distributed=self.distributed
            )
            self.val_loader = create_dataloader(
                config, is_training=False, distributed=self.distributed
            )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _build_losses(self):
        """Build loss functions."""
        loss_config = self.config["losses"]
        
        # Mel spectrogram loss
        mel_config = loss_config["mel"]
        self.mel_loss = MultiResolutionMelLoss(
            sample_rate=self.config["data"]["sample_rate"],
            n_ffts=mel_config["n_ffts"],
            hop_lengths=mel_config["hop_lengths"],
            win_lengths=mel_config["win_lengths"],
            n_mels=mel_config["n_mels"],
        ).to(self.device)
        
        # Adversarial losses
        adv_config = loss_config["adversarial"]
        self.disc_loss_fn = DiscriminatorLoss(loss_type=adv_config["loss_type"])
        self.gen_adv_loss_fn = GeneratorAdversarialLoss(loss_type=adv_config["loss_type"])
        self.fm_loss_fn = FeatureMatchingLoss()
        
        # Semantic loss
        sem_config = loss_config["semantic"]
        self.semantic_loss = SemanticReconstructionLoss(
            loss_type=sem_config["loss_type"]
        )
        
        # Loss weights
        self.loss_weights = {
            "mel": mel_config["weight"],
            "adversarial": adv_config["weight"],
            "feature_matching": loss_config["feature_matching"]["weight"],
            "semantic": sem_config["weight"],
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        audio = batch["audio"].to(self.device)  # [B, 1, T]
        
        train_config = self.config["training"]
        disc_start = train_config.get("discriminator_start_step", 0)
        use_disc = self.global_step >= disc_start
        
        losses = {}
        
        # ========== Generator Step ==========
        self.optimizer_g.zero_grad()
        
        with autocast('cuda', enabled=train_config["mixed_precision"] != "fp32"):
            # Forward through generator
            gen_module = self.generator.module if self.distributed else self.generator
            
            # Encode
            fsq_codes = gen_module.encode_code(audio)
            
            # Decode (output is 24kHz)
            audio_recon = gen_module.decode_code(fsq_codes)
            
            # Resample 24kHz output to 16kHz to match input for loss computation
            # This fixes the sample rate mismatch that caused "fast forward" audio
            audio_recon = self.output_resampler.to(audio_recon.device)(audio_recon)
            
            # Match lengths (now both are 16kHz)
            min_len = min(audio.shape[-1], audio_recon.shape[-1])
            audio = audio[..., :min_len]
            audio_recon = audio_recon[..., :min_len]
            
            # Mel loss
            mel_loss = self.mel_loss(audio_recon, audio)
            losses["mel"] = mel_loss.item()
            
            gen_loss = self.loss_weights["mel"] * mel_loss
            
            # Adversarial losses (after warmup)
            if use_disc:
                # Get discriminator outputs
                disc_module = self.discriminators.module if self.distributed else self.discriminators
                
                all_fake_outputs = []
                all_real_features = []
                all_fake_features = []
                
                for name, disc in disc_module.items():
                    fake_out, fake_feat = disc(audio_recon)
                    with torch.no_grad():
                        real_out, real_feat = disc(audio)
                    
                    all_fake_outputs.append(fake_out)
                    all_real_features.append(real_feat)
                    all_fake_features.append(fake_feat)
                
                # Generator adversarial loss
                adv_loss = self.gen_adv_loss_fn(all_fake_outputs)
                losses["adversarial"] = adv_loss.item()
                gen_loss = gen_loss + self.loss_weights["adversarial"] * adv_loss
                
                # Feature matching loss
                fm_loss = self.fm_loss_fn(all_real_features, all_fake_features)
                losses["feature_matching"] = fm_loss.item()
                gen_loss = gen_loss + self.loss_weights["feature_matching"] * fm_loss
        
        losses["generator_total"] = gen_loss.item()
        
        # Backward and step
        self.scaler.scale(gen_loss).backward()
        
        if train_config.get("gradient_clip_norm"):
            self.scaler.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                train_config["gradient_clip_norm"]
            )
        
        self.scaler.step(self.optimizer_g)
        
        # ========== Discriminator Step ==========
        if use_disc:
            self.optimizer_d.zero_grad()
            
            with autocast('cuda', enabled=train_config["mixed_precision"] != "fp32"):
                disc_module = self.discriminators.module if self.distributed else self.discriminators
                
                all_real_outputs = []
                all_fake_outputs = []
                
                for name, disc in disc_module.items():
                    real_out, _ = disc(audio)
                    fake_out, _ = disc(audio_recon.detach())
                    
                    all_real_outputs.append(real_out)
                    all_fake_outputs.append(fake_out)
                
                disc_loss = self.disc_loss_fn(all_real_outputs, all_fake_outputs)
                losses["discriminator"] = disc_loss.item()
            
            self.scaler.scale(disc_loss).backward()
            
            if train_config.get("gradient_clip_norm"):
                self.scaler.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(
                    self.discriminators.parameters(),
                    train_config["gradient_clip_norm"]
                )
            
            self.scaler.step(self.optimizer_d)
        
        self.scaler.update()
        
        return losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.generator.eval()
        
        total_mel_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", disable=not self.is_main_process):
            audio = batch["audio"].to(self.device)
            
            gen_module = self.generator.module if self.distributed else self.generator
            fsq_codes = gen_module.encode_code(audio)
            audio_recon = gen_module.decode_code(fsq_codes)
            
            # Resample 24kHz output to 16kHz to match input for loss computation
            audio_recon = self.output_resampler.to(audio_recon.device)(audio_recon)
            
            min_len = min(audio.shape[-1], audio_recon.shape[-1])
            audio = audio[..., :min_len]
            audio_recon = audio_recon[..., :min_len]
            
            mel_loss = self.mel_loss(audio_recon, audio)
            total_mel_loss += mel_loss.item()
            num_batches += 1
        
        self.generator.train()
        
        return {"val_mel_loss": total_mel_loss / max(num_batches, 1)}
    
    def save_checkpoint(self, tag: str = "latest"):
        """Save training checkpoint."""
        if not self.is_main_process:
            return
        
        gen_module = self.generator.module if self.distributed else self.generator
        disc_module = self.discriminators.module if self.distributed else self.discriminators
        
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "generator": gen_module.state_dict(),
            "discriminators": disc_module.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scheduler_g is not None:
            checkpoint["scheduler_g"] = self.scheduler_g.state_dict()
            checkpoint["scheduler_d"] = self.scheduler_d.state_dict()
        
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Also save step-specific checkpoint
        step_path = self.checkpoint_dir / f"step_{self.global_step}.pt"
        torch.save(checkpoint, step_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        keep_n = self.config["training"]["keep_last_n_checkpoints"]
        
        step_checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
            reverse=True
        )
        
        for ckpt in step_checkpoints[keep_n:]:
            ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        gen_module = self.generator.module if self.distributed else self.generator
        disc_module = self.discriminators.module if self.distributed else self.discriminators
        
        gen_module.load_state_dict(checkpoint["generator"])
        disc_module.load_state_dict(checkpoint["discriminators"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        if self.scheduler_g is not None and "scheduler_g" in checkpoint:
            self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        
        logger.info(f"Resumed from step {self.global_step}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to wandb/tensorboard."""
        if not self.is_main_process:
            return
        
        # Prefix metrics
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log(prefixed, step=self.global_step)
        
        # Log to tensorboard
        if self.writer is not None:
            for k, v in prefixed.items():
                self.writer.add_scalar(k, v, self.global_step)
    
    def train(self):
        """Main training loop."""
        train_config = self.config["training"]
        total_steps = train_config["total_steps"]
        
        logger.info("Starting training...")
        
        while self.global_step < total_steps:
            # Set epoch for distributed sampler
            self.train_loader.set_epoch(self.epoch)
            
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch}",
                disable=not self.is_main_process
            )
            
            for batch in pbar:
                if self.global_step >= total_steps:
                    break
                
                # Training step
                losses = self.train_step(batch)
                self.global_step += 1
                
                # Update schedulers
                if self.scheduler_g is not None:
                    self.scheduler_g.step()
                    self.scheduler_d.step()
                
                # Always update progress bar with key losses
                pbar.set_postfix({
                    "mel": f"{losses.get('mel', 0):.3f}",
                    "adv": f"{losses.get('adversarial', 0):.3f}",
                    "fm": f"{losses.get('feature_matching', 0):.3f}",
                    "step": self.global_step,
                })
                
                # Detailed logging at intervals
                if self.global_step % train_config["log_every_steps"] == 0:
                    self.log_metrics(losses)
                
                # Validation
                if self.global_step % train_config["eval_every_steps"] == 0:
                    val_metrics = self.validate()
                    self.log_metrics(val_metrics, prefix="val")
                    
                    if val_metrics["val_mel_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_mel_loss"]
                        self.save_checkpoint(tag="best")
                
                # Checkpointing
                if self.global_step % train_config["save_every_steps"] == 0:
                    self.save_checkpoint(tag="latest")
            
            self.epoch += 1
        
        # Final checkpoint
        self.save_checkpoint(tag="final")
        logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train NeuCodec")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override resume path if provided
    if args.resume:
        config["training"]["resume_from"] = args.resume
    
    # Setup distributed training
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend=config["distributed"]["backend"],
            init_method="env://",
        )
    else:
        local_rank = 0
        world_size = 1
    
    # Set seed
    seed = config["training"]["seed"]
    random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)
    
    # Create trainer
    trainer = NeuCodecTrainer(
        config=config,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    # Resume if checkpoint provided
    if config["training"]["resume_from"]:
        trainer.load_checkpoint(config["training"]["resume_from"])
    
    # Train
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
