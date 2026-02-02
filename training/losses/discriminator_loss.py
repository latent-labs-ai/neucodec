"""
Discriminator and Generator Adversarial Losses for NeuCodec training.
Supports hinge, LSGAN, and vanilla GAN losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for GAN training.
    Supports multiple loss types: hinge, lsgan, vanilla.
    """
    
    def __init__(
        self,
        loss_type: Literal["hinge", "lsgan", "vanilla"] = "hinge",
    ):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_outputs: List of discriminator outputs for real samples
            fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            loss: Total discriminator loss
        """
        total_loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if self.loss_type == "hinge":
                real_loss = F.relu(1.0 - real_out).mean()
                fake_loss = F.relu(1.0 + fake_out).mean()
            elif self.loss_type == "lsgan":
                real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
                fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
            elif self.loss_type == "vanilla":
                real_loss = F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out)
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss = total_loss + real_loss + fake_loss
        
        return total_loss


class GeneratorAdversarialLoss(nn.Module):
    """
    Generator adversarial loss for GAN training.
    Supports multiple loss types: hinge, lsgan, vanilla.
    """
    
    def __init__(
        self,
        loss_type: Literal["hinge", "lsgan", "vanilla"] = "hinge",
    ):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        fake_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute generator adversarial loss.
        
        Args:
            fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            loss: Total generator adversarial loss
        """
        total_loss = 0.0
        
        for fake_out in fake_outputs:
            if self.loss_type == "hinge":
                loss = -fake_out.mean()
            elif self.loss_type == "lsgan":
                loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
            elif self.loss_type == "vanilla":
                loss = F.binary_cross_entropy_with_logits(
                    fake_out, torch.ones_like(fake_out)
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss = total_loss + loss
        
        return total_loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss between real and fake discriminator features.
    Encourages the generator to produce features similar to real samples.
    """
    
    def __init__(self, loss_type: Literal["l1", "l2"] = "l1"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: List of feature lists from discriminators for real samples
            fake_features: List of feature lists from discriminators for fake samples
            
        Returns:
            loss: Total feature matching loss
        """
        total_loss = 0.0
        num_features = 0
        
        for real_feat_list, fake_feat_list in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
                if self.loss_type == "l1":
                    loss = F.l1_loss(fake_feat, real_feat.detach())
                else:
                    loss = F.mse_loss(fake_feat, real_feat.detach())
                
                total_loss = total_loss + loss
                num_features += 1
        
        return total_loss / max(num_features, 1)


class CombinedDiscriminatorLoss(nn.Module):
    """
    Combined loss for multiple discriminators (MPD + MSD + MS-STFT).
    """
    
    def __init__(
        self,
        loss_type: Literal["hinge", "lsgan", "vanilla"] = "hinge",
        feature_matching_weight: float = 2.0,
    ):
        super().__init__()
        
        self.disc_loss = DiscriminatorLoss(loss_type)
        self.gen_loss = GeneratorAdversarialLoss(loss_type)
        self.fm_loss = FeatureMatchingLoss()
        self.fm_weight = feature_matching_weight
    
    def discriminator_loss(
        self,
        real_outputs_list: List[List[torch.Tensor]],
        fake_outputs_list: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute total discriminator loss across all discriminators.
        
        Args:
            real_outputs_list: List of outputs from all discriminators for real
            fake_outputs_list: List of outputs from all discriminators for fake
            
        Returns:
            loss: Total discriminator loss
        """
        total_loss = 0.0
        
        for real_outputs, fake_outputs in zip(real_outputs_list, fake_outputs_list):
            total_loss = total_loss + self.disc_loss(real_outputs, fake_outputs)
        
        return total_loss
    
    def generator_loss(
        self,
        fake_outputs_list: List[List[torch.Tensor]],
        real_features_list: List[List[List[torch.Tensor]]],
        fake_features_list: List[List[List[torch.Tensor]]],
    ) -> tuple:
        """
        Compute total generator loss across all discriminators.
        
        Args:
            fake_outputs_list: List of outputs from all discriminators for fake
            real_features_list: List of features from all discriminators for real
            fake_features_list: List of features from all discriminators for fake
            
        Returns:
            (adv_loss, fm_loss): Adversarial and feature matching losses
        """
        adv_loss = 0.0
        fm_loss = 0.0
        
        for fake_outputs in fake_outputs_list:
            adv_loss = adv_loss + self.gen_loss(fake_outputs)
        
        for real_features, fake_features in zip(real_features_list, fake_features_list):
            fm_loss = fm_loss + self.fm_loss(real_features, fake_features)
        
        return adv_loss, fm_loss * self.fm_weight
