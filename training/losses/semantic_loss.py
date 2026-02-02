"""
Semantic Reconstruction Loss for NeuCodec training.
Encourages the codec to preserve semantic information from Wav2Vec2-BERT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class SemanticReconstructionLoss(nn.Module):
    """
    Semantic reconstruction loss using features from a frozen semantic encoder.
    Computes similarity between semantic features of original and reconstructed audio.
    """
    
    def __init__(
        self,
        loss_type: Literal["mse", "cosine", "l1"] = "mse",
        normalize: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
    
    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute semantic reconstruction loss.
        
        Args:
            pred_features: Predicted semantic features [B, T, D] or [B, D, T]
            target_features: Target semantic features [B, T, D] or [B, D, T]
            mask: Optional mask for valid positions [B, T]
            
        Returns:
            loss: Semantic reconstruction loss
        """
        # Ensure same shape
        if pred_features.shape != target_features.shape:
            # Handle length mismatch by truncating to shorter
            min_len = min(pred_features.shape[-1], target_features.shape[-1])
            pred_features = pred_features[..., :min_len]
            target_features = target_features[..., :min_len]
        
        # Normalize features
        if self.normalize:
            pred_features = F.normalize(pred_features, dim=-1, p=2)
            target_features = F.normalize(target_features, dim=-1, p=2)
        
        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(pred_features, target_features, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_features, target_features, reduction="none")
        elif self.loss_type == "cosine":
            # Cosine similarity loss: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(pred_features, target_features, dim=-1)
            loss = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            if self.loss_type == "cosine":
                loss = loss * mask
                return loss.sum() / mask.sum().clamp(min=1)
            else:
                mask = mask.unsqueeze(-1).expand_as(loss)
                loss = loss * mask
                return loss.sum() / mask.sum().clamp(min=1)
        
        return loss.mean()


class MultiLayerSemanticLoss(nn.Module):
    """
    Multi-layer semantic loss using features from multiple layers of the semantic encoder.
    Useful for capturing hierarchical semantic information.
    """
    
    def __init__(
        self,
        layer_weights: Optional[list] = None,
        loss_type: Literal["mse", "cosine", "l1"] = "mse",
        normalize: bool = True,
    ):
        super().__init__()
        self.layer_weights = layer_weights
        self.loss_fn = SemanticReconstructionLoss(loss_type, normalize)
    
    def forward(
        self,
        pred_features_list: list,
        target_features_list: list,
    ) -> torch.Tensor:
        """
        Compute multi-layer semantic loss.
        
        Args:
            pred_features_list: List of predicted features from different layers
            target_features_list: List of target features from different layers
            
        Returns:
            loss: Weighted sum of per-layer losses
        """
        assert len(pred_features_list) == len(target_features_list)
        
        # Default to equal weights
        if self.layer_weights is None:
            weights = [1.0 / len(pred_features_list)] * len(pred_features_list)
        else:
            weights = self.layer_weights
        
        total_loss = 0.0
        for pred, target, weight in zip(
            pred_features_list, target_features_list, weights
        ):
            total_loss = total_loss + weight * self.loss_fn(pred, target)
        
        return total_loss


class ContrastiveSemanticLoss(nn.Module):
    """
    Contrastive semantic loss for learning discriminative representations.
    Pulls together positive pairs (same utterance) and pushes apart negative pairs.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive semantic loss.
        
        Args:
            anchor_features: Anchor features [B, T, D]
            positive_features: Positive features [B, T, D] (reconstructed from same audio)
            negative_features: Negative features [B, T, D] (from different audio)
            
        Returns:
            loss: InfoNCE-style contrastive loss
        """
        # Pool temporal dimension
        anchor = anchor_features.mean(dim=1)  # [B, D]
        positive = positive_features.mean(dim=1)  # [B, D]
        
        # Normalize
        if self.normalize:
            anchor = F.normalize(anchor, dim=-1, p=2)
            positive = F.normalize(positive, dim=-1, p=2)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [B]
        
        if negative_features is not None:
            negative = negative_features.mean(dim=1)  # [B, D]
            if self.normalize:
                negative = F.normalize(negative, dim=-1, p=2)
            
            # Use all other samples in batch as negatives
            # [B, D] x [D, B] -> [B, B]
            all_sim = torch.mm(anchor, torch.cat([positive, negative], dim=0).T) / self.temperature
            labels = torch.arange(anchor.size(0), device=anchor.device)
            loss = F.cross_entropy(all_sim, labels)
        else:
            # Only use in-batch negatives
            all_sim = torch.mm(anchor, positive.T) / self.temperature  # [B, B]
            labels = torch.arange(anchor.size(0), device=anchor.device)
            loss = F.cross_entropy(all_sim, labels)
        
        return loss
