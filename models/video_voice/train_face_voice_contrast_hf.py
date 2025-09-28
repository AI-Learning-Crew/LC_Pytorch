# =============================================
# File: train_face_voice_contrast_hf.py
# Description: ViT + Wav2Vec2 dual encoder contrastive training (InfoNCE)
# =============================================
from __future__ import annotations
import math, os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    ViTModel, ViTImageProcessor,
    Wav2Vec2Model, Wav2Vec2Processor,
    get_cosine_schedule_with_warmup,
)

# If running as a single file, the definitions above would be in the same module.
# Otherwise, import as:
# from dataset_face_voice_hf import FaceVoiceDatasetHF, make_collate_fn

# ---- Projection heads ------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.normalize(x, dim=-1)
        return x

# ---- Dual encoder wrapper --------------------------------------------------

class FaceVoiceDualEncoder(nn.Module):
    def __init__(self,
                 vit_name: str = "google/vit-base-patch16-224-in21k",
                 w2v_name: str = "facebook/wav2vec2-base",
                 embed_dim: int = 256,
                 freeze_backbones: bool = False,
                 average_frame_embeddings: bool = True,
                 ) -> None:
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        self.average_frame_embeddings = average_frame_embeddings

        vit_out = self.vit.config.hidden_size
        w2v_out = self.w2v.config.hidden_size

        self.img_proj = ProjectionHead(vit_out, embed_dim, hidden_dim=vit_out)
        self.aud_proj = ProjectionHead(w2v_out, embed_dim, hidden_dim=w2v_out)

        if freeze_backbones:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.w2v.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _vit_encode_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """frames_tensor: (K, 3, H, W) -> returns (K, D)
        Uses CLS token embedding (last_hidden_state[:,0,:]).
        """
        out = self.vit(frames_tensor, output_hidden_states=False)
        cls = out.last_hidden_state[:, 0, :]  # (K, D)
        return cls

    def encode_images(self, pixel_values_list: list[torch.Tensor]) -> torch.Tensor:
        """pixel_values_list: list of (Ki, 3, H, W) for each item in batch
        Returns image embeddings (B, D) after frame-avg pooling + projection.
        """
        embs = []
        for frames in pixel_values_list:
            # encode each frame, then mean-pool
            frame_feats = self._vit_encode_frames(frames)  # (K, D)
            if self.average_frame_embeddings:
                feat = frame_feats.mean(dim=0, keepdim=True)  # (1, D)
            else:
                feat = frame_feats.mean(dim=0, keepdim=True)  # placeholder; could do attention
            embs.append(feat)
        img_feats = torch.cat(embs, dim=0)  # (B, D)
        img_z = self.img_proj(img_feats)    # (B, d)
        return img_z

    def encode_audios(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        w2v_out = self.w2v(input_values=input_values, attention_mask=attention_mask)
        hidden = w2v_out.last_hidden_state  # (B, T, C)
        # masked mean pooling over time
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        sum_hidden = (hidden * mask).sum(dim=1)  # (B, C)
        len_mask = mask.sum(dim=1).clamp_min(1.0)
        pooled = sum_hidden / len_mask  # (B, C)
        aud_z = self.aud_proj(pooled)   # (B, d)
        return aud_z

# ---- Loss ------------------------------------------------------------------

def contrastive_loss(img_z: torch.Tensor, aud_z: torch.Tensor, tau: float = 0.07):
    img_z = nn.functional.normalize(img_z, dim=-1)
    aud_z = nn.functional.normalize(aud_z, dim=-1)
    logits_ia = img_z @ aud_z.t() / tau
    logits_ai = aud_z @ img_z.t() / tau
    labels = torch.arange(img_z.size(0), device=img_z.device)
    loss = (nn.functional.cross_entropy(logits_ia, labels) +
            nn.functional.cross_entropy(logits_ai, labels)) * 0.5
    acc = (logits_ia.argmax(dim=1) == labels).float().mean()
    return loss, acc

# ---- Train loop ------------------------------------------------------------

@dataclass
class TrainConfig:
    meta_path: str = "meta.json"
    root_dir: str = "dataset_root"
    vit_name: str = "google/vit-base-patch16-224-in21k"
    w2v_name: str = "facebook/wav2vec2-base"
    batch_size: int = 16
    frames_per_sample: int = 4
    frame_sample_mode: str = "uniform"
    audio_seconds: float = 2.0
    lr: float = 3e-5
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 500
    embed_dim: int = 256
    tau: float = 0.07
    freeze_backbones: bool = False
    num_workers: int = 4
    grad_accum_steps: int = 1
    mixed_precision: bool = True