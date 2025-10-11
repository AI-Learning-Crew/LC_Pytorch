# =============================================
# File: face_voice_dual_encoder.py
# Description: Shared TimeSformer + Wav2Vec2 dual encoder module
# =============================================
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TimesformerModel, Wav2Vec2Model


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FaceVoiceDualEncoder(nn.Module):
    def __init__(
        self,
        vit_name: str = "facebook/timesformer-base-finetuned-k400",
        w2v_name: str = "facebook/wav2vec2-base",
        embed_dim: int = 256,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.video_encoder = TimesformerModel.from_pretrained(vit_name)
        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        self.freeze_backbones = freeze_backbones

        vit_out = self.video_encoder.config.hidden_size
        w2v_out = self.w2v.config.hidden_size

        self.img_proj = ProjectionHead(vit_out, embed_dim, hidden_dim=vit_out)
        self.aud_proj = ProjectionHead(w2v_out, embed_dim, hidden_dim=w2v_out)

        self.required_frames = getattr(self.video_encoder.config, "num_frames", None)

        if freeze_backbones:
            for p in self.video_encoder.parameters():
                p.requires_grad = False
            for p in self.w2v.parameters():
                p.requires_grad = False

    def _prepare_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4:
            raise ValueError(f"Expected clip tensor (frames,3,H,W); got {clip.shape}")

        if self.required_frames is None:
            return clip

        current = clip.size(0)
        target = self.required_frames
        if current == target:
            return clip
        if current > target:
            idxs = torch.linspace(0, current - 1, steps=target, device=clip.device).round().long()
            return clip.index_select(0, idxs)

        pad_count = target - current
        pad = clip[-1:].expand(pad_count, -1, -1, -1)
        return torch.cat([clip, pad], dim=0)

    def _video_encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        clip = self._prepare_clip(clip).unsqueeze(0)  # (1,T,3,H,W)
        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            out = self.video_encoder(pixel_values=clip, output_hidden_states=False)
            cls = out.last_hidden_state[:, 0, :]
        return cls

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(images, list):
            feats = []
            for clip in images:
                clip = clip.to(device)
                if clip.ndim == 3:
                    clip = clip.unsqueeze(0)
                if clip.ndim != 4:
                    raise ValueError(f"Unexpected frame tensor shape: {clip.shape}")
                feats.append(self._video_encode_clip(clip))
            img_feats = torch.cat(feats, dim=0)
        elif images.ndim == 4:  # (B,3,H,W)
            clips = images.to(device)
            feats = []
            for img in clips:
                clip = img.unsqueeze(0)
                feats.append(self._video_encode_clip(clip))
            img_feats = torch.cat(feats, dim=0)
        elif images.ndim == 5:  # (B,K,3,H,W)
            B, K = images.shape[:2]
            clips = images.to(device)
            feats = []
            for b in range(B):
                clip = clips[b]
                flat = clip.view(K, -1)
                valid = flat.abs().sum(dim=1) > 0
                clip = clip[valid] if valid.any() else clip[:1]
                feats.append(self._video_encode_clip(clip))
            img_feats = torch.cat(feats, dim=0)
        else:
            raise ValueError(f"Unexpected images shape: {getattr(images, 'shape', None)}")

        return self.img_proj(img_feats)

    @torch.no_grad()
    def encode_audios(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        audio = audio.to(device)
        if attention_mask is None:
            attention_mask = (audio != 0.0).to(audio.dtype)
        attention_mask = attention_mask.to(device)

        out = self.w2v(input_values=audio, attention_mask=attention_mask, output_hidden_states=False)
        hidden = out.last_hidden_state
        aud_feat = hidden.mean(dim=1)
        return self.aud_proj(aud_feat)


__all__ = ["FaceVoiceDualEncoder"]
