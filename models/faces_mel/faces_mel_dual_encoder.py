import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import TimesformerModel
from typing import Optional, Tuple

from utils.voice_encoders import TransEncoder
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

class FacesMelDualEncoder(nn.Module):
    def __init__(
        self,
        vit_name: str = "facebook/timesformer-base-finetuned-k400",
        embed_dim: int = 256,
        mel_freq_bins: int = 40,
        freeze_backbones: bool = False,
        average_frame_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.video_encoder = TimesformerModel.from_pretrained(vit_name)
        self.average_frame_embeddings = average_frame_embeddings
        self.freeze_backbones = freeze_backbones

        vit_out = self.video_encoder.config.hidden_size

        self.img_proj = ProjectionHead(vit_out, embed_dim, hidden_dim=vit_out)

        self.mel_feat_dim = 512
        self.mel_encoder = TransEncoder(
            input_channel=mel_freq_bins,
            cnn_channels=[self.mel_feat_dim, self.mel_feat_dim],
            transformer_dim=self.mel_feat_dim,
            transformer_depth=2,
            return_seq=False,
            pos_embedding_dim=0,
            sin_pos_encoding=True,
        )
        self.mel_proj = ProjectionHead(self.mel_feat_dim, embed_dim, hidden_dim=self.mel_feat_dim)

        self.required_frames = getattr(self.video_encoder.config, "num_frames", None)

        if freeze_backbones:
            for p in self.video_encoder.parameters():
                p.requires_grad = False
            for p in self.mel_encoder.parameters():
                p.requires_grad = False

    def _prepare_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Normalize clip length to match the video encoder requirement.
        clip: (T, 3, H, W)
        """
        if clip.ndim != 4:
            raise ValueError(f"Expected clip tensor (frames,3,H,W); got {clip.shape}")

        if self.required_frames is None:
            return clip

        current = clip.size(0)
        target = self.required_frames
        if current == target:
            return clip
        if current > target:
            idxs = torch.linspace(0, current - 1, steps=target, device=clip.device)
            idxs = idxs.round().long()
            return clip.index_select(0, idxs)
        # current < target: pad by repeating last frame
        pad_count = target - current
        pad = clip[-1:].expand(pad_count, -1, -1, -1)
        return torch.cat([clip, pad], dim=0)

    def _video_encode_clip(self, clip: torch.Tensor, enable_grad: bool) -> torch.Tensor:
        """
        clip: (T,3,H,W) -> returns (1,D) CLS embedding
        """
        clip = self._prepare_clip(clip)
        clip = clip.unsqueeze(0)  # (1,T,3,H,W)
        with torch.set_grad_enabled(enable_grad):
            out = self.video_encoder(pixel_values=clip, output_hidden_states=False)
            cls = out.last_hidden_state[:, 0, :]  # (1,D)
        return cls

    def encode_images(self, images, *, enable_grad: Optional[bool] = None) -> torch.Tensor:
        """
        images:
          - (B,3,H,W)        if averaged in collate, or
          - (B,K,3,H,W)      if not averaged, or
          - list[(K,3,H,W)]  legacy
        returns: (B, d) L2-normalized
        """
        device = next(self.parameters()).device
        enable_grad = (self.training and not self.freeze_backbones) if enable_grad is None else enable_grad

        if isinstance(images, list):
            feats = []
            for clip in images:
                clip = clip.to(device)
                if clip.ndim == 3:
                    clip = clip.unsqueeze(0)  # (1,3,H,W)
                if clip.ndim != 4:
                    raise ValueError(f"Unexpected frame tensor shape: {clip.shape}")
                feat = self._video_encode_clip(clip, enable_grad)
                feats.append(feat)
            img_feats = torch.cat(feats, dim=0)

        elif images.ndim == 4:  # (B,3,H,W)
            clips = images.to(device)
            feats = []
            for img in clips:
                clip = img.unsqueeze(0)  # (1,3,H,W)
                feat = self._video_encode_clip(clip, enable_grad)
                feats.append(feat)
            img_feats = torch.cat(feats, dim=0)

        elif images.ndim == 5:  # (B,K,3,H,W)
            B, K = images.shape[:2]
            clips = images.to(device)
            feats = []
            for b in range(B):
                clip = clips[b]
                flat = clip.view(K, -1)
                valid = flat.abs().sum(dim=1) > 0
                if valid.any():
                    clip = clip[valid]
                else:
                    clip = clip[:1]
                feat = self._video_encode_clip(clip, enable_grad)
                feats.append(feat)
            img_feats = torch.cat(feats, dim=0)
        else:
            raise ValueError(f"Unexpected images shape: {getattr(images, 'shape', None)}")

        img_z = self.img_proj(img_feats)               # (B,d)
        return F.normalize(img_z, p=2, dim=1)

    def encode_mel(
        self,
        mel: torch.Tensor,
        *,
        enable_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        mel: (B, T, F) log-mel spectrograms
        returns: (B, d) L2-normalized
        """
        device = next(self.parameters()).device
        mel = mel.to(device)

        if mel.ndim != 3:
            raise ValueError(f"Expected mel tensor of shape (B,T,F); got {mel.shape}")

        mel_seq = mel.transpose(1, 2).contiguous()

        grad_enabled = (self.training and not self.freeze_backbones) if enable_grad is None else enable_grad
        with torch.set_grad_enabled(grad_enabled):
            mel_embeddings = self.mel_encoder(mel_seq)

        if mel_embeddings.ndim > 2:
            mel_feat = mel_embeddings.flatten(start_dim=1)
        else:
            mel_feat = mel_embeddings

        aud_z = self.mel_proj(mel_feat)
        return F.normalize(aud_z, p=2, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        mel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_enabled = self.training and not self.freeze_backbones
        img_z = self.encode_images(images, enable_grad=grad_enabled)
        mel_z = self.encode_mel(mel, enable_grad=grad_enabled)
        return img_z, mel_z
