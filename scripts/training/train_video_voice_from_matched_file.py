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

import argparse
import os
import sys
from pathlib import Path

from transformers import (
    ViTModel, ViTImageProcessor,
    Wav2Vec2Model, Wav2Vec2Processor,
    get_cosine_schedule_with_warmup,
)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.video_voice.dataset_face_voice_hf import FaceVoiceDatasetHF, make_collate_fn

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
    save_dir: str = "checkpoints"


def main(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # processors
    image_processor = ViTImageProcessor.from_pretrained(cfg.vit_name)
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(cfg.w2v_name)

    # dataset & loader
    ds = FaceVoiceDatasetHF(
        meta_path=cfg.meta_path,
        root_dir=cfg.root_dir,
        frames_per_sample=cfg.frames_per_sample,
        frame_sample_mode=cfg.frame_sample_mode,
        audio_seconds=cfg.audio_seconds,
        target_sr=wav2vec2_processor.feature_extractor.sampling_rate,
        average_frames=True,
    )
    collate_fn = make_collate_fn(image_processor, wav2vec2_processor, average_frames=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    collate_fn=collate_fn, drop_last=True, pin_memory=True)

    # model
    model = FaceVoiceDualEncoder(
        vit_name=cfg.vit_name,
        w2v_name=cfg.w2v_name,
        embed_dim=cfg.embed_dim,
        freeze_backbones=cfg.freeze_backbones,
        average_frame_embeddings=True,
    ).to(device)

    # optimizer & scheduler
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.max_epochs * math.ceil(len(ds) / (cfg.batch_size * max(1, torch.cuda.device_count())))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    model.train()
    step = 0
    for epoch in range(cfg.max_epochs):
        for batch in dl:
            try:
                # move to device
                pixel_values_list = [t.to(device, non_blocking=True) for t in batch["pixel_values_list"]]
                audio_input_values = batch["audio_input_values"].to(device, non_blocking=True)
                audio_attention_mask = batch["audio_attention_mask"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    img_z = model.encode_images(pixel_values_list)
                    aud_z = model.encode_audios(audio_input_values, audio_attention_mask)
                    loss, acc = contrastive_loss(img_z, aud_z, tau=cfg.tau)

                scaler.scale(loss).step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}: loss={loss.item():.4f} acc@1={acc.item():.3f}")

            except Exception as e:
                print(f"오류 발생: {e}")
                print("해당 배치를 건너뜁니다.")
                continue
    # save
    from pathlib import Path
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "config": cfg.__dict__,
        "img_proj": model.img_proj.state_dict(),
        "aud_proj": model.aud_proj.state_dict(),
        "vit_name": cfg.vit_name,
        "w2v_name": cfg.w2v_name,
    }, save_dir / "dual_encoder_heads.pth")
    # Optionally save full model (large):
    # torch.save(model.state_dict(), save_dir / "full_dual_encoder.pth")


if __name__ == "__main__":
    # Quick config override via env or edit directly
    parser = argparse.ArgumentParser(description='얼굴-음성 매칭 모델을 학습합니다.')
    
    parser.add_argument('--meta_path', type=str, default=os.environ.get("META", "meta.json"),
                    help='메타 파일 경로 (기본값: meta.json)')
    parser.add_argument('--root_dir', type=str, default=os.environ.get("ROOT", "dataset_root"),
                        help='데이터셋 루트 디렉토리 (기본값: dataset_root)')
    parser.add_argument('--save_dir', type=str, default=os.environ.get("SAVE_DIR", "checkpoints"),
                        help='모델 저장 경로 (기본값: checkpoints)')
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get("BATCH", 16)),
                        help='배치 크기 (기본값: 16)')
    parser.add_argument('--frames_per_sample', type=int, default=int(os.environ.get("K", 4)),
                        help='샘플당 프레임 수 (기본값: 4)')
    parser.add_argument('--max_epochs', type=int, default=int(os.environ.get("EPOCHS", 5)),
                        help='최대 에포크 수 (기본값: 5)')
    parser.add_argument('--freeze_backbones', type=lambda x: bool(int(x)), default=bool(int(os.environ.get("FREEZE", 0))),
                        help='백본 동결 여부 (0: False, 1: True, 기본값: 0)')
    parser.add_argument('--lr', type=float, default=float(os.environ.get("LR", 3e-5)),
                        help='학습률 (기본값: 3e-5)')
    parser.add_argument('--audio_seconds', type=float, default=float(os.environ.get("ASEC", 2.0)),
                        help='오디오 길이 (초) (기본값: 2.0)')
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        meta_path=args.meta_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        frames_per_sample=args.frames_per_sample,
        max_epochs=args.max_epochs,
        freeze_backbones=args.freeze_backbones,
        lr=args.lr,
        audio_seconds=args.audio_seconds,
        save_dir=args.save_dir,
    )
    
    main(cfg)