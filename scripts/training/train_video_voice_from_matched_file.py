# =============================================
# File: train_face_voice_contrast_hf.py
# Description: TimeSformer (video ViT) + Wav2Vec2 dual encoder contrastive training (InfoNCE)
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

import torch.nn.functional as F

from transformers import (
    TimesformerModel, AutoImageProcessor,
    Wav2Vec2Model, Wav2Vec2Processor,
    get_cosine_schedule_with_warmup,
)

from tqdm import tqdm # tqdm 추가

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
                 vit_name: str = "facebook/timesformer-base-finetuned-k400",
                 w2v_name: str = "facebook/wav2vec2-base",
                 embed_dim: int = 256,
                 freeze_backbones: bool = False,
                 average_frame_embeddings: bool = True):
        super().__init__()
        self.video_encoder = TimesformerModel.from_pretrained(vit_name)
        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name)
        self.average_frame_embeddings = average_frame_embeddings
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

    def _video_encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (T,3,H,W) -> returns (1,D) CLS embedding
        """
        clip = self._prepare_clip(clip)
        clip = clip.unsqueeze(0)  # (1,T,3,H,W)
        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            out = self.video_encoder(pixel_values=clip, output_hidden_states=False)
            cls = out.last_hidden_state[:, 0, :]  # (1,D)
        return cls

    def encode_images(self, images) -> torch.Tensor:
        """
        images:
          - (B,3,H,W)        if averaged in collate, or
          - (B,K,3,H,W)      if not averaged, or
          - list[(K,3,H,W)]  legacy
        returns: (B, d) L2-normalized
        """
        device = next(self.parameters()).device

        if isinstance(images, list):
            feats = []
            for clip in images:
                clip = clip.to(device)
                if clip.ndim == 3:
                    clip = clip.unsqueeze(0)  # (1,3,H,W)
                if clip.ndim != 4:
                    raise ValueError(f"Unexpected frame tensor shape: {clip.shape}")
                feat = self._video_encode_clip(clip)
                feats.append(feat)
            img_feats = torch.cat(feats, dim=0)

        elif images.ndim == 4:  # (B,3,H,W)
            clips = images.to(device)
            feats = []
            for img in clips:
                clip = img.unsqueeze(0)  # (1,3,H,W)
                feat = self._video_encode_clip(clip)
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
                feat = self._video_encode_clip(clip)
                feats.append(feat)
            img_feats = torch.cat(feats, dim=0)
        else:
            raise ValueError(f"Unexpected images shape: {getattr(images, 'shape', None)}")

        img_z = self.img_proj(img_feats)               # (B,d)
        return F.normalize(img_z, p=2, dim=1)

    def encode_audios(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        audio: (B, T) padded with zeros
        attention_mask: (B, T) with 1 for real, 0 for pad. If None, infer from audio!=0.
        returns: (B, d) L2-normalized
        """
        device = next(self.parameters()).device
        audio = audio.to(device)

        if attention_mask is None:
            attention_mask = (audio != 0.0).to(audio.dtype)
        attention_mask = attention_mask.to(device)

        ctx = torch.no_grad() if self.freeze_backbones else torch.enable_grad()
        with ctx:
            out = self.w2v(input_values=audio, attention_mask=attention_mask, output_hidden_states=False)
            # 마스킹 평균
            hidden = out.last_hidden_state  # (B, T', D); W2V는 stride로 T' < T일 수 있음
            # Wav2Vec2는 내부 다운샘플로 attention_mask도 다운샘플 안 됨.
            # 간단 대안: hidden의 시간축 mean (실무에선 CTC mask/특징 길이로 정교화 권장)
            aud_feat = hidden.mean(dim=1)   # (B, D)

        aud_z = self.aud_proj(aud_feat)     # (B, d)
        return F.normalize(aud_z, p=2, dim=1)



# ---- Loss ------------------------------------------------------------------

def contrastive_loss(img_z: torch.Tensor, aud_z: torch.Tensor, tau: float = 0.07):
    # img_z, aud_z: (B, d)
    img_z = F.normalize(img_z, dim=-1)
    aud_z = F.normalize(aud_z, dim=-1)

    logits = img_z @ aud_z.t() / tau   # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)  # (B,)

    loss_i = F.cross_entropy(logits, targets)         # 이미지→오디오
    loss_a = F.cross_entropy(logits.t(), targets)     # 오디오→이미지
    loss = (loss_i + loss_a) / 2.0
    return loss

# ---- Train loop ------------------------------------------------------------

@dataclass
class TrainConfig:
    meta_path: str = "meta.json"
    root_dir: str = "dataset_root"
    vit_name: str = "facebook/timesformer-base-finetuned-k400"
    w2v_name: str = "facebook/wav2vec2-base"
    batch_size: int = 16
    frames_per_sample: int = 8
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
    image_processor = AutoImageProcessor.from_pretrained(cfg.vit_name)
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(cfg.w2v_name)

    # dataset & loader
    ds = FaceVoiceDatasetHF(
        meta_path=cfg.meta_path,
        root_dir=cfg.root_dir,
        
        frames_per_sample=cfg.frames_per_sample,
        frame_sample_mode=cfg.frame_sample_mode,
        
        audio_processor=wav2vec2_processor,
        audio_seconds=cfg.audio_seconds,
        target_sr=wav2vec2_processor.feature_extractor.sampling_rate,
    )
    collate_fn = make_collate_fn(image_processor)
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
        print(f"Epoch {epoch + 1}/{cfg.max_epochs}")
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0

        # tqdm을 사용하여 진행률 바 추가
        for batch in tqdm(dl, desc=f"Training Epoch {epoch + 1}/{cfg.max_epochs}", leave=False):
            try:
                # --- 디버그 출력 (원하면 유지/비활성)
                # print("Batch structure:")
                # for key, value in batch.items():
                #     if isinstance(value, torch.Tensor):
                #         print(f"  {key}: {value.shape}")
                #     elif isinstance(value, list):
                #         print(f"  {key}: List of length {len(value)}")
                #     else:
                #         print(f"  {key}: {type(value)}")

                # collate 반환 키에 맞게 수정
                images = batch["images"].to(device, non_blocking=True)  # (B,3,H,W) 또는 (B,K,3,H,W)
                audios = batch["audio"].to(device, non_blocking=True)   # (B,T)
                attn_mask = (audios != 0).to(torch.long)                 # zero-padding 기반 마스크

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    img_z = model.encode_images(images)
                    aud_z = model.encode_audios(audios, attention_mask=attn_mask)
                    loss = contrastive_loss(img_z, aud_z, tau=cfg.tau)

                    # 선택: Top-1 정합(acc) 계산 (모니터링용)
                    with torch.no_grad():
                        logits = (F.normalize(img_z, dim=-1) @ F.normalize(aud_z, dim=-1).t())
                        tgt = torch.arange(logits.size(0), device=logits.device)
                        acc_i = (logits.argmax(dim=1) == tgt).float().mean()
                        acc_a = (logits.argmax(dim=0) == tgt).float().mean()
                        acc = 0.5 * (acc_i + acc_a)

                # AMP backward & step
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

                step += 1
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                batch_count += 1

                if step % 50 == 0:
                    print(f"Step {step}: Loss={loss.item():.4f}, Acc@1={acc.item():.3f}")

            except Exception as e:
                print(f"오류 발생: {e}")
                print("해당 배치를 건너뜁니다.")
                continue


        # 에포크별 평균 손실 및 정확도 출력
        epoch_loss /= batch_count
        epoch_acc /= batch_count
        print(f"Epoch {epoch + 1} Completed: Avg Loss={epoch_loss:.4f}, Avg Acc@1={epoch_acc:.3f}")
    
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
    parser.add_argument('--frames_per_sample', type=int, default=int(os.environ.get("K", 8)),
                        help='샘플당 프레임 수 (기본값: 8)')
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
