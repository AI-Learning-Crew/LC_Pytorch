# =============================================
# File: train_faces_mel.py
# Description: TimeSformer (video ViT) + Transformer mel encoder contrastive training (InfoNCE)
# =============================================

from __future__ import annotations
import os
import math
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from models.faces_mel.dataset_faces_mel import FacesMelDataset, make_collate_fn
from models.faces_mel.faces_mel_dual_encoder import FacesMelDualEncoder


# ---- Loss ------------------------------------------------------------------

def contrastive_loss(img_z: torch.Tensor, mel_z: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Computes the contrastive loss using InfoNCE."""
    img_z = F.normalize(img_z, dim=-1)
    mel_z = F.normalize(mel_z, dim=-1)

    logits = img_z @ mel_z.t() / tau  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)  # (B,)

    loss_i = F.cross_entropy(logits, targets)         # Image → Mel
    loss_a = F.cross_entropy(logits.t(), targets)     # Mel → Image
    return (loss_i + loss_a) / 2.0


# ---- Configuration ---------------------------------------------------------

@dataclass
class TrainConfig:
    meta_path: str = "meta.json"
    root_dir: str = "dataset_root"
    mel_root: Optional[str] = None
    vit_name: str = "facebook/timesformer-base-finetuned-k400"
    batch_size: int = 16
    frames_per_sample: int = 8
    frame_sample_mode: str = "uniform"
    mel_freq_bins: int = 40
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
    resume_ckpt: Optional[str] = None


# ---- Checkpoint Management -------------------------------------------------

def save_epoch_checkpoint(
    save_dir: Path,
    model: FacesMelDualEncoder,
    opt: torch.optim.Optimizer,
    sched,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
) -> Tuple[Path, Path, Path]:
    """Saves model, optimizer, and scheduler states."""
    cfg_dict = dict(cfg.__dict__)
    cfg_dict["resume_ckpt"] = None  # Avoid recursive resume references
    checkpoint = {
        "config": cfg_dict,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    ckpt_path = save_dir / f"dual_encoder_full_epoch{epoch:03d}.pth"
    torch.save(checkpoint, ckpt_path)

    video_path = save_dir / f"video_encoder_epoch{epoch:03d}.pth"
    mel_path = save_dir / f"mel_encoder_epoch{epoch:03d}.pth"
    torch.save(model.video_encoder.state_dict(), video_path)
    torch.save(model.mel_encoder.state_dict(), mel_path)
    return ckpt_path, video_path, mel_path


# ---- Training Loop ---------------------------------------------------------

def train_one_epoch(
    model: FacesMelDualEncoder,
    dataloader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    sched,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    step: int,
) -> Tuple[float, float, int]:
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_count = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{cfg.max_epochs}", leave=False):
        try:
            images = batch["images"].to(device, non_blocking=True)
            mels = batch["mel"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                img_z, mel_z = model(images, mels)
                loss = contrastive_loss(img_z, mel_z, tau=cfg.tau)

                with torch.no_grad():
                    logits = F.normalize(img_z, dim=-1) @ F.normalize(mel_z, dim=-1).t()
                    tgt = torch.arange(logits.size(0), device=logits.device)
                    acc_i = (logits.argmax(dim=1) == tgt).float().mean()
                    acc_a = (logits.argmax(dim=0) == tgt).float().mean()
                    acc = 0.5 * (acc_i + acc_a)

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
            print(f"Error occurred: {e}. Skipping batch.")
            continue

    epoch_loss /= batch_count
    epoch_acc /= batch_count
    return epoch_loss, epoch_acc, step


def main(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize processor for image frames
    image_processor = AutoImageProcessor.from_pretrained(cfg.vit_name)

    # Dataset and DataLoader
    ds = FacesMelDataset(
        meta_path=cfg.meta_path,
        root_dir=cfg.root_dir,
        mel_root=cfg.mel_root,
        frames_per_sample=cfg.frames_per_sample,
        frame_sample_mode=cfg.frame_sample_mode,
    )
    collate_fn = make_collate_fn(image_processor)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    # Model
    mel_freq_bins = cfg.mel_freq_bins
    if mel_freq_bins <= 0:
        if len(ds) == 0:
            raise ValueError("Dataset is empty; cannot infer mel frequency bins.")
        sample = ds[0]["mel"]
        mel_freq_bins = sample.size(1)

    model = FacesMelDualEncoder(
        vit_name=cfg.vit_name,
        embed_dim=cfg.embed_dim,
        mel_freq_bins=mel_freq_bins,
        freeze_backbones=cfg.freeze_backbones,
        average_frame_embeddings=True,
    ).to(device)

    # Optimizer and Scheduler
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.max_epochs * math.ceil(len(ds) / cfg.batch_size)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if cfg.resume_ckpt:
        resume_path = Path(cfg.resume_ckpt)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt.get("scheduler_state_dict", {}))
        scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("global_step", 0)
        print(f"Resumed from checkpoint {resume_path} (epoch={start_epoch}, step={step})")

    # Training loop
    for epoch in range(start_epoch, cfg.max_epochs):
        print(f"Epoch {epoch + 1}/{cfg.max_epochs}")
        epoch_loss, epoch_acc, step = train_one_epoch(
            model, dl, opt, scaler, sched, cfg, device, epoch, step
        )
        print(f"Epoch {epoch + 1} Completed: Avg Loss={epoch_loss:.4f}, Avg Acc@1={epoch_acc:.3f}")

        save_epoch_checkpoint(
            save_dir=save_dir,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch + 1,
            global_step=step,
        )

    # Save latest aliases
    shutil.copy2(save_dir / f"dual_encoder_full_epoch{cfg.max_epochs:03d}.pth", save_dir / "dual_encoder_full.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face-mel contrastive model.")
    parser.add_argument("--meta_path", type=str, default="meta.json", help="Path to meta file.")
    parser.add_argument("--root_dir", type=str, default="dataset_root", help="Dataset root directory.")
    parser.add_argument("--mel_root", type=str, default="", help="Optional directory that mirrors mel pickles.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--frames_per_sample", type=int, default=8, help="Frames per sample.")
    parser.add_argument("--frame_sample_mode", type=str, default="uniform", help="How to sample frames per video.")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of epochs.")
    parser.add_argument("--freeze_backbones", type=int, default=0, help="Freeze backbones (0: False, 1: True).")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--mel_freq_bins", type=int, default=40, help="Mel frequency bins (set 0 to infer from dataset).")
    parser.add_argument("--resume_ckpt", type=str, default="", help="Path to resume checkpoint.")
    args = parser.parse_args()

    cfg = TrainConfig(
        meta_path=args.meta_path,
        root_dir=args.root_dir,
        mel_root=args.mel_root or None,
        batch_size=args.batch_size,
        frames_per_sample=args.frames_per_sample,
        frame_sample_mode=args.frame_sample_mode,
        max_epochs=args.max_epochs,
        freeze_backbones=bool(args.freeze_backbones),
        lr=args.lr,
        mel_freq_bins=args.mel_freq_bins,
        save_dir=args.save_dir,
        resume_ckpt=args.resume_ckpt or None,
    )
    main(cfg)
