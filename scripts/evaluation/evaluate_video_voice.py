# =============================================
# File: evaluate_face_voice_contrast_hf.py
# Description: Evaluate TimeSformer+Wav2Vec2 dual-encoder (InfoNCE) on 118 IDs × 5 items
# Metrics: Recall@1 / @5 / @10 for image→audio and audio→image; macro average
# =============================================
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoImageProcessor,
    Wav2Vec2Processor,
)

# ---- project imports (assumes same project as training script) ------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from models.video_voice.dataset_face_voice_hf import FaceVoiceDatasetHF, make_collate_fn
from models.video_voice.face_voice_dual_encoder import FaceVoiceDualEncoder




# ---- Metrics ---------------------------------------------------------------
@torch.no_grad()
def recall_at_k(sim: torch.Tensor, labels_q: List[str], labels_g: List[str], ks=(1,5,10)) -> Dict[int, float]:
    """
    sim: (Nq, Ng) similarity matrix (higher is better)
    labels_q: list[str] size Nq
    labels_g: list[str] size Ng
    Success if *any* of the top-k gallery items share the same label as the query.
    """
    Nq, Ng = sim.shape
    ks = tuple(k for k in ks if k <= Ng)
    # argsort descending
    ranks = torch.argsort(sim, dim=1, descending=True)
    recalls = {k: 0 for k in ks}
    for i in range(Nq):
        qlab = labels_q[i]
        top = ranks[i, :max(ks)].tolist()
        
        top10 = ranks[i, : min(10, Ng)].tolist()
        top10_labels = [labels_g[idx] for idx in top10]
        print(f"[recall_at_k] Query {i} ({qlab}) top-10: {top10_labels}")
        
        top_ok = {k: False for k in ks}
        for j, gidx in enumerate(top, start=1):
            if labels_g[gidx] == qlab:
                for k in ks:
                    if j <= k:
                        top_ok[k] = True
        for k in ks:
            recalls[k] += 1 if top_ok[k] else 0
    for k in ks:
        recalls[k] /= Nq
    return recalls


# ---- Helpers ---------------------------------------------------------------
@torch.no_grad()
def compute_embeddings(
    model: FaceVoiceDualEncoder,
    dl: DataLoader,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    device = next(model.parameters()).device
    img_embs, aud_embs = [], []
    labels: List[str] = []

    total_batches = len(dl) if hasattr(dl, "__len__") else None

    for step, batch in enumerate(dl, start=1):
        images = batch["images"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        bsz = images.size(0)
        if total_batches:
            print(f"[compute_embeddings] Batch {step}/{total_batches} (size={bsz}) | keys={list(batch.keys())}")
        else:
            print(f"[compute_embeddings] Batch {step} (size={bsz}) | keys={list(batch.keys())}")
        # Try get labels (speaker/ID). Collate may keep a list of strings or ints.
        meta = batch.get("meta")
        if isinstance(meta, list):
            batch_labels = []
            for entry in meta:
                pid = entry.get("pid") if isinstance(entry, dict) else None
                batch_labels.append(str(pid) if pid is not None else "unknown")
        else:
            B = images.size(0)
            batch_labels = ["unknown"] * B

        with torch.cuda.amp.autocast(enabled=use_amp):
            ie = model.encode_images(images)
            ae = model.encode_audios(audio, attention_mask=(audio != 0).to(torch.long))
        img_embs.append(F.normalize(ie, dim=-1).cpu())
        aud_embs.append(F.normalize(ae, dim=-1).cpu())
        labels.extend(batch_labels)

    img_embs = torch.cat(img_embs, dim=0)
    aud_embs = torch.cat(aud_embs, dim=0)
    
    print(f"Computed embeddings: images {img_embs.shape}, audios {aud_embs.shape}, labels {len(labels)}")
    print(f"Sample labels: {labels}")
    return img_embs, aud_embs, labels


def collapse_by_id(embs: torch.Tensor, labels: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Average embeddings for items with the same label (ID)."""
    from collections import defaultdict
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        buckets[lab].append(idx)
    new_embs = []
    new_labels = []
    for lab, idxs in buckets.items():
        new_embs.append(embs[idxs].mean(dim=0, keepdim=True))
        new_labels.append(lab)
    return torch.cat(new_embs, dim=0), new_labels


# ---- Main -----------------------------------------------------------------
@dataclass
class EvalConfig:
    meta_path: str = "meta_eval.json"
    root_dir: str = "dataset_root"
    ckpt_path: str = "checkpoints/dual_encoder_full.pth"
    vit_name: Optional[str] = None
    w2v_name: Optional[str] = None
    video_encoder_path: Optional[str] = None
    voice_encoder_path: Optional[str] = None
    batch_size: int = 32
    frames_per_sample: int = 8
    frame_sample_mode: str = "uniform"
    audio_seconds: float = 2.0
    num_workers: int = 4
    mixed_precision: bool = True
    average_by_id: bool = False  # if True, average the 5 items per ID before computing recall


def _load_model_state_dict(model: FaceVoiceDualEncoder, state_dict: Dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] Missing keys when loading state dict: {missing}")
    if unexpected:
        print(f"[load_model] Unexpected keys when loading state dict: {unexpected}")


def load_model_from_checkpoint(cfg: EvalConfig) -> FaceVoiceDualEncoder:
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")

    train_cfg = ckpt.get("config", {})
    embed_dim = train_cfg.get("embed_dim", 256)

    vit_name = cfg.vit_name or train_cfg.get("vit_name") or ckpt.get("vit_name") or "facebook/timesformer-base-finetuned-k400"
    w2v_name = cfg.w2v_name or train_cfg.get("w2v_name") or ckpt.get("w2v_name") or "facebook/wav2vec2-base"

    model = FaceVoiceDualEncoder(
        vit_name=vit_name,
        w2v_name=w2v_name,
        embed_dim=embed_dim,
        freeze_backbones=False,
        average_frame_embeddings=train_cfg.get("average_frame_embeddings", True),
    )

    if "model_state_dict" in ckpt:
        _load_model_state_dict(model, ckpt["model_state_dict"])
    else:
        # Backward compatibility with head-only checkpoints
        if "img_proj" in ckpt:
            model.img_proj.load_state_dict(ckpt["img_proj"])
        if "aud_proj" in ckpt:
            model.aud_proj.load_state_dict(ckpt["aud_proj"])
        if cfg.video_encoder_path:
            video_state = torch.load(cfg.video_encoder_path, map_location="cpu")
            _load_model_state_dict(model.video_encoder, video_state)
        if cfg.voice_encoder_path:
            voice_state = torch.load(cfg.voice_encoder_path, map_location="cpu")
            _load_model_state_dict(model.w2v, voice_state)
        model.freeze_backbones = True
        for p in model.parameters():
            p.requires_grad = False

    # Optionally load external encoder weights if provided
    if cfg.video_encoder_path and "model_state_dict" in ckpt:
        video_state = torch.load(cfg.video_encoder_path, map_location="cpu")
        _load_model_state_dict(model.video_encoder, video_state)
    if cfg.voice_encoder_path and "model_state_dict" in ckpt:
        voice_state = torch.load(cfg.voice_encoder_path, map_location="cpu")
        _load_model_state_dict(model.w2v, voice_state)

    vit_name = vit_name or ckpt.get("vit_name", "facebook/timesformer-base-finetuned-k400")
    w2v_name = w2v_name or ckpt.get("w2v_name", "facebook/wav2vec2-base")

    model.eval()
    return model


def main(cfg: EvalConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # processors
    image_processor = AutoImageProcessor.from_pretrained(cfg.vit_name or "facebook/timesformer-base-finetuned-k400")
    w2v_proc = Wav2Vec2Processor.from_pretrained(cfg.w2v_name or "facebook/wav2vec2-base")

    # dataset (NO shuffle)
    ds = FaceVoiceDatasetHF(
        meta_path=cfg.meta_path,
        root_dir=cfg.root_dir,
        frames_per_sample=cfg.frames_per_sample,
        frame_sample_mode=cfg.frame_sample_mode,
        audio_processor=w2v_proc,
        audio_seconds=cfg.audio_seconds,
        target_sr=w2v_proc.feature_extractor.sampling_rate,
    )
    collate_fn = make_collate_fn(image_processor)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    collate_fn=collate_fn, drop_last=False, pin_memory=True)

    # model
    model = load_model_from_checkpoint(cfg).to(device)

    # embeddings
    img_embs, aud_embs, labels = compute_embeddings(model, dl, use_amp=cfg.mixed_precision)

    # optionally average per ID (118×118 retrieval)
    if cfg.average_by_id:
        img_embs, labels_img = collapse_by_id(img_embs, labels)
        aud_embs, labels_aud = collapse_by_id(aud_embs, labels)
    else:
        labels_img = labels_aud = labels

    # similarity
    sim_i2a = img_embs @ aud_embs.t()   # (Ni, Na)
    sim_a2i = sim_i2a.t()

    # recalls
    ks = (1, 5, 10)
    r_i2a = recall_at_k(sim_i2a, labels_img, labels_aud, ks)
    r_a2i = recall_at_k(sim_a2i, labels_aud, labels_img, ks)

    # macro average
    avg = {k: (r_i2a.get(k, 0.0) + r_a2i.get(k, 0.0)) / 2.0 for k in ks}

    # pretty print
    def pct(x):
        return f"{x*100:.2f}%"

    print("=== Evaluation (Recall@K) ===")
    print(f"Items: images={img_embs.shape[0]}, audios={aud_embs.shape[0]} | IDs={len(set(labels_img))}")
    print(f"Averaged by ID: {cfg.average_by_id}")
    print("Image→Audio:")
    for k in ks:
        if k <= aud_embs.shape[0]:
            print(f"  R@{k}: {pct(r_i2a[k])}")
    print("Audio→Image:")
    for k in ks:
        if k <= img_embs.shape[0]:
            print(f"  R@{k}: {pct(r_a2i[k])}")
    print("Macro Avg:")
    for k in ks:
        k_ok = (k <= img_embs.shape[0]) and (k <= aud_embs.shape[0])
        if k_ok:
            print(f"  R@{k}: {pct(avg[k])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dual-encoder on face↔voice retrieval")
    parser.add_argument('--meta_path', type=str, default=os.environ.get("META", "meta_eval.json"))
    parser.add_argument('--root_dir', type=str, default=os.environ.get("ROOT", "dataset_root"))
    parser.add_argument('--ckpt_path', type=str, default=os.environ.get("CKPT", "checkpoints/dual_encoder_full.pth"))
    parser.add_argument('--vit_name', type=str, default=os.environ.get("VIT", ""))
    parser.add_argument('--w2v_name', type=str, default=os.environ.get("W2V", ""))
    parser.add_argument('--video_encoder_path', type=str, default=os.environ.get("VIDEO_CKPT", ""))
    parser.add_argument('--voice_encoder_path', type=str, default=os.environ.get("VOICE_CKPT", ""))
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get("BATCH", 32)))
    parser.add_argument('--frames_per_sample', type=int, default=int(os.environ.get("K", 8)))
    parser.add_argument('--audio_seconds', type=float, default=float(os.environ.get("ASEC", 2.0)))
    parser.add_argument('--num_workers', type=int, default=int(os.environ.get("WORKERS", 4)))
    parser.add_argument('--average_by_id', type=lambda x: bool(int(x)), default=bool(int(os.environ.get("AVG_ID", 0))))
    args = parser.parse_args()

    cfg = EvalConfig(
        meta_path=args.meta_path,
        root_dir=args.root_dir,
        ckpt_path=args.ckpt_path,
        vit_name=(args.vit_name or None),
        w2v_name=(args.w2v_name or None),
        video_encoder_path=(args.video_encoder_path or None),
        voice_encoder_path=(args.voice_encoder_path or None),
        batch_size=args.batch_size,
        frames_per_sample=args.frames_per_sample,
        audio_seconds=args.audio_seconds,
        num_workers=args.num_workers,
        average_by_id=args.average_by_id,
    )
    main(cfg)
