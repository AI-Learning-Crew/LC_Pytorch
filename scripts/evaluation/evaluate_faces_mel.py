"""
Evaluate FacesMelDualEncoder on a faces+mel dataset.

Metrics:
  - Recall@1/@5/@10 for image→mel and mel→image retrieval.

This script mirrors the training setup in `train_faces_mel.py` and the evaluation
helpers used in `evaluate_video_voice.py`, but operates on the Faces+Mel pipeline.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoImageProcessor

# ---------------------------------------------------------------------------
# Project imports (same root assumption as training script)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from models.faces_mel.dataset_faces_mel import FacesMelDataset, make_collate_fn  # noqa: E402
from models.faces_mel.faces_mel_dual_encoder import FacesMelDualEncoder  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def recall_at_k(
    sim: torch.Tensor,
    labels_q: List[str],
    labels_g: List[str],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    """
    Compute Recall@K for retrieval where a match is a shared label.
    """
    nq, ng = sim.shape
    ks = tuple(k for k in ks if k <= ng)
    ranks = torch.argsort(sim, dim=1, descending=True)
    hits = {k: 0 for k in ks}

    for i in range(nq):
        target = labels_q[i]
        top_indices = ranks[i, : max(ks)].tolist()
        for rank_pos, gidx in enumerate(top_indices, start=1):
            if labels_g[gidx] == target:
                for k in ks:
                    if rank_pos <= k:
                        hits[k] += 1
                break

    return {k: hits[k] / nq for k in ks}


def collapse_by_label(embs: torch.Tensor, labels: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """
    Average embeddings for items sharing the same label.
    """
    from collections import defaultdict

    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        buckets[str(label)].append(idx)

    reduced_embs = []
    reduced_labels = []
    for label, idxs in buckets.items():
        reduced_embs.append(embs[idxs].mean(dim=0, keepdim=True))
        reduced_labels.append(label)
    return torch.cat(reduced_embs, dim=0), reduced_labels


# ---------------------------------------------------------------------------
# Configuration / loading helpers
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    meta_path: str = "meta_eval.json"
    root_dir: str = "dataset_root"
    mel_root: Optional[str] = None
    ckpt_path: str = "checkpoints/dual_encoder_full.pth"
    vit_name: Optional[str] = None
    batch_size: int = 32
    frames_per_sample: int = 8
    frame_sample_mode: str = "uniform"
    mel_freq_bins: int = 40
    num_workers: int = 4
    mixed_precision: bool = True
    average_by_id: bool = False


def _init_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _load_model(cfg: EvalConfig, device: torch.device) -> FacesMelDualEncoder:
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    train_cfg = ckpt.get("config", {})
    embed_dim = train_cfg.get("embed_dim", 256)
    mel_freq_bins = cfg.mel_freq_bins or train_cfg.get("mel_freq_bins", 40)
    vit_name = cfg.vit_name or train_cfg.get("vit_name") or "facebook/timesformer-base-finetuned-k400"

    model = FacesMelDualEncoder(
        vit_name=vit_name,
        embed_dim=embed_dim,
        mel_freq_bins=mel_freq_bins,
        freeze_backbones=False,
        average_frame_embeddings=True,
    )

    state = ckpt.get("model_state_dict")
    if state is None:
        raise ValueError(f"Checkpoint {cfg.ckpt_path} missing 'model_state_dict'.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys while loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %s", unexpected)

    model.eval()
    model.to(device)
    logger.info(
        "Loaded model from %s (vit=%s, embed_dim=%d, mel_bins=%d).",
        cfg.ckpt_path,
        vit_name,
        embed_dim,
        mel_freq_bins,
    )
    return model


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_embeddings(
    model: FacesMelDualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    img_embs: List[torch.Tensor] = []
    mel_embs: List[torch.Tensor] = []
    labels: List[str] = []

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

    for step, batch in enumerate(tqdm(dataloader, desc="Embedding", leave=False), start=1):
        images = batch["images"].to(device, non_blocking=True)
        mel = batch["mel"].to(device, non_blocking=True)
        meta = batch.get("meta")

        if isinstance(meta, list):
            batch_labels = [str(item.get("pid", "unknown")) for item in meta]
        else:
            batch_labels = ["unknown"] * images.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            img_z = model.encode_images(images)
            mel_z = model.encode_mel(mel)

        img_embs.append(F.normalize(img_z, dim=-1).cpu())
        mel_embs.append(F.normalize(mel_z, dim=-1).cpu())
        labels.extend(batch_labels)

        if total_batches:
            logger.info(
                "Processed batch %d/%d (size=%d)",
                step,
                total_batches,
                images.size(0),
            )

    img_embs = torch.cat(img_embs, dim=0)
    mel_embs = torch.cat(mel_embs, dim=0)
    logger.info("Collected embeddings: images %s | mel %s", tuple(img_embs.shape), tuple(mel_embs.shape))
    return img_embs, mel_embs, labels


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(cfg: EvalConfig) -> None:
    _init_logging()
    logger.info("Evaluation config: %s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Dataset & loader ------------------------------------------------------
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.vit_name or "facebook/timesformer-base-finetuned-k400",
        use_fast=False,
    )
    dataset = FacesMelDataset(
        meta_path=cfg.meta_path,
        root_dir=cfg.root_dir,
        mel_root=cfg.mel_root,
        frames_per_sample=cfg.frames_per_sample,
        frame_sample_mode=cfg.frame_sample_mode,
    )
    logger.info("Loaded dataset: %d samples.", len(dataset))

    # Infer mel bins if requested
    if cfg.mel_freq_bins <= 0:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty; cannot infer mel frequency bins.")
        cfg.mel_freq_bins = dataset[0]["mel"].size(1)
        logger.info("Inferred mel frequency bins: %d", cfg.mel_freq_bins)

    collate_fn = make_collate_fn(image_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    logger.info("Prepared dataloader with batch size %d.", cfg.batch_size)

    # Model -----------------------------------------------------------------
    model = _load_model(cfg, device)

    # Embeddings ------------------------------------------------------------
    img_embs, mel_embs, labels = compute_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        use_amp=cfg.mixed_precision,
    )

    if cfg.average_by_id:
        img_embs, labels_img = collapse_by_label(img_embs, labels)
        mel_embs, labels_mel = collapse_by_label(mel_embs, labels)
    else:
        labels_img = labels_mel = labels

    # Retrieval metrics -----------------------------------------------------
    sim_img_to_mel = img_embs @ mel_embs.t()
    sim_mel_to_img = sim_img_to_mel.t()

    recalls_img2mel = recall_at_k(sim_img_to_mel, labels_img, labels_mel)
    recalls_mel2img = recall_at_k(sim_mel_to_img, labels_mel, labels_img)

    logger.info("Image → Mel Recall: %s", recalls_img2mel)
    logger.info("Mel → Image Recall: %s", recalls_mel2img)

    # Pretty print summary
    print("\n=== Faces→Mel Evaluation ===")
    print(f"Image → Mel Recall: {recalls_img2mel}")
    print(f"Mel → Image Recall: {recalls_mel2img}")


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate FacesMelDualEncoder retrieval performance.")
    parser.add_argument("--meta_path", type=str, default="meta_eval.json", help="Evaluation meta JSON.")
    parser.add_argument("--root_dir", type=str, default="dataset_root", help="Dataset root directory.")
    parser.add_argument("--mel_root", type=str, default="", help="Optional directory for mel pickles.")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/dual_encoder_full.pth", help="Model checkpoint.")
    parser.add_argument("--vit_name", type=str, default="", help="Override video backbone name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--frames_per_sample", type=int, default=8, help="Frames per video sample.")
    parser.add_argument("--frame_sample_mode", type=str, default="uniform", help="Frame sampling strategy.")
    parser.add_argument("--mel_freq_bins", type=int, default=40, help="Mel bins (set <=0 to infer).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--average_by_id", action="store_true", help="Average embeddings per identity before metrics.")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision during embedding extraction.")
    args = parser.parse_args()

    return EvalConfig(
        meta_path=args.meta_path,
        root_dir=args.root_dir,
        mel_root=args.mel_root or None,
        ckpt_path=args.ckpt_path,
        vit_name=args.vit_name or None,
        batch_size=args.batch_size,
        frames_per_sample=args.frames_per_sample,
        frame_sample_mode=args.frame_sample_mode,
        mel_freq_bins=args.mel_freq_bins,
        num_workers=args.num_workers,
        mixed_precision=not args.no_amp,
        average_by_id=args.average_by_id,
    )


if __name__ == "__main__":
    evaluate(parse_args())
