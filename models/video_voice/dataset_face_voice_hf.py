# =============================================
# File: dataset_face_voice_hf.py
# Description: Dataset & collate for ViT (images) + Wav2Vec2 (audio)
# Requirements: transformers, torch, torchaudio, pillow
# =============================================
from __future__ import annotations
from pathlib import Path
import json, random
from typing import List, Dict, Any, Tuple

import numpy as np             # ⬅️ 추가
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image
import librosa

from transformers import Wav2Vec2Processor


# ---- Meta helpers -----------------------------------------------------------

def flatten_meta(meta_obj):
    """
    meta_obj: dict({"idXXXX": {...}}) 또는 list[dict({...})] 모두 지원.
    빈 faces/누락 voice는 스킵.
    """
    # 형태 정규화
    if isinstance(meta_obj, dict):
        items = meta_obj.items()
    elif isinstance(meta_obj, list):
        merged = {}
        for block in meta_obj:
            if isinstance(block, dict):
                for k, v in block.items():
                    merged[k] = v
        items = merged.items()
    else:
        raise TypeError(f"Unsupported meta type: {type(meta_obj)}")

    records = []
    for pid, sessions in items:
        for sess, d in sessions.items():
            faces = d.get("faces", []) or []
            voice = d.get("voice", None)
            # 빈 faces 또는 voice 누락은 스킵
            if not faces or not voice:
                continue
            records.append({
                "id": pid,
                "session": sess,
                "faces": faces,
                "voice": voice,
            })
    if not records:
        raise ValueError("No valid records found in meta (check faces/voice fields).")
    return records


# ---- Dataset ---------------------------------------------------------------

class FaceVoiceDatasetHF(Dataset):
    def __init__(
        self,
        meta_path: str | Path,
        root_dir: str | Path,
        frames_per_sample: int = 4,
        frame_sample_mode: str = "uniform",
        audio_processor: Wav2Vec2Processor | None = None,
        audio_seconds: float = 2.0,
        target_sr: int = 16000,
        average_frames: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.records = flatten_meta(self.meta)

        self.frames_per_sample = max(1, int(frames_per_sample))
        self.frame_sample_mode = frame_sample_mode
        self.audio_seconds = float(audio_seconds)
        self.target_sr = int(target_sr)
        self.num_audio_samples = int(self.target_sr * self.audio_seconds)
        self.average_frames = average_frames

        # 프로세서: 전달받지 않으면 여기서 한 번만 생성
        self.audio_processor = audio_processor or Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def __len__(self):
        return len(self.records)

    def _sample_frames(self, face_list: List[str]) -> List[str]:
        n = len(face_list)
        k = min(self.frames_per_sample, n)
        if self.frame_sample_mode == "uniform" and k > 1:
            idxs = [int(round(i*(n-1)/(k-1))) for i in range(k)]
        else:
            idxs = sorted(random.sample(range(n), k)) if k < n else list(range(n))
        return [face_list[i] for i in idxs]

    def _load_image(self, rel_path: str) -> Image.Image:
        p = self.root / rel_path
        img = Image.open(p).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        chosen = self._sample_frames(r["faces"])
        images = [self._load_image(fp) for fp in chosen]

        audio_path = self.root / r["voice"]
        # librosa가 float32 mono로 반환, sr로 리샘플링 완료
        speech_array, _ = librosa.load(audio_path, sr=self.target_sr)

        # 길이 고정
        if len(speech_array) >= self.num_audio_samples:
            speech_array = speech_array[: self.num_audio_samples]
        else:
            pad = self.num_audio_samples - len(speech_array)
            speech_array = np.pad(speech_array, (0, pad), mode="constant")

        audio_input = self.audio_processor(
            speech_array, sampling_rate=self.target_sr, return_tensors="pt"
        ).input_values.squeeze(0)  # (T,)

        return {
            "images": images,        # list[PIL.Image] (K)
            "audio": audio_input,    # (T,)
            "pid": r["id"],
            "session": r["session"],
        }

# ---- Collate with HF processors -------------------------------------------
    
def make_collate_fn(image_processor, average_frames: bool = True):
    def collate(batch):
        # ----- images -----
        all_frames = []
        frame_counts = []
        for b in batch:
            frames = b["images"]
            frame_counts.append(len(frames))
            all_frames.extend(frames)

        if sum(frame_counts) == 0:
            raise RuntimeError("All items have zero frames in this batch.")

        px = image_processor(images=all_frames, return_tensors="pt")
        pixel_values = px["pixel_values"]                 # (sumK, 3, H, W)
        chunks = torch.split(pixel_values, frame_counts)  # list[(Ki,3,H,W)]

        if average_frames:
            # Ki가 0인 경우를 방어 (이상치 스킵 또는 제로패드)
            pooled = []
            for c in chunks:
                if c.shape[0] == 0:
                    # 제로로 대체 (드문 케이스)
                    pooled.append(torch.zeros((1, *c.shape[1:]), dtype=c.dtype, device=c.device))
                else:
                    pooled.append(c.mean(dim=0, keepdim=True))
            videos = torch.cat(pooled, dim=0)  # (B,3,H,W)
        else:
            max_k = max((c.shape[0] for c in chunks), default=0)
            if max_k == 0:
                raise RuntimeError("No frames to pad in non-averaging path.")
            padded_list = []
            for c in chunks:
                if c.shape[0] < max_k:
                    pad = torch.zeros((max_k - c.shape[0], *c.shape[1:]),
                                      dtype=c.dtype, device=c.device)
                    c = torch.cat([c, pad], dim=0)
                padded_list.append(c)
            videos = torch.stack(padded_list, dim=0)  # (B,K,3,H,W)

        # ----- audios -----
        audios = [b["audio"] for b in batch]  # list[(T_i,)]
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)  # (B, Tmax)

        meta = [{"pid": b["pid"], "session": b["session"]} for b in batch]
        return {"images": videos, "audio": padded_audios, "meta": meta}
    return collate


