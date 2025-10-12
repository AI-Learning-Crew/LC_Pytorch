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
            record = {
                "id": pid,
                "session": sess,
                "faces": faces,
                "voice": voice,
            }
            records.append(record)
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
    ) -> None:
        self.root = Path(root_dir)
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.records = flatten_meta(self.meta)

        self.frames_per_sample = max(1, int(frames_per_sample))
        self.frame_sample_mode = frame_sample_mode
        self.audio_seconds = float(audio_seconds)
        self.target_sr = int(target_sr)
        self.num_audio_samples = int(self.target_sr * self.audio_seconds)

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
        try:
            return Image.open(p).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            print(f"[FaceVoiceDatasetHF] Failed to load image {p}: {exc}")
            return None

    def _get_item(self, idx: int, depth: int = 0) -> Dict[str, Any]:
        if depth >= len(self.records):
            raise RuntimeError("Exceeded max retries while attempting to fetch a valid sample.")

        r = self.records[idx]
        chosen = self._sample_frames(r["faces"])
        images = []
        for fp in chosen:
            img = self._load_image(fp)
            if img is not None:
                images.append(img)

        if not images:
            print(f"[FaceVoiceDatasetHF] No valid frames for pid={r['id']} session={r['session']}. Retrying with next sample.")
            return self._get_item((idx + 1) % len(self.records), depth + 1)

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._get_item(idx)

# ---- Collate with HF processors -------------------------------------------

def make_collate_fn(image_processor):
    def collate(batch):
        # ----- images -----
        clips = []
        frame_counts = []
        for b in batch:
            frames = b["images"]
            if len(frames) == 0:
                clips.append(torch.empty(0))
                frame_counts.append(0)
                continue

            processed = image_processor(images=frames, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            # 일부 프로세서는 (1, T, 3, H, W) 형태로 반환
            # 1: 현재 텐서에 포함된 비디오 클립 수 (batch dimension).
            # T: 비디오의 프레임 개수(시간축).
            # 3: 채널 수(RGB).
            # H, W: 각 프레임의 높이·너비(이미지 공간 해상도).
            if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)  # (T,3,H,W)
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)  # (1,3,H,W)
            clips.append(pixel_values)
            frame_counts.append(pixel_values.shape[0])

        if all(fc == 0 for fc in frame_counts):
            raise RuntimeError("All items have zero frames in this batch.")

        ref_shape = None
        ref_dtype = None
        ref_device = None
        for clip in clips:
            if clip.ndim >= 3:
                ref_shape = clip.shape[1:]
                ref_dtype = clip.dtype
                ref_device = clip.device
                break

        if ref_shape is None:
            raise RuntimeError("No clips with valid frames found in batch.")

        max_k = max(frame_counts)
        if max_k == 0:
            raise RuntimeError("No frames to pad in non-averaging path.")
        padded_list = []
        for clip in clips:
            if clip.ndim <= 1 or clip.shape[0] == 0:
                padded = torch.zeros((max_k, *ref_shape), dtype=ref_dtype, device=ref_device)
            else:
                if clip.shape[0] < max_k:
                    pad = torch.zeros((max_k - clip.shape[0], *clip.shape[1:]),
                                      dtype=clip.dtype, device=clip.device)
                    padded = torch.cat([clip, pad], dim=0)
                else:
                    padded = clip[:max_k]
            padded_list.append(padded)
        videos = torch.stack(padded_list, dim=0)  # (B,K,3,H,W)

        # ----- audios -----
        audios = [b["audio"] for b in batch]  # list[(T_i,)]
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)  # (B, Tmax)

        meta = [{"pid": b["pid"], "session": b["session"]} for b in batch]
        return {"images": videos, "audio": padded_audios, "meta": meta}
    return collate
