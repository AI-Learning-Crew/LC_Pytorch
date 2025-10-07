# =============================================
# File: dataset_face_voice_hf.py
# Description: Dataset & collate for ViT (images) + Wav2Vec2 (audio)
# Requirements: transformers, torch, torchaudio, pillow
# =============================================
from __future__ import annotations
from pathlib import Path
import json, random
from typing import List, Dict, Any, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image
import librosa

from transformers import Wav2Vec2Processor

# ---- Meta helpers -----------------------------------------------------------

def flatten_meta(meta_obj: List[Dict[str, Any]]):
    """
    Convert a nested meta structure into a flat list of sample records.

    This function takes a list of dictionaries, where each dictionary maps an identity ID
    (e.g., "id00019") to its session data. Each session contains multiple face frame paths
    and a corresponding voice file path.

    Example Input:
    [
        {
            "id00019": {
                "00001": {
                    "faces": ["id00019/faces/00001/frame_0000.jpg", ...],
                    "voice": "id00019/voices/00001.wav"
                },
                "00002": {
                    ...
                }
            }
        },
        {
            "id00020": {
                ...
            }
        }
    ]

    Example Output:
    [
        {"id": "id00019", "session": "00001", "faces": [...], "voice": "..."},
        {"id": "id00019", "session": "00002", "faces": [...], "voice": "..."},
        {"id": "id00020", "session": "00001", "faces": [...], "voice": "..."},
        ...
    ]

    Returns:
        List[Dict[str, Any]]: a flattened list where each element contains
        the fields {id, session, faces, voice}.
    """
    
    meta_obj = [{k: v} for k, v in meta_obj.items()]  # ensure list of dicts
    records = []
    for id_block in meta_obj:
        for pid, sessions in id_block.items():
            for sess, d in sessions.items():
                faces = d["faces"]
                voice = d["voice"]
                records.append({
                    "id": pid,
                    "session": sess,
                    "faces": faces,
                    "voice": voice,
                })
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

    def _get_resampler(self, sr: int):
        if sr == self.target_sr:
            return None
        if sr not in self.resamplers:
            self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
        return self.resamplers[sr]

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
    
import torch
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(image_processor, average_frames: bool = True):
    def collate(batch):
        # ----- images -----
        all_frames = []
        frame_counts = []
        for b in batch:
            frames = b["images"]
            frame_counts.append(len(frames))
            all_frames.extend(frames)

        px = image_processor(images=all_frames, return_tensors="pt")
        pixel_values = px["pixel_values"]                 # (sumK, 3, H, W)
        chunks = torch.split(pixel_values, frame_counts)  # list[(Ki,3,H,W)]

        if average_frames:
            # (Ki,3,H,W) -> (1,3,H,W) 평균, B개 concat → (B,3,H,W)
            videos = torch.cat([c.mean(dim=0, keepdim=True) for c in chunks], dim=0)
        else:
            # K를 맞춰 패딩 → (B,K,3,H,W)
            max_k = max(c.shape[0] for c in chunks)
            padded_list = []
            for c in chunks:
                if c.shape[0] < max_k:
                    pad = torch.zeros((max_k - c.shape[0], *c.shape[1:]), dtype=c.dtype, device=c.device)
                    c = torch.cat([c, pad], dim=0)
                padded_list.append(c)
            videos = torch.stack(padded_list, dim=0)

        # ----- audios -----
        audios = [b["audio"] for b in batch]             # (T_i,)
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)  # (B, Tmax)

        meta = [{"pid": b["pid"], "session": b["session"]} for b in batch]

        return {"images": videos, "audio": padded_audios, "meta": meta}
    return collate

