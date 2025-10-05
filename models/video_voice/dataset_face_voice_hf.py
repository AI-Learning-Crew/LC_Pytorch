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
from torch.utils.data import Dataset
from PIL import Image
import torchaudio

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
        frame_sample_mode: str = "uniform",  # "uniform" | "random"
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

        self.resamplers = {}  # lazy build by sr

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

    def _load_audio(self, rel_path: str) -> torch.Tensor:
        p = self.root / rel_path
        wav, sr = torchaudio.load(str(p))  # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # mono
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        wav = wav.squeeze(0)  # (T,)

        # random crop (pad if short)
        T = wav.numel()
        N = self.num_audio_samples
        if T >= N:
            start = random.randint(0, T - N)
            clip = wav[start:start+N]
        else:
            clip = torch.zeros(N)
            clip[:T] = wav

        # simple rms norm (keeps amplitude reasonable)
        rms = clip.pow(2).mean().sqrt().clamp_min(1e-8)
        clip = clip / rms
        return clip  # (N,)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        chosen = self._sample_frames(r["faces"])  # list of paths
        images = [self._load_image(fp) for fp in chosen]
        audio = self._load_audio(r["voice"])  # (N,)
        return {
            "images": images,      # list of PIL images (K)
            "audio": audio,        # (N,)
            "pid": r["id"],
            "session": r["session"],
        }

# ---- Collate with HF processors -------------------------------------------

def make_collate_fn(
    image_processor,           # transformers.ViTImageProcessor
    wav2vec2_processor,        # transformers.Wav2Vec2Processor
    average_frames: bool = True,
):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # images: flatten all frames, process via processor, then regroup/average
        all_frames: List[Image.Image] = []
        frame_counts: List[int] = []
        for b in batch:
            frames = b["images"]
            frame_counts.append(len(frames))
            all_frames.extend(frames)

        pixel_inputs = image_processor(images=all_frames, return_tensors="pt")
        pixel_values = pixel_inputs["pixel_values"]  # (sumK, 3, 224, 224)

        # regroup
        chunks = torch.split(pixel_values, frame_counts)
        if average_frames:
            # average frame embeddings later, so keep as images for encoder.
            # Here we average the pixel tensors to keep memory lower (approximation)
            # Alternatively, encode each frame then average embeddings (better).
            # We'll choose **encode-each-then-average** in the model step, so keep stacks.
            pass

        # audio
        audios = [b["audio"] for b in batch]
        audio_inputs = wav2vec2_processor(
            audios, sampling_rate=wav2vec2_processor.feature_extractor.sampling_rate,
            return_tensors="pt", padding=True
        )  # input_values: (B, Tpad) attention_mask: (B, Tpad)

        return {
            "pixel_values_list": list(chunks),     # list of tensors with shape (Ki, 3, H, W)
            "audio_input_values": audio_inputs.input_values,  # (B, T)
            "audio_attention_mask": audio_inputs.attention_mask,  # (B, T)
        }
    return collate