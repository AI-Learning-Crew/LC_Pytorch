# -*- coding: utf-8 -*-
"""
Given a VoxCeleb-style face track text (e.g., 00001.txt), this script:
1) Parses Identity / Reference / Offset and per-frame bounding boxes.
2) Downloads the YouTube video by Reference (e.g., 5r0dWxy17C8) via yt-dlp.
3) Extracts the frame range [min_frame, max_frame] as a clip (mp4).
4) Saves face crops per frame using (X, Y, W, H).
5) Optionally builds a "cropped-track" video from the face crops.

Tested in Colab and local Python 3.9+.
"""

import os
import re
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

import argparse

# Install deps if missing (for Colab/local convenience)
def _pip_install(pkgs):
    import importlib
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p.split("==")[0].split("[")[0])
        except Exception:
            missing.append(p)
    if missing:
        subprocess.run(["python", "-m", "pip", "install", "-q", *missing], check=True)

_pip_install(["yt_dlp", "opencv-python"])

import cv2
from yt_dlp import YoutubeDL


# -----------------------------
# Parsing 00001.txt
# -----------------------------
def parse_track_txt(txt_path: str) -> Dict[str, Any]:
    """
    Parse a track file like the one you provided.
    Returns:
      {
        'identity': 'id10270',
        'reference': '5r0dWxy17C8',
        'offset': -2,
        'frames': [ (frame_idx, x, y, w, h), ... ]
      }
    """
    identity = None
    reference = None
    offset = 0
    frames: List[Tuple[int, int, int, int, int]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract header fields
    for i, line in enumerate(lines):
        if line.startswith("Identity"):
            identity = line.split(":")[-1].strip()
        elif line.startswith("Reference"):
            reference = line.split(":")[-1].strip()
        elif line.startswith("Offset"):
            try:
                offset = int(line.split(":")[-1].strip())
            except ValueError:
                try:
                    offset = float(line.split(":")[-1].strip())
                except Exception:
                    offset = 0

    # Find frame table start
    try:
        header_idx = lines.index("FRAME \tX \tY \tW \tH")
    except ValueError:
        # sometimes spacing differs; try a loose match
        header_idx = None
        for i, line in enumerate(lines):
            if re.match(r"^FRAME\s+X\s+Y\s+W\s+H$", re.sub(r"\t+", " ", line)):
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("Could not find 'FRAME  X  Y  W  H' header in the txt file.")

    # Parse subsequent rows as frame entries
    for line in lines[header_idx + 1:]:
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue
        # First column might be zero-padded frame index like "000594"
        frame_str = parts[0]
        if not frame_str.isdigit():
            # tolerate trailing brackets or other punctuation
            frame_str = re.sub(r"\D", "", frame_str)
        if not frame_str:
            continue
        frame_idx = int(frame_str)

        try:
            x, y, w, h = map(int, parts[1:5])
        except Exception:
            # if any parsing issue occurs, skip
            continue

        frames.append((frame_idx, x, y, w, h))

    if identity is None or reference is None or len(frames) == 0:
        raise RuntimeError("Parsing failed: missing identity/reference or empty frames.")

    return {
        "identity": identity,
        "reference": reference,
        "offset": offset,
        "frames": frames,
    }


# -----------------------------
# Download YouTube video
# -----------------------------
def download_youtube(reference: str, out_dir: str) -> str:
    """
    Downloads the YouTube video with given ID using yt-dlp.
    Returns the local MP4 path.
    """
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={reference}"
    outtmpl = str(Path(out_dir) / f"{reference}.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",              # best video+audio; fallback best
        "merge_output_format": "mp4",      # ensure mp4 after merge
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file (mp4)
    mp4_path = Path(out_dir) / f"{reference}.mp4"
    if not mp4_path.exists():
        # Sometimes extension differs; try to find any file beginning with reference.
        candidates = list(Path(out_dir).glob(f"{reference}.*"))
        if not candidates:
            raise FileNotFoundError("Video download failed: no output file found.")
        # Convert to mp4 via ffmpeg if needed
        if candidates[0].suffix.lower() != ".mp4":
            tmp_mp4 = Path(out_dir) / f"{reference}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(candidates[0]),
                "-c:v", "copy", "-c:a", "copy", str(tmp_mp4)
            ], check=True)
            return str(tmp_mp4)
        return str(candidates[0])
    return str(mp4_path)


# -----------------------------
# Frame utilities
# -----------------------------
def get_video_props(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h


def frame_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / max(fps, 1e-6)


# -----------------------------
# Extract clip and save crops
# -----------------------------
def extract_clip_ffmpeg(video_path: str, start_sec: float, end_sec: float, out_path: str):
    """
    Cuts [start_sec, end_sec] (inclusive) using ffmpeg without re-encoding when possible.
    """
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        out_path
    ]
    # If copy fails due to keyframe issues, fall back to re-encode
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]
        subprocess.run(cmd, check=True)


def save_face_crops(video_path: str,
                    frames: List[Tuple[int,int,int,int,int]],
                    out_dir: str,
                    pad: int = 0,
                    make_cropped_video: bool = True):
    """
    Saves face crops per frame, and (optionally) builds a cropped-track video.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Determine maximum crop size for a stable cropped video canvas
    max_w = max(w for _,_,_,w,_ in frames)
    max_h = max(h for _,_,_,_,h in frames)

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if make_cropped_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        crop_vid_path = str(Path(out_dir) / "cropped_track.mp4")
        writer = cv2.VideoWriter(crop_vid_path, fourcc, fps, (max_w, max_h))

    # CSV for bookkeeping
    csv_path = Path(out_dir) / "crops_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        cw = csv.writer(cf)
        cw.writerow(["frame", "t_sec", "x", "y", "w", "h", "crop_path"])

        for (fidx, x, y, w, h) in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {fidx}")
                continue

            H, W, _ = frame.shape
            # padding & clamp
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)

            crop = frame[y0:y1, x0:x1].copy()
            # Save
            crop_path = Path(out_dir) / f"crop_{fidx:06d}.jpg"
            cv2.imwrite(str(crop_path), crop)

            # Write to cropped video on a fixed canvas (letterbox if needed)
            if writer is not None:
                canvas = cv2.copyMakeBorder(
                    crop,
                    top=0, bottom=max(0, max_h - crop.shape[0]),
                    left=0, right=max(0, max_w - crop.shape[1]),
                    borderType=cv2.BORDER_CONSTANT, value=(0,0,0)
                )
                # If crop is bigger than max (shouldn't happen), resize down
                canvas = cv2.resize(canvas, (max_w, max_h))
                writer.write(canvas)

            t = frame_to_time(fidx, fps)
            cw.writerow([fidx, f"{t:.3f}", x, y, w, h, str(crop_path)])

    cap.release()
    if writer is not None:
        writer.release()


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
    txt_path: str,
    work_dir: str = "./vox_download",
    pad: int = 0,
    clip_margin_frames: int = 5,
    also_save_raw_clip: bool = True
):
    meta = parse_track_txt(txt_path)
    identity = meta["identity"]
    reference = meta["reference"]
    offset = meta["offset"]
    frames = sorted(meta["frames"], key=lambda t: t[0])

    print(f"[INFO] Identity={identity}  Reference={reference}  Offset={offset}  #frames={len(frames)}")

    # 1) Download video
    vid_dir = str(Path(work_dir) / identity / reference)
    os.makedirs(vid_dir, exist_ok=True)
    video_path = download_youtube(reference, vid_dir)
    print(f"[INFO] Downloaded: {video_path}")

    # 2) Determine clip time range from frame indices
    fps, total, W, H = get_video_props(video_path)
    fmin = max(0, frames[0][0] - clip_margin_frames)
    fmax = min(total - 1, frames[-1][0] + clip_margin_frames)
    t0 = frame_to_time(fmin, fps)
    t1 = frame_to_time(fmax + 1, fps)  # end-exclusive

    # Apply offset (if you want to time-shift the *video* to match audio labels)
    # Here: shift the start/end by 'offset' seconds (negative -> move earlier).
    # If offset refers only to A/V alignment in your training, you can set this to 0.
    t0_shifted = max(0.0, t0 + float(offset))
    t1_shifted = max(t0_shifted + 0.01, t1 + float(offset))

    print(f"[INFO] FPS={fps:.3f} total_frames={total} size=({W}x{H})")
    print(f"[INFO] Clip frames: {fmin}..{fmax}  →  times: {t0:.3f}s..{t1:.3f}s  (offset applied: {t0_shifted:.3f}s..{t1_shifted:.3f}s)")

    # 3) Extract raw clip (optional)
    if also_save_raw_clip:
        raw_clip_path = str(Path(vid_dir) / "track_clip.mp4")
        extract_clip_ffmpeg(video_path, t0_shifted, t1_shifted, raw_clip_path)
        print(f"[INFO] Saved raw track clip → {raw_clip_path}")

    # 4) Save per-frame face crops (+ cropped-track video)
    crops_dir = str(Path(vid_dir) / "crops")
    save_face_crops(video_path, frames, crops_dir, pad=pad, make_cropped_video=True)
    print(f"[INFO] Saved crops → {crops_dir} (with crops_index.csv & cropped_track.mp4)")

    print("\n[DONE] ✅")


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video/crops from VoxCeleb-style track txt.")
    parser.add_argument("--txt_path", type=str, required=True, help="Path to track txt (e.g., 00001.txt)")
    parser.add_argument("--work_dir", type=str, default="/drive/MyDrive/VoxCeleb1", help="Output base directory")
    parser.add_argument("--pad", type=int, default=0, help="Padding (pixels) around face bbox for crops")
    parser.add_argument("--clip_margin_frames", type=int, default=5, help="Extra frames before/after for raw clip")
    parser.add_argument("--also_save_raw_clip", action="store_true", help="Save raw region clip as track_clip.mp4")
    parser.add_argument("--no_apply_offset", action="store_true", help="Do NOT apply offset to clip times")
    args = parser.parse_args()
    
    # Place your '00001.txt' in the current directory or give an absolute path.
    # Adjust work_dir as you like (e.g., a mounted drive in Colab).
    run_pipeline(
        txt_path=args.txt_path,
        work_dir="./vox_videos",
        pad=0,                    # add padding (pixels) around the face box if needed
        clip_margin_frames=5,     # add a few frames before/after the track
        also_save_raw_clip=True   # save original-region clip (not just crops)
    )
