import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from utils.wav2mel import wav_to_mel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="메타 파일을 바탕으로 VAD+Mel 스펙트로그램을 생성합니다.")
    parser.add_argument(
        "--meta_path",
        type=str,
        default=os.environ.get("META", "meta.json"),
        help="메타 JSON 파일 경로 (기본: meta.json)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get("ROOT", "dataset_root"),
        help="데이터셋 루트 디렉토리 (기본: dataset_root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="생성된 Mel 스펙트로그램(.pickle) 저장 경로",
    )
    return parser.parse_args()


def load_meta(meta_path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    print(f"[generate_mel] Loading metadata from {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def iter_voice_entries(
    metadata: Dict[str, Dict[str, Dict[str, object]]]
) -> Iterator[Tuple[str, str, str]]:
    for pid, sessions in metadata.items():
        for session_id, payload in sessions.items():
            voice = payload.get("voice")
            if voice:
                yield pid, session_id, voice


def save_mel_pickle(output_path: Path, log_mel, wav_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_mel": log_mel,
        "wav_path": str(wav_path),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def generate_mel_from_meta(meta_path: Path, root_dir: Path, output_dir: Path) -> None:
    metadata = load_meta(meta_path)
    entries = list(iter_voice_entries(metadata))

    if not entries:
        print("[generate_mel] No voice entries found in metadata.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    print(f"[generate_mel] Found {len(entries)} voice entries. Starting conversion...")
    for idx, (pid, session_id, rel_voice) in enumerate(entries, 1):
        wav_path = root_dir / rel_voice
        print(f"[generate_mel] ({idx}/{len(entries)}) Processing {wav_path}")

        if not wav_path.is_file():
            skipped += 1
            print(f"[generate_mel][WARN] WAV file not found, skipping: {wav_path}")
            continue

        try:
            log_mel = wav_to_mel(wav_path)
        except Exception as exc:
            skipped += 1
            print(f"[generate_mel][ERROR] Failed to convert {wav_path}: {exc}")
            continue

        mel_path = output_dir / pid / "mel" / f"{session_id}.pickle"
        save_mel_pickle(mel_path, log_mel, wav_path)
        print(f"[generate_mel] Saved mel-spectrogram to {mel_path}")
        processed += 1

    print(
        f"[generate_mel] Completed. Processed: {processed}, Skipped: {skipped}, "
        f"Output dir: {output_dir}"
    )


def main() -> int:
    args = parse_args()
    meta_path = Path(args.meta_path)
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    if not meta_path.is_file():
        print(f"[generate_mel][ERROR] meta_path not found: {meta_path}")
        return 1
    if not root_dir.exists():
        print(f"[generate_mel][ERROR] root_dir not found: {root_dir}")
        return 1

    generate_mel_from_meta(meta_path, root_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
