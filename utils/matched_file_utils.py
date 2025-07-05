from pathlib import Path
import logging
from typing import List, Optional, Tuple
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from typing import List

def load_id_list_from_json(meta_path: str) -> List[str]:
    """JSON 파일에서 ID 리스트(key 목록)를 추출하여 반환"""
    try:
        with open(meta_path, 'r') as f:
            data = json.load(f)
        id_list = list(data.keys())
        logging.info(f"Loaded {len(id_list)} IDs from {meta_path}")
        return id_list
    except Exception as e:
        logging.info(f"Error loading ID list from {meta_path}: {e}")
        return []


def get_matched_pair(face_dir: Path, voice_dir: Path, idx: int) -> Optional[Tuple[Path, Path]]:
    """주어진 인덱스에 해당하는 face-voice 쌍 경로를 반환"""
    try:
        face_dirs = sorted([d for d in face_dir.iterdir() if d.is_dir()])
        voice_files = sorted([f for f in voice_dir.iterdir() if f.is_file()])

        if idx < min(len(face_dirs), len(voice_files)):
            face_path = face_dirs[idx] / 'frame_0005.jpg'
            voice_path = voice_files[idx]
            return face_path, voice_path
    except Exception as e:
        logging.warning(f"Error matching face/voice at idx {idx}: {e}")
    return None

def save_matched_files_by_index(
    id_list: List[str],
    train_base_dir: Path,
    output_base_dir: Path,
    max_index: int = 100
):
    """주어진 id 목록에 대해 matched 파일을 인덱스별로 저장"""
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(max_index):
        matched_files = []

        for id in id_list:
            id_dir = train_base_dir / id
            id_face_dir = id_dir / 'faces'
            id_voices_dir = id_dir / 'voices'

            if not id_face_dir.exists() or not id_voices_dir.exists():
                logging.warning(f"Missing directory for ID: {id}")
                continue

            pair = get_matched_pair(id_face_dir, id_voices_dir, idx)
            if pair:
                matched_files.append(pair)

        output_path = output_base_dir / f'matched_files-{idx}.txt'

        try:
            with output_path.open('w') as f:
                for face_path, voice_path in matched_files:
                    f.write(f"{face_path}\t{voice_path}\n")
            logging.info(f"Saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save file {output_path}: {e}")