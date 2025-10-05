#!/usr/bin/env python3
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------
# 로깅 설정
# ---------------------------
LOGGER = logging.getLogger("metadata_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# 유틸 함수
# ---------------------------
_NATNUM_RE = re.compile(r"(\d+)")

def natural_key(s: str):
    """문자열 내 숫자를 기준으로 자연스러운 정렬 키 생성"""
    return [int(t) if t.isdigit() else t.lower() for t in _NATNUM_RE.split(s)]

def resolve_train_dir(dataset_path: Path) -> Path:
    """
    사용자가 dataset 루트 혹은 train 디렉토리를 줄 수 있으므로 안전하게 train 디렉토리를 결정
    """
    if (dataset_path / "train").is_dir():
        return dataset_path / "train"
    return dataset_path  # 이미 train을 가리키는 경우

def list_subdirs(p: Path) -> List[Path]:
    return sorted([d for d in p.iterdir() if d.is_dir()], key=lambda x: x.name)

def list_files(p: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts_lower = tuple(e.lower() for e in exts)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts_lower]
    # 파일명 기준 자연 정렬
    return sorted(files, key=lambda x: natural_key(x.name))

def save_json(obj, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

# ---------------------------
# 메인 로직
# ---------------------------
def build_metadata(
    root_dir: Path,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    audio_ext: str = ".wav",
    log_progress: bool = True,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    주어진 루트(혹은 train) 디렉토리에서 id/세션별 faces와 voice 상대경로 매핑을 생성
    반환 형식:
    {
        "id00001": {
            "00001": {
                "faces": ["train/id00001/faces/00001/frame_0001.jpg", ...],
                "voice":  "train/id00001/voices/00001.wav"
            },
            ...
        },
        ...
    }
    """
    metadata: Dict[str, Dict[str, Dict[str, object]]] = {}

    train_dir = resolve_train_dir(root_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train 디렉토리를 찾을 수 없습니다: {train_dir}")

    id_dirs = list_subdirs(train_dir)
    total_ids = len(id_dirs)
    if total_ids == 0:
        LOGGER.warning("train 하위에 id 디렉토리가 없습니다.")
        return metadata

    t0 = time.time()

    # 통계용
    num_sessions_total = 0
    num_frames_total = 0
    num_voice_missing = 0
    num_faces_missing = 0

    for idx, id_dir in enumerate(id_dirs, 1):
        id_name = id_dir.name
        t_id = time.time()

        faces_root = id_dir / "faces"
        voices_root = id_dir / "voices"

        if not faces_root.exists() or not voices_root.exists():
            LOGGER.warning(f"⚠️ {id_name}: 'faces' 또는 'voices' 폴더가 없어 건너뜀")
            continue

        metadata[id_name] = {}

        # 세션은 faces 하위의 폴더 이름으로 판단 (예: 00001)
        session_dirs = list_subdirs(faces_root)[:5]  # 최대 5개 세션만 사용
        if not session_dirs:
            LOGGER.warning(f"⚠️ {id_name}: faces 하위 세션 폴더가 없어 건너뜀")
            continue

        for session_dir in session_dirs:
            session_name = session_dir.name

            # faces 프레임들
            face_paths = list_files(session_dir, image_exts)
            if not face_paths:
                num_faces_missing += 1
                LOGGER.warning(f"⚠️ {id_name}/{session_name}: face 이미지가 없습니다.")
                faces_rel: List[str] = []
            else:
                faces_rel = [str(p.relative_to(root_dir)) for p in face_paths]
                num_frames_total += len(faces_rel)

            # voice 파일 (세션명 + 확장자)
            voice_path = voices_root / f"{session_name}{audio_ext}"
            voice_rel: Optional[str] = None
            if voice_path.exists() and voice_path.is_file():
                voice_rel = str(voice_path.relative_to(root_dir))
            else:
                num_voice_missing += 1
                LOGGER.warning(f"⚠️ {id_name}/{session_name}: voice 파일이 없습니다 -> {voice_path.name}")

            metadata[id_name][session_name] = {
                "faces": faces_rel,
                "voice": voice_rel,
            }
            num_sessions_total += 1

        if log_progress:
            dt = time.time() - t_id
            LOGGER.info(f"[{idx}/{total_ids}] {id_name} 처리 완료 (세션 {len(session_dirs)}개, {dt:.2f}s)")

    if log_progress:
        dt_total = time.time() - t0
        LOGGER.info(
            "요약 | IDs: %d, 세션: %d, 총 프레임: %d, 음성 누락 세션: %d, 얼굴 누락 세션: %d, 총 소요: %.2fs",
            len(metadata),
            num_sessions_total,
            num_frames_total,
            num_voice_missing,
            num_faces_missing,
            dt_total,
        )

    return metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="id/세션 단위로 faces와 voice 상대경로를 매핑한 학습용 메타데이터 JSON을 생성합니다."
    )
    parser.add_argument(
        "-d", "--dataset_path", type=Path, required=True,
        help="데이터셋 루트 혹은 train 디렉토리 경로 (둘 다 허용)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="출력 디렉토리 (파일명은 자동으로 matched_pair_set.json)"
    )
    parser.add_argument(
        "--image-exts", type=str, default=".jpg,.jpeg,.png",
        help="얼굴 이미지 확장자 목록 (쉼표 구분, 기본: .jpg,.jpeg,.png)"
    )
    parser.add_argument(
        "--audio-ext", type=str, default=".wav",
        help="음성 파일 확장자 (기본: .wav)"
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    if not args.dataset_path.exists():
        LOGGER.error(f"오류: 입력 경로가 존재하지 않습니다 -> {args.dataset_path}")
        return 1
    if not args.output.exists():
        LOGGER.error(f"오류: 출력 디렉토리가 존재하지 않습니다 -> {args.output}")
        return 1

    LOGGER.info(f"Dataset path: {args.dataset_path}")
    LOGGER.info(f"Output path:   {args.output}")

    image_exts = tuple(
        e.strip() if e.strip().startswith(".") else f".{e.strip()}"
        for e in args.image_exts.split(",")
        if e.strip()
    )
    audio_ext = args.audio_ext if args.audio_ext.startswith(".") else f".{args.audio_ext}"

    start = time.time()
    try:
        metadata = build_metadata(
            root_dir=args.dataset_path,
            image_exts=image_exts,
            audio_ext=audio_ext,
            log_progress=True,
        )
    except Exception as e:
        LOGGER.exception(f"메타데이터 생성 중 예외 발생: {e}")
        return 1

    out_path = args.output / "matched_pair_set.json"
    try:
        save_json(metadata, out_path)
    except Exception as e:
        LOGGER.exception(f"JSON 저장 오류: {e}")
        return 1

    LOGGER.info(f"✅ 완료: {out_path} (총 소요: {time.time() - start:.2f}s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
