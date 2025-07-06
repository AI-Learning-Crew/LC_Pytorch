"""
매칭된 파일들을 처리하는 유틸리티 함수들

이 모듈은 얼굴-음성 매칭 데이터셋에서 파일들을 처리하고 관리하는 
다양한 유틸리티 함수들을 제공합니다.
"""

from pathlib import Path
import logging
from typing import List, Optional, Tuple
import json

# 로깅 설정 - INFO 레벨 이상의 메시지를 출력
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from typing import List

def load_id_list_from_json(meta_path: str) -> List[str]:
    """
    JSON 파일에서 ID 리스트(key 목록)를 추출하여 반환
    
    Args:
        meta_path (str): JSON 메타데이터 파일의 경로
        
    Returns:
        List[str]: JSON 파일의 모든 키들을 리스트로 반환
                  파일 로드에 실패하면 빈 리스트 반환
    """
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
    """
    주어진 인덱스에 해당하는 face-voice 쌍 경로를 반환
    
    Args:
        face_dir (Path): 얼굴 이미지들이 저장된 디렉토리 경로
        voice_dir (Path): 음성 파일들이 저장된 디렉토리 경로
        idx (int): 가져올 파일 쌍의 인덱스
        
    Returns:
        Optional[Tuple[Path, Path]]: (얼굴 파일 경로, 음성 파일 경로) 튜플
                                   실패 시 None 반환
    """
    try:
        # 얼굴 디렉토리들을 정렬하여 순서대로 처리
        face_dirs = sorted([d for d in face_dir.iterdir() if d.is_dir()])
        # 음성 파일들을 정렬하여 순서대로 처리
        voice_files = sorted([f for f in voice_dir.iterdir() if f.is_file()])

        # 인덱스가 유효한 범위 내에 있는지 확인
        if idx < min(len(face_dirs), len(voice_files)):
            # 기본적으로 frame_0005.jpg 파일을 사용 (5번째 프레임)
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
    """
    주어진 id 목록에 대해 matched 파일을 인덱스별로 저장
    
    이 함수는 각 ID에 대해 동일한 인덱스의 얼굴-음성 쌍을 찾아서
    인덱스별로 별도의 파일에 저장합니다.
    
    Args:
        id_list (List[str]): 처리할 ID들의 리스트
        train_base_dir (Path): 훈련 데이터의 기본 디렉토리 경로
        output_base_dir (Path): 결과 파일들을 저장할 디렉토리 경로
        max_index (int): 처리할 최대 인덱스 수 (기본값: 100)
    """
    # 출력 디렉토리가 없으면 생성
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 각 인덱스에 대해 처리
    for idx in range(max_index):
        matched_files = []

        # 각 ID에 대해 해당 인덱스의 파일 쌍을 찾기
        for id in id_list:
            id_dir = train_base_dir / id
            id_face_dir = id_dir / 'faces'
            id_voices_dir = id_dir / 'voices'

            # 필요한 디렉토리가 존재하는지 확인
            if not id_face_dir.exists() or not id_voices_dir.exists():
                logging.warning(f"Missing directory for ID: {id}")
                continue

            # 해당 인덱스의 얼굴-음성 쌍 가져오기
            pair = get_matched_pair(id_face_dir, id_voices_dir, idx)
            if pair:
                matched_files.append(pair)

        # 인덱스별로 결과 파일 저장
        output_path = output_base_dir / f'matched_files-{idx}.txt'

        try:
            with output_path.open('w') as f:
                # 각 파일 쌍을 탭으로 구분하여 저장
                for face_path, voice_path in matched_files:
                    f.write(f"{face_path}\t{voice_path}\n")
            logging.info(f"Saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save file {output_path}: {e}")