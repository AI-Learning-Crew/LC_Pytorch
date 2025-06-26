#!/usr/bin/env python3
"""
얼굴 추출 및 중복 제거 사용 예제
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.face_extractor import FaceExtractor
from utils.face_deduplicator import FaceDeduplicator


def example_face_extraction():
    """얼굴 추출 예제"""
    print("=== 얼굴 추출 예제 ===")
    
    # 설정
    dataset_path = "/path/to/your/video/dataset"  # 실제 경로로 변경하세요
    output_dir = "/path/to/output/extracted_faces"  # 실제 경로로 변경하세요
    
    # FaceExtractor 초기화
    extractor = FaceExtractor(
        detector_backend='retinaface',  # 얼굴 감지기 백엔드
        align_faces=True  # 얼굴 정렬 활성화
    )
    
    # 얼굴 추출 실행
    stats = extractor.extract_faces_from_videos(
        dataset_path=dataset_path,
        output_dir=output_dir,
        video_extensions=['*.mp4', '*.avi']  # 처리할 비디오 확장자
    )
    
    print(f"추출 완료: {stats['processed_count']}개의 얼굴을 추출했습니다.")
    return output_dir


def example_face_deduplication(faces_dir):
    """얼굴 중복 제거 예제"""
    print("\n=== 얼굴 중복 제거 예제 ===")
    
    # 설정
    dedupe_dir = "/path/to/output/deduped_faces"  # 실제 경로로 변경하세요
    representative_dir = "/path/to/output/representative_faces"  # 실제 경로로 변경하세요
    
    # FaceDeduplicator 초기화
    deduplicator = FaceDeduplicator(
        model_name='Facenet',  # 얼굴 임베딩 모델
        threshold=0.4  # 동일 인물 판단 임계값
    )
    
    # 중복 제거 실행
    stats = deduplicator.deduplicate_faces(
        faces_dir=faces_dir,
        dedupe_dir=dedupe_dir,
        representative_dir=representative_dir
    )
    
    print(f"중복 제거 완료!")
    print(f"- 총 {stats['copied_files_to_dedupe_count']}개의 파일이 dedupe 디렉토리로 복사되었습니다.")
    print(f"- 총 {stats['copied_files_to_representative_count']}개의 대표 파일이 representative 디렉토리로 복사되었습니다.")


def example_google_colab_usage():
    """Google Colab에서 사용하는 예제"""
    print("\n=== Google Colab 사용 예제 ===")
    
    # Google Drive 마운트 (Colab에서만 실행)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive가 마운트되었습니다.")
    except ImportError:
        print("Google Colab 환경이 아닙니다.")
        return
    
    # 설정
    dataset_path = "/content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k"
    output_base_dir = "/content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k_processed"
    
    # 1단계: 얼굴 추출
    extracted_faces_dir = os.path.join(output_base_dir, 'extracted_faces')
    extractor = FaceExtractor(detector_backend='retinaface')
    
    extraction_stats = extractor.extract_faces_from_videos(
        dataset_path=dataset_path,
        output_dir=extracted_faces_dir
    )
    
    # 2단계: 얼굴 중복 제거
    dedupe_dir = os.path.join(output_base_dir, 'deduped_faces')
    representative_dir = os.path.join(output_base_dir, 'representative_faces')
    
    deduplicator = FaceDeduplicator(model_name='Facenet', threshold=0.4)
    
    deduplication_stats = deduplicator.deduplicate_faces(
        faces_dir=extracted_faces_dir,
        dedupe_dir=dedupe_dir,
        representative_dir=representative_dir
    )
    
    print("Google Colab 워크플로우 완료!")


def main():
    """메인 함수"""
    print("LC_PyTorch 얼굴 추출 및 중복 제거 예제")
    print("=" * 50)
    
    # 예제 1: 기본 사용법
    print("\n1. 기본 사용법 예제")
    print("주석을 해제하고 실제 경로를 설정한 후 실행하세요.")
    
    # example_face_extraction()
    # faces_dir = "/path/to/output/extracted_faces"
    # example_face_deduplication(faces_dir)
    
    # 예제 2: Google Colab 사용법
    print("\n2. Google Colab 사용법 예제")
    print("Google Colab 환경에서 실행하세요.")
    
    # example_google_colab_usage()
    
    print("\n예제 실행을 위해서는 주석을 해제하고 실제 경로를 설정하세요.")


if __name__ == '__main__':
    main() 