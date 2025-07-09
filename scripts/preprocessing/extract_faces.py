#!/usr/bin/env python3
"""
비디오에서 얼굴 추출 스크립트

이 스크립트는 비디오 파일들에서 얼굴을 자동으로 추출하여 이미지 파일로 저장합니다.
얼굴-음성 매칭 데이터셋 구성을 위한 전처리 단계로 사용됩니다.

사용법:
    python scripts/extract_faces.py --dataset_path /path/to/videos --output_dir /path/to/faces

주요 기능:
- 다양한 얼굴 감지기 백엔드 지원 (RetinaFace, MTCNN, OpenCV 등)
- 얼굴 정렬 기능으로 일관된 얼굴 이미지 생성
- 배치 처리로 대용량 데이터셋 효율적 처리
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가하여 모듈 import 가능하게 함
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.face_extractor import FaceExtractor


def main():
    """
    메인 함수: 명령행 인자를 파싱하고 얼굴 추출 작업을 실행합니다.
    
    Returns:
        int: 성공 시 0, 실패 시 1
    """
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(description='비디오 파일에서 얼굴을 추출합니다.')
    
    # 필수 인자들
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='비디오 파일들이 있는 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='추출된 얼굴 이미지를 저장할 디렉토리')
    
    # 선택적 인자들
    parser.add_argument('--detector_backend', type=str, default='retinaface',
                       choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'],
                       help='얼굴 감지기 백엔드 (기본값: retinaface)')
    parser.add_argument('--align_faces', action='store_true', default=True,
                       help='얼굴 정렬 기능 활성화 (기본값: True)')
    parser.add_argument('--video_extensions', nargs='+', default=['*.mp4'],
                       help='처리할 비디오 파일 확장자 (기본값: *.mp4)')
    
    # 명령행 인자 파싱
    args = parser.parse_args()
    
    # 입력 디렉토리 존재 여부 확인
    if not os.path.exists(args.dataset_path):
        print(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    
    # FaceExtractor 초기화 (설정된 백엔드와 옵션으로)
    extractor = FaceExtractor(
        detector_backend=args.detector_backend,
        align_faces=args.align_faces
    )
    
    # 얼굴 추출 작업 실행
    try:
        # extract_faces_from_videos 메서드를 호출하여 배치 처리
        stats = extractor.extract_faces_from_videos(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            video_extensions=args.video_extensions
        )
        
        # 처리 결과 확인 및 출력
        if stats:
            print(f"\n처리 완료! 성공적으로 {stats['processed_count']}개의 얼굴을 추출했습니다.")
            return 0
        else:
            print("처리 중 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1


if __name__ == '__main__':
    # 스크립트가 직접 실행될 때만 main 함수 호출
    exit(main()) 