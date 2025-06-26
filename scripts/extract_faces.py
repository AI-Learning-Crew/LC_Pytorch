#!/usr/bin/env python3
"""
비디오에서 얼굴 추출 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.face_extractor import FaceExtractor


def main():
    parser = argparse.ArgumentParser(description='비디오 파일에서 얼굴을 추출합니다.')
    
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='비디오 파일들이 있는 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='추출된 얼굴 이미지를 저장할 디렉토리')
    parser.add_argument('--detector_backend', type=str, default='retinaface',
                       choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'],
                       help='얼굴 감지기 백엔드 (기본값: retinaface)')
    parser.add_argument('--align_faces', action='store_true', default=True,
                       help='얼굴 정렬 기능 활성화 (기본값: True)')
    parser.add_argument('--video_extensions', nargs='+', default=['*.mp4'],
                       help='처리할 비디오 파일 확장자 (기본값: *.mp4)')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.dataset_path):
        print(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    
    # FaceExtractor 초기화
    extractor = FaceExtractor(
        detector_backend=args.detector_backend,
        align_faces=args.align_faces
    )
    
    # 얼굴 추출 실행
    try:
        stats = extractor.extract_faces_from_videos(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            video_extensions=args.video_extensions
        )
        
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
    exit(main()) 