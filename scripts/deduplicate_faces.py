#!/usr/bin/env python3
"""
얼굴 중복 제거 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.face_deduplicator import FaceDeduplicator


def main():
    parser = argparse.ArgumentParser(description='얼굴 이미지들의 중복을 제거하고 그룹화합니다.')
    
    parser.add_argument('--faces_dir', type=str, required=True,
                       help='원본 얼굴 이미지가 저장된 디렉토리')
    parser.add_argument('--dedupe_dir', type=str, required=True,
                       help='동일 인물 그룹화 후 복사/저장할 디렉토리')
    parser.add_argument('--representative_dir', type=str, required=True,
                       help='대표 얼굴만 별도 복사할 디렉토리')
    parser.add_argument('--model_name', type=str, default='Facenet',
                       choices=['Facenet', 'VGG-Face', 'OpenFace', 'DeepID', 'ArcFace', 'SFace'],
                       help='얼굴 임베딩에 사용할 모델 (기본값: Facenet)')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='동일 인물로 판단할 코사인 거리 임계값 (기본값: 0.4)')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.faces_dir):
        print(f"오류: 입력 디렉토리 '{args.faces_dir}'가 존재하지 않습니다.")
        return 1
    
    # FaceDeduplicator 초기화
    deduplicator = FaceDeduplicator(
        model_name=args.model_name,
        threshold=args.threshold
    )
    
    # 얼굴 중복 제거 실행
    try:
        stats = deduplicator.deduplicate_faces(
            faces_dir=args.faces_dir,
            dedupe_dir=args.dedupe_dir,
            representative_dir=args.representative_dir
        )
        
        if stats:
            print(f"\n처리 완료!")
            print(f"- 총 {stats['copied_files_to_dedupe_count']}개의 파일이 dedupe 디렉토리로 복사되었습니다.")
            print(f"- 총 {stats['copied_files_to_representative_count']}개의 대표 파일이 representative 디렉토리로 복사되었습니다.")
            return 0
        else:
            print("처리 중 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 