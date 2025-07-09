#!/usr/bin/env python3
"""
얼굴 중복 제거 스크립트

이 스크립트는 추출된 얼굴 이미지들 중에서 중복된 인물의 얼굴을 제거하고 
동일한 인물끼리 그룹화하여 대표 얼굴을 선택합니다.

주요 기능:
- 얼굴 임베딩을 사용한 유사도 계산
- 임계값 기반 중복 인물 그룹화
- 각 그룹에서 대표 얼굴 선택
- 그룹화된 결과와 대표 얼굴을 별도 디렉토리에 저장

사용법:
    python scripts/deduplicate_faces.py --faces_dir /path/to/faces --dedupe_dir /path/to/deduped --representative_dir /path/to/representative
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가하여 모듈 import 가능하게 함
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.face_deduplicator import FaceDeduplicator


def main():
    """
    메인 함수: 명령행 인자를 파싱하고 얼굴 중복 제거 작업을 실행합니다.
    
    Returns:
        int: 성공 시 0, 실패 시 1
    """
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(description='얼굴 이미지들의 중복을 제거하고 그룹화합니다.')
    
    # 필수 인자들
    parser.add_argument('--faces_dir', type=str, required=True,
                       help='원본 얼굴 이미지가 저장된 디렉토리')
    parser.add_argument('--dedupe_dir', type=str, required=True,
                       help='동일 인물 그룹화 후 복사/저장할 디렉토리')
    parser.add_argument('--representative_dir', type=str, required=True,
                       help='대표 얼굴만 별도 복사할 디렉토리')
    
    # 선택적 인자들
    parser.add_argument('--model_name', type=str, default='Facenet',
                       choices=['Facenet', 'VGG-Face', 'OpenFace', 'DeepID', 'ArcFace', 'SFace'],
                       help='얼굴 임베딩에 사용할 모델 (기본값: Facenet)')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='동일 인물로 판단할 코사인 거리 임계값 (기본값: 0.4, 낮을수록 엄격)')
    
    # 명령행 인자 파싱
    args = parser.parse_args()
    
    # 입력 디렉토리 존재 여부 확인
    if not os.path.exists(args.faces_dir):
        print(f"오류: 입력 디렉토리 '{args.faces_dir}'가 존재하지 않습니다.")
        return 1
    
    # FaceDeduplicator 초기화 (설정된 모델과 임계값으로)
    deduplicator = FaceDeduplicator(
        model_name=args.model_name,
        threshold=args.threshold
    )
    
    # 얼굴 중복 제거 작업 실행
    try:
        # deduplicate_faces 메서드를 호출하여 배치 처리
        stats = deduplicator.deduplicate_faces(
            faces_dir=args.faces_dir,
            dedupe_dir=args.dedupe_dir,
            representative_dir=args.representative_dir
        )
        
        # 처리 결과 확인 및 출력
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
    # 스크립트가 직접 실행될 때만 main 함수 호출
    exit(main()) 