#!/usr/bin/env python3
"""
얼굴 추출 및 중복 제거 전체 워크플로우 실행 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.face_extractor import FaceExtractor
from utils.face_deduplicator import FaceDeduplicator


def main():
    parser = argparse.ArgumentParser(description='비디오에서 얼굴 추출 및 중복 제거 전체 워크플로우를 실행합니다.')
    
    # 기본 경로 설정
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='비디오 파일들이 있는 디렉토리 경로')
    parser.add_argument('--output_base_dir', type=str, required=True,
                       help='출력 파일들을 저장할 기본 디렉토리')
    
    # 얼굴 추출 설정
    parser.add_argument('--detector_backend', type=str, default='retinaface',
                       choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'],
                       help='얼굴 감지기 백엔드 (기본값: retinaface)')
    parser.add_argument('--align_faces', action='store_true', default=True,
                       help='얼굴 정렬 기능 활성화 (기본값: True)')
    parser.add_argument('--video_extensions', nargs='+', default=['*.mp4'],
                       help='처리할 비디오 파일 확장자 (기본값: *.mp4)')
    
    # 중복 제거 설정
    parser.add_argument('--model_name', type=str, default='Facenet',
                       choices=['Facenet', 'VGG-Face', 'OpenFace', 'DeepID', 'ArcFace', 'SFace'],
                       help='얼굴 임베딩에 사용할 모델 (기본값: Facenet)')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='동일 인물로 판단할 코사인 거리 임계값 (기본값: 0.4)')
    
    # 단계별 실행 옵션
    parser.add_argument('--skip_extraction', action='store_true',
                       help='얼굴 추출 단계를 건너뜁니다')
    parser.add_argument('--skip_deduplication', action='store_true',
                       help='중복 제거 단계를 건너뜁니다')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    extracted_faces_dir = os.path.join(args.output_base_dir, 'extracted_faces')
    dedupe_dir = os.path.join(args.output_base_dir, 'deduped_faces')
    representative_dir = os.path.join(args.output_base_dir, 'representative_faces')
    
    print("=== 얼굴 추출 및 중복 제거 워크플로우 시작 ===")
    print(f"입력 디렉토리: {args.dataset_path}")
    print(f"출력 기본 디렉토리: {args.output_base_dir}")
    print(f"추출된 얼굴 디렉토리: {extracted_faces_dir}")
    print(f"중복 제거 디렉토리: {dedupe_dir}")
    print(f"대표 얼굴 디렉토리: {representative_dir}")
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.dataset_path):
        print(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    
    # 1단계: 얼굴 추출
    if not args.skip_extraction:
        print("\n=== 1단계: 비디오에서 얼굴 추출 ===")
        
        extractor = FaceExtractor(
            detector_backend=args.detector_backend,
            align_faces=args.align_faces
        )
        
        try:
            extraction_stats = extractor.extract_faces_from_videos(
                dataset_path=args.dataset_path,
                output_dir=extracted_faces_dir,
                video_extensions=args.video_extensions
            )
            
            if not extraction_stats:
                print("얼굴 추출 중 오류가 발생했습니다.")
                return 1
                
            print(f"얼굴 추출 완료: {extraction_stats['processed_count']}개의 얼굴을 추출했습니다.")
            
        except Exception as e:
            print(f"얼굴 추출 중 오류 발생: {e}")
            return 1
    else:
        print("\n=== 1단계: 얼굴 추출 건너뜀 ===")
        if not os.path.exists(extracted_faces_dir):
            print(f"오류: 얼굴 추출을 건너뛰었지만 '{extracted_faces_dir}' 디렉토리가 존재하지 않습니다.")
            return 1
    
    # 2단계: 얼굴 중복 제거
    if not args.skip_deduplication:
        print("\n=== 2단계: 얼굴 중복 제거 ===")
        
        deduplicator = FaceDeduplicator(
            model_name=args.model_name,
            threshold=args.threshold
        )
        
        try:
            deduplication_stats = deduplicator.deduplicate_faces(
                faces_dir=extracted_faces_dir,
                dedupe_dir=dedupe_dir,
                representative_dir=representative_dir
            )
            
            if not deduplication_stats:
                print("얼굴 중복 제거 중 오류가 발생했습니다.")
                return 1
                
            print(f"얼굴 중복 제거 완료!")
            print(f"- 총 {deduplication_stats['copied_files_to_dedupe_count']}개의 파일이 dedupe 디렉토리로 복사되었습니다.")
            print(f"- 총 {deduplication_stats['copied_files_to_representative_count']}개의 대표 파일이 representative 디렉토리로 복사되었습니다.")
            
        except Exception as e:
            print(f"얼굴 중복 제거 중 오류 발생: {e}")
            return 1
    else:
        print("\n=== 2단계: 얼굴 중복 제거 건너뜀 ===")
    
    print("\n=== 워크플로우 완료 ===")
    print(f"모든 결과는 '{args.output_base_dir}' 디렉토리에 저장되었습니다.")
    return 0


if __name__ == '__main__':
    exit(main())
