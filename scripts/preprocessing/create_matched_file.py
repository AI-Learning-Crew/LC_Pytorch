#!/usr/bin/env python3
"""
데이터셋의 얼굴 이미지와 음성 파일을 인덱스별로 매칭하여 저장하는 스크립트

이 스크립트는 주어진 데이터셋 디렉토리와 메타 정보(JSON 파일)를 기반으로,
각 ID에 대해 동일한 인덱스의 얼굴-음성 쌍을 찾아서 인덱스별로 별도의 파일에 저장합니다.

사용법:
    python scripts/create_matched_file.py -d /path/to/dataset -m /path/to/meta.json -o /path/to/output -l 100

주요 기능:
- JSON 메타데이터에서 ID 리스트 로드
- 각 ID에 대해 동일한 인덱스의 파일 쌍 매칭
- 인덱스별로 별도 파일에 매칭 결과 저장
- 로깅을 통한 처리 과정 추적
"""

import sys, os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가하여 모듈 import 가능하게 함
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.matched_file_utils import load_id_list_from_json, save_matched_files_by_index

import argparse
import logging

# 로깅 설정 - 시간 정보와 함께 INFO 레벨 이상의 메시지를 출력
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    메인 함수: 명령행 인자를 파싱하고 매칭 파일 생성 작업을 실행합니다.
    
    Returns:
        int: 성공 시 0, 실패 시 1
    """
    # 명령행 인자 파서 설정
    parser = argparse.ArgumentParser(description='데이터셋의 얼굴 이미지와 음성 파일을 인덱스별로 매칭하여 저장합니다.')
    
    # 필수 인자들
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                       help='데이터셋이 저장된 디렉토리 경로')
    parser.add_argument('-m', '--meta_path', type=str, required=True,
                       help='데이터셋 메타데이터 JSON 파일 경로')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='매칭 결과를 저장할 출력 디렉토리')
    parser.add_argument('-s', '--start', type=str, required=True,
                       help='매칭을 시작할 인덱스 (예: 0)')
    parser.add_argument('-l', '--limit', type=str, required=True,
                       help='매칭할 최대 인덱스 수 (예: 100)')
    
    # 명령행 인자 파싱
    args = parser.parse_args()
    
    # 입력 디렉토리 및 파일 존재 여부 확인
    if not os.path.exists(args.dataset_path):
        logging.error(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    if not os.path.exists(args.meta_path):
        logging.error(f"오류: 메타데이터 파일 '{args.meta_path}'가 존재하지 않습니다.")
        return 1
    if not os.path.exists(args.output):
        logging.error(f"오류: 출력 디렉토리 '{args.output}'가 존재하지 않습니다.")
        return 1
    
    # 처리할 인자 정보 출력 (디버깅 및 추적용)
    logging.info("인자 정보:") 
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Metadata path: {args.meta_path}")   
    logging.info(f"Output path: {args.output}")
    logging.info(f"Start index: {args.start}")
    logging.info(f"Limit: {args.limit}")
    
    # JSON 파일에서 ID 리스트 로드
    id_list = load_id_list_from_json(args.meta_path)
    
    # 인덱스별로 매칭 파일 생성 및 저장
    save_matched_files_by_index(
        id_list=id_list,
        train_base_dir=Path(args.dataset_path),
        output_base_dir=Path(args.output),
        start_index=int(args.start),
        max_index=int(args.start + args.limit)
    )
    
    return 0

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 main 함수 호출
    main() 