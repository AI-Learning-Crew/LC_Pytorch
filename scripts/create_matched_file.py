import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.matched_file_utils import load_id_list_from_json, save_matched_files_by_index

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='데이터셋의 얼굴 이미지와 음성 파일을 인덱스별로 매칭하여 저장합니다.')
    
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                       help='데이터셋이 저장된 디렉토리 경로')
    parser.add_argument('-m', '--meta_path', type=str, required=True,
                       help='데이터셋 메타데이터 JSON 파일 경로')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='동일 인물 그룹화 후 복사/저장할 디렉토리')
    parser.add_argument('-l', '--limit', type=str, required=True,
                       help='매칭할 최대 인덱스 수 (예: 100)')
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.dataset_path):
        logging.error(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    if not os.path.exists(args.meta_path):
        logging.error(f"오류: 메타데이터 파일 '{args.meta_path}'가 존재하지 않습니다.")
        return 1
    if not os.path.exists(args.output):
        logging.error(f"오류: 출력 디렉토리 '{args.output}'가 존재하지 않습니다.")
        return 1
    
    # 인자 출력
    logging.info("인자 정보:") 
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Metadata path: {args.meta_path}")   
    logging.info(f"Output path: {args.output}")
    logging.info(f"Limit: {args.limit}")
    
    id_list = load_id_list_from_json(args.meta_path)
    
    save_matched_files_by_index(
        id_list=id_list,
        train_base_dir=Path(args.dataset_path),
        output_base_dir=Path(args.output),
        max_index=int(args.limit)
    );
    
if __name__ == "__main__":
    main() 