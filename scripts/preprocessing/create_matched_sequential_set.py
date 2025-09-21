import sys, os
from pathlib import Path
import os
import json
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_metadata(root_dir):
    metadata = {}
    
    train_dir = os.path.join(root_dir, "train")
    id_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    total_ids = len(id_list)

    for idx, id_name in enumerate(id_list, 1):  # enumerate 시작값 1
        start_id_time = time.time()
        
        id_path = os.path.join(train_dir, id_name)
        metadata[id_name] = {}
        
        faces_root = os.path.join(id_path, "faces")
        voices_root = os.path.join(id_path, "voices")
        
        if not os.path.exists(faces_root) or not os.path.exists(voices_root):
            logging.warning(f"⚠️ {id_name} 은 faces 또는 voices 폴더가 없어 건너뜀")
            continue
        
        for session_name in os.listdir(faces_root):  # ex) 00001
            session_faces_path = os.path.join(faces_root, session_name)
            session_voice_path = os.path.join(voices_root, session_name + ".wav")
            
            if not os.path.isdir(session_faces_path):
                continue
            
            face_files = sorted([
                os.path.relpath(os.path.join(session_faces_path, f), root_dir)
                for f in os.listdir(session_faces_path)
                if f.endswith(".jpg")
            ])
            
            voice_file = (
                os.path.relpath(session_voice_path, root_dir)
                if os.path.exists(session_voice_path)
                else None
            )
            
            metadata[id_name][session_name] = {
                "faces": face_files,
                "voice": voice_file
            }
        
        elapsed_id_time = time.time() - start_id_time
        logging.info(f"[{idx}/{total_ids}] {id_name} 처리 완료 (소요 {elapsed_id_time:.2f} 초)")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='데이터셋의 얼굴 이미지와 음성 파일을 인덱스별로 매칭하여 저장합니다.')
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                       help='데이터셋이 저장된 디렉토리 경로')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='매칭 결과를 저장할 출력 디렉토리')
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        logging.error(f"오류: 입력 디렉토리 '{args.dataset_path}'가 존재하지 않습니다.")
        return 1
    if not os.path.exists(args.output):
        logging.error(f"오류: 출력 디렉토리 '{args.output}'가 존재하지 않습니다.")
        return 1    
    
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Output path: {args.output}") 
    
    start_time = time.time()
    
    result = build_metadata(args.dataset_path)
    
    output_path = os.path.join(args.output, "matched_pair_set.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    logging.info(f"✅ 전체 처리 완료! 총 소요 시간: {elapsed_time:.2f} 초")
    return 0

if __name__ == "__main__":
    main()
