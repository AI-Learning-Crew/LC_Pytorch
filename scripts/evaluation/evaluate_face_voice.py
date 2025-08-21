#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 평가 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.face_voice_model import FaceVoiceModel
from datasets.face_voice_dataset import (
    FaceVoiceDataset, collate_fn, create_data_transforms, match_face_voice_files
)
from utils.evaluator import (
    evaluate_summary_metrics, evaluate_retrieval_ranking, 
    calculate_retrieval_metrics, print_evaluation_summary,
    save_results_to_csv
)


def main():
    parser = argparse.ArgumentParser(description='얼굴-음성 매칭 모델을 평가합니다.')
    
    # 데이터 경로
    parser.add_argument('--image_folder', type=str, required=True,
                       help='얼굴 이미지 폴더 경로')
    parser.add_argument('--audio_folder', type=str, required=True,
                       help='음성 파일 폴더 경로')
    parser.add_argument('--model_path', type=str, required=True,
                       help='평가할 모델 파일(.pth) 경로')
    
    # 평가 설정
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기 (기본값: 32)')
    parser.add_argument('--test_size', type=float, default=0.05,
                       help='테스트 데이터 비율 (기본값: 0.05)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='상세 랭킹 평가에서 사용할 Top-K (기본값: 5)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='상세 결과를 저장할 CSV 파일 경로 (선택사항)')
    
    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=5,
                       help='오디오 길이 (초) (기본값: 5)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    
    args = parser.parse_args()
    
    # 디렉토리 확인
    if not os.path.exists(args.image_folder):
        print(f"오류: 이미지 폴더 '{args.image_folder}'가 존재하지 않습니다.")
        return 1
    
    if not os.path.exists(args.audio_folder):
        print(f"오류: 오디오 폴더 '{args.audio_folder}'가 존재하지 않습니다.")
        return 1
    
    # 파일 존재 여부 확인
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일 '{args.model_path}'가 존재하지 않습니다.")
        return 1

    # --- Pandas 출력 옵션 설정 ---
    # 이 옵션들을 설정하면 DataFrame이 생략 없이 전체 출력됩니다.
    pd.set_option('display.max_columns', None)  # 모든 열을 출력
    pd.set_option('display.width', 1000)        # 출력 너비를 넓게 설정하여 줄바꿈 방지
    pd.set_option('display.max_colwidth', None) # 열 내용이 길어도 생략하지 않음
    # -----------------------------------------
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"평가에 사용될 장치: {device}")
    
    # 데이터 변환기 생성 (테스트 데이터셋은 증강 off)
    image_transform, processor = create_data_transforms(
        use_augmentation=False
    )
    
    # 파일 매칭
    print("파일 매칭 중...")
    matched_files = match_face_voice_files(args.image_folder, args.audio_folder)
    print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 찾았습니다.")
    
    if len(matched_files) == 0:
        print("매칭된 파일이 없습니다. 경로를 확인해주세요.")
        return 1
    
    # 데이터 분할 (학습 시와 동일한 random_state 사용)
    train_files, test_files = train_test_split(
        matched_files, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    print(f"테스트에 사용될 데이터: {len(test_files)}개")
    
    # 테스트 데이터셋 생성
    test_dataset = FaceVoiceDataset(
        test_files, processor, image_transform,
        audio_augmentations=None,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr
    )
    
    # 테스트 데이터로더 생성
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 모델 생성 및 로드
    print("모델 로드 중...")
    model = FaceVoiceModel()
    
    try:
        # 표준 PyTorch 방식으로 모델 가중치 로드
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"모델 로드에 실패했습니다: {e}")
        return 1
    
    model.to(device).eval()
    print("✅ 모델이 평가 모드로 설정되었습니다.")
    
    # 1. 요약 성능 지표 평가
    print("\n=== 요약 성능 지표 평가 ===")
    top1_accuracy, auc_score = evaluate_summary_metrics(model, test_dataloader, device)
    
    # 2. 검색 성능 지표 계산
    print("\n=== 검색 성능 지표 계산 ===")
    retrieval_metrics = calculate_retrieval_metrics(
        model, test_dataset, device, top_ks=[1, 5, 10]
    )
    
    # 3. 상세 랭킹 평가
    print("\n=== 상세 랭킹 평가 ===")
    results_df = evaluate_retrieval_ranking(
        model, test_dataset, device, top_k=args.top_k
    )
    
    # 결과 출력
    print_evaluation_summary(top1_accuracy, auc_score, retrieval_metrics)
    
    # 상세 결과 출력 (처음 50개만)
    print(f"\n--- 이미지 기반 음성 검색 평가 결과 (Top {args.top_k}) ---")
    pd.set_option('display.float_format', '{:.4f}'.format)
    # 이 메서드는 문자 너비를 고려하여 더 정확하게 정렬된 텍스트를 생성합니다.
    print(results_df.head(50).to_string()) 
    
    # CSV 파일로 저장 (선택사항)
    if args.output_file:
        save_results_to_csv(results_df, args.output_file)
        print(f"\n상세 결과가 '{args.output_file}'에 저장되었습니다.")
    
    return 0


if __name__ == '__main__':
    exit(main()) 