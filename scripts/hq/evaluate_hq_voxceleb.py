#!/usr/bin/env python3
"""
HQ VoxCeleb 데이터셋을 위한 얼굴-음성 매칭 모델 평가 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import json

from models.hq_voxceleb_model import HQVoxCelebModel, load_hq_voxceleb_model_components
from datasets.hq_voxceleb_dataset import create_hq_voxceleb_dataloaders


def calculate_top_k_accuracy(face_embeddings, audio_embeddings, k=1):
    """Top-K 정확도를 계산합니다."""
    # 코사인 유사도 계산
    similarities = torch.mm(face_embeddings, audio_embeddings.T)
    
    # 각 얼굴에 대해 가장 유사한 k개의 음성 찾기
    top_k_values, top_k_indices = torch.topk(similarities, k, dim=1)
    
    # 정답 인덱스 (대각선)
    correct_indices = torch.arange(face_embeddings.size(0), device=face_embeddings.device)
    
    # Top-K 정확도 계산
    correct = torch.any(top_k_indices == correct_indices.unsqueeze(1), dim=1)
    accuracy = correct.float().mean().item()
    
    return accuracy


def calculate_roc_auc(face_embeddings, audio_embeddings):
    """ROC-AUC 점수를 계산합니다."""
    # 코사인 유사도 계산
    similarities = torch.mm(face_embeddings, audio_embeddings.T)
    
    # 정답 라벨 생성 (대각선이 positive)
    batch_size = face_embeddings.size(0)
    labels = torch.zeros(batch_size, batch_size, device=face_embeddings.device)
    labels.fill_diagonal_(1)
    
    # ROC-AUC 계산
    similarities_flat = similarities.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()
    
    auc_score = roc_auc_score(labels_flat, similarities_flat)
    
    return auc_score


def evaluate_model(model, test_dataloader, device):
    """모델을 평가합니다."""
    model.eval()
    
    all_face_embeddings = []
    all_audio_embeddings = []
    all_identities = []
    
    with torch.no_grad():
        for mels, faces, identities in tqdm(test_dataloader, desc="평가 중"):
            mels, faces = mels.to(device), faces.to(device)
            
            # 임베딩 추출
            face_embeddings, audio_embeddings = model(mels, faces)
            
            all_face_embeddings.append(face_embeddings.cpu())
            all_audio_embeddings.append(audio_embeddings.cpu())
            all_identities.extend(identities)
    
    # 모든 임베딩을 하나로 합치기
    face_embeddings = torch.cat(all_face_embeddings, dim=0)
    audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
    
    # 성능 지표 계산
    top1_accuracy = calculate_top_k_accuracy(face_embeddings, audio_embeddings, k=1)
    top5_accuracy = calculate_top_k_accuracy(face_embeddings, audio_embeddings, k=5)
    top10_accuracy = calculate_top_k_accuracy(face_embeddings, audio_embeddings, k=10)
    auc_score = calculate_roc_auc(face_embeddings, audio_embeddings)
    
    results = {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
        'roc_auc_score': auc_score,
        'num_samples': len(face_embeddings)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='HQ VoxCeleb 얼굴-음성 매칭 모델을 평가합니다.')
    
    # 데이터 경로
    parser.add_argument('--split_json_path', type=str, 
                       default='./data/HQVoxCeleb/split.json',
                       help='split.json 파일 경로 (기본값: ./data/HQVoxCeleb/split.json)')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='모델이 저장된 디렉토리')
    
    # 모델 설정
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원 (기본값: 512)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='사전 훈련된 모델 사용 (기본값: True)')
    
    # 평가 설정
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기 (기본값: 16)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='데이터 로딩 워커 수 (기본값: 2)')
    
    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=5,
                       help='오디오 길이 (초) (기본값: 5)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='이미지 크기 (기본값: 224)')
    
    # 장치 설정
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='사용할 장치 (기본값: auto)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='강제로 CPU 사용')
    
    args = parser.parse_args()
    
    # 장치 설정
    if args.force_cpu or args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 장치: {device}")
    
    # split.json 파일 확인
    if not os.path.exists(args.split_json_path):
        print(f"오류: split.json 파일 '{args.split_json_path}'가 존재하지 않습니다.")
        return 1
    
    # split.json 파일 로드 및 데이터 확인
    with open(args.split_json_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # split.json에 있는 모든 데이터셋 타입 확인
    available_datasets = list(split_data.keys())
    print(f"split.json에서 발견된 데이터셋: {available_datasets}")
    
    # 데이터 분할 정보 출력
    total_test_identities = 0
    for dataset_type in available_datasets:
        print(f"\nHQ VoxCeleb {dataset_type} 데이터 분할:")
        for split_type in ['train', 'val', 'test']:
            count = len(split_data[dataset_type][split_type])
            print(f"  {split_type}: {count}개 identity")
            if split_type == 'test':
                total_test_identities += count
    
    print(f"\n전체 테스트 데이터: {total_test_identities}개 identity")
    
    # 모델 디렉토리 확인
    if not os.path.exists(args.model_dir):
        print(f"오류: 모델 디렉토리 '{args.model_dir}'가 존재하지 않습니다.")
        return 1
    
    # 설정 파일 로드
    config_path = os.path.join(args.model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("모델 설정:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("경고: config.json 파일을 찾을 수 없습니다.")
    
    # 데이터로더 생성 (test만)
    print("\n테스트 데이터로더 생성 중...")
    dataloaders = create_hq_voxceleb_dataloaders(
        split_json_path=args.split_json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr,
        image_size=args.image_size
    )
    
    test_dataloader = dataloaders['test']
    print(f"테스트 데이터: {len(test_dataloader.dataset)}개 샘플")
    
    # 모델 생성 및 로드
    print("모델 로드 중...")
    model = HQVoxCelebModel(
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    )
    model.to(device)
    
    # 모델 가중치 로드
    load_hq_voxceleb_model_components(model, args.model_dir)
    
    # 평가 실행
    print("모델 평가 시작...")
    results = evaluate_model(model, test_dataloader, device)
    
    # 결과 출력
    print("\n" + "="*50)
    print("평가 결과:")
    print("="*50)
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {results['top10_accuracy']:.4f}")
    print(f"ROC-AUC Score: {results['roc_auc_score']:.4f}")
    print(f"테스트 샘플 수: {results['num_samples']}")
    print("="*50)
    
    # 결과 저장
    results_path = os.path.join(args.model_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"평가 결과가 '{results_path}'에 저장되었습니다.")
    
    return 0


if __name__ == '__main__':
    exit(main()) 