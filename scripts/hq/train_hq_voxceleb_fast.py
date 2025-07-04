#!/usr/bin/env python3
"""
HQ VoxCeleb 데이터셋을 위한 고속 학습 스크립트

학습 속도 최적화 버전:
- 큰 배치 크기 (64-128)
- 많은 워커 수 (8-16)
- 짧은 오디오 길이 (3초)
- 그래디언트 체크포인팅 비활성화
- 메모리 최적화
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import time

from models.hq.hq_voxceleb_model import (
    HQVoxCelebModel, HQVoxCelebInfoNCELoss, save_hq_voxceleb_model_components
)
from data.HQVoxCeleb.hq_voxceleb_dataset import create_hq_voxceleb_dataloaders


def train_epoch_fast(model, train_dataloader, criterion, optimizer, device, epoch, num_epochs):
    """고속 학습을 위한 에포크 함수"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_dataloader)
    
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for batch_idx, batch in enumerate(train_pbar):
        # 배치 데이터를 지정된 장치로 이동
        mels = batch['mel'].to(device, non_blocking=True)      # non_blocking=True로 메모리 전송 최적화
        faces = batch['face'].to(device, non_blocking=True)    # non_blocking=True로 메모리 전송 최적화
        identities = batch['identity']
        
        # 순전파
        face_embeddings, audio_embeddings = model(mels, faces)
        
        # 손실 계산
        loss = criterion(face_embeddings, audio_embeddings)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실 누적 및 진행률 업데이트
        total_loss += loss.item()
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
        
        # 메모리 정리 (매 10배치마다)
        if batch_idx % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch_fast(model, val_dataloader, criterion, device, epoch, num_epochs):
    """고속 검증을 위한 에포크 함수"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_dataloader)
    
    if num_batches == 0:
        print(f"검증 데이터가 없습니다. 검증을 건너뜁니다.")
        return float('inf')
    
    with torch.no_grad():
        val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        for batch_idx, batch in enumerate(val_pbar):
            # 배치 데이터를 지정된 장치로 이동
            mels = batch['mel'].to(device, non_blocking=True)      # non_blocking=True로 메모리 전송 최적화
            faces = batch['face'].to(device, non_blocking=True)    # non_blocking=True로 메모리 전송 최적화
            identities = batch['identity']
            
            # 순전파
            face_embeddings, audio_embeddings = model(mels, faces)
            
            # 손실 계산
            loss = criterion(face_embeddings, audio_embeddings)
            
            # 손실 누적 및 진행률 업데이트
            total_loss += loss.item()
            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 메모리 정리 (매 5배치마다)
            if batch_idx % 5 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model_fast(model, train_dataloader, val_dataloader, criterion, optimizer, 
                    device, num_epochs, save_dir, save_interval=10):
    """고속 학습 관리 함수"""
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    has_validation = len(val_dataloader) > 0
    
    print(f"총 배치 수: 학습={len(train_dataloader)}, 검증={len(val_dataloader)}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 학습
        train_loss = train_epoch_fast(model, train_dataloader, criterion, optimizer, 
                                     device, epoch, num_epochs)
        
        # 검증
        if has_validation:
            val_loss = validate_epoch_fast(model, val_dataloader, criterion, 
                                          device, epoch, num_epochs)
        else:
            val_loss = float('inf')
        
        epoch_time = time.time() - start_time
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 결과 출력
        if has_validation:
            print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): '
                  f'Train Loss: {train_loss:.4f}')
        
        # 모델 저장
        if (epoch + 1) % save_interval == 0:
            save_hq_voxceleb_model_components(model, save_dir)
            print(f'모델 저장 완료 (에포크 {epoch+1})')
        
        # 최고 성능 모델 저장
        if has_validation and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'새로운 최고 성능 모델 저장 (Val Loss: {val_loss:.4f})')
    
    # 최종 모델 저장
    save_hq_voxceleb_model_components(model, save_dir)
    return history


def main():
    parser = argparse.ArgumentParser(description='HQ VoxCeleb 고속 학습')
    
    # 데이터 경로
    parser.add_argument('--split_json_path', type=str, 
                       default='./data/HQVoxCeleb/split.json',
                       help='split.json 파일 경로')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='모델 저장 디렉토리')
    
    # 모델 설정
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='InfoNCE 온도 파라미터')
    
    # 고속 학습 설정
    parser.add_argument('--batch_size', type=int, default=8,  # 64 -> 8로 대폭 감소
                       help='배치 크기 (기본값: 8)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='학습 에포크 수 (기본값: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='학습률 (기본값: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠 (기본값: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=2,  # 8 -> 2로 감소
                       help='데이터 로딩 워커 수 (기본값: 2)')
    
    # 오디오 설정 (단축)
    parser.add_argument('--audio_duration_sec', type=int, default=2,  # 3 -> 2로 감소
                       help='오디오 길이 (초) (기본값: 2)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='이미지 크기 (기본값: 224)')
    
    # 장치 설정
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--force_cpu', action='store_true')
    
    # 저장 설정
    parser.add_argument('--save_interval', type=int, default=10,
                       help='모델 저장 간격')
    
    args = parser.parse_args()
    
    # ===== 장치 설정 =====
    if args.force_cpu or args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 장치: {device}")
    
    # GPU 메모리 최적화 설정
    if device.type == 'cuda':
        # GPU 메모리 캐시 정리
        torch.cuda.empty_cache()
        
        # 메모리 할당 전략 설정
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # GPU 메모리 정보 출력
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 메모리: {gpu_memory:.1f}GB")
        
        # 메모리 사용량 모니터링 활성화
        torch.cuda.memory.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80%만 사용
    
    # 데이터 파일 검증
    if not os.path.exists(args.split_json_path):
        print(f"오류: split.json 파일이 존재하지 않습니다: {args.split_json_path}")
        return 1
    
    # 데이터로더 생성
    print("데이터로더 생성 중...")
    dataloaders = create_hq_voxceleb_dataloaders(
        split_json_path=args.split_json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr,
        image_size=args.image_size
    )
    
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    
    print(f"학습 데이터: {len(train_dataloader.dataset)}개 샘플")
    print(f"검증 데이터: {len(val_dataloader.dataset)}개 샘플")
    print(f"학습 배치 수: {len(train_dataloader)}")
    print(f"검증 배치 수: {len(val_dataloader)}")
    
    # 모델 초기화
    print("모델 초기화 중...")
    model = HQVoxCelebModel(
        embedding_dim=args.embedding_dim,
        pretrained=True
    )
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = HQVoxCelebInfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 설정 저장
    config = {
        'embedding_dim': args.embedding_dim,
        'temperature': args.temperature,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'audio_duration_sec': args.audio_duration_sec,
        'target_sr': args.target_sr,
        'image_size': args.image_size,
        'device': str(device)
    }
    
    with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 학습 실행
    print("고속 학습 시작...")
    start_time = time.time()
    
    history = train_model_fast(
        model, train_dataloader, val_dataloader,
        criterion, optimizer, device,
        args.num_epochs, args.save_dir, args.save_interval
    )
    
    total_time = time.time() - start_time
    print(f"학습 완료! 총 소요 시간: {total_time/3600:.1f}시간")
    print(f"모델이 '{args.save_dir}'에 저장되었습니다.")
    
    # 히스토리 저장
    with open(os.path.join(args.save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    return 0


if __name__ == '__main__':
    exit(main()) 