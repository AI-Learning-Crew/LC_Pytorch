#!/usr/bin/env python3
"""
HQ VoxCeleb 데이터셋을 위한 고속 학습 스크립트 (병렬 처리 최적화)

학습 속도 최적화 버전:
- 큰 배치 크기 (64-128)
- 많은 워커 수 (8-16)
- 짧은 오디오 길이 (3초)
- 그래디언트 체크포인팅 비활성화
- 메모리 최적화
- 조기 종료 (Early Stopping) 추가
- 학습률 스케줄러 추가
- 그래디언트 클리핑 추가
"""

import argparse
import os
import sys
from pathlib import Path
import multiprocessing as mp

# TensorFlow 충돌 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 경고 숨김
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # GPU 0만 사용

# CUDA 경고 숨김
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 멀티프로세싱 시작 방법 설정
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 무시

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm.auto import tqdm
import json
import time
import threading
# Mixed Precision Training 비활성화로 인해 import 제거
# from torch.amp import autocast, GradScaler

from models.hq.hq_voxceleb_model import (
    HQVoxCelebModel, HQVoxCelebInfoNCELoss, save_hq_voxceleb_model_components
)
from data.HQVoxCeleb.hq_voxceleb_dataset import create_hq_voxceleb_dataloaders

# 시스템 최적화 설정
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def train_epoch_fast(model, train_dataloader, criterion, optimizer, device, epoch, num_epochs, grad_clip_norm=1.0):
    """고속 학습을 위한 에포크 함수 (Mixed Precision Training 포함)"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_dataloader)
    
    # Mixed Precision Training을 위한 GradScaler (수정된 버전)
    # 안정성을 위해 Mixed Precision 비활성화
    scaler = None
    
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                     leave=False, ncols=120, mininterval=1.0)
    
    for batch_idx, batch in enumerate(train_pbar):
        try:
            # 배치 데이터를 지정된 장치로 이동
            mels = batch['mel'].to(device, non_blocking=True)      # non_blocking=True로 메모리 전송 최적화
            faces = batch['face'].to(device, non_blocking=True)    # non_blocking=True로 메모리 전송 최적화
            identities = batch['identity']
            
            # 안정성을 위해 Mixed Precision 비활성화
            # 순전파
            face_embeddings, audio_embeddings = model(mels, faces)
            
            # 손실 계산
            loss = criterion(face_embeddings, audio_embeddings)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 (오버피팅 방지)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # 손실 누적 및 진행률 업데이트
            total_loss += loss.item()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 메모리 정리 (매 10배치마다 - 메모리 부족 방지)
            if batch_idx % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"배치 {batch_idx} 처리 중 오류 발생: {e}")
            continue
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch_fast(model, val_dataloader, criterion, device, epoch, num_epochs):
    """고속 검증을 위한 에포크 함수 (Mixed Precision Training 포함)"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_dataloader)
    
    if num_batches == 0:
        print(f"검증 데이터가 없습니다. 검증을 건너뜁니다.")
        return float('inf')
    
    with torch.no_grad():
        val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False, ncols=120, mininterval=1.0)
        
        for batch_idx, batch in enumerate(val_pbar):
            try:
                # 배치 데이터를 지정된 장치로 이동
                mels = batch['mel'].to(device, non_blocking=True)      # non_blocking=True로 메모리 전송 최적화
                faces = batch['face'].to(device, non_blocking=True)    # non_blocking=True로 메모리 전송 최적화
                identities = batch['identity']
                
                # 안정성을 위해 Mixed Precision 비활성화
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
                
                # 메모리 정리 (매 10배치마다 - 메모리 부족 방지)
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"검증 배치 {batch_idx} 처리 중 오류 발생: {e}")
                continue
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model_fast(model, train_dataloader, val_dataloader, criterion, optimizer, 
                    device, num_epochs, save_dir, save_interval=10, patience=5, grad_clip_norm=1.0):
    """고속 학습 관리 함수 (조기 종료 및 학습률 스케줄러 포함)"""
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    has_validation = len(val_dataloader) > 0
    patience_counter = 0
    
    # 학습률 스케줄러 설정 (Cosine Annealing + ReduceLROnPlateau 조합)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, 
        verbose=True, min_lr=1e-7, threshold=1e-4
    )
    
    print(f"총 배치 수: 학습={len(train_dataloader)}, 검증={len(val_dataloader)}")
    print(f"조기 종료 patience: {patience}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # 학습
        train_loss = train_epoch_fast(
            model, train_dataloader, criterion, optimizer, 
            device, epoch, num_epochs, grad_clip_norm
        )
        
        # 검증
        if has_validation:
            val_loss = validate_epoch_fast(model, val_dataloader, criterion, 
                                          device, epoch, num_epochs)
            
            # 학습률 스케줄러 업데이트 (Cosine Annealing은 매 에포크마다, Plateau는 검증 손실 기반)
            cosine_scheduler.step()
            plateau_scheduler.step(val_loss)
        else:
            val_loss = float('inf')
            # 검증이 없으면 Cosine Annealing만 사용
            cosine_scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 결과 출력
        if has_validation:
            print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'LR: {current_lr:.2e}, Best: {best_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): '
                  f'Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}')
        
        # 조기 종료 및 최고 성능 모델 저장
        if has_validation:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'새로운 최고 성능 모델 저장 (Val Loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'검증 성능 개선 없음 ({patience_counter}/{patience})')
                
                if patience_counter >= patience:
                    print(f'조기 종료: {patience} 에포크 동안 검증 성능 개선 없음')
                    break
        
        # 모델 저장
        if (epoch + 1) % save_interval == 0:
            save_hq_voxceleb_model_components(model, save_dir)
            print(f'모델 저장 완료 (에포크 {epoch+1})')
    
    # 최종 모델 저장
    save_hq_voxceleb_model_components(model, save_dir)
    print(f'최종 모델 저장 완료 (총 {epoch+1} 에포크)')
    return history


def main():
    parser = argparse.ArgumentParser(description='HQ VoxCeleb 고속 학습 (병렬 처리 최적화)')
    
    # 데이터 경로
    parser.add_argument('--split_json_path', type=str, 
                       default='./data/HQVoxCeleb/split.json',
                       help='split.json 파일 경로')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='모델 저장 디렉토리')
    
    # 모델 설정
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='InfoNCE 온도 파라미터 (기본값: 0.1)')
    
    # 병렬 처리 최적화 설정
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기 (기본값: 32)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='학습 에포크 수 (기본값: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='학습률 (기본값: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='가중치 감쇠 (기본값: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='데이터 로딩 워커 수 (0=메인 프로세스, 기본값: 0)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='워커당 미리 로드할 배치 수 (기본값: 2)')
    parser.add_argument('--cache_size', type=int, default=1000,
                       help='데이터 캐시 크기 (기본값: 1000)')
    parser.add_argument('--enable_parallel', action='store_true', default=True,
                       help='병렬 처리 활성화')
    
    # 정규화 설정
    parser.add_argument('--patience', type=int, default=5,
                       help='조기 종료 patience (기본값: 5)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='그래디언트 클리핑 노름 (기본값: 1.0)')
    
    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=1,
                       help='오디오 길이 (초) (기본값: 1)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='이미지 크기 (기본값: 224)')
    
    # 장치 설정
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--force_cpu', action='store_true')
    
    # 저장 설정
    parser.add_argument('--save_interval', type=int, default=5,
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
        
        # 메모리 할당 전략 설정 (성능 최적화)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # GPU 메모리 정보 출력
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 메모리: {gpu_memory:.1f}GB")
        
        # 성능 최적화 설정
        torch.backends.cudnn.benchmark = True  # 성능 향상을 위해 True로 설정
        torch.backends.cudnn.deterministic = False  # 성능 향상을 위해 False로 설정
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화 (속도 향상)
        torch.backends.cudnn.allow_tf32 = True  # cuDNN TF32 활성화
        
        # 메모리 할당 최적화 (안전 모드)
        torch.cuda.set_per_process_memory_fraction(0.6)  # GPU 메모리의 60% 사용 (더 보수적)
        
        # CUDA 스트림 최적화
        torch.cuda.synchronize()  # 초기 동기화
        
        # 메모리 할당 전략 최적화 (안전 모드)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
        
        print("GPU 성능 최적화 설정 완료")
    
    # 데이터 파일 검증
    if not os.path.exists(args.split_json_path):
        print(f"오류: split.json 파일이 존재하지 않습니다: {args.split_json_path}")
        return 1
    
    # 병렬 처리 최적화된 데이터로더 생성
    print("병렬 처리 최적화된 데이터로더 생성 중...")
    
    # 성능 최적화를 위한 설정 (GPU 사용 시 최적화)
    if device.type == 'cuda':
        safe_num_workers = 2  # GPU 사용 시 2개 워커로 최적화
        persistent_workers = True  # 워커 재사용으로 성능 향상
        pin_memory = True  # GPU 메모리 전송 최적화
        prefetch_factor = 2
    else:
        safe_num_workers = 0  # CPU 사용 시 단일 프로세스
        persistent_workers = False
        pin_memory = False
        prefetch_factor = None
    
    print(f"데이터로더 설정: 워커={safe_num_workers}, persistent_workers={persistent_workers}, pin_memory={pin_memory}")
    
    dataloaders = create_hq_voxceleb_dataloaders(
        split_json_path=args.split_json_path,
        batch_size=args.batch_size,
        num_workers=safe_num_workers,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr,
        image_size=args.image_size,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        enable_parallel=args.enable_parallel,
        cache_size=args.cache_size
    )
    
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    
    print(f"학습 데이터: {len(train_dataloader.dataset)}개 샘플")
    print(f"검증 데이터: {len(val_dataloader.dataset)}개 샘플")
    print(f"학습 배치 수: {len(train_dataloader)}")
    print(f"검증 배치 수: {len(val_dataloader)}")
    
    # 첫 번째 배치 테스트 (디버깅용)
    print("첫 번째 배치 테스트 중...")
    try:
        test_batch = next(iter(train_dataloader))
        print(f"배치 테스트 성공: mel shape={test_batch['mel'].shape}, face shape={test_batch['face'].shape}")
        print(f"첫 번째 identity: {test_batch['identity'][0]}")
    except Exception as e:
        print(f"배치 테스트 실패: {e}")
        return 1
    
    # 모델 초기화
    print("모델 초기화 중...")
    mel_time_steps = args.audio_duration_sec * 100  # 실제 데이터 차원에 맞춤
    mel_freq_bins = 40  # 실제 데이터에서 확인된 주파수 빈 수
    model = HQVoxCelebModel(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        mel_freq_bins=mel_freq_bins,  # 실제 데이터 차원에 맞춤
        mel_time_steps=mel_time_steps  # 실제 데이터 차원에 맞춤
    )
    model.to(device)
    
    # 모델 성능 최적화 (메모리 절약을 위해 컴파일 비활성화)
    # if device.type == 'cuda':
    #     model = torch.compile(model)  # PyTorch 2.0 컴파일 최적화
    #     print("모델 컴파일 완료 (PyTorch 2.0 최적화)")
    print("모델 컴파일 비활성화 (메모리 절약)")
    
    # 모델을 장치로 이동
    model.to(device)
    
    # 모델 테스트 (디버깅용)
    print("모델 테스트 중...")
    try:
        test_mel = test_batch['mel'].to(device)
        test_face = test_batch['face'].to(device)
        
        with torch.no_grad():
            face_emb, audio_emb = model(test_mel, test_face)
            print(f"모델 테스트 성공: face_emb shape={face_emb.shape}, audio_emb shape={audio_emb.shape}")
            
            # 손실 계산 테스트
            test_criterion = HQVoxCelebInfoNCELoss(temperature=args.temperature)
            test_loss = test_criterion(face_emb, audio_emb)
            print(f"테스트 손실: {test_loss.item():.4f}")
            
    except Exception as e:
        print(f"모델 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
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
        'device': str(device),
        'patience': args.patience,
        'grad_clip_norm': args.grad_clip_norm
    }
    
    with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 학습 실행
    print("고속 학습 시작...")
    start_time = time.time()
    
    history = train_model_fast(
        model, train_dataloader, val_dataloader,
        criterion, optimizer, device,
        args.num_epochs, args.save_dir, args.save_interval,
        args.patience, args.grad_clip_norm
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