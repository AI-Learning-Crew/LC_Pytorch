#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 학습 스크립트 (메모리 효율적 버전)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import gc

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np

try:
    from models.face_voice_model import FaceVoiceModel, InfoNCELoss, save_model_components
    from datasets.face_voice_dataset import (
        FaceVoiceDataset, collate_fn, create_data_transforms, match_face_voice_files
    )
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print(f"현재 Python 경로: {sys.path}")
    print(f"프로젝트 루트: {project_root}")
    print("프로젝트 루트에서 스크립트를 실행해주세요:")
    print(f"python scripts/train_face_voice.py [인자들]")
    sys.exit(1)


def get_optimal_batch_size(device, base_batch_size=16):
    """
    GPU 메모리에 따라 최적 배치 크기 계산
    """
    if device.type == 'cuda':
        # GPU 메모리 확인
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        print(f"GPU 메모리: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 24:  # 24GB 이상 (RTX 3090, A100 등)
            return min(base_batch_size * 2, 32)
        elif gpu_memory >= 12:  # 12GB 이상 (RTX 3080, 4080 등)
            return min(base_batch_size * 1.5, 24)
        elif gpu_memory >= 8:   # 8GB 이상 (RTX 3070, 4070 등)
            return min(base_batch_size * 1.25, 20)
        else:  # 8GB 미만
            return max(base_batch_size // 2, 8)
    else:
        return max(base_batch_size // 2, 8)


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, tensorboard_dir=None, grad_clip_norm=1.0):
    """
    모델 학습 (메모리 효율적 버전)
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # 조기 종료를 위한 patience
    
    # TensorBoard 설정
    writer = None
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard 로그가 '{tensorboard_dir}'에 저장됩니다.")
        print(f"TensorBoard를 실행하려면: tensorboard --logdir={tensorboard_dir}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, audios) in enumerate(train_pbar):
            try:
                # 메모리 정리
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                images, audios = images.to(device, non_blocking=True), audios.to(device, non_blocking=True)
                
                # NaN 체크
                if torch.isnan(images).any() or torch.isnan(audios).any():
                    print(f"경고: 배치 {batch_idx}에서 입력 데이터에 NaN이 발견되었습니다. 건너뜁니다.")
                    continue
                
                # 순전파
                image_embeddings, audio_embeddings = model(images, audios)
                
                # 임베딩 NaN 체크
                if torch.isnan(image_embeddings).any() or torch.isnan(audio_embeddings).any():
                    print(f"경고: 배치 {batch_idx}에서 임베딩에 NaN이 발견되었습니다. 건너뜁니다.")
                    continue
                
                loss = criterion(image_embeddings, audio_embeddings)
                
                # 손실 NaN 체크
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"경고: 배치 {batch_idx}에서 손실이 NaN/Inf입니다. 건너뜁니다.")
                    continue
                
                # 정확도 계산 (메모리 효율적)
                with torch.no_grad():
                    similarity_matrix = torch.matmul(image_embeddings, audio_embeddings.T)
                    predictions = torch.argmax(similarity_matrix, dim=1)
                    labels = torch.arange(images.size(0), device=device)
                    correct = (predictions == labels).sum().item()
                    train_correct += correct
                    train_total += images.size(0)
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (안정성 향상)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_acc = train_correct / train_total if train_total > 0 else 0
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'acc': f'{train_acc:.3f}'
                })
                
                # TensorBoard에 배치별 손실 기록 (메모리 효율적)
                if writer and batch_idx % 5 == 0:  # 5배치마다만 기록
                    writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                    writer.add_scalar('Accuracy/Train_Batch', train_acc, global_step)
                
                global_step += 1
                
                # 메모리에서 텐서 제거
                del images, audios, image_embeddings, audio_embeddings, loss, similarity_matrix, predictions, labels
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM 발생: 배치 {batch_idx} 건너뜀")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        train_loss /= len(train_dataloader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, audios in val_pbar:
                try:
                    images, audios = images.to(device, non_blocking=True), audios.to(device, non_blocking=True)
                    
                    image_embeddings, audio_embeddings = model(images, audios)
                    loss = criterion(image_embeddings, audio_embeddings)
                    
                    # 정확도 계산
                    similarity_matrix = torch.matmul(image_embeddings, audio_embeddings.T)
                    predictions = torch.argmax(similarity_matrix, dim=1)
                    labels = torch.arange(images.size(0), device=device)
                    correct = (predictions == labels).sum().item()
                    val_correct += correct
                    val_total += images.size(0)
                    
                    val_loss += loss.item()
                    val_acc = val_correct / val_total if val_total > 0 else 0
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'acc': f'{val_acc:.3f}'
                    })
                    
                    # 메모리에서 텐서 제거
                    del images, audios, image_embeddings, audio_embeddings, loss, similarity_matrix, predictions, labels
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"검증 중 OOM 발생: 배치 건너뜀")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        val_loss /= len(val_dataloader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'          Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')
        
        # TensorBoard에 에포크별 메트릭 기록
        if writer:
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/Train_Epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/Val_Epoch', val_acc, epoch)
            
            # 학습률 기록
            for param_group in optimizer.param_groups:
                writer.add_scalar('Learning_Rate', param_group['lr'], epoch)
            
            # 모델 파라미터 분포 히스토그램 (매 20 에포크마다, 메모리 절약)
            if (epoch + 1) % 20 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Parameters/{name}', param.data, epoch)
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_components(model, save_dir)
            print(f"새로운 최고 성능! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"성능 개선 없음. 현재 최고: {best_val_loss:.4f} (patience: {patience_counter}/{patience})")
        
        # 조기 종료 체크
        if patience_counter >= patience:
            print(f"조기 종료: {patience} 에포크 동안 성능 개선이 없었습니다.")
            break
        
        # 스케줄러 스텝 (검증 후)
        if scheduler:
            scheduler.step(val_loss)
        
        # 에포크 끝에 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    # TensorBoard 종료
    if writer:
        writer.close()
        print(f"TensorBoard 로그가 '{tensorboard_dir}'에 저장되었습니다.")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='얼굴-음성 매칭 모델을 학습합니다.')
    
    # 데이터 경로
    parser.add_argument('--image_folder', type=str, required=True,
                       help='얼굴 이미지 폴더 경로')
    parser.add_argument('--audio_folder', type=str, required=True,
                       help='음성 파일 폴더 경로')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='모델 저장 디렉토리')
    
    # 모델 설정
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원 (기본값: 512)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='InfoNCE 온도 파라미터 (기본값: 0.07)')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기 (기본값: 16)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='학습률 (기본값: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠 (기본값: 1e-4)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='그래디언트 클리핑 노름 (기본값: 1.0)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='테스트 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=3,
                       help='오디오 길이 (초) (기본값: 3)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    
    # 파일 매칭 설정
    parser.add_argument('--skip_file_matching', action='store_true',
                       help='파일 매칭 과정을 건너뜁니다 (이미 매칭된 파일 목록이 있는 경우)')
    parser.add_argument('--matched_files_path', type=str, default=None,
                       help='이미 매칭된 파일 목록이 저장된 경로 (JSON 파일)')
    
    # TensorBoard 설정
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                       help='TensorBoard 로그 디렉토리 (기본값: save_dir/runs)')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='TensorBoard 로깅을 비활성화합니다')
    
    # 메모리 최적화 설정
    parser.add_argument('--num_workers', type=int, default=2,
                       help='데이터로더 워커 수 (기본값: 2)')
    parser.add_argument('--pin_memory', action='store_true',
                       help='pin_memory 활성화 (GPU 사용 시 권장)')
    
    args = parser.parse_args()
    
    # 디렉토리 확인
    if not os.path.exists(args.image_folder):
        print(f"오류: 이미지 폴더 '{args.image_folder}'가 존재하지 않습니다.")
        return 1
    
    if not os.path.exists(args.audio_folder):
        print(f"오류: 오디오 폴더 '{args.audio_folder}'가 존재하지 않습니다.")
        return 1
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # GPU 메모리 정보 출력
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"사용 가능한 GPU 메모리: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
    
    # 최적 배치 크기 계산
    optimal_batch_size = get_optimal_batch_size(device, args.batch_size)
    if optimal_batch_size != args.batch_size:
        print(f"GPU 메모리에 따라 배치 크기를 {args.batch_size} → {optimal_batch_size}로 조정합니다.")
        args.batch_size = optimal_batch_size
    
    # TensorBoard 디렉토리 설정
    tensorboard_dir = None
    if not args.no_tensorboard:
        if args.tensorboard_dir:
            tensorboard_dir = args.tensorboard_dir
        else:
            # 기본 TensorBoard 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_dir = os.path.join(args.save_dir, 'runs', f'train_{timestamp}')
        
        os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 데이터 변환기 생성
    image_transform, processor = create_data_transforms()
    
    # 파일 매칭 (선택적)
    if args.skip_file_matching:
        if args.matched_files_path and os.path.exists(args.matched_files_path):
            import json
            print(f"저장된 파일 매칭 결과를 불러오는 중: {args.matched_files_path}")
            with open(args.matched_files_path, 'r', encoding='utf-8') as f:
                matched_files = json.load(f)
            print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 불러왔습니다.")
        else:
            print("오류: --skip_file_matching이 설정되었지만 유효한 --matched_files_path가 제공되지 않았습니다.")
            return 1
    else:
        print("파일 매칭 중...")
        matched_files = match_face_voice_files(args.image_folder, args.audio_folder)
        print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 찾았습니다.")
        
        # 매칭 결과 저장 (선택적)
        if args.matched_files_path:
            import json
            os.makedirs(os.path.dirname(args.matched_files_path), exist_ok=True)
            with open(args.matched_files_path, 'w', encoding='utf-8') as f:
                json.dump(matched_files, f, ensure_ascii=False, indent=2)
            print(f"파일 매칭 결과가 '{args.matched_files_path}'에 저장되었습니다.")
    
    if len(matched_files) == 0:
        print("매칭된 파일이 없습니다. 경로를 확인해주세요.")
        return 1
    
    # 데이터 분할
    train_files, test_files = train_test_split(
        matched_files, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    print(f"학습 데이터: {len(train_files)}개, 테스트 데이터: {len(test_files)}개")
    
    # 데이터셋 생성
    train_dataset = FaceVoiceDataset(
        train_files, processor, image_transform, 
        args.audio_duration_sec, args.target_sr
    )
    test_dataset = FaceVoiceDataset(
        test_files, processor, image_transform, 
        args.audio_duration_sec, args.target_sr
    )
    
    # 메모리 효율적인 데이터로더 생성
    pin_memory = args.pin_memory and device.type == 'cuda'
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=True  # 마지막 배치가 불완전하면 버림
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=True  # 마지막 배치가 불완전하면 버림
    )
    
    # 모델 생성
    print("모델 초기화 중...")
    model = FaceVoiceModel(embedding_dim=args.embedding_dim)
    
    # 그래디언트 체크포인팅 활성화 (메모리 절약)
    if hasattr(model.image_encoder, 'gradient_checkpointing_enable'):
        model.image_encoder.gradient_checkpointing_enable()
    if hasattr(model.audio_encoder, 'gradient_checkpointing_enable'):
        model.audio_encoder.gradient_checkpointing_enable()
    
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 개선된 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=7, 
        verbose=True, 
        min_lr=1e-6,
        threshold=1e-4
    )
    
    # 모델 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 학습 실행
    print("학습 시작...")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"학습률: {args.learning_rate}")
    print(f"배치 크기: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"워커 수: {args.num_workers}")
    print(f"Pin Memory: {pin_memory}")
    
    history = train_model(
        model, train_dataloader, test_dataloader, 
        criterion, optimizer, scheduler, device, args.num_epochs, args.save_dir, 
        tensorboard_dir, args.grad_clip_norm
    )
    
    print(f"학습 완료! 모델이 '{args.save_dir}'에 저장되었습니다.")
    if tensorboard_dir:
        print(f"TensorBoard 로그가 '{tensorboard_dir}'에 저장되었습니다.")
        print(f"TensorBoard를 실행하려면: tensorboard --logdir={tensorboard_dir}")
    return 0


if __name__ == '__main__':
    exit(main()) 