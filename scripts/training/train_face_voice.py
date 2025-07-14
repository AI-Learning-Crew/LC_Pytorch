#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 학습 스크립트
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

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


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, 
                device, num_epochs, save_dir, tensorboard_dir=None):
    """
    모델 학습
    
    Args:
        model: 학습할 모델
        train_dataloader: 학습 데이터로더
        val_dataloader: 검증 데이터로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 계산 장치
        num_epochs: 학습 에포크 수
        save_dir: 모델 저장 디렉토리
        tensorboard_dir: TensorBoard 로그 디렉토리
        
    Returns:
        학습 히스토리
    """
    history = {'train_loss': [], 'val_loss': []}
    
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
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, audios) in enumerate(train_pbar):
            images, audios = images.to(device), audios.to(device)
            
            # 순전파
            image_embeddings, audio_embeddings = model(images, audios)
            loss = criterion(image_embeddings, audio_embeddings)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # TensorBoard에 배치별 손실 기록
            if writer:
                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            
            global_step += 1
        
        train_loss /= len(train_dataloader)
        history['train_loss'].append(train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, audios in val_pbar:
                images, audios = images.to(device), audios.to(device)
                
                image_embeddings, audio_embeddings = model(images, audios)
                loss = criterion(image_embeddings, audio_embeddings)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_dataloader)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # TensorBoard에 에포크별 메트릭 기록
        if writer:
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            
            # 학습률 기록
            for param_group in optimizer.param_groups:
                writer.add_scalar('Learning_Rate', param_group['lr'], epoch)
            
            # 모델 파라미터 분포 히스토그램 (매 10 에포크마다)
            if (epoch + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # 모델 저장 (매 에포크마다)
        save_model_components(model, save_dir)
    
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
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기 (기본값: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='학습률 (기본값: 1e-4)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='테스트 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=5,
                       help='오디오 길이 (초) (기본값: 5)')
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
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 모델 생성
    print("모델 초기화 중...")
    model = FaceVoiceModel(embedding_dim=args.embedding_dim)
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 모델 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 학습 실행
    print("학습 시작...")
    history = train_model(
        model, train_dataloader, test_dataloader, 
        criterion, optimizer, device, args.num_epochs, args.save_dir, tensorboard_dir
    )
    
    print(f"학습 완료! 모델이 '{args.save_dir}'에 저장되었습니다.")
    if tensorboard_dir:
        print(f"TensorBoard 로그가 '{tensorboard_dir}'에 저장되었습니다.")
        print(f"TensorBoard를 실행하려면: tensorboard --logdir={tensorboard_dir}")
    return 0


if __name__ == '__main__':
    exit(main()) 