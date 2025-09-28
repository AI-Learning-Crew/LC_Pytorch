#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 학습 스크립트
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path
import random
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

try:
    from models.face_voice_model import (
        FaceVoiceModel, InfoNCELoss
    )
    from datasets.face_voice_dataset import (
        FaceVoiceDataset, collate_fn, create_data_transforms,
        create_audio_augmentations, match_face_voice_files
    )
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print(f"현재 Python 경로: {sys.path}")
    print(f"프로젝트 루트: {project_root}")
    print("프로젝트 루트에서 스크립트를 실행해주세요:")
    print(f"python scripts/train_face_voice.py [인자들]")
    sys.exit(1)


def set_seed(seed):
    """
    재현성을 위해 랜덤 시드를 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_atomic(data, path):
    """데이터를 임시 파일에 저장 후, 최종 경로로 원자적으로 이동(rename)합니다."""
    temp_path = path + ".tmp"
    try:
        torch.save(data, temp_path)
        os.rename(temp_path, path)
    except Exception as e:
        print(f"❌ 파일 저장 실패: {path} - {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

class CheckpointManager:
    """체크포인트 관리를 위한 클래스 (저장, 로드, 정리)"""
    def __init__(self, save_dir, max_to_keep=2):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, epoch, model, optimizer, scheduler, best_val_loss):
        """재개에 필요한 모든 정보를 포함하는 체크포인트를 저장합니다."""
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        filename = f"checkpoint_epoch_{epoch+1:04d}.pth"
        filepath = os.path.join(self.save_dir, filename)
        save_atomic(checkpoint_data, filepath)
        print(f"Epoch {epoch+1}: 체크포인트가 '{os.path.basename(filepath)}'에 저장되었습니다.")
        self._rotate_checkpoints()

    def load_latest(self, model, optimizer, scheduler, device):
        """가장 최신의 체크포인트를 찾아 로드합니다."""
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, "checkpoint_epoch_*.pth")))
        if not checkpoints:
            print("INFO: 저장된 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
            return 0, float('inf')

        latest_checkpoint_path = checkpoints[-1]
        try:
            print(f"가장 최신 체크포인트를 불러옵니다: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            print(f"✅ Epoch {start_epoch} 체크포인트 로드 완료. Epoch {start_epoch+1}부터 학습을 재개합니다.")
            return start_epoch, best_val_loss
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {latest_checkpoint_path} - {e}. 학습을 처음부터 시작합니다.")
            return 0, float('inf')

    def _rotate_checkpoints(self):
        """오래된 체크포인트를 삭제하여 max_to_keep 개수만 유지합니다."""
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, "checkpoint_epoch_*.pth")))
        if len(checkpoints) > self.max_to_keep:
            for ckpt_to_delete in checkpoints[:-self.max_to_keep]:
                print(f"오래된 체크포인트 삭제: {os.path.basename(ckpt_to_delete)}")
                os.remove(ckpt_to_delete)

def save_best_model_weights(model, filepath):
    """추론을 위한 최고 성능 모델의 가중치(state_dict)만 단일 파일로 원자적으로 저장합니다."""
    try:
        save_atomic(model.state_dict(), filepath)
        print(f"✅ 최고 성능 모델 가중치가 '{os.path.basename(filepath)}'에 안전하게 저장되었습니다.")
    except Exception as e:
        print(f"❌ 최고 성능 모델 저장 중 오류 발생: {e}")

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer,
                scheduler, device, num_epochs, save_dir,
                start_epoch, best_val_loss, checkpoint_manager):
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
        start_epoch: 학습을 시작할 에포크 번호. 학습 재개 시 사용
        best_val_loss: 이전 학습에서 기록된 가장 낮은 검증 손실 값. 학습 재개 시 사용

    Returns:
        학습 히스토리
    """
    history = {'train_loss': [], 'val_loss': []}


    # 학습 재개를 위해 global_step 초기화
    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, audios in train_pbar:
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

        # 매 에포크가 끝난 후 스케줄러의 step()을 호출하여 학습률을 업데이트
        scheduler.step()

        # 스케줄러에서 모든 그룹의 학습률 리스트를 가져옴
        lrs = scheduler.get_last_lr()

        # 그룹별 학습률 명시
        lr_info = (f"Pretrained: {lrs[0]:.7f}, "
                   f"Projection: {lrs[1]:.7f}")
        # 학습 진행 상황 출력
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LRs: [{lr_info}]")

        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🎉 새로운 최고 성능 모델 발견! (Val Loss: {best_val_loss:.4f})")
            best_model_path = os.path.join(save_dir, "best_model.pth")
            save_best_model_weights(model, best_model_path)

        # 재개에 필요한 모든 정보를 포함하는 체크포인트를 저장
        checkpoint_manager.save(epoch, model, optimizer, scheduler, best_val_loss)

    return history


def main():
    parser = argparse.ArgumentParser(description='얼굴-음성 매칭 모델을 학습합니다.')

    # 데이터 경로
    parser.add_argument('--matched_pair', type=str, default=None,
                       help='이미 매칭된 파일 목록이 저장된 경로 (JSON 파일)')
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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='(신규 레이어용) 기본 학습률 (기본값: 1e-3)')
    
    # 사전 학습된 모델을 위한 학습률 인자
    parser.add_argument('--pretrained_lr', type=float, default=1e-5,
                        help='사전 학습된 레이어의 학습률 (기본값: 1e-5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='테스트 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')

    # 오디오 설정
    parser.add_argument('--audio_duration_sec', type=int, default=5,
                       help='오디오 길이 (초) (기본값: 5)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')

    # 학습 재개 설정
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='재개할 학습의 타임스탬프 디렉토리 경로 (예: saved_models/20250819_122430)')

    args = parser.parse_args()

    # --- 초기 설정 ---
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # --- 경로 설정 ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_dir:
        # 재개 모드: 지정된 디렉토리 사용
        final_save_dir = args.resume_dir
        if not os.path.exists(final_save_dir):
            print(f"❌ 오류: 학습을 재개할 디렉토리를 찾을 수 없습니다: {final_save_dir}")
            return 1
        print(f"학습을 재개합니다. 저장 디렉토리: {final_save_dir}")
    else:
        # 신규 학습 모드: 한국 시간(KST) 기준으로 타임스탬프 디렉토리 생성
        KST = ZoneInfo("Asia/Seoul")
        timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        final_save_dir = os.path.join(args.save_dir, timestamp)
        os.makedirs(final_save_dir, exist_ok=True)
        print(f"새로운 학습을 시작합니다. 저장 디렉토리: {final_save_dir}")
        with open(os.path.join(final_save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
            print(f"학습 설정이 '{os.path.join(final_save_dir, 'config.json')}'에 저장되었습니다.")


    # --- 데이터 준비 ---
    
    # 데이터셋 생성
    train_dataset = FaceVoiceDataset(
        train_files, processor, image_transform,
        audio_augmentations=audio_augmentations,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr
    )
    val_dataset = FaceVoiceDataset(
        val_files, processor, create_data_transforms(use_augmentation=False)[0],
        audio_augmentations=None,
        audio_duration_sec=args.audio_duration_sec,
        target_sr=args.target_sr
    )
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # --- 모델 및 옵티마이저 준비 ---
    print("모델 초기화 중...")
    model = FaceVoiceModel(embedding_dim=args.embedding_dim)
    model.to(device)

    # 차등 학습률을 적용한 옵티마이저 생성
    # 사전 학습된 인코더(ViT, Wav2Vec2) 파라미터
    pretrained_params = list(model.image_encoder.parameters()) + list(model.audio_encoder.parameters())
    # 처음부터 학습해야 하는 프로젝션 레이어 파라미터
    projection_params = list(model.image_projection.parameters()) + list(model.audio_projection.parameters())

    # 각 그룹에 다른 학습률을 설정하여 옵티마이저 생성
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': args.pretrained_lr},  # 사전 학습된 부분은 낮은 학습률로 미세 조정
        {'params': projection_params, 'lr': args.learning_rate}   # 새로 추가된 부분은 상대적으로 높은 학습률로 빠르게 학습
    ])

    # 학습률 스케줄러 생성
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs, # 전체 에포크 수
        eta_min=1e-7          # 도달할 최소 학습률
    )

    # 체크포인트 매니저 생성
    checkpoint_manager = CheckpointManager(save_dir=final_save_dir, max_to_keep=2)

    # 학습 재개 시 가장 최신의 체크포인트를 찾아 로드
    if args.resume_dir:
        start_epoch, best_val_loss = checkpoint_manager.load_latest(model, optimizer, scheduler, device)

    # --- 학습 실행 ---
    print("학습 시작...")
    history = train_model(
        model, train_dataloader, val_dataloader,
        InfoNCELoss(args.temperature),
        optimizer, scheduler, device, args.num_epochs, 
        final_save_dir, '', start_epoch, best_val_loss, checkpoint_manager
    )

    print(f"학습 완료! 모델이 '{final_save_dir}'에 저장되었습니다.")


if __name__ == '__main__':
    sys.exit(main())
