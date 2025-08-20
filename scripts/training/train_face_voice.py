#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 학습 스크립트
"""

import argparse
import os
import sys
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
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

try:
    from models.face_voice_model import (
        FaceVoiceModel, InfoNCELoss, save_model_components, load_model_components
    )
    from datasets.face_voice_dataset import (
        FaceVoiceDataset, collate_fn, create_data_transforms, create_audio_augmentations, match_face_voice_files
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

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, current_val_loss, save_dir):
    """
    학습 상태(최신, 최고 성능, 체크포인트)를 저장합니다.

    Args:
        epoch (int): 현재 에포크 번호.
        model: 저장할 모델.
        optimizer: 저장할 옵티마이저.
        scheduler: 저장할 스케줄러.
        best_val_loss (float): 현재까지의 최고 검증 손실.
        current_val_loss (float): 이번 에포크의 검증 손실.
        save_dir (str): 저장할 디렉토리.
    
    Returns:
        float: 업데이트된 최고 검증 손실.
    """
    # 최신 모델 컴포넌트 저장 (매 에포크마다 덮어쓰기)
    # 이 파일들은 학습 재개 시 사용
    save_model_components(model, save_dir)

    # 학습 재개를 위한 체크포인트 저장 (모델 가중치 제외)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save({
        'epoch': epoch + 1,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)

    # 최고 성능 모델 저장
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        print(f"🎉 새로운 최고 성능 모델 발견! (Val Loss: {best_val_loss:.4f}). 'best_model/' 디렉토리에 저장합니다.")
        best_model_dir = os.path.join(save_dir, 'best_model')
        save_model_components(model, best_model_dir)
    
    return best_val_loss

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer,
                scheduler, device, num_epochs, save_dir, tensorboard_dir,
                start_epoch, best_val_loss):
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
        tensorboard_dir: TensorBoard 로그를 저장할 경로. None일 경우 로깅하지 않음
        start_epoch: 학습을 시작할 에포크 번호. 학습 재개 시 사용
        best_val_loss: 이전 학습에서 기록된 가장 낮은 검증 손실 값. 학습 재개 시 사용

    Returns:
        학습 히스토리
    """
    history = {'train_loss': [], 'val_loss': []}

    # TensorBoard 설정
    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None 
    if writer:
        print(f"TensorBoard 로그가 '{tensorboard_dir}'에 저장됩니다.")
        print(f"실행: tensorboard --logdir={os.path.dirname(tensorboard_dir)}")

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

        # 매 에포크가 끝난 후 스케줄러의 step()을 호출하여 학습률을 업데이트
        scheduler.step()

        # 스케줄러에서 모든 그룹의 학습률 리스트를 가져옴
        lrs = scheduler.get_last_lr()

        # 그룹별 학습률 명시
        lr_info = (f"Pretrained: {lrs[0]:.7f}, "
                   f"Projection: {lrs[1]:.7f}")
        # 학습 진행 상황 출력
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LRs: [{lr_info}]")

        # 학습 상태(최신, 최고 성능, 체크포인트)를 한 번에 저장
        best_val_loss = save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            current_val_loss=val_loss,
            save_dir=save_dir
        )

        # TensorBoard에 에포크별 메트릭 기록
        if writer:
            # 훈련 및 검증 손실 기록
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)

            # 그룹별 학습률을 구분하여 기록
            writer.add_scalar('Learning_Rate/Pretrained', lrs[0], epoch)
            writer.add_scalar('Learning_Rate/Projection', lrs[1], epoch)

            # 모델 파라미터 및 그래디언트 분포 히스토그램 (매 10 에포크마다)
            if (epoch + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='(신규 레이어용) 기본 학습률 (기본값: 1e-3)')
    # 사전 학습된 모델을 위한 학습률 인자
    parser.add_argument('--pretrained_lr', type=float, default=1e-5,
                        help='사전 학습된 레이어의 학습률 (기본값: 1e-5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='테스트 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--disable_image_augmentation', action='store_true',
                        help='이미지 데이터 증강을 비활성화합니다.')
    parser.add_argument('--disable_audio_augmentation', action='store_true',
                        help='오디오 데이터 증강을 비활성화합니다.')


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

    # TensorBoard 디렉토리 설정
    tensorboard_dir = None if args.no_tensorboard else os.path.join(final_save_dir, 'runs')

    # --- 데이터 준비 ---

    # 디렉토리 확인
    if not os.path.exists(args.image_folder):
        print(f"❌ 오류: 이미지 폴더 '{args.image_folder}'가 존재하지 않습니다.")
        return 1

    if not os.path.exists(args.audio_folder):
        print(f"❌ 오류: 오디오 폴더 '{args.audio_folder}'가 존재하지 않습니다.")
        return 1

    # 파일 매칭 (선택적)
    if args.skip_file_matching:
        if args.matched_files_path and os.path.exists(args.matched_files_path):
            import json
            print(f"저장된 파일 매칭 결과를 불러오는 중: {args.matched_files_path}")
            with open(args.matched_files_path, 'r', encoding='utf-8') as f:
                matched_files = json.load(f)
            print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 불러왔습니다.")
        else:
            print(f"❌ 오류: --skip_file_matching이 설정되었지만 유효한 --matched_files_path가 제공되지 않았습니다.")
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
        print(f"❌ 오류: 매칭된 파일이 없습니다. 경로를 확인해주세요.")
        return 1
    
    # 데이터 분할
    train_files, val_files = train_test_split(
        matched_files,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"학습 데이터: {len(train_files)}개, 검증 데이터: {len(val_files)}개")
    
    # 데이터 변환기 및 증강 파이프라인 생성
    image_transform, processor = create_data_transforms(
        use_augmentation=not args.disable_image_augmentation
    )
    audio_augmentations = create_audio_augmentations(
        sample_rate=args.target_sr,
        use_augmentation=not args.disable_audio_augmentation
    )

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

    # 체크포인트 로드
    if args.resume_dir:
        # 모델 가중치는 항상 컴포넌트 파일에서 로드
        print(f"모델 컴포넌트를 불러옵니다: {final_save_dir}")
        model = load_model_components(model, final_save_dir, device)
        if model is None:
            print(f"❌ 오류: 모델 로드에 실패하여 학습을 중단합니다.")
            return 1
        
        checkpoint_path = os.path.join(args.resume_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"체크포인트 파일을 불러옵니다: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"✅ 체크포인트 로드 완료. Epoch {start_epoch}부터 학습을 재개합니다.")
            print(f"   (이전 최고 Val Loss: {best_val_loss:.4f})")
        else:
            print(f"❌ 오류: --resume_dir이 지정되었지만 체크포인트 파일 '{checkpoint_path}'을 찾을 수 없습니다.")
            return 1

    # --- 학습 실행 ---
    print("학습 시작...")
    history = train_model(
        model, train_dataloader, val_dataloader,
        InfoNCELoss(args.temperature),
        optimizer, scheduler, device, args.num_epochs, 
        final_save_dir, tensorboard_dir, start_epoch, best_val_loss
    )

    print(f"학습 완료! 모델이 '{final_save_dir}'에 저장되었습니다.")
    if tensorboard_dir:
        print(f"TensorBoard 로그 확인: tensorboard --logdir={os.path.dirname(tensorboard_dir)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
