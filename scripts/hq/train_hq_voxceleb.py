#!/usr/bin/env python3
"""
HQ VoxCeleb 데이터셋을 위한 얼굴-음성 매칭 모델 학습 스크립트

이 스크립트는 HQ VoxCeleb 데이터셋을 사용하여 얼굴과 음성을 매칭하는 멀티모달 모델을 학습합니다.
모델은 Vision Transformer (ViT)와 Mel spectrogram을 사용하여 얼굴과 음성의 임베딩을 생성하고,
InfoNCE 손실 함수를 통해 contrastive learning을 수행합니다.

주요 특징:
- 동적 입력 차원 처리 (다양한 mel spectrogram 크기 지원)
- .npy 및 .pickle 파일 형식 지원
- 검증 데이터 없음 처리
- macOS OpenMP 호환성
- 자동 모델 저장 및 체크포인팅

사용법:
    python scripts/hq/train_hq_voxceleb.py \
        --save_dir ./saved_models/hq_voxceleb \
        --force_cpu \
        --batch_size 8 \
        --num_epochs 10 \
        --dataset_type vox2
"""

# ===== 표준 라이브러리 import =====
import argparse  # 명령행 인자 파싱을 위한 라이브러리
import os        # 운영체제 인터페이스 (파일 경로, 디렉토리 생성 등)
import sys       # 시스템 관련 파라미터와 함수들
from pathlib import Path  # 경로 처리를 위한 객체지향 인터페이스

# ===== 프로젝트 루트 경로 설정 =====
# 이 스크립트의 위치를 기준으로 프로젝트 루트 디렉토리를 찾습니다
# scripts/hq/train_hq_voxceleb.py -> 프로젝트 루트
project_root = Path(__file__).parent.parent.parent
# 프로젝트 루트를 Python 경로의 맨 앞에 추가하여 상대 경로 import가 가능하도록 합니다
sys.path.insert(0, str(project_root))

# 디버깅을 위한 경로 출력 (필요시 주석 해제)
# print(f"프로젝트 루트: {project_root}")
# print(f"Python 경로: {sys.path[:3]}")

# ===== 딥러닝 및 데이터 처리 라이브러리 import =====
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # PyTorch 신경망 모듈들
from torch.utils.data import DataLoader  # 데이터 배치 처리를 위한 로더
from tqdm.auto import tqdm  # 진행률 표시를 위한 라이브러리 (자동 감지)
import json  # JSON 파일 읽기/쓰기를 위한 라이브러리

# ===== 프로젝트 내부 모듈 import =====
# HQ VoxCeleb 전용 모델 관련 클래스들과 함수들
from models.hq.hq_voxceleb_model import (
    HQVoxCelebModel,           # HQ VoxCeleb 전용 멀티모달 모델 클래스
    HQVoxCelebInfoNCELoss,     # InfoNCE 손실 함수 클래스 (contrastive learning용)
    save_hq_voxceleb_model_components  # 모델 컴포넌트들을 개별적으로 저장하는 함수
)
# HQ VoxCeleb 데이터셋 관련 클래스들과 함수들
from data.HQVoxCeleb.hq_voxceleb_dataset import (
    HQVoxCelebDataset,         # HQ VoxCeleb 데이터셋 클래스
    create_hq_voxceleb_dataloaders,  # 데이터로더들을 생성하는 함수
    collate_hq_voxceleb_fn     # 배치 데이터를 정렬하는 함수
)


def train_epoch(model, train_dataloader, criterion, optimizer, device, epoch, num_epochs):
    """
    한 에포크 동안 모델을 학습하는 함수
    
    이 함수는 전체 학습 데이터셋을 한 번 순회하면서 모델을 학습시킵니다.
    각 배치마다 순전파, 손실 계산, 역전파, 파라미터 업데이트를 수행합니다.
    
    Args:
        model (HQVoxCelebModel): 학습할 멀티모달 모델
            - 얼굴 이미지와 mel spectrogram을 입력받아 임베딩을 생성
            - Vision Transformer와 오디오 인코더로 구성
        train_dataloader (DataLoader): 학습 데이터 로더
            - 배치 단위로 데이터를 제공
            - 각 배치는 {'mel': tensor, 'face': tensor, 'identity': list} 형태
        criterion (HQVoxCelebInfoNCELoss): 손실 함수
            - InfoNCE 손실을 계산하여 얼굴-음성 매칭 학습
            - Contrastive learning을 위한 temperature-scaled cosine similarity 사용
        optimizer (torch.optim.AdamW): 옵티마이저
            - AdamW 알고리즘으로 모델 파라미터를 업데이트
            - 가중치 감쇠(weight decay) 포함
        device (torch.device): 학습 장치 (CPU 또는 GPU)
            - 모델과 데이터가 위치할 장치
        epoch (int): 현재 에포크 번호 (0부터 시작)
            - 진행률 표시용
        num_epochs (int): 전체 에포크 수
            - 진행률 표시용
    
    Returns:
        float: 해당 에포크의 평균 학습 손실
            - 모든 배치의 손실을 평균낸 값
            - 학습 진행 상황을 모니터링하는 데 사용
    """
    # 모델을 학습 모드로 설정
    # - 드롭아웃 레이어가 활성화되어 랜덤하게 뉴런을 비활성화
    # - 배치 정규화 레이어가 학습 통계를 업데이트
    # - 그래디언트 계산이 활성화됨
    model.train()
    
    # 손실 누적을 위한 변수 초기화
    total_loss = 0.0  # 전체 배치의 손실 합계
    num_batches = len(train_dataloader)  # 총 배치 수
    
    # 진행률 표시를 위한 tqdm 설정
    # desc: 진행률 바에 표시될 설명
    # 자동으로 남은 시간과 처리 속도를 계산하여 표시
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    # 각 배치에 대해 학습 수행
    # enumerate를 사용하여 배치 인덱스와 데이터를 동시에 가져옴
    for batch_idx, batch in enumerate(train_pbar):
        # 배치 데이터를 지정된 장치로 이동
        # .to(device)는 텐서를 CPU 또는 GPU로 이동시킴
        mels = batch['mel'].to(device)      # Mel spectrogram 데이터 (오디오 특성)
        faces = batch['face'].to(device)    # 얼굴 이미지 데이터 (시각적 특성)
        identities = batch['identity']      # identity 정보 (문자열 리스트, 장치 이동 불필요)
        
        # ===== 순전파 (Forward Pass) =====
        # 모델에 mel spectrogram과 얼굴 이미지를 입력하여 임베딩 생성
        # model(mels, faces)는 내부적으로 다음을 수행:
        # 1. 얼굴 이미지를 Vision Transformer로 처리하여 face_embeddings 생성
        # 2. Mel spectrogram을 오디오 인코더로 처리하여 audio_embeddings 생성
        # 3. 두 임베딩을 동일한 차원으로 정규화
        face_embeddings, audio_embeddings = model(mels, faces)
        
        # ===== 손실 계산 =====
        # InfoNCE 손실을 사용하여 얼굴과 음성 임베딩 간의 매칭 학습
        # criterion 내부에서:
        # 1. 배치 내 모든 얼굴-음성 쌍 간의 cosine similarity 계산
        # 2. Positive pair (같은 identity)는 높은 similarity를, 
        #    negative pair (다른 identity)는 낮은 similarity를 학습하도록 유도
        # 3. Temperature scaling을 통해 similarity 분포 조정
        loss = criterion(face_embeddings, audio_embeddings)
        
        # ===== 역전파 (Backward Pass) =====
        optimizer.zero_grad()  # 이전 그래디언트 초기화 (중요: 누적 방지)
        loss.backward()        # 그래디언트 계산 (chain rule 적용)
        optimizer.step()       # 파라미터 업데이트 (AdamW 알고리즘)
        
        # ===== 손실 누적 및 진행률 업데이트 =====
        total_loss += loss.item()  # .item()으로 텐서를 Python 스칼라로 변환
        
        # 진행률 바에 현재 상태 표시
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',           # 현재 배치 손실 (소수점 4자리)
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'  # 평균 손실 (현재까지)
        })
    
    # 에포크 평균 손실 계산
    # 전체 배치의 손실 합계를 배치 수로 나누어 평균 계산
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, val_dataloader, criterion, device, epoch, num_epochs):
    """
    한 에포크 동안 모델을 검증하는 함수
    
    이 함수는 검증 데이터셋을 사용하여 모델의 성능을 평가합니다.
    학습과 달리 그래디언트 계산을 비활성화하여 메모리를 절약하고 속도를 향상시킵니다.
    
    Args:
        model (HQVoxCelebModel): 검증할 멀티모달 모델
            - 학습된 모델을 평가 모드로 설정하여 검증
        val_dataloader (DataLoader): 검증 데이터 로더
            - 학습 데이터와 동일한 구조의 배치 데이터 제공
        criterion (HQVoxCelebInfoNCELoss): 손실 함수
            - 학습과 동일한 손실 함수 사용
        device (torch.device): 검증 장치 (CPU 또는 GPU)
        epoch (int): 현재 에포크 번호 (0부터 시작)
        num_epochs (int): 전체 에포크 수
    
    Returns:
        float: 해당 에포크의 평균 검증 손실
            - 검증 데이터가 없으면 float('inf') 반환
            - 무한대 값을 반환하여 최고 성능 모델 저장을 방지
    """
    # 모델을 평가 모드로 설정
    # - 드롭아웃 레이어가 비활성화되어 모든 뉴런 사용
    # - 배치 정규화 레이어가 학습된 통계를 고정
    # - 그래디언트 계산이 비활성화됨 (메모리 절약)
    model.eval()
    
    # 손실 누적을 위한 변수 초기화
    total_loss = 0.0
    num_batches = len(val_dataloader)
    
    # 검증 데이터가 없는 경우 처리
    # HQ VoxCeleb 데이터셋에서는 검증 데이터가 없을 수 있음
    if num_batches == 0:
        print(f"검증 데이터가 없습니다. 검증을 건너뜁니다.")
        return float('inf')  # 무한대 값 반환하여 최고 성능 모델 저장 방지
    
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    # with torch.no_grad(): 블록 내에서는 그래디언트가 계산되지 않음
    with torch.no_grad():
        # 진행률 표시를 위한 tqdm 설정
        val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        # 각 배치에 대해 검증 수행
        for batch_idx, batch in enumerate(val_pbar):
            # 배치 데이터를 지정된 장치로 이동
            mels = batch['mel'].to(device)      # Mel spectrogram 데이터
            faces = batch['face'].to(device)    # 얼굴 이미지 데이터
            identities = batch['identity']      # identity 정보 (문자열 리스트)
            
            # 순전파 (그래디언트 계산 없음)
            # 학습과 동일한 모델 순전파 수행
            face_embeddings, audio_embeddings = model(mels, faces)
            
            # 손실 계산
            # 학습과 동일한 InfoNCE 손실 계산
            loss = criterion(face_embeddings, audio_embeddings)
            
            # 손실 누적 및 진행률 업데이트
            total_loss += loss.item()
            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',           # 현재 배치 손실
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'  # 평균 손실
            })
    
    # 에포크 평균 손실 계산
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, 
                device, num_epochs, save_dir, save_interval=5):
    """
    전체 모델 학습을 관리하는 함수
    
    이 함수는 전체 학습 과정을 조율합니다:
    1. 각 에포크마다 학습과 검증을 수행
    2. 학습 히스토리를 기록
    3. 주기적으로 모델을 저장 (체크포인팅)
    4. 최고 성능 모델을 자동으로 저장
    
    Args:
        model (HQVoxCelebModel): 학습할 멀티모달 모델
        train_dataloader (DataLoader): 학습 데이터 로더
        val_dataloader (DataLoader): 검증 데이터 로더
        criterion (HQVoxCelebInfoNCELoss): 손실 함수
        optimizer (torch.optim.AdamW): 옵티마이저
        device (torch.device): 학습 장치 (CPU 또는 GPU)
        num_epochs (int): 전체 에포크 수
        save_dir (str): 모델 저장 디렉토리 경로
        save_interval (int): 모델 저장 간격 (에포크 단위, 기본값: 5)
            - 예: save_interval=5이면 5, 10, 15... 에포크마다 저장
    
    Returns:
        dict: 학습 히스토리
            - 'train_loss': 각 에포크의 학습 손실 리스트
            - 'val_loss': 각 에포크의 검증 손실 리스트
            - 나중에 학습 곡선을 그리거나 분석하는 데 사용
    """
    # 학습 히스토리 초기화
    # 각 에포크의 손실을 기록하여 학습 진행 상황을 추적
    history = {'train_loss': [], 'val_loss': []}
    
    # 최고 성능 모델 추적용 변수
    # 검증 손실이 낮을수록 좋은 성능이므로, 초기값은 무한대
    best_val_loss = float('inf')
    
    # 검증 데이터가 있는지 확인
    # HQ VoxCeleb 데이터셋에서는 검증 데이터가 없을 수 있음
    has_validation = len(val_dataloader) > 0
    
    # 각 에포크에 대해 학습 및 검증 수행
    for epoch in range(num_epochs):
        # ===== 1. 학습 단계 =====
        # train_epoch 함수를 호출하여 한 에포크 동안 모델 학습
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, 
                                device, epoch, num_epochs)
        
        # ===== 2. 검증 단계 =====
        if has_validation:
            # 검증 데이터가 있으면 validate_epoch 함수를 호출하여 검증 수행
            val_loss = validate_epoch(model, val_dataloader, criterion, 
                                     device, epoch, num_epochs)
        else:
            # 검증 데이터가 없으면 무한대 값으로 설정
            val_loss = float('inf')
        
        # ===== 3. 히스토리 저장 =====
        # 현재 에포크의 손실을 히스토리에 추가
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # ===== 4. 학습 결과 출력 =====
        if has_validation:
            # 검증 데이터가 있으면 학습 손실과 검증 손실을 모두 출력
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            # 검증 데이터가 없으면 학습 손실만 출력
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}')
        
        # ===== 5. 주기적 모델 저장 (체크포인팅) =====
        # save_interval마다 모델을 저장하여 중간 결과 보존
        # 긴 학습 시간 동안 중간 결과를 잃지 않도록 보장
        if (epoch + 1) % save_interval == 0:
            # 모델 컴포넌트들을 개별적으로 저장
            # - 얼굴 인코더, 오디오 인코더, 프로젝션 레이어 등을 별도 파일로 저장
            save_hq_voxceleb_model_components(model, save_dir)
            print(f'모델이 {save_dir}에 저장되었습니다.')
        
        # ===== 6. 최고 성능 모델 저장 =====
        # 검증 데이터가 있을 때만 최고 성능 모델 저장
        if has_validation and val_loss < best_val_loss:
            # 현재 검증 손실이 이전 최고 성능보다 좋으면 업데이트
            best_val_loss = val_loss
            
            # 최고 성능 모델을 별도 파일로 저장
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'새로운 최고 성능 모델이 저장되었습니다. (Val Loss: {val_loss:.4f})')
    
    # ===== 7. 최종 모델 저장 =====
    # 학습 완료 후 최종 모델을 저장
    save_hq_voxceleb_model_components(model, save_dir)
    
    return history


def main():
    """
    메인 함수: 명령행 인자 파싱, 데이터 로딩, 모델 초기화, 학습 실행
    
    이 함수는 전체 학습 파이프라인의 진입점입니다:
    1. 명령행 인자를 파싱하여 학습 설정을 구성
    2. 데이터셋을 로드하고 데이터로더를 생성
    3. 모델, 손실 함수, 옵티마이저를 초기화
    4. 학습을 실행하고 결과를 저장
    
    Returns:
        int: 종료 코드 (0: 성공, 1: 오류)
            - 0: 학습이 성공적으로 완료됨
            - 1: 오류가 발생하여 학습이 중단됨
    """
    # ===== 명령행 인자 파서 생성 =====
    # argparse를 사용하여 명령행에서 전달된 인자들을 파싱
    parser = argparse.ArgumentParser(description='HQ VoxCeleb 얼굴-음성 매칭 모델을 학습합니다.')
    
    # ===== 데이터 경로 관련 인자 =====
    parser.add_argument('--split_json_path', type=str, 
                       default='./data/HQVoxCeleb/split.json',
                       help='split.json 파일 경로 (기본값: ./data/HQVoxCeleb/split.json)')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='모델 저장 디렉토리')
    
    # ===== 모델 설정 관련 인자 =====
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원 (기본값: 512)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='InfoNCE 온도 파라미터 (기본값: 0.07)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='사전 훈련된 모델 사용 (기본값: True)')
    
    # ===== 학습 설정 관련 인자 =====
    parser.add_argument('--batch_size', type=int, default=64,
                       help='배치 크기 (기본값: 64)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='학습 에포크 수 (기본값: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='학습률 (기본값: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠 (기본값: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='데이터 로딩 워커 수 (기본값: 8)')
    
    # ===== 오디오 설정 관련 인자 =====
    parser.add_argument('--audio_duration_sec', type=int, default=3,
                       help='오디오 길이 (초) (기본값: 3)')
    parser.add_argument('--target_sr', type=int, default=16000,
                       help='오디오 샘플링 레이트 (기본값: 16000)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='이미지 크기 (기본값: 224)')
    
    # ===== 장치 설정 관련 인자 =====
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='사용할 장치 (기본값: auto)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='강제로 CPU 사용')
    
    # ===== 저장 설정 관련 인자 =====
    parser.add_argument('--save_interval', type=int, default=5,
                       help='모델 저장 간격 (에포크) (기본값: 5)')
    
    # 명령행 인자 파싱
    # sys.argv에서 전달된 인자들을 파싱하여 args 객체에 저장
    args = parser.parse_args()
    
    # ===== 장치 설정 =====
    # 학습에 사용할 장치(CPU/GPU)를 결정
    if args.force_cpu or args.device == 'cpu':
        # 강제로 CPU 사용하거나 명시적으로 CPU 선택
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # 명시적으로 CUDA 선택 (GPU 사용 가능 여부 확인)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:  # auto
        # 자동 감지: GPU가 있으면 GPU, 없으면 CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 장치: {device}")
    
    # ===== 데이터 파일 검증 =====
    # split.json 파일 존재 여부 확인
    # 이 파일은 데이터셋의 train/val/test 분할 정보를 담고 있음
    if not os.path.exists(args.split_json_path):
        print(f"오류: split.json 파일 '{args.split_json_path}'가 존재하지 않습니다.")
        print("먼저 scripts/create_voxceleb_split.py를 실행하여 split.json을 생성하세요.")
        return 1  # 오류 코드 반환
    
    # split.json 파일 로드 및 데이터 확인
    # JSON 파일을 읽어서 데이터셋 분할 정보를 가져옴
    with open(args.split_json_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # split.json에 있는 모든 데이터셋 타입 확인
    # 예: ['vox1', 'vox2'] 등
    available_datasets = list(split_data.keys())
    print(f"split.json에서 발견된 데이터셋: {available_datasets}")
    
    # 데이터 분할 정보 출력
    # 각 데이터셋 타입별로 train/val/test identity 수를 계산
    total_train_identities = 0
    total_val_identities = 0
    total_test_identities = 0
    
    for dataset_type in available_datasets:
        print(f"\nHQ VoxCeleb {dataset_type} 데이터 분할:")
        for split_type in ['train', 'val', 'test']:
            # 각 분할의 identity 수 계산
            count = len(split_data[dataset_type][split_type])
            print(f"  {split_type}: {count}개 identity")
            
            # 전체 통계에 추가
            if split_type == 'train':
                total_train_identities += count
            elif split_type == 'val':
                total_val_identities += count
            elif split_type == 'test':
                total_test_identities += count
    
    print(f"\n전체 데이터:")
    print(f"  train: {total_train_identities}개 identity")
    print(f"  val: {total_val_identities}개 identity")
    print(f"  test: {total_test_identities}개 identity")
    
    # ===== 데이터로더 생성 =====
    print("\n데이터로더 생성 중...")
    # create_hq_voxceleb_dataloaders 함수를 호출하여 데이터로더들을 생성
    # 이 함수는 내부적으로:
    # 1. HQVoxCelebDataset 인스턴스들을 생성
    # 2. DataLoader들을 생성하여 배치 처리를 위한 설정
    # 3. collate_fn을 사용하여 배치 데이터를 정렬
    dataloaders = create_hq_voxceleb_dataloaders(
        split_json_path=args.split_json_path,  # 데이터 분할 정보 파일 경로
        batch_size=args.batch_size,            # 배치 크기
        num_workers=args.num_workers,          # 데이터 로딩 워커 수
        audio_duration_sec=args.audio_duration_sec,  # 오디오 길이 (초)
        target_sr=args.target_sr,              # 목표 샘플링 레이트
        image_size=args.image_size             # 이미지 크기
    )
    
    # 학습 및 검증 데이터로더 추출
    # dataloaders 딕셔너리에서 train과 val 키에 해당하는 DataLoader 객체들을 가져옴
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    
    # 데이터셋 크기 정보 출력
    # len(dataloader.dataset)으로 각 데이터셋의 총 샘플 수를 확인
    print(f"학습 데이터: {len(train_dataloader.dataset)}개 샘플")
    print(f"검증 데이터: {len(val_dataloader.dataset)}개 샘플")
    
    # ===== 모델 초기화 =====
    print("모델 초기화 중...")
    # HQVoxCelebModel 인스턴스를 생성
    # 이 모델은:
    # 1. 얼굴 이미지를 처리하는 Vision Transformer (ViT)
    # 2. Mel spectrogram을 처리하는 오디오 인코더
    # 3. 두 임베딩을 동일한 차원으로 프로젝션하는 레이어들
    model = HQVoxCelebModel(
        embedding_dim=args.embedding_dim,  # 최종 임베딩 차원
        pretrained=args.pretrained         # 사전 훈련된 가중치 사용 여부
    )
    model.to(device)  # 모델을 지정된 장치(CPU/GPU)로 이동
    
    # ===== 손실 함수 및 옵티마이저 설정 =====
    # InfoNCE 손실 함수 초기화
    # InfoNCE는 contrastive learning에서 사용되는 손실 함수로:
    # 1. Positive pair (같은 identity의 얼굴-음성)는 높은 similarity를 학습
    # 2. Negative pair (다른 identity의 얼굴-음성)는 낮은 similarity를 학습
    # 3. Temperature scaling을 통해 similarity 분포를 조정
    criterion = HQVoxCelebInfoNCELoss(temperature=args.temperature)
    
    # AdamW 옵티마이저 초기화
    # AdamW는 Adam의 개선된 버전으로 가중치 감쇠(weight decay)를 포함
    # - learning_rate: 파라미터 업데이트 크기
    # - weight_decay: 정규화를 위한 가중치 감쇠 (과적합 방지)
    optimizer = torch.optim.AdamW(
        model.parameters(),     # 모델의 모든 학습 가능한 파라미터
        lr=args.learning_rate,  # 학습률
        weight_decay=args.weight_decay  # 가중치 감쇠
    )
    
    # ===== 저장 디렉토리 생성 =====
    # 모델과 설정을 저장할 디렉토리를 생성
    # exist_ok=True: 디렉토리가 이미 존재해도 오류를 발생시키지 않음
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ===== 학습 설정 저장 =====
    # 학습에 사용된 모든 설정을 JSON 파일로 저장
    # 이는 나중에 실험을 재현하거나 비교하는 데 사용됨
    config = {
        'available_datasets': available_datasets,  # 사용 가능한 데이터셋 목록
        'embedding_dim': args.embedding_dim,       # 임베딩 차원
        'temperature': args.temperature,           # InfoNCE 온도 파라미터
        'batch_size': args.batch_size,             # 배치 크기
        'num_epochs': args.num_epochs,             # 에포크 수
        'learning_rate': args.learning_rate,       # 학습률
        'weight_decay': args.weight_decay,         # 가중치 감쇠
        'audio_duration_sec': args.audio_duration_sec,  # 오디오 길이
        'target_sr': args.target_sr,               # 샘플링 레이트
        'image_size': args.image_size,             # 이미지 크기
        'device': str(device)                      # 사용 장치
    }
    
    # 설정을 JSON 파일로 저장
    # indent=2: JSON 파일을 읽기 쉽게 들여쓰기
    # ensure_ascii=False: 한글 등 유니코드 문자를 그대로 저장
    with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # ===== 학습 실행 =====
    print("학습 시작...")
    # train_model 함수를 호출하여 전체 학습 과정을 실행
    # 이 함수는:
    # 1. 각 에포크마다 학습과 검증을 수행
    # 2. 학습 히스토리를 기록
    # 3. 주기적으로 모델을 저장
    # 4. 최고 성능 모델을 자동으로 저장
    history = train_model(
        model, train_dataloader, val_dataloader,  # 모델과 데이터로더들
        criterion, optimizer, device,             # 손실 함수, 옵티마이저, 장치
        args.num_epochs, args.save_dir,           # 에포크 수, 저장 디렉토리
        args.save_interval                        # 저장 간격
    )
    
    # ===== 학습 히스토리 저장 =====
    # 학습 과정에서의 손실 변화를 JSON 파일로 저장
    # 이는 나중에 학습 곡선을 그리거나 분석하는 데 사용됨
    with open(os.path.join(args.save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"학습 완료! 모델이 '{args.save_dir}'에 저장되었습니다.")
    return 0  # 성공 코드 반환


if __name__ == '__main__':
    # 스크립트가 직접 실행될 때만 main 함수 호출
    # 이는 스크립트를 import할 때 main 함수가 자동으로 실행되지 않도록 하기 위함
    exit(main())  # main 함수의 반환값을 종료 코드로 사용 