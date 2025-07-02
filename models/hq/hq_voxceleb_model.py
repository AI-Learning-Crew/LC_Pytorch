"""
HQ VoxCeleb 데이터셋을 위한 전용 모델 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from torchvision.models import vit_b_16
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np


class HQVoxCelebModel(nn.Module):
    """
    HQ VoxCeleb 데이터셋을 위한 얼굴-음성 매칭 모델
    
    - 얼굴 인코더: Vision Transformer (ViT-Base)
    - 음성 인코더: Wav2Vec2-Base
    - 투영층: 얼굴과 음성을 공통 임베딩 공간으로 매핑
    """
    
    def __init__(self, embedding_dim=512, pretrained=True):
        super(HQVoxCelebModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 얼굴 인코더 (ViT-Base)
        self.face_encoder = vit_b_16(pretrained=pretrained)
        # 분류 헤드 제거
        self.face_encoder.heads = nn.Identity()
        face_feature_dim = 768  # ViT-Base의 출력 차원
        
        # 음성 인코더 (Wav2Vec2-Base)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        audio_feature_dim = 768  # Wav2Vec2-Base의 출력 차원
        
        # 투영층
        self.face_projection = nn.Sequential(
            nn.Linear(face_feature_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 동적 음성 투영층 (나중에 초기화)
        self.audio_projection = None
        self.audio_input_dim = None
        
        # L2 정규화를 위한 파라미터
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def encode_face(self, face_images):
        """얼굴 이미지를 인코딩합니다."""
        # ViT는 [B, 3, H, W] 입력을 받음
        face_features = self.face_encoder(face_images)  # [B, 768]
        face_embeddings = self.face_projection(face_features)  # [B, embedding_dim]
        face_embeddings = F.normalize(face_embeddings, p=2, dim=1)
        return face_embeddings
    
    def encode_audio(self, mel_spectrograms):
        """Mel spectrogram을 인코딩합니다."""
        # Wav2Vec2는 [B, T] 입력을 받지만, mel spectrogram은 [B, F, T]
        # 따라서 시간 차원을 평균하여 [B, F]로 만든 후 처리
        if len(mel_spectrograms.shape) == 3:
            # [B, F, T] -> [B, F] (시간 차원 평균)
            mel_spectrograms = mel_spectrograms.mean(dim=2)
        
        # Wav2Vec2 특성 추출을 위해 가상의 오디오 시퀀스 생성
        batch_size = mel_spectrograms.shape[0]
        # mel spectrogram을 1D로 평탄화하여 오디오 특성으로 사용
        audio_features = mel_spectrograms.view(batch_size, -1)
        
        # 동적으로 투영층 생성
        if self.audio_projection is None or self.audio_input_dim != audio_features.shape[1]:
            self.audio_input_dim = audio_features.shape[1]
            print(f"음성 투영층 생성: {self.audio_input_dim} -> {self.embedding_dim}")
            self.audio_projection = nn.Sequential(
                nn.Linear(self.audio_input_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            ).to(audio_features.device)
        
        # Wav2Vec2 인코더 통과 (실제로는 mel spectrogram을 직접 처리)
        # 여기서는 mel spectrogram을 직접 투영층에 통과시킴
        audio_embeddings = self.audio_projection(audio_features)  # [B, embedding_dim]
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
        return audio_embeddings
    
    def forward(self, mel_spectrograms, face_images):
        """순전파"""
        face_embeddings = self.encode_face(face_images)
        audio_embeddings = self.encode_audio(mel_spectrograms)
        return face_embeddings, audio_embeddings


class HQVoxCelebInfoNCELoss(nn.Module):
    """
    HQ VoxCeleb를 위한 InfoNCE 손실 함수
    """
    
    def __init__(self, temperature=0.07):
        super(HQVoxCelebInfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, face_embeddings, audio_embeddings):
        """
        Args:
            face_embeddings: [B, D] 얼굴 임베딩
            audio_embeddings: [B, D] 음성 임베딩
        
        Returns:
            InfoNCE 손실
        """
        batch_size = face_embeddings.shape[0]
        
        # 코사인 유사도 계산
        logits = torch.mm(face_embeddings, audio_embeddings.T) / self.temperature
        
        # 대각선이 positive pair
        labels = torch.arange(batch_size, device=face_embeddings.device)
        
        # 얼굴->음성 방향 손실
        loss_face_to_audio = F.cross_entropy(logits, labels)
        
        # 음성->얼굴 방향 손실
        loss_audio_to_face = F.cross_entropy(logits.T, labels)
        
        # 평균 손실
        loss = (loss_face_to_audio + loss_audio_to_face) / 2
        
        return loss


def save_hq_voxceleb_model_components(model, save_dir):
    """
    HQ VoxCeleb 모델 컴포넌트들을 저장합니다.
    
    Args:
        model: HQVoxCelebModel 인스턴스
        save_dir: 저장할 디렉토리 경로
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 얼굴 인코더 저장
    torch.save(model.face_encoder.state_dict(), 
               os.path.join(save_dir, 'face_encoder.pth'))
    
    # 얼굴 투영층 저장
    torch.save(model.face_projection.state_dict(), 
               os.path.join(save_dir, 'face_projection.pth'))
    
    # 음성 인코더 저장
    torch.save(model.audio_encoder.state_dict(), 
               os.path.join(save_dir, 'audio_encoder.pth'))
    
    # 음성 투영층 저장
    torch.save(model.audio_projection.state_dict(), 
               os.path.join(save_dir, 'audio_projection.pth'))
    
    # 전체 모델 저장
    torch.save(model.state_dict(), 
               os.path.join(save_dir, 'full_model.pth'))
    
    print(f"모델 컴포넌트들이 {save_dir}에 저장되었습니다.")


def load_hq_voxceleb_model_components(model, save_dir):
    """
    HQ VoxCeleb 모델 컴포넌트들을 로드합니다.
    
    Args:
        model: HQVoxCelebModel 인스턴스
        save_dir: 로드할 디렉토리 경로
    """
    import os
    
    # 얼굴 인코더 로드
    face_encoder_path = os.path.join(save_dir, 'face_encoder.pth')
    if os.path.exists(face_encoder_path):
        model.face_encoder.load_state_dict(torch.load(face_encoder_path))
    
    # 얼굴 투영층 로드
    face_projection_path = os.path.join(save_dir, 'face_projection.pth')
    if os.path.exists(face_projection_path):
        model.face_projection.load_state_dict(torch.load(face_projection_path))
    
    # 음성 인코더 로드
    audio_encoder_path = os.path.join(save_dir, 'audio_encoder.pth')
    if os.path.exists(audio_encoder_path):
        model.audio_encoder.load_state_dict(torch.load(audio_encoder_path))
    
    # 음성 투영층 로드
    audio_projection_path = os.path.join(save_dir, 'audio_projection.pth')
    if os.path.exists(audio_projection_path):
        model.audio_projection.load_state_dict(torch.load(audio_projection_path))
    
    print(f"모델 컴포넌트들이 {save_dir}에서 로드되었습니다.")


def _get_image_transform(self):
    """개선된 이미지 변환기"""
    if self.split_type == 'train':
        # 학습용: 데이터 증강 적용
        return transforms.Compose([
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 검증/테스트용: 기본 변환만 적용
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# 옵티마이저 설정 후 추가
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# 코사인 어닐링 스케줄러
scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

# 또는 검증 손실 기반 스케줄러
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
#                               patience=5, verbose=True) 

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy() 

# 역전파 후 추가
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step() 

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 