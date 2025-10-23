"""
Cross-Modal Autoencoder 모델
이미지↔음성 교차 모달 재구성을 통한 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import numpy as np


class MelSpectrogramEncoder(nn.Module):
    """멜스펙트로그램 인코더 (기존 HQVoxCelebModel에서 가져옴)"""
    
    def __init__(self, embedding_dim: int = 512):
        super(MelSpectrogramEncoder, self).__init__()
        
        # CNN 기반 멜스펙트로그램 인코더
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # x: (batch_size, 1, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MelSpectrogramDecoder(nn.Module):
    """멜스펙트로그램 디코더 (이미지 임베딩 → 멜스펙트로그램)"""
    
    def __init__(self, embedding_dim: int = 512, output_height: int = 40, output_width: int = 128):
        super(MelSpectrogramDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_height = output_height  # 멜스펙트로그램 높이 (40)
        self.output_width = output_width    # 멜스펙트로그램 너비 (가변 길이를 고정)
        
        # 임베딩을 2D로 변환
        self.fc = nn.Linear(embedding_dim, 512 * 4 * 4)
        
        # Transposed CNN으로 멜스펙트로그램 재구성
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # x: (batch_size, embedding_dim)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        
        x = F.relu(self.bn1(self.deconv1(x)))  # 8x8
        x = F.relu(self.bn2(self.deconv2(x)))  # 16x16
        x = F.relu(self.bn3(self.deconv3(x)))  # 32x32
        x = torch.sigmoid(self.deconv4(x))     # 64x64
        
        # 멜스펙트로그램 크기로 리사이즈 (40 x 128)
        x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)
        
        return x


class ImageDecoder(nn.Module):
    """이미지 디코더 (음성 임베딩 → 이미지)"""
    
    def __init__(self, embedding_dim: int = 512, output_size: int = 224):
        super(ImageDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        
        # 임베딩을 2D로 변환
        self.fc = nn.Linear(embedding_dim, 512 * 7 * 7)
        
        # Transposed CNN으로 이미지 재구성
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        # x: (batch_size, embedding_dim)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # 이미지는 0-1 범위
        
        return x


class CrossModalAutoencoder(nn.Module):
    """Cross-Modal Autoencoder 모델
    
    이미지와 음성을 서로 변환하는 능력을 학습합니다.
    - 이미지 → 음성 재구성
    - 음성 → 이미지 재구성
    """
    
    def __init__(self, 
                 image_model_name: str = "google/vit-base-patch16-224-in21k",
                 embedding_dim: int = 512,
                 mel_height: int = 128,
                 mel_width: int = 128,
                 image_size: int = 224,
                 pretrained: bool = True):
        """
        CrossModalAutoencoder 초기화
        
        Args:
            image_model_name: 이미지 인코더 모델명
            embedding_dim: 임베딩 차원
            mel_height: 멜스펙트로그램 높이
            mel_width: 멜스펙트로그램 너비
            image_size: 이미지 크기
            pretrained: 사전 훈련된 가중치 사용 여부
        """
        super(CrossModalAutoencoder, self).__init__()
        
        # 이미지 인코더 (ViT)
        self.image_encoder = ViTModel.from_pretrained(
            image_model_name, 
            add_pooling_layer=False
        )
        self.image_projection = nn.Linear(
            self.image_encoder.config.hidden_size, 
            embedding_dim
        )
        
        # 멜스펙트로그램 인코더 (CNN)
        self.mel_encoder = MelSpectrogramEncoder(embedding_dim)
        
        # 디코더들
        self.mel_decoder = MelSpectrogramDecoder(
            embedding_dim, mel_height, mel_width
        )
        self.image_decoder = ImageDecoder(embedding_dim, image_size)
        
        self.embedding_dim = embedding_dim
        self.mel_height = mel_height
        self.mel_width = mel_width
        self.image_size = image_size
        
    def encode_image(self, images):
        """이미지 인코딩"""
        # ViT로 이미지 인코딩
        outputs = self.image_encoder(images)
        # CLS 토큰 사용
        image_features = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)
        image_embedding = self.image_projection(image_features)  # (batch_size, embedding_dim)
        
        return image_embedding
    
    def encode_mel(self, mels):
        """멜스펙트로그램 인코딩"""
        mel_embedding = self.mel_encoder(mels)
        return mel_embedding
    
    def encode(self, mels, images):
        """두 모달리티 모두 인코딩"""
        mel_embedding = self.encode_mel(mels)
        image_embedding = self.encode_image(images)
        return mel_embedding, image_embedding
    
    def decode_to_mel(self, image_embedding):
        """이미지 임베딩을 멜스펙트로그램으로 디코딩"""
        reconstructed_mel = self.mel_decoder(image_embedding)
        return reconstructed_mel
    
    def decode_to_image(self, mel_embedding):
        """멜스펙트로그램 임베딩을 이미지로 디코딩"""
        reconstructed_image = self.image_decoder(mel_embedding)
        return reconstructed_image
    
    def forward(self, mels, images):
        """순전파"""
        # 인코딩
        mel_embedding, image_embedding = self.encode(mels, images)
        
        # 교차 모달 재구성
        reconstructed_mel = self.decode_to_mel(image_embedding)
        reconstructed_image = self.decode_to_image(mel_embedding)
        
        return {
            'mel_embedding': mel_embedding,
            'image_embedding': image_embedding,
            'reconstructed_mel': reconstructed_mel,
            'reconstructed_image': reconstructed_image
        }


class CrossModalReconstructionLoss(nn.Module):
    """Cross-Modal 재구성 손실 함수"""
    
    def __init__(self, 
                 mel_weight: float = 0.5,
                 image_weight: float = 0.5,
                 similarity_weight: float = 0.0):
        """
        Args:
            mel_weight: 멜스펙트로그램 재구성 손실 가중치
            image_weight: 이미지 재구성 손실 가중치
            similarity_weight: 임베딩 유사도 손실 가중치
        """
        super(CrossModalReconstructionLoss, self).__init__()
        self.mel_weight = mel_weight
        self.image_weight = image_weight
        self.similarity_weight = similarity_weight
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def forward(self, outputs, mels, images):
        """
        Args:
            outputs: 모델 출력 딕셔너리
            mels: 원본 멜스펙트로그램
            images: 원본 이미지
        """
        reconstructed_mel = outputs['reconstructed_mel']
        reconstructed_image = outputs['reconstructed_image']
        mel_embedding = outputs['mel_embedding']
        image_embedding = outputs['image_embedding']
        
        # 재구성 손실
        mel_recon_loss = self.mse_loss(reconstructed_mel, mels)
        image_recon_loss = self.mse_loss(reconstructed_image, images)
        
        # 총 손실
        total_loss = (self.mel_weight * mel_recon_loss + 
                     self.image_weight * image_recon_loss)
        
        # 임베딩 유사도 손실 (선택적)
        if self.similarity_weight > 0:
            # 같은 샘플의 임베딩은 유사해야 함
            targets = torch.ones(mel_embedding.size(0), device=mel_embedding.device)
            similarity_loss = self.cosine_loss(mel_embedding, image_embedding, targets)
            total_loss += self.similarity_weight * similarity_loss
        
        return {
            'total_loss': total_loss,
            'mel_recon_loss': mel_recon_loss,
            'image_recon_loss': image_recon_loss,
            'mel_weight': self.mel_weight,
            'image_weight': self.image_weight
        }


def calculate_psnr_sf2f_style(reconstructed, original, data_type='mel'):
    """
    SF2F_PyTorch 방식의 PSNR 계산
    정규화된 데이터를 역정규화한 후 PSNR 계산
    """
    if data_type == 'mel':
        # 멜스펙트로그램 역정규화 (SF2F_PyTorch 방식)
        VOX_MEL_MEAN = 10.9915
        VOX_MEL_STD = 3.1661
        
        recon_denorm = reconstructed * VOX_MEL_STD + VOX_MEL_MEAN
        orig_denorm = original * VOX_MEL_STD + VOX_MEL_MEAN
        
        # 실제 데이터 범위 계산
        min_val = min(torch.min(recon_denorm), torch.min(orig_denorm))
        max_val = max(torch.max(recon_denorm), torch.max(orig_denorm))
        data_range = max_val - min_val
        
    elif data_type == 'image':
        # 이미지 역정규화 (ImageNet 방식)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        recon_denorm = reconstructed.clone()
        orig_denorm = original.clone()
        
        for i in range(3):
            recon_denorm[:, i] = recon_denorm[:, i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
            orig_denorm[:, i] = orig_denorm[:, i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
        
        # 0-1 범위로 클리핑
        recon_denorm = torch.clamp(recon_denorm, 0, 1)
        orig_denorm = torch.clamp(orig_denorm, 0, 1)
        data_range = 1.0
    
    # MSE 계산
    mse = F.mse_loss(recon_denorm, orig_denorm)
    
    if mse < 1e-8:
        return float('inf')
    
    # PSNR 계산
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def calculate_batch_psnr_sf2f(reconstructed, original, data_type='mel'):
    """
    SF2F_PyTorch 방식의 배치별 PSNR 계산
    """
    batch_size = reconstructed.size(0)
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr_sf2f_style(
            reconstructed[i:i+1], 
            original[i:i+1], 
            data_type
        )
        psnr_values.append(psnr)
    
    # 무한대 값 제외하고 평균 계산
    valid_psnr = [p for p in psnr_values if p != float('inf')]
    if valid_psnr:
        return sum(valid_psnr) / len(valid_psnr)
    else:
        return 0.0


def calculate_psnr(img1, img2, max_val=1.0):
    """Peak Signal-to-Noise Ratio 계산 (기존 방식 유지)"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_similarity_score(embedding1, embedding2):
    """임베딩 간 코사인 유사도 계산"""
    similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
    return similarity.mean().item()
