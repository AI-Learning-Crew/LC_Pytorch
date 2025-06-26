"""
얼굴-음성 매칭을 위한 멀티모달 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, Wav2Vec2Model


class FaceVoiceModel(nn.Module):
    """얼굴-음성 매칭을 위한 멀티모달 모델"""
    
    def __init__(self, 
                 image_model_name: str = "google/vit-base-patch16-224-in21k",
                 audio_model_name: str = "facebook/wav2vec2-base-960h",
                 embedding_dim: int = 512):
        """
        FaceVoiceModel 초기화
        
        Args:
            image_model_name: 이미지 인코더 모델명
            audio_model_name: 오디오 인코더 모델명
            embedding_dim: 임베딩 차원
        """
        super(FaceVoiceModel, self).__init__()
        
        # 이미지 인코더 (ViT)
        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        self.image_projection = nn.Linear(self.image_encoder.config.hidden_size, embedding_dim)
        
        # 오디오 인코더 (Wav2Vec2)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.audio_projection = nn.Linear(self.audio_encoder.config.hidden_size, embedding_dim)
        
        self.embedding_dim = embedding_dim
        
    def encode_image(self, images):
        """
        이미지 인코딩
        
        Args:
            images: 이미지 텐서 (batch_size, channels, height, width)
            
        Returns:
            이미지 임베딩 (batch_size, embedding_dim)
        """
        image_outputs = self.image_encoder(images)
        # [CLS] 토큰 사용 (첫 번째 토큰)
        image_embeddings_raw = image_outputs.last_hidden_state[:, 0, :]
        image_embeddings = self.image_projection(image_embeddings_raw)
        return F.normalize(image_embeddings, p=2, dim=1)
    
    def encode_audio(self, audios):
        """
        오디오 인코딩
        
        Args:
            audios: 오디오 텐서 (batch_size, sequence_length)
            
        Returns:
            오디오 임베딩 (batch_size, embedding_dim)
        """
        audio_outputs = self.audio_encoder(audios)
        # 시퀀스 평균 사용
        audio_embeddings_raw = torch.mean(audio_outputs.last_hidden_state, dim=1)
        audio_embeddings = self.audio_projection(audio_embeddings_raw)
        return F.normalize(audio_embeddings, p=2, dim=1)
    
    def forward(self, images, audios):
        """
        순전파
        
        Args:
            images: 이미지 텐서
            audios: 오디오 텐서
            
        Returns:
            이미지 임베딩, 오디오 임베딩
        """
        image_embeddings = self.encode_image(images)
        audio_embeddings = self.encode_audio(audios)
        return image_embeddings, audio_embeddings


class InfoNCELoss(nn.Module):
    """InfoNCE 손실 함수"""
    
    def __init__(self, temperature: float = 0.07):
        """
        InfoNCELoss 초기화
        
        Args:
            temperature: 온도 파라미터
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, image_embeddings, audio_embeddings):
        """
        InfoNCE 손실 계산
        
        Args:
            image_embeddings: 이미지 임베딩 (batch_size, embedding_dim)
            audio_embeddings: 오디오 임베딩 (batch_size, embedding_dim)
            
        Returns:
            InfoNCE 손실 값
        """
        batch_size = image_embeddings.size(0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(image_embeddings, audio_embeddings.T) / self.temperature
        
        # 정답 라벨 (대각선)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # 이미지->오디오 손실
        image_to_audio_loss = F.cross_entropy(similarity_matrix, labels)
        
        # 오디오->이미지 손실
        audio_to_image_loss = F.cross_entropy(similarity_matrix.T, labels)
        
        # 평균 손실
        total_loss = (image_to_audio_loss + audio_to_image_loss) / 2
        
        return total_loss


def save_model_components(model, save_dir):
    """
    모델 컴포넌트들을 개별적으로 저장
    
    Args:
        model: FaceVoiceModel 인스턴스
        save_dir: 저장할 디렉토리 경로
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.image_encoder.state_dict(), os.path.join(save_dir, 'image_model.pth'))
    torch.save(model.image_projection.state_dict(), os.path.join(save_dir, 'image_projection.pth'))
    torch.save(model.audio_encoder.state_dict(), os.path.join(save_dir, 'audio_model.pth'))
    torch.save(model.audio_projection.state_dict(), os.path.join(save_dir, 'audio_projection.pth'))
    
    print(f"모델 컴포넌트들이 '{save_dir}'에 저장되었습니다.")


def load_model_components(model, save_dir, device):
    """
    모델 컴포넌트들을 개별적으로 로드
    
    Args:
        model: FaceVoiceModel 인스턴스
        save_dir: 모델이 저장된 디렉토리 경로
        device: 로드할 장치
        
    Returns:
        로드된 모델
    """
    import os
    
    try:
        model.image_encoder.load_state_dict(
            torch.load(os.path.join(save_dir, 'image_model.pth'), map_location=device)
        )
        model.image_projection.load_state_dict(
            torch.load(os.path.join(save_dir, 'image_projection.pth'), map_location=device)
        )
        model.audio_encoder.load_state_dict(
            torch.load(os.path.join(save_dir, 'audio_model.pth'), map_location=device)
        )
        model.audio_projection.load_state_dict(
            torch.load(os.path.join(save_dir, 'audio_projection.pth'), map_location=device)
        )
        
        print("모델 컴포넌트들이 성공적으로 로드되었습니다.")
        return model
        
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        return None 