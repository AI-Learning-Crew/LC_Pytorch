"""
얼굴-음성 매칭을 위한 데이터셋
"""

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from transformers import Wav2Vec2Processor
from PIL import Image
import librosa
import numpy as np


class FaceVoiceDataset(Dataset):
    """얼굴-음성 매칭 데이터셋"""
    
    def __init__(self, 
                 file_pairs, 
                 processor, 
                 image_transform, 
                 audio_duration_sec: int = 5,
                 target_sr: int = 16000):
        """
        FaceVoiceDataset 초기화
        
        Args:
            file_pairs: (이미지 경로, 오디오 경로) 튜플 리스트
            processor: Wav2Vec2 프로세서
            image_transform: 이미지 변환기
            audio_duration_sec: 오디오 길이 (초)
            target_sr: 오디오 샘플링 레이트
        """
        self.file_pairs = file_pairs
        self.processor = processor
        self.image_transform = image_transform
        self.audio_duration_sec = audio_duration_sec
        self.target_sr = target_sr
        self.max_length = self.target_sr * audio_duration_sec
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        """
        데이터셋에서 아이템 가져오기
        
        Args:
            idx: 인덱스
            
        Returns:
            이미지 텐서, 오디오 텐서
        """
        image_path, audio_path = self.file_pairs[idx]
        
        # 이미지 로드 및 변환
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        
        # 오디오 로드 및 전처리
        speech_array, _ = librosa.load(audio_path, sr=self.target_sr)
        
        # 오디오 길이 조정
        if len(speech_array) > self.max_length:
            speech_array = speech_array[:self.max_length]
        
        # Wav2Vec2 입력 형식으로 변환
        audio_input = self.processor(
            speech_array, 
            sampling_rate=self.target_sr, 
            return_tensors="pt"
        ).input_values.squeeze(0)
        
        return image_tensor, audio_input


def collate_fn(batch):
    """
    배치 데이터를 위한 collate 함수
    
    Args:
        batch: 배치 데이터
        
    Returns:
        패딩된 이미지 텐서, 패딩된 오디오 텐서
    """
    images = torch.stack([item[0] for item in batch])
    audios = [item[1] for item in batch]
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
    
    return images, padded_audios


def create_data_transforms():
    """
    데이터 변환기 생성
    
    Returns:
        이미지 변환기, 오디오 프로세서
    """
    # 이미지 변환기 설정
    image_transform = transforms.Compose([
        # 이미지를 224x224 크기로 리사이즈 (표준 CNN 입력 크기)
        transforms.Resize((224, 224)),
        # PIL 이미지를 PyTorch 텐서로 변환 (0-1 범위의 float32)
        transforms.ToTensor(),
        # ImageNet 데이터셋의 평균과 표준편차로 정규화
        # RGB 채널별로 정규화하여 모델 학습 안정성 향상
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Wav2Vec2 오디오 프로세서 초기화
    # facebook/wav2vec2-base-960h: 960시간의 LibriSpeech로 사전 훈련된 모델
    # 오디오를 모델이 이해할 수 있는 형태로 변환
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    return image_transform, processor


def match_face_voice_files(image_folder, audio_folder):
    """
    얼굴 이미지와 음성 파일을 매칭
    
    Args:
        image_folder: 얼굴 이미지 폴더 경로
        audio_folder: 음성 파일 폴더 경로
        
    Returns:
        매칭된 파일 쌍 리스트
    """
    # 파일 확장자별로 분리
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.jpg')}
    audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_folder) if f.endswith('.wav')}
    
    # 교집합으로 매칭
    matched_base_names = sorted(list(image_files.intersection(audio_files)))
    
    # 전체 경로로 변환
    matched_files = [
        (os.path.join(image_folder, name + '.jpg'), os.path.join(audio_folder, name + '.wav'))
        for name in matched_base_names
    ]
    
    return matched_files 