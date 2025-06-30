"""
HQ VoxCeleb 데이터셋을 위한 전용 데이터셋 클래스
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import librosa
from transformers import Wav2Vec2Processor
from torchvision import transforms


class HQVoxCelebDataset(Dataset):
    """
    HQ VoxCeleb 데이터셋을 위한 데이터셋 클래스
    
    데이터 구조:
    data/HQVoxCeleb/
    ├── vox1/
    │   ├── vox1_meta.csv
    │   ├── mel_spectograms/ (또는 mel_spectrograms)
    │   └── masked_faces/
    ├── vox2/
    │   ├── full_vox2_meta.csv
    │   ├── mel_spectograms/ (또는 mel_spectrograms)
    │   └── masked_faces/
    └── split.json
    """
    
    def __init__(self, split_json_path, split_type='train', 
                 audio_duration_sec=5, target_sr=16000, image_size=224):
        """
        Args:
            split_json_path (str): split.json 파일 경로
            split_type (str): 'train', 'val', 'test' 중 하나
            audio_duration_sec (int): 오디오 길이 (초)
            target_sr (int): 오디오 샘플링 레이트
            image_size (int): 이미지 크기
        """
        self.split_type = split_type
        self.audio_duration_sec = audio_duration_sec
        self.target_sr = target_sr
        self.image_size = image_size
        
        # split.json 로드
        with open(split_json_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        # 모든 데이터셋 타입에서 해당 split_type의 identity들을 수집
        self.identities = []
        for dataset_type in split_data.keys():
            if split_type in split_data[dataset_type]:
                self.identities.extend(split_data[dataset_type][split_type])
        
        # 데이터 경로 설정 (vox1과 vox2 디렉토리 모두 포함)
        self.vox_dir = Path(split_json_path).parent
        self.vox1_mel_dir = self.vox_dir / 'vox1' / 'mel_spectograms'
        self.vox1_face_dir = self.vox_dir / 'vox1' / 'masked_faces'
        self.vox2_mel_dir = self.vox_dir / 'vox2' / 'mel_spectograms'
        self.vox2_face_dir = self.vox_dir / 'vox2' / 'masked_faces'
        
        # 디렉토리 존재 확인 및 대체 경로 시도
        for mel_dir in [self.vox1_mel_dir, self.vox2_mel_dir]:
            if not mel_dir.exists():
                alt_mel_dir = mel_dir.parent / 'mel_spectrograms'
                if alt_mel_dir.exists():
                    if mel_dir == self.vox1_mel_dir:
                        self.vox1_mel_dir = alt_mel_dir
                    else:
                        self.vox2_mel_dir = alt_mel_dir
        
        for face_dir in [self.vox1_face_dir, self.vox2_face_dir]:
            if not face_dir.exists():
                alt_face_dir = face_dir.parent / 'faces'
                if alt_face_dir.exists():
                    if face_dir == self.vox1_face_dir:
                        self.vox1_face_dir = alt_face_dir
                    else:
                        self.vox2_face_dir = alt_face_dir
        
        # 파일 쌍 생성
        self.file_pairs = self._create_file_pairs()
        
        # 데이터 변환기 설정
        self.image_transform = self._get_image_transform()
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        print(f"HQ VoxCeleb {split_type}: {len(self.file_pairs)}개 파일 쌍")
    
    def _create_file_pairs(self):
        """얼굴과 음성 파일 쌍을 생성합니다."""
        file_pairs = []
        
        for identity in self.identities:
            # vox1과 vox2 디렉토리 모두에서 identity 확인
            mel_identity_dir_vox1 = self.vox1_mel_dir / identity
            face_identity_dir_vox1 = self.vox1_face_dir / identity
            mel_identity_dir_vox2 = self.vox2_mel_dir / identity
            face_identity_dir_vox2 = self.vox2_face_dir / identity
            
            # vox1에서 찾기
            if mel_identity_dir_vox1.exists() and face_identity_dir_vox1.exists():
                mel_files = list(mel_identity_dir_vox1.glob("*.npy")) + list(mel_identity_dir_vox1.glob("*.pickle"))
                face_files = list(face_identity_dir_vox1.glob("*.jpg")) + list(face_identity_dir_vox1.glob("*.png"))
                
                print(f"{identity} (vox1): mel 파일 {len(mel_files)}개, face 파일 {len(face_files)}개")
                
                # 파일 매칭 로직
                if len(face_files) == 1:
                    # 얼굴 파일이 하나만 있으면 모든 mel 파일과 매칭
                    face_file = face_files[0]
                    for mel_file in mel_files:
                        file_pairs.append({
                            'mel_path': str(mel_file),
                            'face_path': str(face_file),
                            'identity': identity
                        })
                else:
                    # 파일명 기반으로 매칭
                    for mel_file in mel_files:
                        mel_stem = mel_file.stem
                        
                        # 대응하는 얼굴 파일 찾기
                        matching_face_files = [
                            f for f in face_files 
                            if f.stem == mel_stem or f.stem.startswith(mel_stem)
                        ]
                        
                        if matching_face_files:
                            file_pairs.append({
                                'mel_path': str(mel_file),
                                'face_path': str(matching_face_files[0]),
                                'identity': identity
                            })
            
            # vox2에서 찾기
            elif mel_identity_dir_vox2.exists() and face_identity_dir_vox2.exists():
                mel_files = list(mel_identity_dir_vox2.glob("*.npy")) + list(mel_identity_dir_vox2.glob("*.pickle"))
                face_files = list(face_identity_dir_vox2.glob("*.jpg")) + list(face_identity_dir_vox2.glob("*.png"))
                
                print(f"{identity} (vox2): mel 파일 {len(mel_files)}개, face 파일 {len(face_files)}개")
                
                # 파일 매칭 로직
                if len(face_files) == 1:
                    # 얼굴 파일이 하나만 있으면 모든 mel 파일과 매칭
                    face_file = face_files[0]
                    for mel_file in mel_files:
                        file_pairs.append({
                            'mel_path': str(mel_file),
                            'face_path': str(face_file),
                            'identity': identity
                        })
                else:
                    # 파일명 기반으로 매칭
                    for mel_file in mel_files:
                        mel_stem = mel_file.stem
                        
                        # 대응하는 얼굴 파일 찾기
                        matching_face_files = [
                            f for f in face_files 
                            if f.stem == mel_stem or f.stem.startswith(mel_stem)
                        ]
                        
                        if matching_face_files:
                            file_pairs.append({
                                'mel_path': str(mel_file),
                                'face_path': str(matching_face_files[0]),
                                'identity': identity
                            })
            else:
                print(f"경고: {identity} 디렉토리가 vox1 또는 vox2에 존재하지 않습니다.")
                continue
        
        return file_pairs
    
    def _get_image_transform(self):
        """이미지 변환기를 반환합니다."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_mel_spectrogram(self, mel_path):
        """Mel spectrogram을 로드하고 처리합니다."""
        import pickle
        
        if mel_path.endswith('.pickle'):
            # pickle 파일 로드
            with open(mel_path, 'rb') as f:
                mel_data = pickle.load(f)
                # pickle 데이터 구조에 따라 mel spectrogram 추출
                if isinstance(mel_data, dict):
                    # 딕셔너리인 경우 mel spectrogram 키 찾기
                    if 'mel' in mel_data:
                        mel = mel_data['mel']
                    elif 'mel_spectrogram' in mel_data:
                        mel = mel_data['mel_spectrogram']
                    elif 'spectrogram' in mel_data:
                        mel = mel_data['spectrogram']
                    else:
                        # 첫 번째 값 사용
                        mel = list(mel_data.values())[0]
                else:
                    mel = mel_data
        else:
            # numpy 파일 로드
            mel = np.load(mel_path)
        
        # numpy 배열로 변환
        if not isinstance(mel, np.ndarray):
            mel = np.array(mel)
        
        # 디버깅: 차원 출력
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"Mel spectrogram shape: {mel.shape}")
            self._debug_printed = True
        
        # 시간 차원에서 지정된 길이만큼 자르기
        target_frames = int(self.audio_duration_sec * 100)  # 10ms per frame
        if mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
        elif mel.shape[1] < target_frames:
            # 패딩
            pad_width = target_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        
        return torch.FloatTensor(mel)
    
    def _load_face_image(self, face_path):
        """얼굴 이미지를 로드하고 처리합니다."""
        image = Image.open(face_path).convert('RGB')
        return self.image_transform(image)
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        
        # Mel spectrogram 로드
        mel = self._load_mel_spectrogram(pair['mel_path'])
        
        # 얼굴 이미지 로드
        face = self._load_face_image(pair['face_path'])
        
        return {
            'mel': mel,
            'face': face,
            'identity': pair['identity']
        }


def create_hq_voxceleb_dataloaders(split_json_path, 
                                  batch_size=32, num_workers=4,
                                  audio_duration_sec=5, target_sr=16000, 
                                  image_size=224):
    """
    HQ VoxCeleb 데이터셋의 train/val/test 데이터로더를 생성합니다.
    
    Args:
        split_json_path (str): split.json 파일 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩 워커 수
        audio_duration_sec (int): 오디오 길이 (초)
        target_sr (int): 오디오 샘플링 레이트
        image_size (int): 이미지 크기
    
    Returns:
        dict: train, val, test 데이터로더가 포함된 딕셔너리
    """
    from torch.utils.data import DataLoader
    
    dataloaders = {}
    
    for split_type in ['train', 'val', 'test']:
        dataset = HQVoxCelebDataset(
            split_json_path=split_json_path,
            split_type=split_type,
            audio_duration_sec=audio_duration_sec,
            target_sr=target_sr,
            image_size=image_size
        )
        
        dataloaders[split_type] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_type == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def collate_hq_voxceleb_fn(batch):
    """
    HQ VoxCeleb 데이터셋을 위한 collate 함수
    """
    mels = torch.stack([item['mel'] for item in batch])
    faces = torch.stack([item['face'] for item in batch])
    identities = [item['identity'] for item in batch]
    
    return mels, faces, identities 