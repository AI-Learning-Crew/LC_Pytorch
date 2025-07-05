"""
HQ VoxCeleb 데이터셋을 위한 전용 데이터셋 클래스 (멀티프로세싱 최적화)
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
import concurrent.futures
import multiprocessing as mp
from functools import lru_cache
import pickle
from typing import Dict, List, Tuple, Any
import time


class HQVoxCelebDataset(Dataset):
    """
    HQ VoxCeleb 데이터셋을 위한 데이터셋 클래스 (멀티프로세싱 최적화)
    
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
                 audio_duration_sec=5, target_sr=16000, image_size=224,
                 enable_parallel=True, cache_size=2000):
        """
        Args:
            split_json_path (str): split.json 파일 경로
            split_type (str): 'train', 'val', 'test' 중 하나
            audio_duration_sec (int): 오디오 길이 (초)
            target_sr (int): 오디오 샘플링 레이트
            image_size (int): 이미지 크기
            enable_parallel (bool): 병렬 처리 활성화
            cache_size (int): 캐시 크기
        """
        self.split_type = split_type
        self.audio_duration_sec = audio_duration_sec
        self.target_sr = target_sr
        self.image_size = image_size
        self.enable_parallel = enable_parallel
        self.cache_size = cache_size
        
        # 멀티프로세싱에서 pickle 불가능한 객체들을 제거
        # 캐시는 각 워커에서 개별적으로 관리
        self.cache = {}
        self.access_count = {}
        
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
        self._check_alternative_paths()
        
        # 파일 쌍 생성 (병렬 처리 가능)
        if self.enable_parallel:
            self.file_pairs = self._create_file_pairs_parallel()
        else:
            self.file_pairs = self._create_file_pairs()
        
        # 데이터 변환기 설정
        self.image_transform = self._get_image_transform()
        
        # 오디오 프로세서는 필요할 때만 로드
        self.audio_processor = None
        
        print(f"HQ VoxCeleb {split_type}: {len(self.file_pairs)}개 파일 쌍 {'(병렬 처리)' if enable_parallel else ''}")
    
    def _check_alternative_paths(self):
        """대체 경로 확인"""
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
    
    def _create_file_pairs(self):
        """기존 방식으로 파일 쌍을 생성합니다."""
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
                
                file_pairs.extend(self._match_files(mel_files, face_files, identity))
                
            # vox2에서 찾기
            elif mel_identity_dir_vox2.exists() and face_identity_dir_vox2.exists():
                mel_files = list(mel_identity_dir_vox2.glob("*.npy")) + list(mel_identity_dir_vox2.glob("*.pickle"))
                face_files = list(face_identity_dir_vox2.glob("*.jpg")) + list(face_identity_dir_vox2.glob("*.png"))
                
                file_pairs.extend(self._match_files(mel_files, face_files, identity))
            else:
                continue
        
        return file_pairs
    
    def _process_identity_parallel(self, identity):
        """단일 identity 처리 (병렬 처리용)"""
        file_pairs = []
        
        # vox1과 vox2 디렉토리 확인
        mel_identity_dir_vox1 = self.vox1_mel_dir / identity
        face_identity_dir_vox1 = self.vox1_face_dir / identity
        mel_identity_dir_vox2 = self.vox2_mel_dir / identity
        face_identity_dir_vox2 = self.vox2_face_dir / identity
        
        # vox1에서 찾기
        if mel_identity_dir_vox1.exists() and face_identity_dir_vox1.exists():
            mel_files = list(mel_identity_dir_vox1.glob("*.npy")) + list(mel_identity_dir_vox1.glob("*.pickle"))
            face_files = list(face_identity_dir_vox1.glob("*.jpg")) + list(face_identity_dir_vox1.glob("*.png"))
            
            file_pairs.extend(self._match_files(mel_files, face_files, identity))
            
        # vox2에서 찾기
        elif mel_identity_dir_vox2.exists() and face_identity_dir_vox2.exists():
            mel_files = list(mel_identity_dir_vox2.glob("*.npy")) + list(mel_identity_dir_vox2.glob("*.pickle"))
            face_files = list(face_identity_dir_vox2.glob("*.jpg")) + list(face_identity_dir_vox2.glob("*.png"))
            
            file_pairs.extend(self._match_files(mel_files, face_files, identity))
        
        return file_pairs
    
    def _match_files(self, mel_files, face_files, identity):
        """파일 매칭 로직"""
        file_pairs = []
        
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
        
        return file_pairs
    
    def _create_file_pairs_parallel(self):
        """병렬 처리로 파일 쌍을 생성합니다."""
        print(f"병렬 처리로 {len(self.identities)}개 identity 처리 중...")
        
        # 병렬 처리로 identity별 파일 쌍 생성
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._process_identity_parallel, identity) 
                      for identity in self.identities]
            
            file_pairs = []
            for future in concurrent.futures.as_completed(futures):
                file_pairs.extend(future.result())
        
        print(f"총 {len(file_pairs)}개 파일 쌍 생성 완료")
        return file_pairs
    
    def _get_image_transform(self):
        """이미지 변환기를 반환합니다."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_audio_processor(self):
        """오디오 프로세서를 지연 로드합니다."""
        if self.audio_processor is None:
            self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        return self.audio_processor
    
    def _load_mel_spectrogram(self, mel_path):
        """Mel spectrogram을 로드하고 처리합니다."""
        if mel_path.endswith('.pickle'):
            with open(mel_path, 'rb') as f:
                mel_data = pickle.load(f)
                if isinstance(mel_data, dict):
                    if 'mel' in mel_data:
                        mel = mel_data['mel']
                    elif 'mel_spectrogram' in mel_data:
                        mel = mel_data['mel_spectrogram']
                    elif 'spectrogram' in mel_data:
                        mel = mel_data['spectrogram']
                    else:
                        mel = list(mel_data.values())[0]
                else:
                    mel = mel_data
        else:
            mel = np.load(mel_path)
        
        if not isinstance(mel, np.ndarray):
            mel = np.array(mel)
        
        # 시간 차원 조정
        target_frames = int(self.audio_duration_sec * 100)
        if mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
        elif mel.shape[1] < target_frames:
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
        try:
            # 간단한 캐시 확인 (lock 없이)
            if idx in self.cache:
                return self.cache[idx]
            
            pair = self.file_pairs[idx]
            
            # 데이터 로드
            mel = self._load_mel_spectrogram(pair['mel_path'])
            face = self._load_face_image(pair['face_path'])
            
            data = {
                'mel': mel,
                'face': face,
                'identity': pair['identity']
            }
            
            # 캐시에 저장 (크기 제한)
            if len(self.cache) < self.cache_size:
                self.cache[idx] = data
            
            return data
            
        except Exception as e:
            print(f"데이터 로드 오류 (idx={idx}): {e}")
            # 오류 발생 시 첫 번째 샘플을 반환 (안전장치)
            if idx > 0:
                return self.__getitem__(0)
            else:
                # 첫 번째 샘플도 오류인 경우 더미 데이터 반환
                dummy_mel = torch.zeros(80, int(self.audio_duration_sec * 100))
                dummy_face = torch.zeros(3, self.image_size, self.image_size)
                return {
                    'mel': dummy_mel,
                    'face': dummy_face,
                    'identity': 'dummy'
                }


def create_hq_voxceleb_dataloaders(split_json_path, 
                                  batch_size=64, num_workers=2,
                                  audio_duration_sec=3, target_sr=16000, 
                                  image_size=224, prefetch_factor=2,
                                  pin_memory=True, persistent_workers=True,
                                  enable_parallel=True, cache_size=500):
    """
    HQ VoxCeleb 데이터셋의 train/val/test 데이터로더를 생성합니다.
    
    Args:
        split_json_path (str): split.json 파일 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩 워커 수
        audio_duration_sec (int): 오디오 길이 (초)
        target_sr (int): 오디오 샘플링 레이트
        image_size (int): 이미지 크기
        prefetch_factor (int): 워커당 미리 로드할 배치 수
        pin_memory (bool): GPU 메모리 고정
        persistent_workers (bool): 워커 재사용
        enable_parallel (bool): 병렬 처리 활성화
        cache_size (int): 캐시 크기
    
    Returns:
        dict: train, val, test 데이터로더가 포함된 딕셔너리
    """
    from torch.utils.data import DataLoader
    
    # 시스템 최적화 설정
    if enable_parallel:
        torch.set_num_threads(1)
    
    dataloaders = {}
    
    for split_type in ['train', 'val', 'test']:
        dataset = HQVoxCelebDataset(
            split_json_path=split_json_path,
            split_type=split_type,
            audio_duration_sec=audio_duration_sec,
            target_sr=target_sr,
            image_size=image_size,
            enable_parallel=enable_parallel,
            cache_size=cache_size
        )
        
        # 기본 데이터로더 설정
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': (split_type == 'train'),
            'drop_last': True,
        }
        
        # 워커 수에 따른 설정 분기
        if num_workers > 0:
            # 멀티프로세싱 모드
            dataloader_kwargs.update({
                'num_workers': num_workers,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor,
                'pin_memory': pin_memory and torch.cuda.is_available(),
                'timeout': 60,
                'multiprocessing_context': 'spawn',
            })
        else:
            # 단일 프로세스 모드 (안정성 우선)
            dataloader_kwargs.update({
                'num_workers': 0,
                'persistent_workers': False,
                'prefetch_factor': None,  # 중요: 단일 프로세스일 때는 None
                'pin_memory': False,  # 단일 프로세스에서는 pin_memory 비활성화
            })
        
        dataloaders[split_type] = DataLoader(dataset, **dataloader_kwargs)
    
    return dataloaders


def collate_hq_voxceleb_fn(batch):
    """
    HQ VoxCeleb 데이터셋을 위한 collate 함수
    """
    mels = torch.stack([item['mel'] for item in batch])
    faces = torch.stack([item['face'] for item in batch])
    identities = [item['identity'] for item in batch]
    
    return {
        'mel': mels,
        'face': faces,
        'identity': identities
    } 