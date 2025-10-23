"""
대조학습을 위한 VoxCeleb 데이터셋

이 데이터셋은 얼굴-음성 매칭을 위한 대조학습에 최적화되어 있습니다.
각 identity별로 이미지와 음성을 명확히 매칭하고, 배치 내에서 positive/negative pair를 구성합니다.
"""

import os
import json
import pickle
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import librosa

# 윈도우 환경에서 UTF-8 인코딩 지원을 위한 설정
if sys.platform == "win32":
    import locale
    # 윈도우에서 UTF-8 인코딩 강제 설정
    try:
        # 환경 변수 설정으로 UTF-8 모드 활성화
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        # 로케일 설정
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    except:
        try:
            # 대안 로케일 설정
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            # 로케일 설정 실패 시 기본값 사용
            pass


def safe_file_open(file_path, mode='r', encoding='utf-8'):
    """
    윈도우 환경에서 특수 문자가 포함된 파일명을 안전하게 처리하는 함수
    
    Args:
        file_path: 파일 경로
        mode: 파일 열기 모드
        encoding: 인코딩 방식 (바이너리 모드에서는 무시됨)
    
    Returns:
        파일 객체
    """
    if sys.platform == "win32":
        try:
            # 바이너리 모드인 경우 인코딩 인자 제외
            if 'b' in mode:
                return open(file_path, mode)
            else:
                return open(file_path, mode, encoding=encoding)
        except (UnicodeDecodeError, UnicodeEncodeError):
            # 대안: 파일명을 바이트로 처리
            if isinstance(file_path, str):
                file_path_bytes = file_path.encode('utf-8')
                if 'b' in mode:
                    return open(file_path_bytes, mode)
                else:
                    return open(file_path_bytes, mode, encoding=encoding)
            else:
                raise
    else:
        # 바이너리 모드인 경우 인코딩 인자 제외
        if 'b' in mode:
            return open(file_path, mode)
        else:
            return open(file_path, mode, encoding=encoding)


def safe_listdir(dir_path):
    """
    윈도우 환경에서 특수 문자가 포함된 디렉토리를 안전하게 스캔하는 함수
    
    Args:
        dir_path: 디렉토리 경로
    
    Returns:
        파일/디렉토리 이름 리스트
    """
    if sys.platform == "win32":
        try:
            # 먼저 일반적인 방법 시도
            return os.listdir(dir_path)
        except (UnicodeDecodeError, UnicodeEncodeError, OSError):
            # 대안: pathlib 사용
            try:
                from pathlib import Path
                path_obj = Path(dir_path)
                return [item.name for item in path_obj.iterdir()]
            except Exception:
                # 최후의 수단: 빈 리스트 반환
                print(f"경고: 디렉토리 '{dir_path}' 스캔 실패 - 특수문자 처리 불가")
                return []
    else:
        return os.listdir(dir_path)


class ContrastiveVoxCelebDataset(Dataset):
    """대조학습을 위한 VoxCeleb 데이터셋
    
    각 identity별로 이미지와 음성을 명확히 매칭하여 대조학습을 수행합니다.
    """
    
    def __init__(self, 
                 voxceleb_root: str,
                 dataset_type: str = 'vox1',
                 split_type: str = 'train',
                 image_transform: Optional[transforms.Compose] = None,
                 audio_transform: Optional[callable] = None,
                 min_samples_per_identity: int = 2,
                 max_samples_per_identity: Optional[int] = None,
                 fixed_identities: Optional[List[str]] = None,
                 random_seed: int = 42):
        """
        ContrastiveVoxCelebDataset 초기화
        
        Args:
            voxceleb_root: VoxCeleb 데이터 루트 디렉토리
            dataset_type: 'vox1' 또는 'vox2'
            split_type: 'train', 'val', 'test'
            image_transform: 이미지 변환기
            audio_transform: 오디오 변환기
            min_samples_per_identity: identity당 최소 샘플 수
            max_samples_per_identity: identity당 최대 샘플 수
            random_seed: 랜덤 시드
        """
        self.voxceleb_root = Path(voxceleb_root)
        self.dataset_type = dataset_type
        self.split_type = split_type
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.min_samples_per_identity = min_samples_per_identity
        self.max_samples_per_identity = max_samples_per_identity
        self.fixed_identities = fixed_identities
        self.random_seed = random_seed
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        
        # 데이터 경로 설정
        # voxceleb_root는 이미 HQ-VoxCeleb까지의 경로
        self.faces_dir = self.voxceleb_root / dataset_type / 'masked_faces'
        # 멜스펙트로그램은 상위 디렉토리의 vox1/vox2에 있음
        mel_root = self.voxceleb_root.parent.parent / dataset_type
        self.mel_dir = mel_root / 'mel_spectrograms'
        
        # 디렉토리 존재 확인
        if not self.faces_dir.exists():
            raise FileNotFoundError(f"얼굴 이미지 디렉토리를 찾을 수 없습니다: {self.faces_dir}")
        if not self.mel_dir.exists():
            raise FileNotFoundError(f"멜스펙트로그램 디렉토리를 찾을 수 없습니다: {self.mel_dir}")
        
        # 각 identity별로 데이터 스캔 및 샘플 생성
        self.samples = self._create_samples()
        print(f"Contrastive VoxCeleb {dataset_type} {split_type}: {len(self.samples)}개 샘플")
        print(f"이미지와 멜스펙트로그램이 모두 존재하는 쌍만 학습 데이터에 포함되었습니다.")
        
    def _create_samples(self) -> List[Dict]:
        """각 identity별로 이미지와 오디오를 스캔하여 샘플 생성"""
        samples = []
        
        # 모든 identity 디렉토리 스캔 (인코딩 문제 해결)
        identity_dirs = []
        try:
            # 윈도우에서 특수 문자 처리 개선
            for d in self.faces_dir.iterdir():
                if d.is_dir():
                    identity_dirs.append(d)
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            print(f"경고: 디렉토리 스캔 중 인코딩 오류 발생: {e}")
            # 대안 방법: 안전한 디렉토리 스캔 사용
            try:
                dir_names = safe_listdir(str(self.faces_dir))
                for dir_name in dir_names:
                    dir_path = self.faces_dir / dir_name
                    if dir_path.is_dir():
                        identity_dirs.append(dir_path)
            except Exception as e2:
                print(f"경고: 대안 방법도 실패: {e2}")
                return []
        
        # identity별로 샘플 수를 확인하고 분할
        identity_samples = {}
        
        for identity_dir in identity_dirs:
            try:
                identity = identity_dir.name
                
                # 이미지 파일 목록 (인코딩 안전하게 처리)
                image_files = []
                try:
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_files.extend(identity_dir.glob(ext))
                    image_files = [str(f) for f in image_files]
                except (UnicodeDecodeError, UnicodeEncodeError) as e:
                    print(f"경고: {identity} 이미지 파일 스캔 중 인코딩 오류: {e}")
                    # 대안 방법 사용: 안전한 디렉토리 스캔
                    try:
                        file_names = safe_listdir(str(identity_dir))
                        image_files = []
                        for file_name in file_names:
                            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                image_files.append(str(identity_dir / file_name))
                    except Exception as e2:
                        print(f"경고: {identity} 대안 이미지 스캔도 실패: {e2}")
                        continue
                
                # 오디오 파일 목록 (같은 identity의 멜스펙트로그램 디렉토리에서)
                mel_dir_path = self.mel_dir / identity
                
                # 윈도우에서 특수문자 포함 디렉토리 존재 여부 안전하게 확인
                mel_dir_exists = False
                actual_mel_dir_name = None
                
                try:
                    mel_dir_exists = mel_dir_path.exists()
                    if mel_dir_exists:
                        actual_mel_dir_name = identity
                except (UnicodeDecodeError, UnicodeEncodeError, OSError):
                    # 대안: 안전한 디렉토리 스캔으로 존재 여부 확인
                    try:
                        mel_parent_files = safe_listdir(str(self.mel_dir))
                        # 정확한 매칭을 위해 여러 방법 시도
                        if identity in mel_parent_files:
                            mel_dir_exists = True
                            actual_mel_dir_name = identity
                        else:
                            # 유니코드 정규화를 통한 매칭 시도
                            import unicodedata
                            normalized_identity = unicodedata.normalize('NFC', identity)
                            for dir_name in mel_parent_files:
                                normalized_dir_name = unicodedata.normalize('NFC', dir_name)
                                if normalized_identity == normalized_dir_name:
                                    mel_dir_exists = True
                                    actual_mel_dir_name = dir_name
                                    break
                            
                            # 여전히 매칭되지 않으면 유사한 이름 찾기
                            if not mel_dir_exists:
                                # 대소문자 무시하고 매칭
                                identity_lower = identity.lower()
                                for dir_name in mel_parent_files:
                                    if identity_lower == dir_name.lower():
                                        mel_dir_exists = True
                                        actual_mel_dir_name = dir_name
                                        break
                                
                                # 여전히 매칭되지 않으면 부분 매칭 시도
                                if not mel_dir_exists:
                                    for dir_name in mel_parent_files:
                                        # 공백이나 언더스코어를 제거하고 비교
                                        clean_identity = identity.replace('_', '').replace(' ', '').lower()
                                        clean_dir_name = dir_name.replace('_', '').replace(' ', '').lower()
                                        if clean_identity == clean_dir_name:
                                            mel_dir_exists = True
                                            actual_mel_dir_name = dir_name
                                            break
                                
                                # 여전히 매칭되지 않으면 더 유연한 매칭 시도
                                if not mel_dir_exists:
                                    # 이름의 주요 부분만 추출하여 비교
                                    identity_parts = identity.replace('_', ' ').split()
                                    for dir_name in mel_parent_files:
                                        dir_parts = dir_name.replace('_', ' ').split()
                                        # 주요 부분이 일치하는지 확인 (최소 2개 부분 일치)
                                        common_parts = set(identity_parts) & set(dir_parts)
                                        if len(common_parts) >= min(2, len(identity_parts), len(dir_parts)):
                                            mel_dir_exists = True
                                            actual_mel_dir_name = dir_name
                                            break
                                
                                # 여전히 매칭되지 않으면 문자열 유사도 기반 매칭
                                if not mel_dir_exists:
                                    def levenshtein_distance(s1, s2):
                                        """레벤슈타인 거리 계산"""
                                        if len(s1) < len(s2):
                                            return levenshtein_distance(s2, s1)
                                        if len(s2) == 0:
                                            return len(s1)
                                        
                                        previous_row = list(range(len(s2) + 1))
                                        for i, c1 in enumerate(s1):
                                            current_row = [i + 1]
                                            for j, c2 in enumerate(s2):
                                                insertions = previous_row[j + 1] + 1
                                                deletions = current_row[j] + 1
                                                substitutions = previous_row[j] + (c1 != c2)
                                                current_row.append(min(insertions, deletions, substitutions))
                                            previous_row = current_row
                                        return previous_row[-1]
                                    
                                    identity_clean = identity.replace('_', '').replace(' ', '').lower()
                                    best_match = None
                                    best_score = float('inf')
                                    
                                    for dir_name in mel_parent_files:
                                        dir_clean = dir_name.replace('_', '').replace(' ', '').lower()
                                        distance = levenshtein_distance(identity_clean, dir_clean)
                                        # 유사도가 높고 길이가 비슷한 경우
                                        if distance < best_score and distance <= max(len(identity_clean), len(dir_clean)) * 0.3:
                                            best_score = distance
                                            best_match = dir_name
                                    
                                    if best_match:
                                        mel_dir_exists = True
                                        actual_mel_dir_name = best_match
                    except Exception as e:
                        print(f"경고: 디렉토리 매칭 중 오류: {e}")
                        mel_dir_exists = False
                
                if not mel_dir_exists:
                    print(f"경고: {identity}의 멜스펙트로그램 디렉토리가 없습니다. 건너뜁니다.")
                    continue
                
                # 실제 디렉토리명으로 경로 재설정
                if actual_mel_dir_name and actual_mel_dir_name != identity:
                    mel_dir_path = self.mel_dir / actual_mel_dir_name
                    print(f"{identity} -> {actual_mel_dir_name} 디렉토리명 매칭 성공")
                    
                mel_files = []
                try:
                    for ext in ['*.pickle']:
                        mel_files.extend(mel_dir_path.glob(ext))
                    mel_files = [str(f) for f in mel_files]
                except (UnicodeDecodeError, UnicodeEncodeError) as e:
                    print(f"경고: {identity} 멜스펙트로그램 파일 스캔 중 인코딩 오류: {e}")
                    # 대안 방법 사용: 안전한 디렉토리 스캔
                    try:
                        file_names = safe_listdir(str(mel_dir_path))
                        mel_files = []
                        for file_name in file_names:
                            if file_name.lower().endswith('.pickle'):
                                mel_files.append(str(mel_dir_path / file_name))
                    except Exception as e2:
                        print(f"경고: {identity} 대안 멜스펙트로그램 스캔도 실패: {e2}")
                        continue
            except Exception as e:
                print(f"경고: {identity_dir} 처리 중 오류 발생: {e}")
                continue
            
            # 이미지와 멜스펙트로그램이 모두 존재하는지 엄격하게 확인
            if not image_files:
                print(f"경고: {identity}에 이미지 파일이 없습니다. 건너뜁니다.")
                continue
                
            if not mel_files:
                print(f"경고: {identity}에 멜스펙트로그램 파일이 없습니다. 건너뜁니다.")
                continue
            
            # 최소 샘플 수 확인 (양쪽 모두 존재해야 함)
            min_samples = min(len(image_files), len(mel_files))
            if min_samples < self.min_samples_per_identity:
                print(f"경고: {identity}의 샘플 수({min_samples})가 최소 요구사항({self.min_samples_per_identity})보다 적습니다. 건너뜁니다.")
                continue
                
            print(f"{identity}: 이미지 {len(image_files)}개, 멜스펙트로그램 {len(mel_files)}개 - 학습 데이터에 포함")
            
            # identity당 샘플 수 제한
            if self.max_samples_per_identity:
                max_samples = min(len(image_files), len(mel_files), self.max_samples_per_identity)
                image_files = random.sample(image_files, max_samples)
                mel_files = random.sample(mel_files, max_samples)
            
            # 샘플 생성: 각 identity 내에서 이미지와 멜스펙트로그램을 매칭
            identity_samples[identity] = []
            for i in range(min_samples):
                # 같은 identity 내에서 랜덤하게 선택
                image_path = random.choice(image_files)
                mel_path = random.choice(mel_files)
                
                # 실제 파일 존재 여부 확인
                if not os.path.exists(image_path):
                    print(f"경고: 이미지 파일이 존재하지 않습니다: {image_path}")
                    continue
                if not os.path.exists(mel_path):
                    print(f"경고: 멜스펙트로그램 파일이 존재하지 않습니다: {mel_path}")
                    continue
                
                identity_samples[identity].append({
                    'identity': identity,
                    'image_path': image_path,
                    'mel_path': mel_path
                })
        
        # Identity 목록 준비
        all_identities = list(identity_samples.keys())
        
        # 고정된 identity 목록이 주어진 경우 우선 사용
        if self.fixed_identities is not None:
            # 존재하는 identity만 필터링
            selected_identities = [i for i in self.fixed_identities if i in identity_samples]
        else:
            # 무작위 분할 수행
            identities = all_identities[:]
            random.shuffle(identities)
            
            # 분할 비율: train 80%, val 10%, test 10%
            n_identities = len(identities)
            train_end = int(0.8 * n_identities)
            val_end = int(0.9 * n_identities)
            
            train_identities = identities[:train_end]
            val_identities = identities[train_end:val_end]
            test_identities = identities[val_end:]
            
            # split_type에 따라 해당하는 identity들의 샘플만 선택
            if self.split_type == 'train':
                selected_identities = train_identities
            elif self.split_type == 'val':
                selected_identities = val_identities
            elif self.split_type == 'test':
                selected_identities = test_identities
            else:
                # 전체 사용
                selected_identities = identities
        
        # 선택된 identity들의 샘플을 모두 추가
        for identity in selected_identities:
            samples.extend(identity_samples[identity])
        
        print(f"데이터 분할: {self.split_type} - {len(selected_identities)}개 identity, {len(samples)}개 샘플")
        print(f"📊 유효한 identity 목록: {selected_identities[:5]}{'...' if len(selected_identities) > 5 else ''}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """데이터셋에서 아이템 가져오기"""
        sample = self.samples[idx]
        
        # 파일 존재 여부 재확인
        if not os.path.exists(sample['image_path']):
            print(f"경고: 이미지 파일이 존재하지 않습니다: {sample['image_path']}")
            # 기본 이미지 생성
            image_tensor = torch.zeros(3, 224, 224)
        elif not os.path.exists(sample['mel_path']):
            print(f"경고: 멜스펙트로그램 파일이 존재하지 않습니다: {sample['mel_path']}")
            # 기본 이미지 생성
            image_tensor = torch.zeros(3, 224, 224)
        else:
            # 이미지 로드 및 변환 (인코딩 안전하게 처리)
            try:
                image = Image.open(sample['image_path']).convert("RGB")
                if self.image_transform:
                    image_tensor = self.image_transform(image)
                else:
                    # 기본 변환
                    image_tensor = transforms.ToTensor()(image)
            except Exception as e:
                print(f"이미지 로드 오류: {sample['image_path']} - {e}")
                # 기본 이미지 생성
                image_tensor = torch.zeros(3, 224, 224)
        
        # 멜스펙트로그램 로드 (인코딩 안전하게 처리)
        mel_path = sample['mel_path']
        
        # 파일 존재 여부 재확인
        if not os.path.exists(mel_path):
            print(f"경고: 멜스펙트로그램 파일이 존재하지 않습니다: {mel_path}")
            # 기본 멜스펙트로그램 생성
            mel_tensor = torch.zeros(1, 40, 100)
        else:
            try:
                # 안전한 파일 열기 사용
                with safe_file_open(mel_path, 'rb') as f:
                    mel_data = pickle.load(f)
            
                # 딕셔너리인 경우 'mel' 키로 멜스펙트로그램 추출
                if isinstance(mel_data, dict):
                    if 'mel' in mel_data:
                        mel_tensor = torch.tensor(mel_data['mel'], dtype=torch.float32)
                    elif 'mel_spectrogram' in mel_data:
                        mel_tensor = torch.tensor(mel_data['mel_spectrogram'], dtype=torch.float32)
                    else:
                        # 딕셔너리의 첫 번째 값 사용
                        first_key = list(mel_data.keys())[0]
                        mel_tensor = torch.tensor(mel_data[first_key], dtype=torch.float32)
                else:
                    # 딕셔너리가 아닌 경우 직접 텐서로 변환
                    mel_tensor = torch.tensor(mel_data, dtype=torch.float32)
                
                # 2D를 3D로 변환 (채널 차원 추가)
                if mel_tensor.dim() == 2:
                    mel_tensor = mel_tensor.unsqueeze(0)  # (1, freq, time)
                    
            except Exception as e:
                print(f"멜스펙트로그램 로드 오류: {mel_path} - {e}")
                # 기본 멜스펙트로그램 생성
                mel_tensor = torch.zeros(1, 40, 100)
        
        return {
            'image': image_tensor,
            'mel': mel_tensor,
            'identity': sample['identity']
        }


def collate_contrastive_fn(batch):
    """
    대조학습을 위한 collate 함수
    
    배치 내에서 같은 identity의 여러 샘플을 포함하여 positive pair를 생성합니다.
    """
    # Identity별로 그룹화
    identity_groups = {}
    for item in batch:
        identity = item['identity']
        if identity not in identity_groups:
            identity_groups[identity] = []
        identity_groups[identity].append(item)
    
    # 각 identity에서 여러 샘플을 선택하여 균형 잡힌 배치 구성
    balanced_batch = []
    identities = list(identity_groups.keys())
    
    if len(identities) == 0:
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'mel': torch.stack([item['mel'] for item in batch]),
            'identity': [item['identity'] for item in batch]
        }
    
    # 원래 배치 구성 (자연스러운 positive pair 생성)
    # 각 identity에서 최소 1개씩 샘플을 선택하되, 가능하면 2개씩 선택
    samples_per_identity = max(1, len(batch) // len(identities))
    
    # 먼저 각 identity에서 기본 샘플 수만큼 선택
    for identity in identities:
        samples = identity_groups[identity]
        for _ in range(min(samples_per_identity, len(samples))):
            balanced_batch.append(random.choice(samples))
    
    # 배치 크기가 부족하면 추가 샘플로 채움
    while len(balanced_batch) < len(batch):
        identity = random.choice(identities)
        balanced_batch.append(random.choice(identity_groups[identity]))
    
    # 배치 크기가 초과하면 자르기
    balanced_batch = balanced_batch[:len(batch)]
    
    # 배치 구성 확인 (로깅 제거)
    identity_counts = {}
    for item in balanced_batch:
        identity = item['identity']
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    
    positive_pairs = sum(1 for count in identity_counts.values() if count >= 2)
    # 로깅 제거 (너무 많은 출력 방지)
    
    # 이미지와 멜스펙트로그램 스택
    images = torch.stack([item['image'] for item in balanced_batch])
    
    # 멜스펙트로그램 패딩 처리
    mels = [item['mel'] for item in balanced_batch]
    
    # 각 멜스펙트로그램을 (freq, time) 형태로 변환
    mels_2d = []
    for mel in mels:
        if mel.dim() == 3:  # (1, freq, time)
            mel = mel.squeeze(0)  # (freq, time)
        mels_2d.append(mel)
    
    # 2D 텐서 패딩을 위한 최대 크기 찾기
    max_freq = max(mel.shape[0] for mel in mels_2d)
    max_time = max(mel.shape[1] for mel in mels_2d)
    
    # 패딩된 텐서 생성
    mels_padded = []
    for mel in mels_2d:
        # (freq, time) -> (max_freq, max_time)로 패딩
        padded_mel = torch.zeros(max_freq, max_time)
        padded_mel[:mel.shape[0], :mel.shape[1]] = mel
        mels_padded.append(padded_mel)
    
    # 배치로 스택하고 (batch, 1, freq, time) 형태로 변환
    mels_padded = torch.stack(mels_padded).unsqueeze(1)
    
    identities = [item['identity'] for item in balanced_batch]
    
    return {
        'image': images,
        'mel': mels_padded,
        'identity': identities
    }


def create_contrastive_transforms(use_augmentation: bool = True, image_size: int = 224):
    """
    대조학습용 이미지 변환기 생성
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if use_augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def create_contrastive_voxceleb_dataloaders(voxceleb_root: str,
                                           dataset_types: List[str] = ['vox1'],
                                           batch_size: int = 16,
                                           num_workers: int = 4,
                                           image_size: int = 224,
                                           min_samples_per_identity: int = 2,
                                           max_samples_per_identity: Optional[int] = None,
                                           random_seed: int = 42,
                                           split_save_dir: Optional[str] = None,
                                           split_load_dir: Optional[str] = None):
    """
    대조학습용 VoxCeleb 데이터로더들을 생성
    
    Args:
        voxceleb_root: VoxCeleb 데이터 루트 디렉토리
        dataset_types: 사용할 데이터셋 타입 리스트
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        image_size: 이미지 크기
        min_samples_per_identity: identity당 최소 샘플 수
        max_samples_per_identity: identity당 최대 샘플 수
        random_seed: 랜덤 시드
        
    Returns:
        train, val 데이터로더 딕셔너리
    """
    from torch.utils.data import DataLoader, ConcatDataset
    
    # 변환기 생성
    train_transform = create_contrastive_transforms(use_augmentation=True, image_size=image_size)
    val_transform = create_contrastive_transforms(use_augmentation=False, image_size=image_size)
    
    # 각 데이터셋 타입별로 데이터셋 생성
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for dataset_type in dataset_types:
        print(f"대조학습 데이터셋 {dataset_type} 로딩 중...")
        
        # 고정 split 로드 (존재하면 사용) - 인코딩 안전하게 처리
        fixed_splits = None
        if split_load_dir is not None:
            split_path = Path(split_load_dir) / f"contrastive_split_{dataset_type}.json"
            if split_path.exists():
                try:
                    # 안전한 파일 열기 사용
                    with safe_file_open(str(split_path), 'r', encoding='utf-8') as f:
                        fixed_splits = json.load(f)
                    print(f"고정 split 로드: {split_path}")
                except Exception as e:
                    print(f"경고: split 파일 로드 실패: {split_path} - {e}")
        
        # 훈련 데이터셋
        train_dataset = ContrastiveVoxCelebDataset(
            voxceleb_root=voxceleb_root,
            dataset_type=dataset_type,
            split_type='train',
            image_transform=train_transform,
            min_samples_per_identity=min_samples_per_identity,
            max_samples_per_identity=max_samples_per_identity,
            fixed_identities=(fixed_splits['train'] if fixed_splits and 'train' in fixed_splits else None),
            random_seed=random_seed
        )
        train_datasets.append(train_dataset)
        
        # 검증 데이터셋
        val_dataset = ContrastiveVoxCelebDataset(
            voxceleb_root=voxceleb_root,
            dataset_type=dataset_type,
            split_type='val',
            image_transform=val_transform,
            min_samples_per_identity=min_samples_per_identity,
            max_samples_per_identity=max_samples_per_identity,
            fixed_identities=(fixed_splits['val'] if fixed_splits and 'val' in fixed_splits else None),
            random_seed=random_seed
        )
        val_datasets.append(val_dataset)
        
        # 테스트 데이터셋
        test_dataset = ContrastiveVoxCelebDataset(
            voxceleb_root=voxceleb_root,
            dataset_type=dataset_type,
            split_type='test',
            image_transform=val_transform,  # 테스트는 augmentation 없이
            min_samples_per_identity=min_samples_per_identity,
            max_samples_per_identity=max_samples_per_identity,
            fixed_identities=(fixed_splits['test'] if fixed_splits and 'test' in fixed_splits else None),
            random_seed=random_seed
        )
        test_datasets.append(test_dataset)

        # 고정 split 저장 (요청된 경우, 그리고 이번에 무작위 분할을 생성한 경우)
        if split_save_dir is not None and fixed_splits is None:
            # 방금 생성한 분할을 저장하기 위해 datasets에서 identity 목록 수집
            def collect_identities(ds: ContrastiveVoxCelebDataset) -> List[str]:
                # samples에서 identity를 수집하여 unique 정렬
                return sorted(list({s['identity'] for s in ds.samples}))
            to_save = {
                'train': collect_identities(train_dataset),
                'val': collect_identities(val_dataset),
                'test': collect_identities(test_dataset)
            }
            Path(split_save_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(split_save_dir) / f"contrastive_split_{dataset_type}.json"
            try:
                # 안전한 파일 열기 사용
                with safe_file_open(str(out_path), 'w', encoding='utf-8') as f:
                    json.dump(to_save, f, ensure_ascii=False, indent=2)
                print(f"고정 split 저장: {out_path}")
            except Exception as e:
                print(f"경고: split 저장 실패: {out_path} - {e}")
    
    # 여러 데이터셋을 하나로 합치기
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)
        print(f"여러 데이터셋 통합: {dataset_types}")
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
        test_dataset = test_datasets[0]
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_contrastive_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_contrastive_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_contrastive_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
