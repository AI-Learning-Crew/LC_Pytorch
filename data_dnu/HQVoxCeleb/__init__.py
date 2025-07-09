"""
HQ VoxCeleb 데이터 패키지

이 패키지는 고품질 VoxCeleb 데이터셋을 위한 특화된 데이터 로딩 클래스들을 포함합니다.
VoxCeleb는 유명인의 인터뷰 영상에서 추출한 얼굴-음성 쌍 데이터셋으로,
얼굴-음성 매칭 연구에 널리 사용됩니다.

주요 구성요소:
- HQVoxCelebDataset: 고품질 VoxCeleb 데이터셋 클래스
- create_hq_voxceleb_dataloaders: 데이터 로더 생성 함수
- collate_hq_voxceleb_fn: 배치 데이터 처리 함수
"""

# 고품질 VoxCeleb 데이터셋 관련 클래스와 함수들
from .hq_voxceleb_dataset import (
    HQVoxCelebDataset, create_hq_voxceleb_dataloaders, collate_hq_voxceleb_fn
)

# 외부에서 import할 수 있는 클래스와 함수들의 목록
__all__ = ['HQVoxCelebDataset', 'create_hq_voxceleb_dataloaders', 'collate_hq_voxceleb_fn'] 