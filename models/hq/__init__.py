"""
HQ (High Quality) 모델 패키지

이 패키지는 고품질 VoxCeleb 데이터셋을 위한 특화된 모델들을 포함합니다.
기본 모델보다 더 정교한 아키텍처와 손실 함수를 사용하여 
더 나은 얼굴-음성 매칭 성능을 제공합니다.
"""

# 고품질 VoxCeleb 모델 관련 클래스와 함수들
from .hq_voxceleb_model import (
    HQVoxCelebModel, HQVoxCelebInfoNCELoss, save_hq_voxceleb_model_components
)

# 외부에서 import할 수 있는 클래스와 함수들의 목록
__all__ = ['HQVoxCelebModel', 'HQVoxCelebInfoNCELoss', 'save_hq_voxceleb_model_components'] 