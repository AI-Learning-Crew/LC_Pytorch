"""
모델 패키지

이 패키지는 얼굴-음성 매칭을 위한 딥러닝 모델들을 포함합니다.
주요 구성요소:
- FaceVoiceModel: 기본 얼굴-음성 매칭 모델
- InfoNCELoss: InfoNCE 손실 함수
- HQVoxCelebModel: 고품질 VoxCeleb 데이터셋용 모델
"""

# 기본 얼굴-음성 매칭 모델 관련 클래스들
from .face_voice_model import FaceVoiceModel, InfoNCELoss, save_model_components, load_model_components
# 고품질 VoxCeleb 모델 관련 클래스들
from .hq.hq_voxceleb_model import HQVoxCelebModel, HQVoxCelebInfoNCELoss, save_hq_voxceleb_model_components

# 외부에서 import할 수 있는 클래스와 함수들의 목록
__all__ = [
    'FaceVoiceModel', 'InfoNCELoss', 'save_model_components', 'load_model_components',
    'HQVoxCelebModel', 'HQVoxCelebInfoNCELoss', 'save_hq_voxceleb_model_components'
]
