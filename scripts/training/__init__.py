"""
모델 학습 관련 스크립트들

이 디렉토리는 다음과 같은 학습 작업을 담당합니다:
- 일반 얼굴-음성 매칭 모델 학습 (train_face_voice.py)
- 매칭 파일 기반 모델 학습 (train_face_voice_from_matched_file.py)
"""

from .train_face_voice import *
from .train_face_voice_from_matched_file import *

__all__ = [
    'train_face_voice',
    'train_face_voice_from_matched_file'
] 