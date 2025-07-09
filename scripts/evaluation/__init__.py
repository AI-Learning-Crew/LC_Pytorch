"""
모델 평가 관련 스크립트들

이 디렉토리는 다음과 같은 평가 작업을 담당합니다:
- 얼굴-음성 매칭 모델 평가 (evaluate_face_voice.py)
"""

from .evaluate_face_voice import *

__all__ = [
    'evaluate_face_voice'
] 