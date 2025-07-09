"""
스크립트 모듈 패키지

이 패키지는 얼굴-음성 매칭 프로젝트의 실행 가능한 스크립트들을 포함합니다.

디렉토리 구조:
- preprocessing/: 데이터 전처리 관련 스크립트들
  - extract_faces.py: 비디오에서 얼굴 추출
  - deduplicate_faces.py: 중복된 얼굴 이미지 제거
  - create_matched_file.py: 얼굴-음성 매칭 파일 생성
- training/: 모델 학습 관련 스크립트들
  - train_face_voice.py: 얼굴-음성 매칭 모델 훈련
  - train_face_voice_from_matched_file.py: 매칭 파일 기반 모델 훈련
- evaluation/: 모델 평가 관련 스크립트들
  - evaluate_face_voice.py: 모델 성능 평가
- hq/: 고품질 VoxCeleb 데이터셋 관련 스크립트들
"""

from .preprocessing import *
from .training import *
from .evaluation import *

__all__ = [
    # Preprocessing
    'extract_faces',
    'deduplicate_faces',
    'create_matched_file',
    # Training
    'train_face_voice',
    'train_face_voice_from_matched_file',
    # Evaluation
    'evaluate_face_voice'
] 