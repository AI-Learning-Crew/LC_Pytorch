"""
유틸리티 모듈 패키지

이 패키지는 얼굴-음성 매칭 프로젝트에서 사용되는 다양한 유틸리티 함수들과 클래스들을 포함합니다.
주요 구성요소:
- FaceExtractor: 비디오에서 얼굴을 추출하는 클래스
- FaceDeduplicator: 중복된 얼굴 이미지를 제거하는 클래스
- evaluator: 모델 성능 평가를 위한 함수들
- matched_file_utils: 매칭된 파일들을 처리하는 유틸리티 함수들
"""

# 얼굴 추출 및 중복 제거 관련 클래스들
from .face_extractor import FaceExtractor
from .face_deduplicator import FaceDeduplicator

# 모델 성능 평가 관련 함수들
from .evaluator import (
    evaluate_summary_metrics, evaluate_retrieval_ranking, 
    calculate_retrieval_metrics, print_evaluation_summary
)

# 매칭된 파일 처리 관련 유틸리티 함수들
from .matched_file_utils import (
    load_id_list_from_json, get_matched_pair, save_matched_files_by_index
)

# 외부에서 import할 수 있는 클래스와 함수들의 목록
__all__ = [
    'FaceExtractor', 'FaceDeduplicator',
    'evaluate_summary_metrics', 'evaluate_retrieval_ranking', 
    'calculate_retrieval_metrics', 'print_evaluation_summary',
    'load_id_list_from_json', 'get_matched_pair', 'save_matched_files_by_index'
]
