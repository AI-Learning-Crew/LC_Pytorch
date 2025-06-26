"""
비디오에서 얼굴 추출 기능을 담당하는 모듈
"""

import cv2
import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from deepface import DeepFace


class FaceExtractor:
    """비디오 파일에서 얼굴을 추출하는 클래스"""
    
    def __init__(self, detector_backend: str = 'retinaface', align_faces: bool = True):
        """
        FaceExtractor 초기화
        
        Args:
            detector_backend: 얼굴 감지기 백엔드 ('opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe')
            align_faces: 얼굴 정렬 기능 활성화 여부
        """
        self.detector_backend = detector_backend
        self.align_faces = align_faces
        
    def extract_faces_from_videos(self, 
                                 dataset_path: str, 
                                 output_dir: str,
                                 video_extensions: List[str] = None) -> Dict[str, int]:
        """
        비디오 파일들에서 얼굴을 추출하여 저장
        
        Args:
            dataset_path: 비디오 파일들이 있는 디렉토리 경로
            output_dir: 추출된 얼굴 이미지를 저장할 디렉토리
            video_extensions: 처리할 비디오 파일 확장자 리스트 (기본값: ['*.mp4'])
            
        Returns:
            처리 결과 통계 딕셔너리
        """
        if video_extensions is None:
            video_extensions = ['*.mp4']
            
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"'{output_dir}' 디렉토리를 생성했습니다.")
            
        # 비디오 파일 찾기
        video_files = []
        for ext in video_extensions:
            pattern = os.path.join(dataset_path, ext)
            video_files.extend(glob.glob(pattern))
            
        if not video_files:
            print(f"경고: '{dataset_path}' 경로에서 비디오 파일을 찾을 수 없습니다.")
            return {}
            
        print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
        
        # 처리 통계 초기화
        stats = {
            'total_files': len(video_files),
            'processed_count': 0,
            'failed_to_open_count': 0,
            'failed_to_read_frame_count': 0,
            'no_face_detected_count': 0,
            'extraction_error_count': 0,
            'failed_to_open_files': [],
            'failed_to_read_frame_files': [],
            'no_face_detected_files': [],
            'extraction_error_files': []
        }
        
        # 각 비디오 파일 처리
        for video_path in tqdm(video_files, desc="비디오 처리 중"):
            result = self._process_single_video(video_path, output_dir)
            
            # 통계 업데이트
            for key in result:
                if key in stats:
                    if isinstance(result[key], list):
                        stats[key].extend(result[key])
                    else:
                        stats[key] += result[key]
                        
        self._print_processing_summary(stats)
        return stats
    
    def _process_single_video(self, video_path: str, output_dir: str) -> Dict[str, int]:
        """
        단일 비디오 파일에서 얼굴 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            처리 결과 딕셔너리
        """
        result = {
            'processed_count': 0,
            'failed_to_open_count': 0,
            'failed_to_read_frame_count': 0,
            'no_face_detected_count': 0,
            'extraction_error_count': 0,
            'failed_to_open_files': [],
            'failed_to_read_frame_files': [],
            'no_face_detected_files': [],
            'extraction_error_files': []
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"  오류: 비디오 파일 '{video_path}'을(를) 열 수 없습니다.")
                result['failed_to_open_count'] = 1
                result['failed_to_open_files'].append(video_path)
                return result
                
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"  오류: 비디오 파일 '{video_path}'에서 첫 번째 프레임을 읽을 수 없습니다.")
                result['failed_to_read_frame_count'] = 1
                result['failed_to_read_frame_files'].append(video_path)
                cap.release()
                return result
                
            try:
                # DeepFace를 사용하여 얼굴 추출
                extracted_faces_data = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                    align=self.align_faces
                )
                
                if extracted_faces_data and len(extracted_faces_data) > 0:
                    # 첫 번째 감지된 얼굴 사용
                    face_data_dict = extracted_faces_data[0]
                    extracted_face_image_rgb_float = face_data_dict['face']
                    
                    face_to_save_uint8 = (extracted_face_image_rgb_float * 255.0).astype(np.uint8)
                    face_to_save_bgr = cv2.cvtColor(face_to_save_uint8, cv2.COLOR_RGB2BGR)
                    
                    base_filename = os.path.basename(video_path)
                    filename_no_ext = os.path.splitext(base_filename)[0]
                    output_filename = f"{filename_no_ext}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    cv2.imwrite(output_path, face_to_save_bgr)
                    print(f"  성공: 얼굴을 추출하여 '{output_path}'에 저장했습니다.")
                    result['processed_count'] = 1
                    
            except ValueError as e:
                if "Face could not be detected" in str(e) or "No face detected" in str(e):
                    print(f"  정보: '{video_path}'의 첫 프레임에서 얼굴을 감지하지 못했습니다.")
                    result['no_face_detected_count'] = 1
                    result['no_face_detected_files'].append(video_path)
                else:
                    print(f"  오류 (DeepFace 얼굴 추출 중): {e}")
                    result['extraction_error_count'] = 1
                    result['extraction_error_files'].append(video_path)
            except Exception as e:
                print(f"  오류 (DeepFace 처리 중 예기치 않은 오류): {e}")
                result['extraction_error_count'] = 1
                result['extraction_error_files'].append(video_path)
            finally:
                cap.release()
                
        except Exception as e:
            print(f"  오류 (비디오 파일 '{video_path}' 처리 중 전역 오류): {e}")
            result['extraction_error_count'] = 1
            result['extraction_error_files'].append(video_path)
            
        return result
    
    def _print_processing_summary(self, stats: Dict[str, int]):
        """처리 결과 요약 출력"""
        print("\n--- 모든 비디오 파일 처리 완료 ---")
        print(f"총 비디오 파일 수: {stats['total_files']}")
        print(f"성공적으로 얼굴 추출 및 저장: {stats['processed_count']}개")
        print(f"파일 열기 실패: {stats['failed_to_open_count']}개")
        print(f"프레임 읽기 실패: {stats['failed_to_read_frame_count']}개")
        print(f"얼굴 미감지: {stats['no_face_detected_count']}개")
        print(f"추출 중 기타 오류: {stats['extraction_error_count']}개")
        
        # 실패한 파일 목록 출력
        if stats['failed_to_open_files']:
            print("\n>> 파일 열기 실패한 동영상 파일 목록:")
            for f_path in stats['failed_to_open_files']:
                print(f" - {os.path.basename(f_path)}")
                
        if stats['failed_to_read_frame_files']:
            print("\n>> 프레임 읽기 실패한 동영상 파일 목록:")
            for f_path in stats['failed_to_read_frame_files']:
                print(f" - {os.path.basename(f_path)}")
                
        if stats['no_face_detected_files']:
            print("\n>> 얼굴 미감지된 동영상 파일 목록:")
            for f_path in stats['no_face_detected_files']:
                print(f" - {os.path.basename(f_path)}")
                
        if stats['extraction_error_files']:
            print("\n>> 추출 중 오류 발생한 동영상 파일 목록:")
            for f_path in stats['extraction_error_files']:
                print(f" - {os.path.basename(f_path)}") 