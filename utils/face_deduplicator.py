"""
얼굴 중복 제거 및 그룹화 기능을 담당하는 모듈
"""

import os
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from deepface import DeepFace
from scipy.spatial.distance import cosine


class FaceDeduplicator:
    """얼굴 이미지들의 중복을 제거하고 그룹화하는 클래스"""
    
    def __init__(self, model_name: str = 'Facenet', threshold: float = 0.4):
        """
        FaceDeduplicator 초기화
        
        Args:
            model_name: 얼굴 임베딩에 사용할 모델 ('Facenet', 'VGG-Face', 'OpenFace', 'DeepID', 'ArcFace', 'SFace')
            threshold: 동일 인물로 판단할 코사인 거리 임계값
        """
        self.model_name = model_name
        self.threshold = threshold
        
    def deduplicate_faces(self, 
                         faces_dir: str, 
                         dedupe_dir: str, 
                         representative_dir: str) -> Dict[str, int]:
        """
        얼굴 이미지들의 중복을 제거하고 그룹화
        
        Args:
            faces_dir: 원본 얼굴 이미지가 저장된 디렉토리
            dedupe_dir: 동일 인물 그룹화 후 복사/저장할 디렉토리
            representative_dir: 대표 얼굴만 별도 복사할 디렉토리
            
        Returns:
            처리 결과 통계 딕셔너리
        """
        # 디렉토리 생성
        for directory in [dedupe_dir, representative_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"'{directory}' 디렉토리를 생성했습니다.")
            else:
                print(f"'{directory}' 디렉토리가 이미 존재합니다.")
                
        # 얼굴 파일 로드 및 정렬
        if not os.path.exists(faces_dir):
            print(f"오류: '{faces_dir}' 디렉토리를 찾을 수 없습니다.")
            return {}
            
        all_face_files_list = os.listdir(faces_dir)
        face_files_jpg = [f for f in all_face_files_list if f.lower().endswith('.jpg')]
        
        if not face_files_jpg:
            print(f"'{faces_dir}' 디렉토리에서 .jpg 파일을 찾을 수 없습니다.")
            return {}
            
        # 파일명 기준 정렬
        face_files_jpg.sort(key=self._get_filenumber)
        print(f"처리할 얼굴 이미지 파일 수: {len(face_files_jpg)}")
        
        # 모든 얼굴 이미지에 대한 임베딩 미리 계산
        print(f"\n'{self.model_name}' 모델을 사용하여 모든 얼굴 이미지의 임베딩을 계산합니다...")
        all_embeddings = self._compute_all_embeddings(faces_dir, face_files_jpg)
        
        # 동일 인물 식별 및 파일 복사
        stats = self._identify_and_copy_faces(faces_dir, face_files_jpg, all_embeddings, 
                                            dedupe_dir, representative_dir)
        
        self._print_deduplication_summary(stats)
        return stats
    
    def _get_filenumber(self, filename: str) -> int:
        """
        파일명에서 번호 추출 (정렬용)
        
        Args:
            filename: 파일명 (예: "vid_123.jpg")
            
        Returns:
            추출된 번호
        """
        try:
            # "vid_NUMBER.jpg" 형식으로 가정, NUMBER 부분 추출하여 정수 변환
            return int(filename.split('_')[1].split('.')[0])
        except:
            # 예외 발생 시 정렬 순서에 영향을 주지 않도록 큰 값 반환
            return float('inf')
    
    def _compute_all_embeddings(self, faces_dir: str, face_files_jpg: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        모든 얼굴 이미지의 임베딩을 미리 계산
        
        Args:
            faces_dir: 얼굴 이미지 디렉토리
            face_files_jpg: 처리할 얼굴 파일 리스트
            
        Returns:
            파일명을 키로 하는 임베딩 딕셔너리
        """
        all_embeddings = {}
        
        for face_filename in tqdm(face_files_jpg, desc="임베딩 계산 중"):
            face_path = os.path.join(faces_dir, face_filename)
            try:
                embedding_objs = DeepFace.represent(
                    img_path=face_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    align=True
                )
                # DeepFace.represent는 리스트를 반환, 첫 번째 요소의 임베딩 사용
                if embedding_objs and len(embedding_objs) > 0:
                    all_embeddings[face_filename] = np.array(embedding_objs[0]['embedding'])
                else:
                    print(f"  경고: {face_filename}에서 임베딩을 추출하지 못했습니다 (결과 없음).")
                    all_embeddings[face_filename] = None
                    
            except Exception as e:
                print(f"  오류: {face_filename} 임베딩 계산 중 오류 발생: {e}")
                all_embeddings[face_filename] = None
                
        none_count = sum(1 for v in all_embeddings.values() if v is None)
        valid_embeddings_count = len(all_embeddings) - none_count
        print(f"총 {valid_embeddings_count}개의 얼굴에 대한 임베딩 계산 완료.")
        
        return all_embeddings

    def _identify_and_copy_faces(self,
                                faces_dir: str, 
                                face_files_jpg: List[str], 
                                all_embeddings: Dict[str, Optional[np.ndarray]],
                                dedupe_dir: str, 
                                representative_dir: str) -> Dict[str, int]:
        """
        동일 인물을 식별하고 파일을 복사
        
        Args:
            faces_dir: 원본 얼굴 이미지 디렉토리
            face_files_jpg: 처리할 얼굴 파일 리스트
            all_embeddings: 모든 얼굴의 임베딩 딕셔너리
            dedupe_dir: 중복 제거된 파일 저장 디렉토리
            representative_dir: 대표 얼굴 저장 디렉토리
            
        Returns:
            처리 결과 통계
        """
        representative_info = {}
        copied_files_to_dedupe_count = 0
        copied_files_to_representative_count = 0
        processed_source_files = set()
        
        print("\n미리 계산된 임베딩을 사용하여 동일 인물을 식별하고 파일을 복사합니다...")
        
        for current_face_filename in face_files_jpg:
            current_embedding = all_embeddings.get(current_face_filename)
            if current_embedding is None or current_face_filename in processed_source_files:
                if current_embedding is None and current_face_filename not in processed_source_files:
                    print(f"  정보: {current_face_filename}은(는) 임베딩이 없어 건너뜁니다.")
                continue
                
            current_face_path_original = os.path.join(faces_dir, current_face_filename)
            current_face_id_num = self._get_filenumber(current_face_filename)
            
            print(f"\n기준 얼굴 처리 중: {current_face_filename}")
            
            is_new_representative = True
            matched_representative_original_filename = None
            
            # 기존 대표 얼굴들과 비교
            for rep_original_filename, rep_data in representative_info.items():
                rep_embedding = rep_data['embedding']
                distance = cosine(rep_embedding, current_embedding)
                if distance <= self.threshold:
                    is_new_representative = False
                    matched_representative_original_filename = rep_original_filename
                    print(f"  >> '{current_face_filename}'은(는) 대표얼굴 '{rep_original_filename}'과(와) 동일 인물입니다 (코사인 거리: {distance:.4f}).")
                    break
                    
            if is_new_representative:
                print(f"  >> '{current_face_filename}'을(를) 새로운 대표 얼굴로 지정합니다.")
                
                # 1. dedupe_dir로 복사
                destination_path_for_dedupe = os.path.join(dedupe_dir, current_face_filename)
                shutil.copy2(current_face_path_original, destination_path_for_dedupe)
                copied_files_to_dedupe_count += 1
                
                # 2. representative_dir로 복사
                destination_path_for_representative = os.path.join(representative_dir, current_face_filename)
                shutil.copy2(current_face_path_original, destination_path_for_representative)
                copied_files_to_representative_count += 1
                
                processed_source_files.add(current_face_filename)

                print(f"  -- 대표 파일 복사 (dedupe): '{current_face_path_original}' -> '{destination_path_for_dedupe}'")
                print(f"  -- 대표 파일 복사 (representative): '{current_face_path_original}' -> '{destination_path_for_representative}'")
                
                representative_info[current_face_filename] = {
                    'embedding': current_embedding,
                    'id_num': current_face_id_num
                }
                
            else:  # 기존 대표 얼굴과 동일 인물인 경우 (dedupe_dir에만 이름 변경하여 복사)
                if matched_representative_original_filename:
                    rep_id_num = representative_info[matched_representative_original_filename]['id_num']
                    
                    new_filename_for_duplicate = f"vid_{rep_id_num}_dedupe_vid_{current_face_id_num}.jpg"
                    destination_path_for_duplicate = os.path.join(dedupe_dir, new_filename_for_duplicate)
                    
                    shutil.copy2(current_face_path_original, destination_path_for_duplicate)
                    copied_files_to_dedupe_count += 1
                    processed_source_files.add(current_face_filename)
                    print(f"  -- 중복 파일 복사 및 이름 변경 (dedupe): '{current_face_path_original}' -> '{destination_path_for_duplicate}'")
                    
        return {
            'copied_files_to_dedupe_count': copied_files_to_dedupe_count,
            'copied_files_to_representative_count': copied_files_to_representative_count,
            'total_processed_files': len(processed_source_files)
        }
    
    def _print_deduplication_summary(self, stats: Dict[str, int]):
        """중복 제거 결과 요약 출력"""
        print("\n--- 모든 파일 처리 완료 ---")
        print(f"총 {stats['copied_files_to_dedupe_count']}개의 파일이 dedupe 디렉토리로 복사되었습니다.")
        print(f"총 {stats['copied_files_to_representative_count']}개의 대표 파일이 representative 디렉토리로 복사되었습니다.")
        print(f"총 {stats['total_processed_files']}개의 파일이 처리되었습니다.") 