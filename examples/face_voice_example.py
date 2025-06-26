#!/usr/bin/env python3
"""
얼굴-음성 매칭 모델 사용 예제
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.face_voice_model import FaceVoiceModel, InfoNCELoss, save_model_components, load_model_components
from datasets.face_voice_dataset import (
    FaceVoiceDataset, collate_fn, create_data_transforms, match_face_voice_files
)
from utils.evaluator import (
    evaluate_summary_metrics, evaluate_retrieval_ranking, 
    calculate_retrieval_metrics, print_evaluation_summary
)


def example_google_colab_usage():
    """Google Colab에서 사용하는 예제"""
    print("=== Google Colab 얼굴-음성 매칭 예제 ===")
    
    # Google Drive 마운트 (Colab에서만 실행)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive가 마운트되었습니다.")
    except ImportError:
        print("Google Colab 환경이 아닙니다.")
        return
    
    # 설정
    base_path = '/content/drive/MyDrive/myProject/pjt_face_voice'
    image_folder = os.path.join(base_path, 'face_video_5k_representative_faces_manu')
    audio_folder = os.path.join(base_path, 'face_video_5k_representative_faces_manu_audio_wav')
    save_dir = os.path.join(base_path, 'saved_models_InfoNCELoss_batch64_100epoch_ViT')
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 1. 모델 아키텍처 정의
    print("\n1. 모델 아키텍처 정의 중...")
    model = FaceVoiceModel(
        image_model_name="google/vit-base-patch16-224-in21k",
        audio_model_name="facebook/wav2vec2-base-960h",
        embedding_dim=512
    )
    
    # 2. 저장된 모델 가중치 불러오기
    print("\n2. 모델 가중치 로드 중...")
    model = load_model_components(model, save_dir, device)
    if model is None:
        print("모델 로드에 실패했습니다.")
        return
    
    model.to(device).eval()
    print("✅ 모델이 평가 모드로 설정되었습니다.")
    
    # 3. 데이터 준비
    print("\n3. 데이터 준비 중...")
    image_transform, processor = create_data_transforms()
    
    # 파일 매칭
    matched_files = match_face_voice_files(image_folder, audio_folder)
    print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 찾았습니다.")
    
    # 데이터 분할 (학습 시와 동일한 random_state 사용)
    train_files, test_files = train_test_split(matched_files, test_size=0.05, random_state=42)
    print(f"테스트에 사용될 데이터: {len(test_files)}개")
    
    # 테스트 데이터셋 생성
    test_dataset = FaceVoiceDataset(test_files, processor, image_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 4. 요약 성능 지표 평가
    print("\n4. 요약 성능 지표 평가 중...")
    top1_accuracy, auc_score = evaluate_summary_metrics(model, test_dataloader, device)
    
    # 5. 검색 성능 지표 계산
    print("\n5. 검색 성능 지표 계산 중...")
    retrieval_metrics = calculate_retrieval_metrics(model, test_dataset, device, top_ks=[1, 5, 10])
    
    # 6. 상세 랭킹 평가
    print("\n6. 상세 랭킹 평가 중...")
    results_df = evaluate_retrieval_ranking(model, test_dataset, device, top_k=5)
    
    # 7. 결과 출력
    print_evaluation_summary(top1_accuracy, auc_score, retrieval_metrics)
    
    # 상세 결과 출력 (처음 20개만)
    print("\n--- 이미지 기반 음성 검색 평가 결과 (Top 5) ---")
    import pandas as pd
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.head(20))
    
    print("\nGoogle Colab 워크플로우 완료!")


def example_basic_usage():
    """기본 사용법 예제"""
    print("=== 기본 얼굴-음성 매칭 예제 ===")
    
    # 설정 (실제 경로로 변경하세요)
    image_folder = "/path/to/face/images"  # 실제 경로로 변경
    audio_folder = "/path/to/audio/files"   # 실제 경로로 변경
    save_dir = "/path/to/save/model"        # 실제 경로로 변경
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 1. 모델 생성
    print("\n1. 모델 생성 중...")
    model = FaceVoiceModel(embedding_dim=512)
    model.to(device)
    
    # 2. 데이터 준비
    print("\n2. 데이터 준비 중...")
    image_transform, processor = create_data_transforms()
    
    # 파일 매칭
    matched_files = match_face_voice_files(image_folder, audio_folder)
    print(f"총 {len(matched_files)}개의 매칭된 파일 쌍을 찾았습니다.")
    
    # 데이터 분할
    train_files, test_files = train_test_split(matched_files, test_size=0.2, random_state=42)
    
    # 데이터셋 생성
    train_dataset = FaceVoiceDataset(train_files, processor, image_transform)
    test_dataset = FaceVoiceDataset(test_files, processor, image_transform)
    
    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 3. 학습 설정
    print("\n3. 학습 설정 중...")
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 4. 학습 (간단한 예제)
    print("\n4. 학습 중... (1 에포크만)")
    model.train()
    for epoch in range(1):
        for images, audios in train_dataloader:
            images, audios = images.to(device), audios.to(device)
            
            # 순전파
            image_embeddings, audio_embeddings = model(images, audios)
            loss = criterion(image_embeddings, audio_embeddings)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")
            break  # 첫 번째 배치만
    
    # 5. 모델 저장
    print("\n5. 모델 저장 중...")
    save_model_components(model, save_dir)
    
    # 6. 평가
    print("\n6. 평가 중...")
    model.eval()
    top1_accuracy, auc_score = evaluate_summary_metrics(model, test_dataloader, device)
    
    print_evaluation_summary(top1_accuracy, auc_score)
    
    print("\n기본 사용법 예제 완료!")


def main():
    """메인 함수"""
    print("LC_PyTorch 얼굴-음성 매칭 모델 예제")
    print("=" * 50)
    
    # 예제 1: Google Colab 사용법
    print("\n1. Google Colab 사용법 예제")
    print("Google Colab 환경에서 실행하세요.")
    
    # example_google_colab_usage()
    
    # 예제 2: 기본 사용법
    print("\n2. 기본 사용법 예제")
    print("주석을 해제하고 실제 경로를 설정한 후 실행하세요.")
    
    # example_basic_usage()
    
    print("\n예제 실행을 위해서는 주석을 해제하고 실제 경로를 설정하세요.")


if __name__ == '__main__':
    main() 