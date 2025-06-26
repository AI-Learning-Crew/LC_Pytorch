"""
모델 평가를 위한 유틸리티
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Dict


def evaluate_summary_metrics(model, dataloader, device) -> Tuple[float, float]:
    """
    요약 성능 지표 계산 (Top-1 Accuracy, ROC-AUC)
    
    Args:
        model: 평가할 모델
        dataloader: 테스트 데이터로더
        device: 계산 장치
        
    Returns:
        Top-1 Accuracy, ROC-AUC Score
    """
    model.eval()
    all_image_embeddings, all_audio_embeddings = [], []
    
    with torch.no_grad():
        for images, audios in tqdm(dataloader, desc="테스트 임베딩 추출 중"):
            images, audios = images.to(device), audios.to(device)
            
            # 임베딩 계산
            image_embeddings, audio_embeddings = model(images, audios)
            
            all_image_embeddings.append(image_embeddings.cpu())
            all_audio_embeddings.append(audio_embeddings.cpu())
    
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_audio_embeddings = torch.cat(all_audio_embeddings)
    
    # 유사도 행렬 계산
    similarity_matrix = torch.matmul(all_image_embeddings, all_audio_embeddings.T)
    
    # Top-1 Accuracy 계산
    top1_indices = torch.argmax(similarity_matrix, dim=1)
    correct_labels = torch.arange(len(all_image_embeddings))
    top1_accuracy = (top1_indices == correct_labels).float().mean().item()
    
    # ROC-AUC Score 계산
    num_samples = len(similarity_matrix)
    labels = torch.eye(num_samples).flatten()  # 긍정적 쌍(대각선)은 1, 나머지는 0
    scores = similarity_matrix.flatten()
    auc_score = roc_auc_score(labels.numpy(), scores.numpy())
    
    return top1_accuracy, auc_score


def evaluate_retrieval_ranking(model, test_dataset, device, top_k: int = 5) -> pd.DataFrame:
    """
    상세 랭킹 평가 (Top-K 검색 결과)
    
    Args:
        model: 평가할 모델
        test_dataset: 테스트 데이터셋
        device: 계산 장치
        top_k: 상위 K개 결과
        
    Returns:
        평가 결과 DataFrame
    """
    model.eval()
    
    # 모든 테스트 데이터 임베딩 사전 계산
    all_image_embeddings, all_audio_embeddings = [], []
    image_file_paths, audio_file_paths = [], []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="전체 테스트 임베딩 계산 중"):
            image_tensor, audio_tensor = test_dataset[i]
            image_path, audio_path = test_dataset.file_pairs[i]
            
            # 임베딩 계산
            image_embeddings, audio_embeddings = model(
                image_tensor.unsqueeze(0).to(device), 
                audio_tensor.unsqueeze(0).to(device)
            )
            
            all_image_embeddings.append(image_embeddings.cpu())
            all_audio_embeddings.append(audio_embeddings.cpu())
            
            image_file_paths.append(image_path)
            audio_file_paths.append(audio_path)
    
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_audio_embeddings = torch.cat(all_audio_embeddings)
    
    # 유사도 행렬 계산 및 랭킹 생성
    similarity_matrix = torch.matmul(all_image_embeddings, all_audio_embeddings.T)
    sorted_audio_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # 결과를 DataFrame으로 정리
    evaluation_results = []
    for i in range(len(image_file_paths)):
        row_data = {'Image_File': os.path.basename(image_file_paths[i])}
        top_k_indices = sorted_audio_indices[i, :top_k].tolist()
        
        is_correct_in_topk = False
        for rank, audio_idx in enumerate(top_k_indices):
            row_data[f'Rank_{rank+1}_Audio'] = os.path.basename(audio_file_paths[audio_idx])
            row_data[f'Rank_{rank+1}_Score'] = similarity_matrix[i, audio_idx].item()
            
            # 정답 여부 확인 (확장자 제외한 파일명 비교)
            if (os.path.basename(image_file_paths[i]).split('.')[0] == 
                os.path.basename(audio_file_paths[audio_idx]).split('.')[0]):
                if rank == 0:
                    row_data['Correct_at_Rank_1'] = "✅"
                is_correct_in_topk = True
        
        row_data[f'Correct_in_Top_{top_k}'] = "✅" if is_correct_in_topk else "❌"
        evaluation_results.append(row_data)
    
    results_df = pd.DataFrame(evaluation_results)
    
    # 컬럼 순서 정리
    display_cols = ['Image_File', 'Correct_at_Rank_1', f'Correct_in_Top_{top_k}']
    for i in range(1, top_k + 1):
        display_cols.extend([f'Rank_{i}_Audio', f'Rank_{i}_Score'])
    
    return results_df.reindex(columns=display_cols).fillna('')


def calculate_retrieval_metrics(model, test_dataset, device, top_ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    다양한 Top-K에서의 검색 성능 계산
    
    Args:
        model: 평가할 모델
        test_dataset: 테스트 데이터셋
        device: 계산 장치
        top_ks: 계산할 Top-K 리스트
        
    Returns:
        각 Top-K에서의 정확도 딕셔너리
    """
    model.eval()
    
    # 모든 테스트 데이터 임베딩 사전 계산
    all_image_embeddings, all_audio_embeddings = [], []
    image_file_paths, audio_file_paths = [], []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="임베딩 계산 중"):
            image_tensor, audio_tensor = test_dataset[i]
            image_path, audio_path = test_dataset.file_pairs[i]
            
            image_embeddings, audio_embeddings = model(
                image_tensor.unsqueeze(0).to(device), 
                audio_tensor.unsqueeze(0).to(device)
            )
            
            all_image_embeddings.append(image_embeddings.cpu())
            all_audio_embeddings.append(audio_embeddings.cpu())
            
            image_file_paths.append(image_path)
            audio_file_paths.append(audio_path)
    
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_audio_embeddings = torch.cat(all_audio_embeddings)
    
    # 유사도 행렬 계산
    similarity_matrix = torch.matmul(all_image_embeddings, all_audio_embeddings.T)
    sorted_audio_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # 각 Top-K에서의 정확도 계산
    metrics = {}
    for k in top_ks:
        correct_count = 0
        for i in range(len(image_file_paths)):
            top_k_indices = sorted_audio_indices[i, :k]
            
            # 정답이 Top-K에 있는지 확인
            for audio_idx in top_k_indices:
                if (os.path.basename(image_file_paths[i]).split('.')[0] == 
                    os.path.basename(audio_file_paths[audio_idx]).split('.')[0]):
                    correct_count += 1
                    break
        
        metrics[f'Top_{k}_Accuracy'] = correct_count / len(image_file_paths)
    
    return metrics


def print_evaluation_summary(top1_accuracy: float, auc_score: float, retrieval_metrics: Dict[str, float] = None):
    """
    평가 결과 요약 출력
    
    Args:
        top1_accuracy: Top-1 정확도
        auc_score: ROC-AUC 점수
        retrieval_metrics: 검색 성능 지표
    """
    print("\n--- 최종 평가 결과 ---")
    print(f"✅ Top-1 Accuracy : {top1_accuracy * 100:.2f}%")
    print(f"✅ ROC-AUC Score  : {auc_score:.4f}")
    
    if retrieval_metrics:
        print("\n--- 검색 성능 지표 ---")
        for metric_name, value in retrieval_metrics.items():
            print(f"✅ {metric_name}: {value * 100:.2f}%")
    
    print("--------------------")


# 편의를 위한 import 추가
import os 