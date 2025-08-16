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


def calculate_all_metrics(model, test_dataset, device, top_ks: List[int]) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """
    모든 평가 지표(검색 성능, ROC-AUC, 상세 랭킹)를 한 번에 계산합니다.
    
    Args:
        model: 평가할 모델
        test_dataset: 테스트 데이터셋
        device: 계산 장치
        top_ks: 계산할 Top-K 리스트 (예: [1, 5, 10])
        
    Returns:
        A tuple containing:
        - retrieval_metrics (Dict[str, float]): Top-K 검색 정확도 딕셔너리
        - auc_score (float): ROC-AUC 점수
        - ranking_df (pd.DataFrame): 상세 랭킹 결과 DataFrame
    """
    model.eval()
    
    # --- 1. 임베딩 및 파일 경로 사전 계산 ---
    all_image_embeddings, all_audio_embeddings = [], []
    image_file_paths, audio_file_paths = [], []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="테스트 임베딩 계산 중"):
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

    # --- 2. 유사도 행렬 계산 ---
    similarity_matrix = torch.matmul(all_image_embeddings, all_audio_embeddings.T)
    sorted_audio_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    num_samples = len(similarity_matrix)

    # --- 3. ROC-AUC 점수 계산 (신뢰성 있는 방식) ---
    labels = torch.eye(num_samples).flatten()
    scores = similarity_matrix.flatten()
    auc_score = roc_auc_score(labels.numpy(), scores.numpy())
    
    # --- 4. 검색 성능(Top-K) 및 상세 랭킹 DataFrame 계산 ---
    retrieval_metrics = {f'Top_{k}_Accuracy': 0 for k in top_ks}
    evaluation_results = []

    # 상세 랭킹 DF의 Top-K를 위한 최대 K 값 (예: [1, 5, 10] 중 10)
    max_k_for_df = max(top_ks) 

    for i in range(num_samples):
        # 상세 랭킹 DataFrame을 위한 데이터 준비
        row_data = {'Image_File': os.path.basename(image_file_paths[i])}
        # 상세 랭킹 DF는 항상 모든 필요한 K까지 결과를 저장
        top_k_indices_for_df = sorted_audio_indices[i, :max_k_for_df].tolist()
        
        # 각 Top-K에 해당하는 정답 여부 플래그
        is_correct_at_rank_flags = {k: False for k in top_ks} 

        for rank, audio_idx in enumerate(top_k_indices_for_df):
            # 상세 랭킹 DF 채우기
            row_data[f'Rank_{rank+1}_Audio'] = os.path.basename(audio_file_paths[audio_idx])
            row_data[f'Rank_{rank+1}_Score'] = similarity_matrix[i, audio_idx].item()
            
            # 정답 여부 확인 (i번째 이미지의 정답은 i번째 오디오)
            if audio_idx == i:
                # 각 Top-K 정확도 계산을 위한 플래그 업데이트
                for k_val in top_ks:
                    if rank < k_val: # 현재 랭크가 해당 k_val 안에 들면 (0-indexed rank)
                        is_correct_at_rank_flags[k_val] = True

        # 상세 랭킹 DF의 마커 할당
        for k_val in top_ks:
            col_name = f'Correct_at_Rank_{k_val}' if k_val != max_k_for_df else f'Correct_in_Top_{k_val}'
            row_data[col_name] = "✅" if is_correct_at_rank_flags[k_val] else "❌"

        evaluation_results.append(row_data)

        # 각 Top-K의 전체 정확도에 합산
        for k_val in top_ks:
            if is_correct_at_rank_flags[k_val]:
                retrieval_metrics[f'Top_{k_val}_Accuracy'] += 1

    # 최종 정확도 계산 (비율로 변환)
    for k_val in top_ks:
        retrieval_metrics[f'Top_{k_val}_Accuracy'] /= num_samples

    # 상세 랭킹 DataFrame 생성 및 컬럼 정리
    ranking_df = pd.DataFrame(evaluation_results)
    
    # 동적으로 display_cols 생성
    display_cols = ['Image_File']
    for k_val in sorted(top_ks): # 정렬된 순서대로 Correct 열 추가
        col_name = f'Correct_at_Rank_{k_val}' if k_val != max_k_for_df else f'Correct_in_Top_{k_val}'
        display_cols.append(col_name)

    for i in range(1, max_k_for_df + 1):
        display_cols.extend([f'Rank_{i}_Audio', f'Rank_{i}_Score'])
    
    ranking_df = ranking_df.reindex(columns=display_cols).fillna('')
    
    return retrieval_metrics, auc_score, ranking_df

def print_evaluation_summary(retrieval_metrics: Dict[str, float], auc_score: float, user_topk_val: int):
    """
    통합된 평가 결과 요약 출력
    Args:
        retrieval_metrics: 검색 성능 지표 딕셔너리
        auc_score: ROC-AUC 점수
        user_topk_val: 사용자가 입력한 --top_k 값
    """
    print("\n--- 통합 평가 결과 ---")
    print(f"✅ ROC-AUC Score       : {auc_score:.4f}")

    # 출력할 Top-K 리스트 생성: 항상 1, 5를 포함하고 user_topk_val도 추가
    display_top_ks = sorted(list(set([1, 5, user_topk_val])))
    
    print("\n--- 검색 성능 지표 ---")
    for k_val in display_top_ks:
        metric_name = f'Top_{k_val}_Accuracy'
        if metric_name in retrieval_metrics: # calculate_all_metrics에서 계산된 K만 출력
            # 패딩을 위한 최대 길이 계산
            max_name_len = max(len(f'Top_{k}_Accuracy') for k in display_top_ks)
            padding = " " * (max_name_len - len(metric_name)) 
            print(f"✅ {metric_name}{padding}: {retrieval_metrics[metric_name] * 100:.2f}%")
    print("----------------------")

def save_results_to_csv(df: pd.DataFrame, file_path: str):
    """
    평가 결과 DataFrame을 CSV 파일로 저장합니다.
    이모지를 텍스트로 변환하고, UTF-8 인코딩을 사용합니다.

    Args:
        df (pd.DataFrame): 저장할 평가 결과 DataFrame
        file_path (str): 저장할 CSV 파일 경로
    """
    if df.empty:
        print("경고: 저장할 데이터가 없습니다. CSV 파일을 생성하지 않습니다.")
        return

    # 원본 DataFrame을 수정하지 않기 위해 복사본을 만듭니다.
    df_to_save = df.copy()

    # CSV 저장을 위해 이모지 마커를 텍스트로 변환합니다.
    # 터미널 출력용 마커를 CSV 저장용 텍스트로 대체합니다.
    df_to_save.replace({
        "✅": "Correct",
        "❌": "Incorrect"
    }, inplace=True)
    
    try:
        # UTF-8 인코딩(utf-8-sig)을 지정하여 이모지 및 다국어 깨짐 방지
        # 'utf-8-sig'는 Excel에서 파일을 열 때 인코딩을 올바르게 인식하도록 돕습니다.
        df_to_save.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"✅ 평가 결과가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 오류: CSV 파일 저장에 실패했습니다. - {e}")


# 편의를 위한 import 추가
import os 