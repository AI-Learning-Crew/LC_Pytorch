"""
모델 평가를 위한 유틸리티
"""

import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

def calculate_all_metrics(model, test_dataset, device, top_ks: List[int]) -> Tuple[Dict, Dict, float, pd.DataFrame, pd.DataFrame]:
    """
    모든 평가 지표(양방향 검색 성능, ROC-AUC, 상세 랭킹)를 한 번에 계산합니다.
    
    Args:
        model: 평가할 모델
        test_dataset: 테스트 데이터셋
        device: 계산 장치
        top_ks: 계산할 Top-K 리스트 (예: [1, 5, 10])
        
    Returns:
        A tuple containing:
        - retrieval_metrics_i2a (Dict): 이미지->음성 검색 정확도
        - retrieval_metrics_a2i (Dict): 음성->이미지 검색 정확도
        - auc_score (float): ROC-AUC 점수
        - ranking_df_i2a (pd.DataFrame): 이미지->음성 상세 랭킹
        - ranking_df_a2i (pd.DataFrame): 음성->이미지 상세 랭킹
    """
    model.eval()
    
    # --- 임베딩 및 파일 경로 사전 계산 (가장 오래 걸리는 작업이므로 한 번만 수행) ---
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

    # --- 유사도 행렬 계산 ---
    similarity_matrix = torch.matmul(all_image_embeddings, all_audio_embeddings.T)
    num_samples = len(similarity_matrix)

    # --- ROC-AUC 점수 계산 ---
    labels = torch.eye(num_samples).flatten()
    scores = similarity_matrix.flatten()
    auc_score = roc_auc_score(labels.numpy(), scores.numpy())
    
    # --- 양방향 랭킹 및 정확도 계산 ---
    # 이미지 -> 음성 검색
    sorted_audio_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    # 음성 -> 이미지 검색 (유사도 행렬을 전치하여 계산)
    sorted_image_indices = torch.argsort(similarity_matrix.T, dim=1, descending=True)

    # 결과를 저장할 변수들 초기화
    retrieval_metrics_i2a = {f'Top_{k}_Accuracy': 0 for k in top_ks}
    retrieval_metrics_a2i = {f'Top_{k}_Accuracy': 0 for k in top_ks}
    eval_results_i2a, eval_results_a2i = [], []
    max_k_for_df = max(top_ks)

    for i in range(num_samples):
        # --- 이미지 -> 음성 평가 ---
        row_data_i2a = {'Image_File': os.path.basename(image_file_paths[i])}
        is_correct_i2a = {k: False for k in top_ks}
        # 상위 K개 결과
        for rank, audio_idx in enumerate(sorted_audio_indices[i, :max_k_for_df]):
            row_data_i2a[f'Rank_{rank+1}_Audio'] = os.path.basename(audio_file_paths[audio_idx])
            row_data_i2a[f'Rank_{rank+1}_Score'] = similarity_matrix[i, audio_idx].item()
            if audio_idx == i:
                for k in top_ks:
                    if rank < k: is_correct_i2a[k] = True
        # 하위 K개 결과
        for rank, audio_idx in enumerate(sorted_audio_indices[i, -max_k_for_df:]):
            bottom_rank_num = max_k_for_df - rank
            row_data_i2a[f'Bottom_Rank_{bottom_rank_num}_Audio'] = os.path.basename(audio_file_paths[audio_idx])
            row_data_i2a[f'Bottom_Rank_{bottom_rank_num}_Score'] = similarity_matrix[i, audio_idx].item()

        for k in top_ks:
            col_name = f'Correct_at_Rank_{k}' if k == 1 else f'Correct_in_Top_{k}'
            row_data_i2a[col_name] = "✅" if is_correct_i2a[k] else "❌"
        eval_results_i2a.append(row_data_i2a)
        
        # --- 음성 -> 이미지 평가 ---
        row_data_a2i = {'Audio_File': os.path.basename(audio_file_paths[i])}
        is_correct_a2i = {k: False for k in top_ks}
        # 상위 K개 결과
        for rank, image_idx in enumerate(sorted_image_indices[i, :max_k_for_df]):
            row_data_a2i[f'Rank_{rank+1}_Image'] = os.path.basename(image_file_paths[image_idx])
            row_data_a2i[f'Rank_{rank+1}_Score'] = similarity_matrix[image_idx, i].item()
            if image_idx == i:
                for k in top_ks:
                    if rank < k: is_correct_a2i[k] = True
        # 하위 K개 결과
        for rank, image_idx in enumerate(sorted_image_indices[i, -max_k_for_df:]):
            bottom_rank_num = max_k_for_df - rank
            row_data_a2i[f'Bottom_Rank_{bottom_rank_num}_Image'] = os.path.basename(image_file_paths[image_idx])
            row_data_a2i[f'Bottom_Rank_{bottom_rank_num}_Score'] = similarity_matrix[image_idx, i].item()
        for k in top_ks:
            col_name = f'Correct_at_Rank_{k}' if k == 1 else f'Correct_in_Top_{k}'
            row_data_a2i[col_name] = "✅" if is_correct_a2i[k] else "❌"
        eval_results_a2i.append(row_data_a2i)

        # 양방향 Top-K 정확도 누적
        for k in top_ks:
            if is_correct_i2a[k]: retrieval_metrics_i2a[f'Top_{k}_Accuracy'] += 1
            if is_correct_a2i[k]: retrieval_metrics_a2i[f'Top_{k}_Accuracy'] += 1

    # 최종 정확도 계산 (비율로 변환)
    for k in top_ks:
        retrieval_metrics_i2a[f'Top_{k}_Accuracy'] /= num_samples
        retrieval_metrics_a2i[f'Top_{k}_Accuracy'] /= num_samples

    # --- 양방향 DataFrame 생성 및 컬럼 정리 ---
    def create_df(results, query_col, rank_col_prefix):
        df = pd.DataFrame(results)
        cols = [query_col]
        for k in sorted(top_ks):
            cols.append(f'Correct_at_Rank_{k}' if k == 1 else f'Correct_in_Top_{k}')
        # 상위 K개 컬럼 추가
        for i in range(1, max_k_for_df + 1):
            cols.extend([f'Rank_{i}_{rank_col_prefix}', f'Rank_{i}_Score'])
        # 구분자 컬럼 추가
        separator_col_name = '...'
        cols.append(separator_col_name)
        df[separator_col_name] = '...'
        # 하위 K개 컬럼 추가
        for i in reversed(range(1, max_k_for_df + 1)):
            cols.extend([f'Bottom_Rank_{i}_{rank_col_prefix}', f'Bottom_Rank_{i}_Score'])
        return df.reindex(columns=cols).fillna('')

    df_i2a = create_df(eval_results_i2a, 'Image_File', 'Audio')
    df_a2i = create_df(eval_results_a2i, 'Audio_File', 'Image')
    
    return retrieval_metrics_i2a, retrieval_metrics_a2i, auc_score, df_i2a, df_a2i

def print_evaluation_summary(metrics_i2a, metrics_a2i, auc_score, user_topk):
    """ 양방향 검색 결과를 포함한 통합 평가 요약 출력 """
    print("\n--- 통합 평가 결과 ---")
    print(f"✅ ROC-AUC Score : {auc_score:.4f}")
    
    display_ks = sorted(list(set([1, 5, user_topk])))
    max_name_len = max(len(f'Top_{k}_Accuracy') for k in display_ks)

    print("\n--- 이미지 -> 음성 검색 성능 ---")
    for k in display_ks:
        metric_name = f'Top_{k}_Accuracy'
        if metric_name in metrics_i2a:
            padding = " " * (max_name_len - len(metric_name))
            print(f"✅ {metric_name}{padding}: {metrics_i2a[metric_name] * 100:.2f}%")
            
    print("\n--- 음성 -> 이미지 검색 성능 ---")
    for k in display_ks:
        metric_name = f'Top_{k}_Accuracy'
        if metric_name in metrics_a2i:
            padding = " " * (max_name_len - len(metric_name))
            print(f"✅ {metric_name}{padding}: {metrics_a2i[metric_name] * 100:.2f}%")
    print("--------------------------------")

def save_results_to_csv(df_i2a: pd.DataFrame, df_a2i: pd.DataFrame, file_path: str):
    """
    양방향 평가 결과 DataFrame을 하나의 CSV 파일에 순차적으로 저장합니다.
    Args:
        df_i2a (pd.DataFrame): 이미지 -> 음성 검색 결과
        df_a2i (pd.DataFrame): 음성 -> 이미지 검색 결과
        file_path (str): 저장할 CSV 파일 경로
    """
    if df_i2a.empty or df_a2i.empty:
        print("경고: 저장할 데이터가 없어 CSV 파일을 생성하지 않습니다.")
        return

    # CSV 저장을 위해 이모지를 텍스트로 변환
    df_i2a_to_save = df_i2a.copy()
    df_a2i_to_save = df_a2i.copy()
    df_i2a_to_save.replace({"✅": "O", "❌": "X"}, inplace=True)
    df_a2i_to_save.replace({"✅": "O", "❌": "X"}, inplace=True)

    try:
        # 하나의 파일에 순차적으로 쓰기
        # UTF-8 인코딩(utf-8-sig)을 지정하여 이모지 및 다국어 깨짐 방지
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            # 첫 번째 섹션: 이미지 -> 음성
            f.write("--- Image to Audio Retrieval Results ---\n")
            df_i2a_to_save.to_csv(f, index=False, lineterminator='\n')
            
            # 섹션 구분을 위한 공백
            f.write("\n\n")

            # 두 번째 섹션: 음성 -> 이미지
            f.write("--- Audio to Image Retrieval Results ---\n")
            df_a2i_to_save.to_csv(f, index=False, lineterminator='\n')
            
        print(f"✅ 양방향 상세 결과가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 오류: CSV 파일 저장에 실패했습니다 - {e}")
