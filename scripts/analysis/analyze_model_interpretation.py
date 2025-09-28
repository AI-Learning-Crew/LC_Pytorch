#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 해석 및 시각화 스크립트.

특정 얼굴 이미지와 음성 데이터에 대해 학습된 모델의 판단 근거를
Grad-CAM과 Attention Rollout을 사용하여 시각화합니다.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from models.face_voice_model import FaceVoiceModel
from datasets.face_voice_dataset import FaceVoiceDataset, create_data_transforms

from transformers import ViTModel
from captum.attr import LayerGradCam

class ImageCamWrapper(nn.Module):
    """
    Captum의 LayerGradCam에 사용하기 위한 이미지 모델 래퍼(Wrapper).
    모델의 forward 출력을 스칼라 값으로 변환합니다.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, audio):
        """
        이미지와 오디오를 입력받아, 두 임베딩 간의 코사인 유사도의 합을 반환합니다.
        """
        img_emb, aud_emb = self.model(image, audio)
        # 유사도 계산 및 차원 축소
        similarity = torch.sum(img_emb * aud_emb, dim=-1)
        return similarity

def attention_rollout(attentions_list, add_residual=True, head_fusion='mean'):
    """
    ViT(Vision Transformer)의 어텐션 맵을 종합하여 시각화하는 Attention Rollout을 계산합니다.

    Args:
        attentions_list (list of torch.Tensor): 모델의 각 레이어에서 나온 어텐션 텐서 리스트.
        add_residual (bool): 잔차 연결(residual connection)을 고려할지 여부.
        head_fusion (str): "mean", "max", "min" 중 어텐션 헤드 병합 방식.

    Returns:
        torch.Tensor: 최종적으로 종합된 어텐션 맵.
    """
    with torch.no_grad():
        # 초기 어텐션 행렬은 항등 행렬로 설정
        result = torch.eye(attentions_list[0].size(-1), device=attentions_list[0].device).unsqueeze(0)
        
        for attn in attentions_list:
            # 헤드 차원에 대해 평균/최대/최소 연산
            if head_fusion == 'mean':
                attn_fused = attn.mean(dim=1)
            elif head_fusion == 'max':
                attn_fused = attn.max(dim=1).values
            elif head_fusion == 'min':
                attn_fused = attn.min(dim=1).values
            else:
                raise ValueError("Invalid head_fusion. Choose 'mean', 'max', or 'min'.")

            # 잔차 연결을 고려하여 어텐션 가중치 업데이트
            if add_residual:
                I = torch.eye(attn_fused.size(-1), device=attn_fused.device)
                attn_fused = (attn_fused + I) / 2.0

            # 각 행의 합이 1이 되도록 정규화
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            
            # 행렬 곱을 통해 이전 레이어의 어텐션과 현재 레이어의 어텐션을 결합
            result = attn_fused @ result
            
        return result

def plot_spectrogram(spec, title=None, ylabel='freq_bin', ax=None):
    """멜 스펙트로그램을 시각화하는 함수"""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    img = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=16000,
                                   fmax=8000, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax

def plot_attention_maps(original_img, attention_maps, save_path):
    """
    원본 이미지와 어텐션 맵들을 나란히 시각화하고 저장합니다.

    Args:
        original_img (PIL.Image): 원본 이미지.
        attention_maps (dict): {'title': attention_map (np.ndarray)} 형태의 딕셔너리.
        save_path (str): 이미지를 저장할 경로.
    """
    num_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(15, 5))
    
    # 원본 이미지 출력
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 어텐션 맵 출력
    for i, (title, amap) in enumerate(attention_maps.items()):
        ax = axes[i+1]
        ax.imshow(amap, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention map to {save_path}")

def run_and_visualize_analysis(model, device, image_tensor, audio_tensor, speech_array, image_path, audio_path, target_sr, save_path):
    """
    모델 분석을 수행하고 결과를 2x2 그리드로 시각화합니다.
    
    Args:
        model (nn.Module): 분석할 학습된 모델.
        device (torch.device): 연산을 수행할 장치 (CPU 또는 CUDA).
        image_tensor (torch.Tensor): 전처리된 이미지 텐서.
        audio_tensor (torch.Tensor): 전처리된 오디오 텐서.
        speech_array (np.ndarray): 리샘플링 및 길이 조절된 원본 오디오 배열.
        image_path (str): 원본 이미지 파일 경로.
        audio_path (str): 원본 오디오 파일 경로.
        target_sr (int): 오디오 샘플링 레이트.
        save_path (str): 결과 이미지를 저장할 경로.
    """
    model.eval()
    
    # 분석을 위해 배치 차원(batch dimension) 추가
    image_tensor_b1 = image_tensor.unsqueeze(0).to(device)
    audio_tensor_b1 = audio_tensor.unsqueeze(0).to(device)

    # Captum Grad-CAM을 위한 모델 래퍼(wrapper) 생성
    image_cam_wrapper = ImageCamWrapper(model).to(device)

    # 최종 매칭 스코어 계산
    with torch.no_grad():
        score = image_cam_wrapper(image_tensor_b1, audio_tensor_b1).item()

    # 이미지에 대한 Grad-CAM 계산
    # Vision Transformer의 첫 번째 컨볼루션 레이어를 타겟으로 설정
    img_target_layer = model.image_encoder.embeddings.patch_embeddings.projection
    layer_gc_img = LayerGradCam(image_cam_wrapper, img_target_layer)
    # LayerGradCam 속성 계산
    img_attr = layer_gc_img.attribute(
        image_tensor_b1,
        additional_forward_args=(audio_tensor_b1,)
    )
    # 결과를 2D 이미지로 변환
    img_attr_2d = img_attr.squeeze(0).mean(dim=0).cpu().detach().numpy()

    # Grad-CAM 후처리 및 정규화
    img_attr_2d[img_attr_2d < 0] = 0
    max_val = img_attr_2d.max()
    if max_val > 0:
        img_attr_2d = img_attr_2d / max_val
    # 최종 유사도 점수를 곱하여 히트맵 강도 조절
    final_grad_cam = img_attr_2d * np.clip(score, 0, 1)

    # Attention Rollout 계산
    with torch.no_grad():
        outputs = model.image_encoder(image_tensor_b1, output_attentions=True)
        attentions = outputs.attentions
        rollout = attention_rollout(attentions, head_fusion='mean')

        # CLS 토큰의 어텐션 맵을 가져와 grid_size x grid_size로 재구성
        rollout_map = rollout[0, 0, 1:]
        grid_size = int(math.sqrt(rollout_map.size(0)))
        rollout_map = rollout_map.reshape(grid_size, grid_size).cpu().numpy()
        # 시각화를 위해 정규화
        rollout_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-6)

    # 오디오 멜-스펙트로그램 생성
    mel_spectrogram = librosa.feature.melspectrogram(y=speech_array, sr=target_sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 최종 시각화 (2x2 그리드)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Model Interpretation: Face-Voice Matching Analysis (Score: {score:.4f})", fontsize=16)

    original_image = Image.open(image_path).convert("RGB").resize((224, 224))

    # (0, 0): 원본 얼굴 이미지
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title(f"Face Image\n({os.path.basename(image_path)})")
    axs[0, 0].axis('off')

    # (0, 1): 음성 스펙트로그램
    librosa.display.specshow(log_mel_spectrogram, sr=target_sr, x_axis='time', y_axis='mel', ax=axs[0, 1])
    axs[0, 1].set_title(f"Voice Spectrogram\n({os.path.basename(audio_path)})")
    
    # (1, 0): Attention Rollout (모델이 '무엇을' 보았는가)
    axs[1, 0].imshow(rollout_map, cmap='jet', vmin=0, vmax=1)
    axs[1, 0].set_title("Attention Rollout (What it saw)")
    axs[1, 0].axis('off')
    
    # (1, 1): Grad-CAM (모델이 '왜' 매칭했는가)
    axs[1, 1].imshow(final_grad_cam, cmap='jet', vmin=0, vmax=1)
    axs[1, 1].set_title("Grad-CAM (Why it matched)")
    axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 분석 결과가 '{save_path}'에 저장되었습니다. (Score: {score:.4f})")


def main():
    """스크립트의 메인 실행 함수"""
    parser = argparse.ArgumentParser(description='얼굴-음성 쌍에 대한 모델 해석을 시각화합니다.')
    parser.add_argument('--image_path', type=str, required=True, help='분석할 이미지 파일 경로')
    parser.add_argument('--audio_path', type=str, required=True, help='분석할 오디오 파일 경로')
    parser.add_argument('--model_path', type=str, required=True, help='학습된 모델 파일(.pth) 경로')
    parser.add_argument('--output_dir', type=str, default="visualization_results", help='시각화 결과 저장 디렉토리')
    parser.add_argument('--target_sr', type=int, default=16000, help='오디오 샘플링 레이트')
    parser.add_argument('--audio_duration_sec', type=int, default=5, help='오디오 클립 길이 (초)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터 전처리 도구 생성
    image_transform, processor = create_data_transforms(use_augmentation=False)

    # FaceVoiceDataset을 통해 이미지와 오디오 로드 및 전처리
    try:
        dataset = FaceVoiceDataset(
            file_pairs=[(args.image_path, args.audio_path)],
            processor=processor,
            image_transform=image_transform,
            audio_augmentations=None,
            audio_duration_sec=args.audio_duration_sec,
            target_sr=args.target_sr,
            return_processed_audio=True  # 스펙트로그램 생성을 위해 원본 배열도 반환
        )
        image_tensor, audio_tensor, speech_array = dataset[0]
    except Exception as e:
        print(f"❌ 오류: 데이터 파일을 로드하거나 처리하는 중 문제가 발생했습니다: {e}")
        return

    # 모델 로드
    model = FaceVoiceModel()
    try:
        # Attention Rollout을 위해 'eager' 모드로 ViT 모델을 다시 로드
        eager_vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            attn_implementation="eager",  # 어텐션 맵 추출을 위해 'eager' 모드 사용
            output_attentions=True
        )
        model.image_encoder = eager_vit
        
        # 학습된 가중치(state_dict) 로드
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("✅ 모델 로드 및 어텐션 출력 설정 완료")
    except Exception as e:
        print(f"❌ 오류: 모델을 로드하는 데 실패했습니다: {e}")
        return

    # 분석/시각화 함수 호출
    img_basename = Path(args.image_path).stem
    aud_basename = Path(args.audio_path).stem
    save_filename = f"analysis_{img_basename}_vs_{aud_basename}.png"
    save_path = os.path.join(args.output_dir, save_filename)

    run_and_visualize_analysis(
        model, device, image_tensor, audio_tensor, speech_array,
        args.image_path, args.audio_path, args.target_sr, save_path
    )


if __name__ == '__main__':
    main()
