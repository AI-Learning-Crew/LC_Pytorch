import argparse
import torch
from transformers import ViTModel


def main(model_path: str, output_path: str = "vit_only.pth"):
    # 모델 파라미터 불러오기
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"Model parameters loaded successfully from: {model_path}")

    # state_dict 추출
    if "model_state_dict" in checkpoint:  # checkpoint 형태
        state_dict = checkpoint["model_state_dict"]
    else:  # 그냥 state_dict만 저장된 경우
        state_dict = checkpoint

    # ViT 부분만 추출
    vit_state_dict = {k: v for k, v in state_dict.items() if k.startswith("image_encoder")}
    torch.save(vit_state_dict, output_path)
    print(f"ViT parameters saved to: {output_path}")

    # 동일한 구조의 ViT 준비
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # 키 이름에서 'image_encoder.' 제거
    new_state_dict = {k.replace("image_encoder.", ""): v for k, v in vit_state_dict.items()}

    # 파라미터 로드
    vit.load_state_dict(new_state_dict)
    print("ViT model loaded successfully with the extracted parameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ViT parameters from a multimodal model checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--output_path", type=str, default="vit_only.pth", help="Path to save extracted ViT parameters")
    args = parser.parse_args()

    main(args.model_path, args.output_path)
