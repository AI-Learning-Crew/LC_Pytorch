# LC_PyTorch - 얼굴 추출, 중복 제거 및 얼굴-음성 매칭 프로젝트

이 프로젝트는 비디오 파일에서 얼굴을 추출하고, 중복된 얼굴을 제거하여 고유한 인물별로 그룹화하며, 얼굴과 음성을 매칭하는 멀티모달 모델을 제공합니다.

## 주요 기능

1. **얼굴 추출**: 비디오 파일의 첫 프레임에서 얼굴을 자동으로 감지하고 추출
2. **얼굴 중복 제거**: 얼굴 임베딩을 사용하여 동일 인물을 식별하고 그룹화
3. **대표 얼굴 선택**: 각 인물 그룹에서 대표 얼굴을 선택하여 별도 저장
4. **얼굴-음성 매칭**: ViT + Wav2Vec2 기반 멀티모달 모델로 얼굴과 음성 매칭
5. **모델 평가**: Top-K 정확도, ROC-AUC 등 다양한 성능 지표 제공

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. DeepFace 설치 (선택사항)

더 나은 성능을 위해 DeepFace를 별도로 설치할 수 있습니다:

```bash
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

## 사용법

### 1. 얼굴 추출 및 중복 제거

#### 전체 워크플로우 실행

```bash
python train.py --dataset_path /path/to/videos --output_base_dir /path/to/output
```

#### 단계별 실행

```bash
# 얼굴 추출만
python scripts/extract_faces.py \
    --dataset_path /path/to/videos \
    --output_dir /path/to/extracted_faces \
    --detector_backend retinaface

# 중복 제거만
python scripts/deduplicate_faces.py \
    --faces_dir /path/to/extracted_faces \
    --dedupe_dir /path/to/deduped_faces \
    --representative_dir /path/to/representative_faces \
    --model_name Facenet \
    --threshold 0.4
```

### 2. 얼굴-음성 매칭 모델

#### 모델 학습

```bash
python scripts/train_face_voice.py \
    --image_folder /path/to/face/images \
    --audio_folder /path/to/audio/files \
    --save_dir /path/to/save/model \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4
```

#### 모델 평가

```bash
python scripts/evaluate_face_voice.py \
    --image_folder /path/to/face/images \
    --audio_folder /path/to/audio/files \
    --model_dir /path/to/saved/model \
    --test_size 0.05 \
    --top_k 5
```

### 3. Google Colab에서 사용

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 얼굴 추출 및 중복 제거
!python train.py \
    --dataset_path /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k \
    --output_base_dir /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k_processed

# 얼굴-음성 매칭 모델 학습
!python scripts/train_face_voice.py \
    --image_folder /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k_representative_faces_manu \
    --audio_folder /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k_representative_faces_manu_audio_wav \
    --save_dir /content/drive/MyDrive/myProject/pjt_face_voice/saved_models_InfoNCELoss_batch64_100epoch_ViT
```

## 매개변수 설명

### 얼굴 추출 매개변수

- `--dataset_path`: 비디오 파일들이 있는 디렉토리 경로
- `--output_dir`: 추출된 얼굴 이미지를 저장할 디렉토리
- `--detector_backend`: 얼굴 감지기 백엔드
  - `opencv`: OpenCV Haar Cascade
  - `ssd`: Single Shot Detector
  - `dlib`: Dlib CNN
  - `mtcnn`: MTCNN
  - `retinaface`: RetinaFace (기본값)
  - `mediapipe`: MediaPipe
- `--align_faces`: 얼굴 정렬 기능 활성화 (기본값: True)
- `--video_extensions`: 처리할 비디오 파일 확장자 (기본값: *.mp4)

### 중복 제거 매개변수

- `--faces_dir`: 원본 얼굴 이미지가 저장된 디렉토리
- `--dedupe_dir`: 동일 인물 그룹화 후 복사/저장할 디렉토리
- `--representative_dir`: 대표 얼굴만 별도 복사할 디렉토리
- `--model_name`: 얼굴 임베딩에 사용할 모델
  - `Facenet`: FaceNet (기본값)
  - `VGG-Face`: VGG Face
  - `OpenFace`: OpenFace
  - `DeepID`: DeepID
  - `ArcFace`: ArcFace
  - `SFace`: SFace
- `--threshold`: 동일 인물로 판단할 코사인 거리 임계값 (기본값: 0.4)

### 얼굴-음성 매칭 매개변수

- `--image_folder`: 얼굴 이미지 폴더 경로
- `--audio_folder`: 음성 파일 폴더 경로
- `--save_dir`: 모델 저장 디렉토리
- `--embedding_dim`: 임베딩 차원 (기본값: 512)
- `--temperature`: InfoNCE 온도 파라미터 (기본값: 0.07)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--num_epochs`: 학습 에포크 수 (기본값: 100)
- `--learning_rate`: 학습률 (기본값: 1e-4)
- `--audio_duration_sec`: 오디오 길이 (초) (기본값: 5)

## 출력 구조

```
output_base_dir/
├── extracted_faces/          # 추출된 원본 얼굴 이미지들
├── deduped_faces/           # 중복 제거된 얼굴 이미지들 (그룹화됨)
└── representative_faces/    # 각 인물 그룹의 대표 얼굴들

saved_models/
├── image_model.pth          # 이미지 인코더 가중치
├── image_projection.pth     # 이미지 투영층 가중치
├── audio_model.pth          # 오디오 인코더 가중치
└── audio_projection.pth     # 오디오 투영층 가중치
```

## 파일 명명 규칙

- **원본 얼굴**: `vid_123.jpg` (비디오 파일명 기반)
- **중복 제거된 얼굴**: 
  - 대표 얼굴: `vid_123.jpg`
  - 중복 얼굴: `vid_123_dedupe_vid_456.jpg`
- **대표 얼굴**: `vid_123.jpg` (각 인물 그룹당 하나)

## 모델 아키텍처

### 얼굴-음성 매칭 모델

- **이미지 인코더**: Vision Transformer (ViT-Base)
- **오디오 인코더**: Wav2Vec2-Base
- **손실 함수**: InfoNCE (Contrastive Learning)
- **임베딩 차원**: 512 (기본값)

### 성능 지표

- **Top-1 Accuracy**: 정확히 매칭되는 비율
- **ROC-AUC Score**: 이진 분류 성능
- **Top-K Accuracy**: 상위 K개 내에 정답이 포함되는 비율

## 예제

### 기본 사용법

```python
from models.face_voice_model import FaceVoiceModel
from datasets.face_voice_dataset import FaceVoiceDataset, create_data_transforms
from utils.evaluator import evaluate_summary_metrics

# 모델 생성
model = FaceVoiceModel(embedding_dim=512)

# 데이터 준비
image_transform, processor = create_data_transforms()
dataset = FaceVoiceDataset(file_pairs, processor, image_transform)

# 평가
top1_accuracy, auc_score = evaluate_summary_metrics(model, dataloader, device)
```

자세한 예제는 `examples/` 디렉토리를 참조하세요.

## 성능 최적화 팁

1. **GPU 사용**: CUDA가 지원되는 환경에서 실행하면 더 빠른 처리 속도를 얻을 수 있습니다.
2. **배치 크기 조정**: 메모리 사용량에 따라 적절한 배치 크기를 설정하세요.
3. **임계값 조정**: `--threshold` 값을 조정하여 중복 제거의 민감도를 조절할 수 있습니다.
   - 낮은 값 (0.3): 더 엄격한 중복 제거
   - 높은 값 (0.5): 더 관대한 중복 제거
4. **모델 선택**: 얼굴 임베딩 모델을 변경하여 성능과 속도의 균형을 맞출 수 있습니다.

## 문제 해결

### 일반적인 오류

1. **"Face could not be detected"**: 비디오의 첫 프레임에서 얼굴을 찾을 수 없음
   - 해결책: 다른 `detector_backend`를 시도하거나 `enforce_detection=False` 사용

2. **메모리 부족**: 대용량 데이터셋 처리 시 발생
   - 해결책: 배치 크기 줄이기 또는 더 작은 단위로 분할 처리

3. **임베딩 계산 오류**: 일부 이미지에서 임베딩 추출 실패
   - 해결책: 이미지 품질 확인 또는 `enforce_detection=False` 사용

4. **CUDA 메모리 부족**: GPU 메모리 부족 시 발생
   - 해결책: 배치 크기 줄이기, 그래디언트 누적 사용

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.
