# LC_PyTorch - 얼굴-음성 매칭 프로젝트

**Vision Transformer (ViT) + Wav2Vec2 기반 멀티모달 얼굴-음성 매칭 모델**

## 주요 기능

- **얼굴 추출**: 비디오에서 얼굴 자동 추출 및 중복 제거
- **얼굴-음성 매칭**: ViT + Wav2Vec2 기반 contrastive learning
- **모델 평가**: Top-K 정확도, ROC-AUC 등 성능 지표
- **학습 모니터링**: TensorBoard를 통한 실시간 학습 과정 시각화

## 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
LC_PyTorch/
├── datasets/           # PyTorch 데이터셋 클래스
├── models/            # 신경망 모델 아키텍처
├── scripts/           # 실행 스크립트
│   ├── preprocessing/  # 데이터 전처리
│   │   ├── extract_faces.py          # 비디오에서 얼굴 추출
│   │   ├── deduplicate_faces.py      # 얼굴 중복 제거
│   │   └── create_matched_file.py    # 매칭 파일 생성
│   ├── training/      # 모델 학습
│   │   ├── train_face_voice.py       # 일반 학습 (TensorBoard 지원)
│   │   └── train_face_voice_from_matched_file.py  # 매칭 파일 기반 학습
│   └── evaluation/    # 모델 평가
│       └── evaluate_face_voice.py    # 모델 성능 평가
└── utils/            # 유틸리티 함수
```

## 사용법

### 1. 얼굴 추출

```bash
python scripts/preprocessing/extract_faces.py \
    --dataset_path /path/to/videos \
    --output_dir /path/to/faces \
    --detector_backend retinaface \
    --align_faces \
    --video_extensions "*.mp4"
```

**주요 파라미터:**
- `--dataset_path`: 비디오 파일들이 있는 디렉토리
- `--output_dir`: 추출된 얼굴 이미지 저장 디렉토리
- `--detector_backend`: 얼굴 감지기 (opencv, ssd, dlib, mtcnn, retinaface, mediapipe)
- `--align_faces`: 얼굴 정렬 기능 활성화
- `--video_extensions`: 처리할 비디오 파일 확장자

### 2. 중복 제거

```bash
python scripts/preprocessing/deduplicate_faces.py \
    --faces_dir /path/to/faces \
    --dedupe_dir /path/to/deduped \
    --representative_dir /path/to/representative \
    --model_name Facenet \
    --threshold 0.4
```

**주요 파라미터:**
- `--faces_dir`: 원본 얼굴 이미지가 저장된 디렉토리
- `--dedupe_dir`: 중복 제거된 얼굴 이미지 저장 디렉토리
- `--representative_dir`: 대표 얼굴 이미지 저장 디렉토리
- `--model_name`: 얼굴 임베딩 모델 (Facenet, VGG-Face, OpenFace, DeepID, ArcFace, SFace)
- `--threshold`: 동일 인물 판단 임계값 (기본값: 0.4)

### 3. 매칭 파일 생성

```bash
python scripts/preprocessing/create_matched_file.py \
    -d /path/to/dataset \
    -m /path/to/metadata.json \
    -o datasets/output \
    -l 100
```

**주요 파라미터:**
- `-d, --dataset_path`: 얼굴 이미지 및 음성 파일이 있는 디렉토리
- `-m, --meta_path`: 메타데이터 JSON 파일 경로
- `-o, --output`: 매칭 결과 저장 디렉토리
- `-l, --limit`: 매칭할 최대 인덱스 수

### 4. 모델 학습

**일반 학습 (TensorBoard 지원):**
```bash
python scripts/training/train_face_voice.py \
    --image_folder /path/to/images \
    --audio_folder /path/to/audio \
    --save_dir /path/to/save \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --pretrained_lr 1e-5
```

**커스텀 TensorBoard 디렉토리 지정:**
```bash
python scripts/training/train_face_voice.py \
    --image_folder /path/to/images \
    --audio_folder /path/to/audio \
    --save_dir /path/to/save \
    --tensorboard_dir /path/to/tensorboard_logs \
    --batch_size 32 \
    --num_epochs 100
```

**TensorBoard 비활성화:**
```bash
python scripts/training/train_face_voice.py \
    --image_folder /path/to/images \
    --audio_folder /path/to/audio \
    --save_dir /path/to/save \
    --no_tensorboard \
    --batch_size 32 \
    --num_epochs 100
```

**매칭 파일 기반 학습:**

**⚠️ 중요: 매칭 파일 기반 학습을 사용하기 전에 다음 경로 설정이 필요합니다:**

```bash
# 코랩에서 공유폴더 바로가기 추가
# Google Drive의 공유폴더를 datasets/voxceleb2/voxceleb2 경로에 바로가기 생성
```

```bash
python scripts/training/train_face_voice_from_matched_file.py \
    --matched_file ./datasets/output/matched-file.txt \
    --save_dir /path/to/save \
    --batch_size 32 \
    --num_epochs 100
```

**주요 파라미터:**
- `--image_folder`: 얼굴 이미지 폴더 경로
- `--audio_folder`: 음성 파일 폴더 경로
- `--matched_file`: 매칭된 파일 목록 경로
- `--save_dir`: 모델 저장 디렉토리
- `--embedding_dim`: 임베딩 차원 (기본값: 512)
- `--temperature`: InfoNCE 온도 파라미터 (기본값: 0.07)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--num_epochs`: 학습 에포크 수 (기본값: 100)
- `--learning_rate`: 신규 레이어용 기본 학습률 (기본값: 1e-3)
- `--test_size`: 테스트 데이터 비율 (기본값: 0.2)
- `--audio_duration_sec`: 오디오 길이 (초, 기본값: 5)
- `--target_sr`: 오디오 샘플링 레이트 (기본값: 16000)
- `--tensorboard_dir`: TensorBoard 로그 디렉토리 (기본값: save_dir/runs)
- `--no_tensorboard`: TensorBoard 로깅 비활성화
- `--pretrained_lr`: 사전 학습된 레이어의 학습률 (기본값: 1e-5)
- `--disable_image_augmentation`: 이미지 데이터 증강 비활성화 (매개변수 제거 시 이미지 증강 활성화)
- `--disable_audio_augmentation`: 오디오 데이터 증강 비활성화 (매개변수 제거 시 오디오 증강 활성화)

### 5. TensorBoard 모니터링

학습 중 실시간으로 모델 성능을 모니터링하려면:

```bash
# TensorBoard 실행 스크립트 사용
python scripts/tensorboard/run_tensorboard_colab.py --log_dir /path/to/logs
```

**모니터링 가능한 메트릭:**
- **Loss/Train_Batch**: 배치별 학습 손실
- **Loss/Train_Epoch**: 에포크별 평균 학습 손실
- **Loss/Val_Epoch**: 에포크별 검증 손실
- **Learning_Rate**: 학습률 변화
- **Parameters/**: 모델 파라미터 분포 (10 에포크마다)
- **Gradients/**: 그래디언트 분포 (10 에포크마다)

### 6. 모델 평가

```bash
python scripts/evaluation/evaluate_face_voice.py \
    --image_folder /path/to/images \
    --audio_folder /path/to/audio \
    --model_dir /path/to/model \
    --batch_size 32 \
    --test_size 0.7 \
    --top_k 5 \
    --output_file results.csv
```

**주요 파라미터:**
- `--image_folder`: 얼굴 이미지 폴더 경로
- `--audio_folder`: 음성 파일 폴더 경로
- `--model_dir`: 학습된 모델 디렉토리
- `--batch_size`: 배치 크기 (기본값: 32)
- `--test_size`: 테스트 데이터 비율 (기본값: 0.05)
- `--top_k`: 상위 K개 결과 평가 (기본값: 5)
- `--output_file`: 결과 CSV 파일 저장 경로
- `--audio_duration_sec`: 오디오 길이 (초, 기본값: 5)
- `--target_sr`: 오디오 샘플링 레이트 (기본값: 16000)

## 모델 아키텍처

- **얼굴 인코더**: Vision Transformer (ViT-Base)
- **음성 인코더**: Wav2Vec2-Base
- **손실 함수**: InfoNCE Loss
- **임베딩 차원**: 512 (기본값)

## 학습 모니터링

이 프로젝트는 TensorBoard를 통한 실시간 학습 모니터링을 지원합니다:

- **실시간 손실 추적**: 배치별 및 에포크별 손실 변화
- **학습률 모니터링**: 학습률 스케줄링 효과 확인
- **모델 파라미터 분석**: 파라미터 및 그래디언트 분포 시각화
- **성능 비교**: 여러 실험 결과 비교 분석

TensorBoard를 통해 학습 과정을 시각적으로 확인하여 모델 성능을 최적화할 수 있습니다.

