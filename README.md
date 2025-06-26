# LC_PyTorch - 얼굴 추출 및 중복 제거 프로젝트

이 프로젝트는 비디오 파일에서 얼굴을 추출하고, 중복된 얼굴을 제거하여 고유한 인물별로 그룹화하는 도구입니다.

## 주요 기능

1. **얼굴 추출**: 비디오 파일의 첫 프레임에서 얼굴을 자동으로 감지하고 추출
2. **얼굴 중복 제거**: 얼굴 임베딩을 사용하여 동일 인물을 식별하고 그룹화
3. **대표 얼굴 선택**: 각 인물 그룹에서 대표 얼굴을 선택하여 별도 저장

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

### 전체 워크플로우 실행

```bash
python train.py --dataset_path /path/to/videos --output_base_dir /path/to/output
```

### 단계별 실행

#### 1. 얼굴 추출만 실행

```bash
python scripts/extract_faces.py \
    --dataset_path /path/to/videos \
    --output_dir /path/to/extracted_faces \
    --detector_backend retinaface
```

#### 2. 중복 제거만 실행

```bash
python scripts/deduplicate_faces.py \
    --faces_dir /path/to/extracted_faces \
    --dedupe_dir /path/to/deduped_faces \
    --representative_dir /path/to/representative_faces \
    --model_name Facenet \
    --threshold 0.4
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

## 출력 구조

```
output_base_dir/
├── extracted_faces/          # 추출된 원본 얼굴 이미지들
├── deduped_faces/           # 중복 제거된 얼굴 이미지들 (그룹화됨)
└── representative_faces/    # 각 인물 그룹의 대표 얼굴들
```

## 파일 명명 규칙

- **원본 얼굴**: `vid_123.jpg` (비디오 파일명 기반)
- **중복 제거된 얼굴**: 
  - 대표 얼굴: `vid_123.jpg`
  - 중복 얼굴: `vid_123_dedupe_vid_456.jpg`
- **대표 얼굴**: `vid_123.jpg` (각 인물 그룹당 하나)

## 예제

### Google Colab에서 사용 예제

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 디렉토리로 이동
%cd /content/drive/MyDrive/myProject/pjt_face_voice

# 전체 워크플로우 실행
!python train.py \
    --dataset_path /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k \
    --output_base_dir /content/drive/MyDrive/myProject/pjt_face_voice/face_video_5k_processed
```

## 성능 최적화 팁

1. **GPU 사용**: CUDA가 지원되는 환경에서 실행하면 더 빠른 처리 속도를 얻을 수 있습니다.
2. **배치 크기 조정**: 메모리 사용량에 따라 적절한 배치 크기를 설정하세요.
3. **임계값 조정**: `--threshold` 값을 조정하여 중복 제거의 민감도를 조절할 수 있습니다.
   - 낮은 값 (0.3): 더 엄격한 중복 제거
   - 높은 값 (0.5): 더 관대한 중복 제거

## 문제 해결

### 일반적인 오류

1. **"Face could not be detected"**: 비디오의 첫 프레임에서 얼굴을 찾을 수 없음
   - 해결책: 다른 `detector_backend`를 시도하거나 `enforce_detection=False` 사용

2. **메모리 부족**: 대용량 데이터셋 처리 시 발생
   - 해결책: 배치 크기 줄이기 또는 더 작은 단위로 분할 처리

3. **임베딩 계산 오류**: 일부 이미지에서 임베딩 추출 실패
   - 해결책: 이미지 품질 확인 또는 `enforce_detection=False` 사용

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.
