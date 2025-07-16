#!/usr/bin/env python3
"""
TensorBoard 실행 스크립트 (Colab 지원)

• Colab 환경이면:
  └─ (선택) 구글 드라이브 마운트
  └─ 필요한 패키지 설치
  └─ %tensorboard 매직 명령어로 즉시 화면 표시

• 로컬 환경이면:
  └─ tensorboard 프로세스를 백그라운드로 실행
  └─ http://localhost:포트 로 접속 안내
"""

import os
import sys
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────
def is_colab() -> bool:
    """현재 실행 환경이 Colab인지 판별"""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def mount_google_drive() -> None:
    """Colab에서 구글 드라이브 마운트"""
    if not is_colab():
        return

    print("🔗 구글 드라이브 마운트 중...")
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        print("✅ 구글 드라이브 마운트 완료")
    except Exception as e:
        print(f"⚠️  구글 드라이브 마운트 실패: {e}")


def install_required_packages() -> None:
    """TensorBoard 및 Colab‑전용 패키지 설치"""
    print("📦 필요한 패키지 설치 중...")
    # TensorBoard
    os.system("pip install -q tensorboard")
    # Colab 환경이면 IPython 확장도 보장
    if is_colab():
        os.system("pip install -q ipython")
    print("✅ 패키지 설치 완료")


# ──────────────────────────────────────────────────────────────
# TensorBoard 실행 (로컬)
# ──────────────────────────────────────────────────────────────
def run_tensorboard_local(log_dir: str, port: int = 6006, host: str = "0.0.0.0") -> None:
    """로컬 환경에서 TensorBoard 백그라운드로 실행"""
    from threading import Thread

    if not os.path.exists(log_dir):
        print(f"📂 로그 디렉토리가 없어 생성합니다: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    cmd = f"tensorboard --logdir {log_dir} --port {port} --host {host}"
    print(f"📊 TensorBoard 실행 중... (cmd: {cmd})")

    def _launch():
        os.system(cmd)

    t = Thread(target=_launch, daemon=True)
    t.start()
    time.sleep(3)  # 기동 대기
    print("✅ TensorBoard 백그라운드 실행 완료")


# ──────────────────────────────────────────────────────────────
# TensorBoard 실행 및 화면 표시 (Colab)
# ──────────────────────────────────────────────────────────────
def run_tensorboard_colab(log_dir: str, port: int = 6006) -> None:
    """Colab에서 %tensorboard 매직으로 실행 + 화면 표시"""
    print("📊 TensorBoard 실행 및 화면 표시 중 (Colab)...")

    # IPython 객체 획득
    from IPython import get_ipython  # type: ignore
    ipython = get_ipython()

    if ipython is None:
        # 이럴 일은 거의 없으나, 방어 로직
        print("⚠️ IPython 환경이 아닌 것 같습니다. 아래 명령을 직접 실행하세요:")
        print("   %load_ext tensorboard")
        print(f"   %tensorboard --logdir {log_dir} --port {port}")
        return

    # 로그 디렉토리가 없으면 생성
    if not os.path.exists(log_dir):
        print(f"📂 로그 디렉토리가 없어 생성합니다: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    # TensorBoard extension 로드 및 실행
    ipython.run_line_magic('load_ext', 'tensorboard')
    ipython.run_line_magic('tensorboard', f'--logdir {log_dir} --port {port}')
    print("✅ TensorBoard 화면이 아래에 표시됩니다")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="TensorBoard 실행 (Colab 지원)")
    parser.add_argument("--log_dir", type=str, default="./output",
                        help="TensorBoard 로그 디렉토리 (기본값: ./output)")
    parser.add_argument("--port", type=int, default=6006,
                        help="TensorBoard 포트 (기본값: 6006)")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="작업 디렉토리 변경 (선택)")
    parser.add_argument("--mount_drive", action="store_true",
                        help="구글 드라이브 마운트 (Colab 한정)")

    args = parser.parse_args()

    print("\n🚀 TensorBoard 실행 스크립트 시작")
    print("=" * 60)

    # Colab 여부 확인
    if is_colab():
        print("🔍 Colab 환경 감지됨")
        if args.mount_drive:
            mount_google_drive()
    else:
        print("🔍 로컬(혹은 기타) 환경 감지됨")

    # 필수 패키지 설치
    install_required_packages()

    # 작업 디렉토리 변경
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        os.chdir(args.work_dir)
        print(f"📁 작업 디렉토리 변경 완료 → {os.getcwd()}")

    # 실행 분기
    if is_colab():
        run_tensorboard_colab(args.log_dir, args.port)
        # Colab에서는 별도 대기 루프不要; %tensorboard가 자체로 실행 유지
    else:
        run_tensorboard_local(args.log_dir, args.port)
        print(f"🌐 브라우저 접속: http://localhost:{args.port}")
        # 무한 대기 (Ctrl+C 로 종료)
        try:
            print("⏳ TensorBoard가 실행 중입니다. 중단하려면 Ctrl+C …")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 TensorBoard 실행을 중단합니다")

    print("=" * 60)
    print("🎉 스크립트 종료")


if __name__ == "__main__":
    main()
