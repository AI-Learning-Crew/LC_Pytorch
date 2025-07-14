#!/usr/bin/env python3
"""
TensorBoard 실행 스크립트 (Colab 지원)

이 스크립트는 TensorBoard를 실행하여 학습 과정을 모니터링할 수 있게 합니다.
Colab 환경에서도 실행 가능하도록 설계되었습니다.
"""

import os
import sys
import threading
import time
import argparse
from pathlib import Path

def is_colab():
    """Colab 환경인지 확인"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_google_drive():
    """구글 드라이브 마운트 (Colab에서만)"""
    if not is_colab():
        return
    
    print("🔗 구글 드라이브 마운트 중...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ 구글 드라이브 마운트 완료")
    except Exception as e:
        print(f"⚠️  구글 드라이브 마운트 실패: {e}")

def install_required_packages():
    """필요한 패키지 설치"""
    print("📦 필요한 패키지 설치 중...")
    
    # TensorBoard 설치
    os.system("pip install -q tensorboard")
    
    # IPython 설치 (Colab에서 필요)
    if is_colab():
        os.system("pip install -q ipython")
    
    print("✅ 패키지 설치 완료")



def run_tensorboard(log_dir, port=6006, host="0.0.0.0"):
    """TensorBoard 실행"""
    print(f"📊 TensorBoard 실행 중... (로그 디렉토리: {log_dir}, 포트: {port})")
    
    # 로그 디렉토리가 존재하는지 확인
    if not os.path.exists(log_dir):
        print(f"⚠️  로그 디렉토리가 존재하지 않습니다: {log_dir}")
        print("📁 디렉토리를 생성합니다...")
        os.makedirs(log_dir, exist_ok=True)
    
    # Colab 환경에서는 host를 0.0.0.0으로 설정
    if is_colab():
        cmd = f"tensorboard --logdir {log_dir} --port {port} --host {host}"
    else:
        cmd = f"tensorboard --logdir {log_dir} --port {port}"
    
    def tensorboard_thread():
        os.system(cmd)
    
    # 백그라운드에서 TensorBoard 실행
    thread = threading.Thread(target=tensorboard_thread)
    thread.daemon = True
    thread.start()
    
    # TensorBoard가 시작될 시간 확보
    time.sleep(3)
    print("✅ TensorBoard 실행 완료")

def display_tensorboard_colab(port=6006):
    """Colab에서 TensorBoard 화면 표시"""
    if not is_colab():
        return
    
    print("🖥️  TensorBoard 화면 표시 중...")
    try:
        # IPython 가져오기
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython is None:
            raise Exception("IPython 환경이 아닙니다")
        
        # TensorBoard extension 로드
        ipython.run_line_magic('load_ext', 'tensorboard')
        print("✅ TensorBoard extension 로드 완료")
        
        # TensorBoard 실행 및 화면 표시
        ipython.run_line_magic('tensorboard', f'--logdir ./output --port {port}')
        print("✅ TensorBoard 화면 표시 완료")
    except Exception as e:
        print(f"⚠️  TensorBoard 화면 표시 실패: {e}")
        print("💡 수동으로 접속: http://localhost:6006")
        print("💡 또는 다음 명령어를 코랩 셀에서 직접 실행하세요:")
        print("   %load_ext tensorboard")
        print(f"   %tensorboard --logdir ./output --port {port}")

def main():
    parser = argparse.ArgumentParser(description="TensorBoard 실행 (Colab 지원)")
    parser.add_argument("--log_dir", type=str, default="./output", 
                       help="TensorBoard 로그 디렉토리 (기본값: ./output)")
    parser.add_argument("--port", type=int, default=6006,
                       help="TensorBoard 포트 (기본값: 6006)")
    parser.add_argument("--work_dir", type=str, default=None,
                       help="작업 디렉토리 변경 (선택사항)")
    parser.add_argument("--mount_drive", action="store_true",
                       help="구글 드라이브 마운트 (Colab에서만)")
    parser.add_argument("--display", action="store_true",
                       help="Colab에서 TensorBoard 화면 표시 (기본적으로 자동 활성화)")
    
    args = parser.parse_args()
    
    print("🚀 TensorBoard 실행 스크립트 시작")
    print("=" * 50)
    
    # 1. Colab 환경 확인
    if is_colab():
        print("🔍 Colab 환경 감지됨")
        
        # 2. 구글 드라이브 마운트 (요청된 경우)
        if args.mount_drive:
            mount_google_drive()
    else:
        print("🔍 로컬 환경에서 실행 중")
    
    # 3. 필요한 패키지 설치
    install_required_packages()
    
    # 4. 작업 디렉토리 변경 (지정된 경우)
    if args.work_dir:
        print(f"📁 작업 디렉토리 변경: {args.work_dir}")
        os.chdir(args.work_dir)
        print(f"✅ 현재 작업 디렉토리: {os.getcwd()}")
    
    # 5. TensorBoard 실행
    run_tensorboard(args.log_dir, args.port)
    
    # 6. Colab에서 화면 표시 (자동으로 활성화)
    if is_colab():
        display_tensorboard_colab(args.port)
    
    print("\n" + "=" * 50)
    print("🎉 TensorBoard 실행 완료!")
    
    if is_colab():
        print(f"💡 Colab에서 접속: http://localhost:{args.port}")
        print("💡 TensorBoard 화면이 자동으로 표시됩니다")
    else:
        print(f"💡 브라우저에서 접속: http://localhost:{args.port}")
    
    print("=" * 50)
    
    # 7. 무한 대기 (TensorBoard가 계속 실행되도록)
    try:
        print("⏳ TensorBoard가 실행 중입니다. 중단하려면 Ctrl+C를 누르세요...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard 실행을 중단합니다...")
        print("✅ 종료 완료")

if __name__ == "__main__":
    main() 