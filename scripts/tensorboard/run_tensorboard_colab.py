#!/usr/bin/env python3
"""
TensorBoard 실행 스크립트

이 스크립트는 TensorBoard를 실행하여 학습 과정을 모니터링할 수 있게 합니다.
"""

import os
import sys
import threading
import time
import argparse
from pathlib import Path

def install_required_packages():
    """필요한 패키지 설치"""
    print("📦 필요한 패키지 설치 중...")
    
    # TensorBoard 설치
    os.system("pip install -q tensorboard")
    
    print("✅ 패키지 설치 완료")

def run_tensorboard(log_dir, port=6006):
    """TensorBoard 실행"""
    print(f"📊 TensorBoard 실행 중... (로그 디렉토리: {log_dir}, 포트: {port})")
    
    # 로그 디렉토리가 존재하는지 확인
    if not os.path.exists(log_dir):
        print(f"⚠️  로그 디렉토리가 존재하지 않습니다: {log_dir}")
        print("📁 디렉토리를 생성합니다...")
        os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard 실행 명령
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

def main():
    parser = argparse.ArgumentParser(description="TensorBoard 실행")
    parser.add_argument("--log_dir", type=str, default="./output", 
                       help="TensorBoard 로그 디렉토리 (기본값: ./output)")
    parser.add_argument("--port", type=int, default=6006,
                       help="TensorBoard 포트 (기본값: 6006)")
    parser.add_argument("--work_dir", type=str, default=None,
                       help="작업 디렉토리 변경 (선택사항)")
    
    args = parser.parse_args()
    
    print("🚀 TensorBoard 실행 스크립트 시작")
    print("=" * 50)
    
    # 1. 필요한 패키지 설치
    install_required_packages()
    
    # 2. 작업 디렉토리 변경 (지정된 경우)
    if args.work_dir:
        print(f"📁 작업 디렉토리 변경: {args.work_dir}")
        os.chdir(args.work_dir)
        print(f"✅ 현재 작업 디렉토리: {os.getcwd()}")
    
    # 3. TensorBoard 실행
    run_tensorboard(args.log_dir, args.port)
    
    print("\n" + "=" * 50)
    print("🎉 TensorBoard 실행 완료!")
    print(f"💡 브라우저에서 접속: http://localhost:{args.port}")
    print("💡 또는 콜랩에서: http://localhost:6006")
    print("=" * 50)
    
    # 4. 무한 대기 (TensorBoard가 계속 실행되도록)
    try:
        print("⏳ TensorBoard가 실행 중입니다. 중단하려면 Ctrl+C를 누르세요...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard 실행을 중단합니다...")
        print("✅ 종료 완료")

if __name__ == "__main__":
    main() 