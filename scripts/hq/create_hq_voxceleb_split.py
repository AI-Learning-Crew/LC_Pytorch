#!/usr/bin/env python3
"""
VoxCeleb 데이터셋의 train/test/val 분할을 위한 매칭 파일 생성 스크립트

이 스크립트는 VoxCeleb1과 VoxCeleb2 데이터셋에서 실제로 사용 가능한 
얼굴과 음성 데이터가 모두 있는 identity들을 찾아 train/test/val 분할을 생성합니다.

vox1_meta.csv는 VoxCeleb 공식 웹사이트에서 다운로드 가능합니다:
    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path


def find_data_directories(base_dir, possible_names):
    """
    가능한 디렉토리 이름들 중에서 실제 존재하는 디렉토리를 찾습니다.
    
    Args:
        base_dir (Path): 검색할 기본 디렉토리
        possible_names (list): 가능한 디렉토리 이름들의 리스트
    
    Returns:
        Path or None: 찾은 디렉토리 경로 또는 None
    """
    for name in possible_names:
        dir_path = base_dir / name
        if dir_path.exists():
            return dir_path
    return None


def create_voxceleb_split(vox_dir, output_json=None, max_identities=None):
    """
    VoxCeleb 데이터셋의 train/test/val 분할을 생성합니다.
    
    Args:
        vox_dir (str): VoxCeleb 데이터가 저장된 디렉토리 경로
        output_json (str): 출력 JSON 파일 경로 (기본값: vox_dir/split.json)
        max_identities (int): 각 데이터셋에서 처리할 최대 identity 개수 (기본값: None, 모든 데이터 처리)
    
    Returns:
        dict: 분할 정보가 담긴 딕셔너리
    """
    vox_dir = Path(vox_dir)
    
    # 메타데이터 파일 경로
    vox1_meta_csv = vox_dir / 'vox1' / 'vox1_meta.csv'
    vox2_meta_csv = vox_dir / 'vox2' / 'full_vox2_meta.csv'
    
    # 출력 파일 경로
    if output_json is None:
        output_json = vox_dir / 'split.json'
    
    # 메타데이터 파일 존재 확인
    if not vox1_meta_csv.exists():
        print(f"Warning: {vox1_meta_csv} 파일을 찾을 수 없습니다.")
        vox1_df = None
    else:
        vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
        vox1_df.columns = vox1_df.columns.str.strip()
        print("VoxCeleb1 메타데이터:")
        print(vox1_df.head())
    
    if not vox2_meta_csv.exists():
        print(f"Warning: {vox2_meta_csv} 파일을 찾을 수 없습니다.")
        vox2_df = None
    else:
        vox2_df = pd.read_csv(vox2_meta_csv, sep='\t')
        vox2_df.columns = vox2_df.columns.str.strip()
        print("VoxCeleb2 메타데이터:")
        print(vox2_df.head())
    
    # 분할 딕셔너리 초기화
    split_dict = {
        'vox1': {'train': [], 'val': [], 'test': []},
        'vox2': {'train': [], 'val': [], 'test': []}
    }
    
    # 실제 사용 가능한 데이터 확인
    vox1_mel_available = set()
    vox1_face_available = set()
    vox2_mel_available = set()
    vox2_face_available = set()
    
    # VoxCeleb1 데이터 확인 - 가능한 디렉토리 이름들 시도
    vox1_mel_dir = find_data_directories(
        vox_dir / 'vox1', 
        ['mel_spectrograms', 'mel_spectograms', 'mel_specs', 'spectrograms']
    )
    vox1_face_dir = find_data_directories(
        vox_dir / 'vox1', 
        ['masked_faces', 'faces', 'face_images', 'cropped_faces']
    )
    
    if vox1_mel_dir:
        vox1_mel_list = os.listdir(vox1_mel_dir)
        if max_identities and len(vox1_mel_list) > max_identities:
            vox1_mel_list = sorted(vox1_mel_list)[:max_identities]
            print(f"VoxCeleb1 mel 데이터를 {max_identities}개로 제한합니다.")
        vox1_mel_available = set(vox1_mel_list)
        print(f"VoxCeleb1 {vox1_mel_dir.name} 디렉토리에서 {len(vox1_mel_available)}개 identity 처리")
        print(f"  예시: {list(vox1_mel_available)[:5]}")
    else:
        print("VoxCeleb1 mel_spectrograms 디렉토리를 찾을 수 없습니다.")
    
    if vox1_face_dir:
        vox1_face_list = os.listdir(vox1_face_dir)
        if max_identities and len(vox1_face_list) > max_identities:
            vox1_face_list = sorted(vox1_face_list)[:max_identities]
            print(f"VoxCeleb1 face 데이터를 {max_identities}개로 제한합니다.")
        vox1_face_available = set(vox1_face_list)
        print(f"VoxCeleb1 {vox1_face_dir.name} 디렉토리에서 {len(vox1_face_available)}개 identity 처리")
        print(f"  예시: {list(vox1_face_available)[:5]}")
    else:
        print("VoxCeleb1 masked_faces 디렉토리를 찾을 수 없습니다.")
    
    # VoxCeleb2 데이터 확인 - 가능한 디렉토리 이름들 시도
    vox2_mel_dir = find_data_directories(
        vox_dir / 'vox2', 
        ['mel_spectrograms', 'mel_spectograms', 'mel_specs', 'spectrograms']
    )
    vox2_face_dir = find_data_directories(
        vox_dir / 'vox2', 
        ['masked_faces', 'faces', 'face_images', 'cropped_faces']
    )
    
    if vox2_mel_dir:
        vox2_mel_list = os.listdir(vox2_mel_dir)
        if max_identities and len(vox2_mel_list) > max_identities:
            vox2_mel_list = sorted(vox2_mel_list)[:max_identities]
            print(f"VoxCeleb2 mel 데이터를 {max_identities}개로 제한합니다.")
        vox2_mel_available = set(vox2_mel_list)
        print(f"VoxCeleb2 {vox2_mel_dir.name} 디렉토리에서 {len(vox2_mel_available)}개 identity 처리")
        print(f"  예시: {list(vox2_mel_available)[:5]}")
    else:
        print("VoxCeleb2 mel_spectrograms 디렉토리를 찾을 수 없습니다.")
    
    if vox2_face_dir:
        vox2_face_list = os.listdir(vox2_face_dir)
        if max_identities and len(vox2_face_list) > max_identities:
            vox2_face_list = sorted(vox2_face_list)[:max_identities]
            print(f"VoxCeleb2 face 데이터를 {max_identities}개로 제한합니다.")
        vox2_face_available = set(vox2_face_list)
        print(f"VoxCeleb2 {vox2_face_dir.name} 디렉토리에서 {len(vox2_face_available)}개 identity 처리")
        print(f"  예시: {list(vox2_face_available)[:5]}")
    else:
        print("VoxCeleb2 masked_faces 디렉토리를 찾을 수 없습니다.")
    
    # 얼굴과 음성 데이터가 모두 있는 identity들의 교집합
    vox1_available = vox1_mel_available.intersection(vox1_face_available)
    vox2_available = vox2_mel_available.intersection(vox2_face_available)
    
    print(f"\nVoxCeleb1 - 실제 사용 가능한 identity 수: {len(vox1_available)}")
    if len(vox1_available) > 0:
        print(f"  예시: {list(vox1_available)[:5]}")
    
    print(f"VoxCeleb2 - 실제 사용 가능한 identity 수: {len(vox2_available)}")
    if len(vox2_available) > 0:
        print(f"  예시: {list(vox2_available)[:5]}")
    
    # VoxCeleb1 분할 (8:1:1 비율)
    vox1_available_list = sorted(list(vox1_available))
    for i, name in enumerate(vox1_available_list):
        if i % 10 == 8:
            split_dict['vox1']['test'].append(name)
        elif i % 10 == 9:
            split_dict['vox1']['val'].append(name)
        else:
            split_dict['vox1']['train'].append(name)
    
    # VoxCeleb2 분할 (8:1:1 비율)
    vox2_available_list = sorted(list(vox2_available))
    for i, name in enumerate(vox2_available_list):
        if i % 10 == 8:
            split_dict['vox2']['test'].append(name)
        elif i % 10 == 9:
            split_dict['vox2']['val'].append(name)
        else:
            split_dict['vox2']['train'].append(name)
    
    # 분할 결과 출력
    print(f"\nVoxCeleb1 분할 결과:")
    print(f"  Train: {len(split_dict['vox1']['train'])}")
    print(f"  Val:   {len(split_dict['vox1']['val'])}")
    print(f"  Test:  {len(split_dict['vox1']['test'])}")
    
    print(f"\nVoxCeleb2 분할 결과:")
    print(f"  Train: {len(split_dict['vox2']['train'])}")
    print(f"  Val:   {len(split_dict['vox2']['val'])}")
    print(f"  Test:  {len(split_dict['vox2']['test'])}")
    
    # max_identities가 설정된 경우 파일명에 표시
    if max_identities:
        output_json = Path(output_json)
        output_json = output_json.parent / f"{output_json.stem}_max{max_identities}{output_json.suffix}"
        print(f"\n제한된 데이터셋이므로 파일명을 {output_json.name}로 변경합니다.")
    
    # JSON 파일로 저장
    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(split_dict, outfile, indent=2, ensure_ascii=False)
    
    print(f"\n분할 정보가 {output_json}에 저장되었습니다.")
    
    return split_dict


def main():
    parser = argparse.ArgumentParser(
        description='VoxCeleb 데이터셋의 train/test/val 분할을 생성합니다.'
    )
    parser.add_argument(
        '--vox_dir',
        type=str,
        default='./data/HQVoxCeleb',
        help='VoxCeleb 데이터가 저장된 디렉토리 경로 (기본값: ./data/HQVoxCeleb)'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='출력 JSON 파일 경로 (기본값: vox_dir/split.json)'
    )
    parser.add_argument(
        '--max_identities',
        type=int,
        default=None,
        help='각 데이터셋에서 처리할 최대 identity 개수 (빠른 테스트용, 예: 100)'
    )
    
    args = parser.parse_args()
    
    # 디렉토리 존재 확인
    if not os.path.exists(args.vox_dir):
        print(f"Error: {args.vox_dir} 디렉토리가 존재하지 않습니다.")
        return 1
    
    # max_identities 값 검증
    if args.max_identities is not None and args.max_identities <= 0:
        print(f"Error: --max_identities는 양수여야 합니다. 입력값: {args.max_identities}")
        return 1
    
    try:
        if args.max_identities:
            print(f"간략화 모드: 각 데이터셋에서 최대 {args.max_identities}개 identity만 처리합니다.")
        
        split_dict = create_voxceleb_split(args.vox_dir, args.output_json, args.max_identities)
        print("\n분할 생성이 완료되었습니다!")
        return 0
    except Exception as e:
        print(f"Error: 분할 생성 중 오류가 발생했습니다: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 