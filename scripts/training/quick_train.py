#!/usr/bin/env python3
"""
빠른 NPU 모델 훈련 스크립트 (사전 훈련 가중치 지원 버전)
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def quick_train(level='level1', epochs=30, pretrained='yolov11n.pt'):
    """빠른 훈련을 위한 단순 스크립트 (사전 훈련 가중치 지원)"""

    # 경로 설정
    project_root = Path(__file__).parent.parent.parent
    data_yaml = project_root / "data" / "dataset" / "data.yaml"
    model_yaml = project_root / "models" / "train" / f"npu_{level}_scales.yaml"

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 MPS (Apple Silicon) 가속 사용")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("🚀 CUDA GPU 가속 사용")
    else:
        device = 'cpu'
        print("⚠️ CPU 사용")

    print(f"\\n🎯 NPU {level.upper()} 모델 빠른 훈련")
    print(f"📂 모델: {model_yaml.name}")
    print(f"📊 데이터: {data_yaml.name}")
    print(f"⏱️ 에폭: {epochs}")
    print(f"🔄 사전 훈련 가중치: {pretrained if pretrained else '없음 (랜덤 초기화)'}")
    print("=" * 50)

    # 모델 로드 및 훈련
    try:
        # 커스텀 구조 로드
        model = YOLO(model_yaml)
        print(f"✅ 커스텀 구조 로드: {model_yaml.name}")

        # 사전 훈련 가중치 로드
        if pretrained:
            try:
                model.load(pretrained)
                print(f"✅ 사전 훈련 가중치 로드: {pretrained}")
            except Exception as e:
                print(f"⚠️ 가중치 로드 실패: {e}, 랜덤 초기화로 진행")

        # 사전 훈련 가중치 사용 시 학습률 조정
        lr0 = 0.005 if pretrained else 0.01

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=8,  # 작은 배치로 빠른 훈련
            imgsz=640,
            device=device,
            project='results/training',
            name=f'quick_{level}',

            # 빠른 훈련을 위한 설정
            patience=10,
            save_period=5,
            val=True,
            plots=False,  # 플롯 생성 안함
            verbose=True,

            # NPU 친화적 설정
            optimizer='SGD',
            lr0=lr0,  # 사전 훈련 시 낮은 학습률
            mosaic=0.0,
            mixup=0.0,
            amp=False
        )

        print(f"\\n✅ 훈련 완료!")
        print(f"📁 결과: {results.save_dir}")
        print(f"📈 학습률: {lr0} ({'사전 훈련' if pretrained else '랜덤 초기화'})")

    except Exception as e:
        print(f"❌ 훈련 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description='빠른 NPU 모델 훈련 (사전 훈련 가중치 지원)')
    parser.add_argument('--level', default='level1',
                       choices=['level1', 'level2', 'level3', 'level4'],
                       help='훈련할 NPU 레벨 (기본값: level1)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='훈련 에폭 수 (기본값: 30)')
    parser.add_argument('--pretrained', type=str, default='yolov11n.pt',
                       help='사전 훈련 가중치 경로 (기본값: yolov11n.pt)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='사전 훈련 가중치 사용하지 않음')

    args = parser.parse_args()

    pretrained = None if args.no_pretrained else args.pretrained

    if pretrained:
        print(f"📦 사전 훈련 가중치 사용: {pretrained}")
    else:
        print(f"⚠️ 랜덤 초기화 모드 (더 오래 걸릴 수 있음)")

    quick_train(args.level, args.epochs, pretrained)

if __name__ == "__main__":
    main()