#!/usr/bin/env python3
"""
빠른 INT8 TFLite 변환 스크립트
- Ultralytics 내장 기능 사용
- 복잡한 변환 과정 생략
"""

import os
import sys
from pathlib import Path

def quick_int8_conversion(model_path: str, output_dir: str):
    """Ultralytics를 이용한 빠른 INT8 변환"""
    print(f"🚀 빠른 INT8 변환 시작: {model_path}")

    try:
        from ultralytics import YOLO

        # 1. 모델 로드
        print("📥 YOLO 모델 로드 중...")
        model = YOLO(model_path)

        # 2. INT8 TFLite 직접 변환 (Ultralytics 최적화 사용)
        print("⚡ INT8 TFLite 변환 실행...")

        # Ultralytics의 내장 INT8 옵션 사용
        export_path = model.export(
            format='tflite',
            imgsz=640,
            int8=True,  # INT8 양자화 활성화
            data='coco128.yaml',  # 캘리브레이션용 데이터셋
            batch=1
        )

        print(f"✅ Ultralytics INT8 변환 완료: {export_path}")

        # 3. 결과 파일을 목적 디렉토리로 이동
        if export_path and os.path.exists(export_path):
            import shutil
            model_name = Path(model_path).stem
            final_path = os.path.join(output_dir, f"{model_name}_ultralytics_int8.tflite")
            shutil.copy2(export_path, final_path)

            # 파일 크기 확인
            size_mb = os.path.getsize(final_path) / (1024 * 1024)
            print(f"📊 생성된 INT8 모델 크기: {size_mb:.2f} MB")

            return final_path
        else:
            print("❌ 변환된 파일을 찾을 수 없음")
            return None

    except Exception as e:
        print(f"❌ Ultralytics INT8 변환 실패: {e}")
        return None

def direct_pytorch_int8(model_path: str, output_dir: str):
    """PyTorch 모델을 직접 INT8로 양자화"""
    print(f"🔧 PyTorch 직접 INT8 양자화: {model_path}")

    try:
        import torch
        import torch.quantization as quant

        # 1. 모델 로드
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        model.eval()

        # 2. 정적 양자화 (더 강력한 INT8)
        print("⚡ 정적 INT8 양자화 실행...")

        # QConfig 설정
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 양자화 준비
        model_prepared = torch.quantization.prepare(model)

        # 캘리브레이션 (간단한 더미 데이터)
        print("📊 캘리브레이션 실행...")
        with torch.no_grad():
            for _ in range(10):  # 빠른 캘리브레이션
                dummy_input = torch.randn(1, 3, 640, 640)
                model_prepared(dummy_input)

        # INT8 변환
        model_int8 = torch.quantization.convert(model_prepared)

        # 저장
        model_name = Path(model_path).stem
        output_path = os.path.join(output_dir, f"{model_name}_pytorch_int8.pt")
        torch.save(model_int8, output_path)

        # 크기 비교
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = ((original_size - quantized_size) / original_size) * 100

        print(f"✅ PyTorch INT8 양자화 완료: {output_path}")
        print(f"📊 크기 감소: {original_size:.2f} MB → {quantized_size:.2f} MB ({reduction:.1f}% 감소)")

        return output_path

    except Exception as e:
        print(f"❌ PyTorch INT8 양자화 실패: {e}")
        return None

def simple_onnx_int8(onnx_path: str, output_dir: str):
    """ONNX 모델을 간단하게 INT8 양자화"""
    print(f"🔧 ONNX INT8 양자화: {onnx_path}")

    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader
        import numpy as np

        # 캘리브레이션 데이터 생성
        class SimpleCalibrationDataReader(CalibrationDataReader):
            def __init__(self):
                self.data_list = []
                # 10개의 캘리브레이션 샘플만 사용 (빠름)
                for _ in range(10):
                    data = np.random.randn(1, 3, 640, 640).astype(np.float32)
                    self.data_list.append({'input': data})
                self.iterator = iter(self.data_list)

            def get_next(self):
                try:
                    return next(self.iterator)
                except StopIteration:
                    return None

        # INT8 정적 양자화
        model_name = Path(onnx_path).stem
        output_path = os.path.join(output_dir, f"{model_name}_static_int8.onnx")

        calibration_reader = SimpleCalibrationDataReader()

        print("⚡ ONNX 정적 INT8 양자화 실행...")
        quantize_static(
            onnx_path,
            output_path,
            calibration_reader,
            quant_format='IntegerOps',
            activation_type='int8',
            weight_type='int8'
        )

        # 크기 비교
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = ((original_size - quantized_size) / original_size) * 100

        print(f"✅ ONNX INT8 양자화 완료: {output_path}")
        print(f"📊 크기 감소: {original_size:.2f} MB → {quantized_size:.2f} MB ({reduction:.1f}% 감소)")

        return output_path

    except Exception as e:
        print(f"❌ ONNX INT8 양자화 실패: {e}")
        return None

def main():
    """모든 INT8 양자화 방법 시도"""
    print("🚀 빠른 INT8 양자화 시작")

    # 입력/출력 디렉토리
    models_dir = "models"
    output_dir = "models/optimized"
    os.makedirs(output_dir, exist_ok=True)

    # 모델 찾기
    model_files = []
    for ext in ['.pt', '.pth']:
        model_files.extend(Path(models_dir).glob(f"*{ext}"))

    onnx_files = list(Path(output_dir).glob("*.onnx"))

    if not model_files and not onnx_files:
        print("❌ 변환할 모델을 찾을 수 없습니다")
        return

    results = []

    # 1. Ultralytics INT8 시도 (가장 빠름)
    for model_path in model_files:
        print(f"\n{'='*50}")
        print(f"🎯 {model_path.name} - Ultralytics INT8 변환")

        result = quick_int8_conversion(str(model_path), output_dir)
        if result:
            results.append(("Ultralytics INT8", result))
            break  # 성공하면 다른 방법 생략

    # 2. PyTorch 직접 INT8 (빠름)
    if not results:
        for model_path in model_files:
            print(f"\n{'='*50}")
            print(f"🎯 {model_path.name} - PyTorch 직접 INT8")

            result = direct_pytorch_int8(str(model_path), output_dir)
            if result:
                results.append(("PyTorch INT8", result))
                break

    # 3. ONNX INT8 (중간 속도)
    for onnx_path in onnx_files:
        if 'quantized' not in onnx_path.name:  # 이미 양자화된 것 제외
            print(f"\n{'='*50}")
            print(f"🎯 {onnx_path.name} - ONNX INT8 양자화")

            result = simple_onnx_int8(str(onnx_path), output_dir)
            if result:
                results.append(("ONNX INT8", result))
                break

    # 결과 요약
    print(f"\n{'='*50}")
    print("🎉 INT8 양자화 결과 요약:")

    if results:
        for method, path in results:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {method}: {Path(path).name} ({size_mb:.2f} MB)")

        print(f"\n📁 결과 위치: {output_dir}/")
        print("🚀 Docker에서 테스트: docker-compose up simple-evaluator")
    else:
        print("❌ 모든 INT8 변환 방법이 실패했습니다")
        print("💡 TensorFlow 버전 확인: pip install tensorflow==2.19.0")

if __name__ == "__main__":
    main()