#!/usr/bin/env python3
"""
PyTorch to ONNX Model Converter
Converts PyTorch (.pt/.pth) models to ONNX format
"""

import os
import sys
import argparse
import torch
import torch.onnx
from pathlib import Path
from ultralytics import YOLO


def convert_yolo_to_onnx(pt_model_path: str, output_path: str = None, img_size: int = 640):
    """
    Convert YOLO PyTorch model to ONNX using Ultralytics
    This is the recommended method for YOLOv5/v8/v11 models
    """
    print(f"Converting YOLO model: {pt_model_path}")

    try:
        # Load YOLO model using ultralytics
        model = YOLO(pt_model_path)

        # Set output path
        if output_path is None:
            pt_path = Path(pt_model_path)
            output_path = pt_path.parent / f"{pt_path.stem}_converted.onnx"

        # Export to ONNX
        success = model.export(
            format='onnx',
            imgsz=img_size,
            optimize=True,
            half=False,  # Use float32 for better compatibility
            dynamic=False,  # Fixed input size for better optimization
            simplify=True,  # Simplify the model
            opset=11  # ONNX opset version (11 is widely supported)
        )

        if success:
            print(f"✅ Successfully converted to: {success}")
            return success
        else:
            print("❌ Conversion failed")
            return None

    except Exception as e:
        print(f"❌ Error during YOLO conversion: {str(e)}")
        return None


def convert_pytorch_to_onnx_manual(pt_model_path: str, output_path: str = None, img_size: int = 640):
    """
    Manual PyTorch to ONNX conversion using torch.onnx.export
    Use this if ultralytics export fails
    """
    print(f"Manual conversion for: {pt_model_path}")

    try:
        # Load PyTorch model
        device = torch.device('cpu')  # Use CPU for conversion
        model = torch.load(pt_model_path, map_location=device, weights_only=False)

        if hasattr(model, 'model'):
            model = model.model  # Extract model if wrapped

        model.eval()

        # Set output path
        if output_path is None:
            pt_path = Path(pt_model_path)
            output_path = pt_path.parent / f"{pt_path.stem}_manual.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"✅ Manual conversion successful: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Manual conversion failed: {str(e)}")
        return None


def verify_onnx_model(onnx_path: str):
    """Verify the converted ONNX model"""
    print(f"\n🔍 Verifying ONNX model: {onnx_path}")

    try:
        import onnx
        import onnxruntime as ort

        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")

        # Test inference with ONNX Runtime
        session = ort.InferenceSession(onnx_path)

        # Get input details
        input_details = session.get_inputs()
        output_details = session.get_outputs()

        print(f"📋 Model Details:")
        print(f"   Input: {input_details[0].name} - Shape: {input_details[0].shape}")
        print(f"   Output: {output_details[0].name} - Shape: {output_details[0].shape}")

        # Test with dummy input
        input_shape = input_details[0].shape
        if input_shape[0] == 'batch_size':
            input_shape = [1] + list(input_shape[1:])

        dummy_input = torch.randn(*input_shape).numpy()

        # Run inference test
        outputs = session.run(None, {input_details[0].name: dummy_input})
        print(f"✅ Inference test passed - Output shape: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"❌ ONNX verification failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--input', '-i',
                       required=True,
                       help='Path to input PyTorch model (.pt or .pth)')
    parser.add_argument('--output', '-o',
                       help='Path to output ONNX model (optional)')
    parser.add_argument('--img-size',
                       type=int,
                       default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--method',
                       choices=['ultralytics', 'manual', 'both'],
                       default='ultralytics',
                       help='Conversion method (default: ultralytics)')
    parser.add_argument('--verify',
                       action='store_true',
                       help='Verify the converted ONNX model')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("PyTorch to ONNX Model Converter")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Method: {args.method}")
    print(f"Image size: {args.img_size}")

    converted_models = []

    # Try Ultralytics method
    if args.method in ['ultralytics', 'both']:
        print(f"\n🔄 Attempting Ultralytics conversion...")
        onnx_path = convert_yolo_to_onnx(args.input, args.output, args.img_size)
        if onnx_path:
            converted_models.append(onnx_path)

    # Try manual method
    if args.method in ['manual', 'both']:
        print(f"\n🔄 Attempting manual conversion...")
        onnx_path = convert_pytorch_to_onnx_manual(args.input, args.output, args.img_size)
        if onnx_path:
            converted_models.append(onnx_path)

    # Verify converted models
    if args.verify and converted_models:
        for model_path in converted_models:
            if os.path.exists(model_path):
                verify_onnx_model(model_path)

    # Summary
    print("\n" + "=" * 60)
    if converted_models:
        print("✅ CONVERSION COMPLETE")
        for model_path in converted_models:
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   📁 {model_path} ({size_mb:.2f} MB)")
    else:
        print("❌ CONVERSION FAILED")
        print("Try different methods or check your PyTorch model")
    print("=" * 60)


if __name__ == "__main__":
    main()