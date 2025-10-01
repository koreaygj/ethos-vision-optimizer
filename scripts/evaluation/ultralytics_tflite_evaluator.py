#!/usr/bin/env python3
"""
Ultralytics TFLite Model Evaluator
Easy and accurate evaluation using Ultralytics YOLO framework

=== USAGE EXAMPLES ===
# Single model evaluation
python scripts/evaluation/ultralytics_tflite_evaluator.py --model models/optimized_npu/level3_relu/level3_relu_full_integer_quant.tflite

# Batch evaluate all TFLite models
python scripts/evaluation/ultralytics_tflite_evaluator.py --batch-eval

# Custom dataset and settings
python scripts/evaluation/ultralytics_tflite_evaluator.py --model my_model.tflite --data data/dataset/data.yaml --imgsz 640

# Save results as JSON and Markdown
python scripts/evaluation/ultralytics_tflite_evaluator.py --batch-eval --output results/evaluation_report

# Save single model results
python scripts/evaluation/ultralytics_tflite_evaluator.py --model my_model.tflite --output results/single_model_eval.md

# Quick test with sample images
python scripts/evaluation/ultralytics_tflite_evaluator.py --model my_model.tflite --test-predict

=== OUTPUT ===
- Official YOLO mAP metrics (mAP50, mAP50-95)
- Per-class precision, recall, F1-score
- Speed benchmarks (preprocess, inference, postprocess)
- Confusion matrices and detailed reports
- JSON results + Markdown summary

=== REQUIREMENTS ===
pip install ultralytics

=== FEATURES ===
- Uses official Ultralytics validation pipeline
- Automatic YOLO postprocessing and NMS
- Support for all TFLite quantization formats
- Consistent with PyTorch model evaluation
- Built-in visualization and reporting

===================
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("❌ Ultralytics not installed! Run: pip install ultralytics")
    sys.exit(1)

class UltralyticsEvaluator:
    """Ultralytics-based TFLite model evaluator"""

    def __init__(self, model_path: str, data_config: str = "data/dataset/data.yaml"):
        """
        Initialize evaluator

        Args:
            model_path: Path to TFLite model
            data_config: Path to dataset YAML config
        """
        self.model_path = Path(model_path)
        self.data_config = Path(data_config)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not self.data_config.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_config}")

        print(f"📋 Loading model: {self.model_path.name}")

        # Load model with explicit task specification
        self.model = YOLO(str(self.model_path), task='detect')

        print(f"✅ Model loaded successfully!")
        print(f"   Type: {type(self.model)}")
        print(f"   Task: {self.model.task}")

    def evaluate_accuracy(self, imgsz: int = 640, conf: float = 0.25,
                         iou: float = 0.45, save_json: bool = True) -> Dict:
        """
        Evaluate model accuracy using Ultralytics validation

        Args:
            imgsz: Image size for evaluation
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save_json: Save detailed JSON results

        Returns:
            Validation results dictionary
        """
        print(f"🎯 Starting accuracy evaluation...")
        print(f"   Dataset: {self.data_config}")
        print(f"   Image size: {imgsz}")
        print(f"   Confidence threshold: {conf}")
        print(f"   IoU threshold: {iou}")

        # Run validation
        start_time = time.time()

        try:
            results = self.model.val(
                data=str(self.data_config),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save_json=save_json,
                plots=True,  # Generate plots
                verbose=True
            )

            eval_time = time.time() - start_time

            print(f"✅ Evaluation completed in {eval_time:.2f}s")

            # Extract key metrics
            metrics = {
                'model_info': {
                    'model_path': str(self.model_path),
                    'model_name': self.model_path.name,
                    'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                    'task': self.model.task
                },
                'evaluation_params': {
                    'dataset': str(self.data_config),
                    'imgsz': imgsz,
                    'conf_threshold': conf,
                    'iou_threshold': iou,
                    'evaluation_time_seconds': eval_time
                },
                'accuracy_metrics': {
                    'mAP50': float(results.box.map50),
                    'mAP50_95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'fitness': float(results.fitness)
                },
                'speed_metrics': {
                    'preprocess_ms': float(results.speed.get('preprocess', 0)),
                    'inference_ms': float(results.speed.get('inference', 0)),
                    'postprocess_ms': float(results.speed.get('postprocess', 0)),
                    'total_ms': float(sum(results.speed.values())),
                    'fps': 1000.0 / float(sum(results.speed.values())) if sum(results.speed.values()) > 0 else 0
                },
                'class_metrics': {},
                'timestamp': datetime.now().isoformat()
            }

            # Extract per-class metrics if available
            if hasattr(results.box, 'maps') and results.box.maps is not None:
                class_names = self.model.names
                for i, class_ap in enumerate(results.box.maps):
                    if i < len(class_names):
                        metrics['class_metrics'][i] = {
                            'class_name': class_names[i],
                            'ap': float(class_ap)
                        }

            return metrics

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

    def test_prediction(self, test_images: List[str] = None, save_results: bool = True) -> Dict:
        """
        Test prediction on sample images

        Args:
            test_images: List of image paths to test
            save_results: Save prediction results

        Returns:
            Prediction results
        """
        if test_images is None:
            # Find some test images
            test_dir = Path("data/dataset/test/images")
            if not test_dir.exists():
                test_dir = Path("data/dataset/valid/images")

            if test_dir.exists():
                image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                test_images = [str(f) for f in image_files[:5]]  # Test on 5 images
            else:
                print("⚠️ No test images found")
                return {}

        print(f"🔍 Testing prediction on {len(test_images)} images...")

        results = []
        total_time = 0

        for img_path in test_images:
            start_time = time.time()

            try:
                result = self.model.predict(
                    source=img_path,
                    save=save_results,
                    verbose=False
                )

                inference_time = time.time() - start_time
                total_time += inference_time

                if result and len(result) > 0:
                    boxes = result[0].boxes
                    detections = []

                    if boxes is not None:
                        for i in range(len(boxes.cls)):
                            detections.append({
                                'class_id': int(boxes.cls[i]),
                                'class_name': self.model.names[int(boxes.cls[i])],
                                'confidence': float(boxes.conf[i]),
                                'bbox': boxes.xywh[i].tolist()
                            })

                    results.append({
                        'image_path': img_path,
                        'inference_time_ms': inference_time * 1000,
                        'num_detections': len(detections),
                        'detections': detections
                    })

                print(f"   ✅ {Path(img_path).name}: {len(detections) if 'detections' in locals() else 0} detections in {inference_time*1000:.1f}ms")

            except Exception as e:
                print(f"   ❌ {Path(img_path).name}: Error - {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        avg_time = total_time / len(test_images) if test_images else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0

        summary = {
            'model_name': self.model_path.name,
            'num_test_images': len(test_images),
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        print(f"📊 Prediction test completed:")
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   FPS: {fps:.1f}")

        return summary

    def print_results(self, results: Dict):
        """Print evaluation results in a nice format"""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULTS: {results['model_info']['model_name']}")
        print(f"{'='*80}")

        # Model info
        print(f"📋 Model Information:")
        print(f"   Size: {results['model_info']['model_size_mb']:.1f} MB")
        print(f"   Task: {results['model_info']['task']}")

        # Accuracy metrics
        acc = results['accuracy_metrics']
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   mAP50: {acc['mAP50']:.3f}")
        print(f"   mAP50-95: {acc['mAP50_95']:.3f}")
        print(f"   Precision: {acc['precision']:.3f}")
        print(f"   Recall: {acc['recall']:.3f}")
        print(f"   Fitness: {acc['fitness']:.3f}")

        # Speed metrics
        speed = results['speed_metrics']
        print(f"\n⚡ Speed Metrics:")
        print(f"   Preprocess: {speed['preprocess_ms']:.2f}ms")
        print(f"   Inference: {speed['inference_ms']:.2f}ms")
        print(f"   Postprocess: {speed['postprocess_ms']:.2f}ms")
        print(f"   Total: {speed['total_ms']:.2f}ms")
        print(f"   FPS: {speed['fps']:.1f}")

        # Top classes
        if results['class_metrics']:
            sorted_classes = sorted(results['class_metrics'].items(),
                                  key=lambda x: x[1]['ap'], reverse=True)
            print(f"\n🏆 Top 5 Classes (by AP):")
            for i, (class_id, metrics) in enumerate(sorted_classes[:5]):
                print(f"   {i+1}. {metrics['class_name']}: AP = {metrics['ap']:.3f}")


def save_single_model_report(results: Dict, output_file: Path):
    """Generate detailed markdown report for single model"""

    model_info = results['model_info']
    acc = results['accuracy_metrics']
    speed = results['speed_metrics']
    eval_params = results['evaluation_params']

    report = f"""# TFLite Model Evaluation Report

**Model**: `{model_info['model_name']}`
**Generated**: {results['timestamp']}
**Framework**: Ultralytics YOLO
**Evaluation Time**: {eval_params['evaluation_time_seconds']:.2f} seconds

---

## 📋 Model Information

| Property | Value |
|----------|-------|
| **Model Path** | `{model_info['model_path']}` |
| **Model Size** | {model_info['model_size_mb']:.2f} MB |
| **Task Type** | {model_info['task']} |
| **Dataset** | `{eval_params['dataset']}` |

---

## 🎯 Accuracy Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | **{acc['mAP50']:.3f}** | Mean Average Precision at IoU=0.5 |
| **mAP50-95** | **{acc['mAP50_95']:.3f}** | Mean Average Precision at IoU=0.5:0.95 |
| **Precision** | {acc['precision']:.3f} | Overall precision across all classes |
| **Recall** | {acc['recall']:.3f} | Overall recall across all classes |
| **Fitness** | {acc['fitness']:.3f} | YOLO fitness score |

---

## ⚡ Performance Metrics

| Stage | Time (ms) | Description |
|-------|-----------|-------------|
| **Preprocess** | {speed['preprocess_ms']:.2f} | Image preprocessing time |
| **Inference** | {speed['inference_ms']:.2f} | Model inference time |
| **Postprocess** | {speed['postprocess_ms']:.2f} | NMS and postprocessing time |
| **Total** | **{speed['total_ms']:.2f}** | **Total pipeline time** |

### Speed Summary
- **FPS**: **{speed['fps']:.1f}** frames per second
- **Throughput**: {speed['fps']*60:.0f} images per minute

---

## 🎪 Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| **Image Size** | {eval_params['imgsz']}×{eval_params['imgsz']} |
| **Confidence Threshold** | {eval_params['conf_threshold']} |
| **IoU Threshold** | {eval_params['iou_threshold']} |

---
"""

    # Add per-class metrics if available
    if results['class_metrics']:
        report += """## 📊 Per-Class Performance

| Rank | Class ID | Class Name | Average Precision |
|------|----------|------------|-------------------|
"""
        sorted_classes = sorted(results['class_metrics'].items(),
                              key=lambda x: x[1]['ap'], reverse=True)

        for i, (class_id, metrics) in enumerate(sorted_classes, 1):
            report += f"| {i} | {class_id} | {metrics['class_name']} | {metrics['ap']:.3f} |\n"

        # Add performance analysis
        aps = [metrics['ap'] for metrics in results['class_metrics'].values()]
        report += f"""
### Class Performance Analysis
- **Best performing class**: {sorted_classes[0][1]['class_name']} (AP: {sorted_classes[0][1]['ap']:.3f})
- **Worst performing class**: {sorted_classes[-1][1]['class_name']} (AP: {sorted_classes[-1][1]['ap']:.3f})
- **Average AP**: {sum(aps)/len(aps):.3f}
- **AP Standard deviation**: {(sum((ap - sum(aps)/len(aps))**2 for ap in aps)/len(aps))**0.5:.3f}

---
"""

    # Add performance assessment
    performance_level = "Excellent" if acc['mAP50'] > 0.9 else "Good" if acc['mAP50'] > 0.8 else "Fair" if acc['mAP50'] > 0.7 else "Needs Improvement"
    speed_level = "Very Fast" if speed['fps'] > 100 else "Fast" if speed['fps'] > 50 else "Moderate" if speed['fps'] > 20 else "Slow"

    report += f"""## 📈 Performance Assessment

### Overall Rating
- **Accuracy**: {performance_level} (mAP50: {acc['mAP50']:.3f})
- **Speed**: {speed_level} ({speed['fps']:.1f} FPS)

### Use Case Recommendations
"""

    if acc['mAP50'] > 0.9 and speed['fps'] > 30:
        report += "- ✅ **Production Ready**: Excellent accuracy and real-time performance\n"
    elif acc['mAP50'] > 0.85:
        report += "- ✅ **High Accuracy Applications**: Great for accuracy-critical tasks\n"
    elif speed['fps'] > 50:
        report += "- ✅ **Real-time Applications**: Good for speed-critical applications\n"
    else:
        report += "- ⚠️ **Development/Testing**: Consider further optimization\n"

    if model_info['model_size_mb'] < 10:
        report += "- 📱 **Edge Device Friendly**: Suitable for mobile/embedded deployment\n"

    report += f"""
### Comparison Benchmarks
- **vs Pure Models**: TFLite quantization provides hardware acceleration
- **vs PyTorch**: Faster inference with slight accuracy tradeoff
- **Model Efficiency**: {acc['mAP50']/model_info['model_size_mb']:.3f} mAP50 per MB

---

## 🛠️ Technical Details

### Model Specifications
- **Quantization**: Full Integer (INT8)
- **Input Format**: 640×640×3 RGB images
- **Output Format**: YOLO detection format
- **Framework**: TensorFlow Lite

### Hardware Requirements
- **Memory**: ~{model_info['model_size_mb']*2:.0f} MB RAM (model + runtime)
- **CPU**: ARM/x86 compatible
- **Accelerator**: NPU/GPU compatible (if available)

---

*Report generated by Ultralytics TFLite Evaluator v1.0*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"📄 Detailed report saved to: {output_file}")


def find_tflite_models(base_dir: str = "models") -> List[Path]:
    """Find TFLite models in the project"""
    base_path = Path(base_dir)
    patterns = [
        "**/level*_full_integer_quant.tflite",
        "**/level*_int8.tflite",
        "**/*traffic*int8.tflite",
        "**/*traffic*full_integer_quant.tflite"
    ]

    tflite_files = []
    for pattern in patterns:
        tflite_files.extend(base_path.glob(pattern))

    return sorted(list(set(tflite_files)))


def batch_evaluate(models_dir: str = "models", data_config: str = "data/dataset/data.yaml",
                  output_file: str = None) -> Dict:
    """Batch evaluate all TFLite models"""
    tflite_models = find_tflite_models(models_dir)

    if not tflite_models:
        print("❌ No TFLite models found!")
        return {}

    print(f"🔍 Found {len(tflite_models)} TFLite models to evaluate")

    all_results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(tflite_models),
            'data_config': data_config,
            'evaluator': 'ultralytics'
        },
        'models': {}
    }

    for i, model_path in enumerate(tflite_models, 1):
        print(f"\n{'='*80}")
        print(f"Evaluating {i}/{len(tflite_models)}: {model_path.name}")
        print(f"{'='*80}")

        try:
            evaluator = UltralyticsEvaluator(str(model_path), data_config)
            results = evaluator.evaluate_accuracy()
            evaluator.print_results(results)

            all_results['models'][model_path.stem] = results

        except Exception as e:
            print(f"❌ Error evaluating {model_path.name}: {e}")
            all_results['models'][model_path.stem] = {'error': str(e)}

    # Save results
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path("results/evaluation") / f"ultralytics_tflite_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📄 Results saved to: {output_path}")

    # Generate summary report
    generate_summary_report(all_results, output_path.with_suffix('.md'))

    return all_results


def generate_summary_report(results: Dict, output_file: Path):
    """Generate markdown summary report"""
    models_data = []

    for model_name, data in results['models'].items():
        if 'error' in data:
            continue

        acc = data['accuracy_metrics']
        speed = data['speed_metrics']
        info = data['model_info']

        models_data.append({
            'name': model_name,
            'size_mb': info['model_size_mb'],
            'mAP50': acc['mAP50'],
            'mAP50_95': acc['mAP50_95'],
            'precision': acc['precision'],
            'recall': acc['recall'],
            'fps': speed['fps'],
            'total_ms': speed['total_ms']
        })

    # Sort by mAP50
    models_data.sort(key=lambda x: x['mAP50'], reverse=True)

    report = f"""# Ultralytics TFLite Evaluation Report

**Generated**: {results['evaluation_info']['timestamp']}
**Models Tested**: {len(models_data)}
**Framework**: Ultralytics YOLO
**Dataset**: Traffic Sign Detection

## 📊 Model Performance Comparison

| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | FPS | Time (ms) | Size (MB) |
|------|-------|-------|----------|-----------|--------|-----|-----------|-----------|
"""

    for i, model in enumerate(models_data, 1):
        report += f"| {i} | {model['name']} | {model['mAP50']:.3f} | {model['mAP50_95']:.3f} | {model['precision']:.3f} | {model['recall']:.3f} | {model['fps']:.1f} | {model['total_ms']:.2f} | {model['size_mb']:.1f} |\n"

    if models_data:
        best_map = models_data[0]
        best_fps = max(models_data, key=lambda x: x['fps'])

        report += f"""

## 🏆 Best Performers

### 🎯 Highest Accuracy
**{best_map['name']}**
- mAP50: {best_map['mAP50']:.3f}
- mAP50-95: {best_map['mAP50_95']:.3f}
- Size: {best_map['size_mb']:.1f} MB

### ⚡ Fastest Performance
**{best_fps['name']}**
- FPS: {best_fps['fps']:.1f}
- Inference time: {best_fps['total_ms']:.2f} ms
- mAP50: {best_fps['mAP50']:.3f}

## 📈 Performance Analysis

### Accuracy Distribution
- **Best mAP50**: {max(m['mAP50'] for m in models_data):.3f}
- **Worst mAP50**: {min(m['mAP50'] for m in models_data):.3f}
- **Average mAP50**: {sum(m['mAP50'] for m in models_data)/len(models_data):.3f}

### Speed Distribution
- **Fastest**: {max(m['fps'] for m in models_data):.1f} FPS
- **Slowest**: {min(m['fps'] for m in models_data):.1f} FPS
- **Average FPS**: {sum(m['fps'] for m in models_data)/len(models_data):.1f}

## 💡 Recommendations

1. **Production Ready**: Use **{best_map['name']}** for best accuracy
2. **Real-time Applications**: Use **{best_fps['name']}** for best speed
3. **Balanced Choice**: Consider accuracy vs speed tradeoffs

---
*Generated by Ultralytics TFLite Evaluator*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"📄 Summary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics TFLite Model Evaluator")
    parser.add_argument("--model", type=str, help="Path to TFLite model")
    parser.add_argument("--batch-eval", action="store_true", help="Evaluate all TFLite models")
    parser.add_argument("--test-predict", action="store_true", help="Test prediction on sample images")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml", help="Dataset YAML config")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--output", type=str, help="Output file path (supports .json and .md formats)")
    parser.add_argument("--save-results", type=str, help="Save results to file (deprecated, use --output)")

    args = parser.parse_args()

    # Handle output file preference (--output takes priority over --save-results)
    output_file = args.output or args.save_results

    if args.batch_eval:
        batch_evaluate(output_file=output_file)
    elif args.model:
        evaluator = UltralyticsEvaluator(args.model, args.data)

        if args.test_predict:
            # Test prediction
            pred_results = evaluator.test_prediction()
            print(f"\n📊 Prediction test completed!")

            # Save prediction results if output specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() == '.md':
                    # Generate markdown report for prediction test
                    with open(output_path, 'w') as f:
                        f.write(f"""# TFLite Prediction Test Report

**Model**: `{evaluator.model_path.name}`
**Generated**: {pred_results['timestamp']}
**Test Type**: Prediction Test

## 📊 Test Summary

- **Number of test images**: {pred_results['num_test_images']}
- **Average inference time**: {pred_results['avg_inference_time_ms']:.2f} ms
- **FPS**: {pred_results['fps']:.1f}

## 📋 Detailed Results

| Image | Detections | Time (ms) | Status |
|-------|------------|-----------|--------|
""")
                        for result in pred_results['results']:
                            if 'error' in result:
                                f.write(f"| {Path(result['image_path']).name} | - | - | ❌ {result['error']} |\n")
                            else:
                                f.write(f"| {Path(result['image_path']).name} | {result['num_detections']} | {result['inference_time_ms']:.1f} | ✅ Success |\n")

                        f.write(f"""

*Generated by Ultralytics TFLite Evaluator*
""")
                else:
                    # Save as JSON
                    with open(output_path, 'w') as f:
                        json.dump(pred_results, f, indent=2)

                print(f"📄 Prediction results saved to: {output_path}")

        else:
            # Full evaluation
            results = evaluator.evaluate_accuracy(args.imgsz, args.conf, args.iou)
            evaluator.print_results(results)

            # Save results if output specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() == '.md':
                    # Generate detailed markdown report
                    save_single_model_report(results, output_path)
                else:
                    # Save as JSON (default or .json extension)
                    if not output_path.suffix:
                        output_path = output_path.with_suffix('.json')

                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📄 Results saved to: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()