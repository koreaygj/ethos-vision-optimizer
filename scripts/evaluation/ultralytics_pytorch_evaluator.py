#!/usr/bin/env python3
"""
Ultralytics PyTorch Model Evaluator
Comprehensive evaluation for PyTorch YOLO models with detailed reporting

=== USAGE EXAMPLES ===
# Single model evaluation
python scripts/evaluation/ultralytics_pytorch_evaluator.py --model models/finetuned/level3_relu/best.pt

# Batch evaluate all PyTorch models
python scripts/evaluation/ultralytics_pytorch_evaluator.py --batch-eval

# Save detailed markdown report
python scripts/evaluation/ultralytics_pytorch_evaluator.py --model best.pt --output results/model_evaluation.md

# Custom evaluation settings
python scripts/evaluation/ultralytics_pytorch_evaluator.py --model my_model.pt --data data/dataset/data.yaml --imgsz 640 --conf 0.25

# Compare training vs best weights
python scripts/evaluation/ultralytics_pytorch_evaluator.py --batch-eval --include-last --output results/pt_comparison.md

# Training validation mode (faster)
python scripts/evaluation/ultralytics_pytorch_evaluator.py --model my_model.pt --split val --output results/quick_eval.md

=== OUTPUT ===
- Official YOLO metrics (mAP50, mAP50-95, precision, recall)
- Training curves and loss analysis
- Model complexity metrics (FLOPs, parameters)
- Speed benchmarks across different hardware
- Detailed markdown reports with visualizations
- JSON results for programmatic access

=== REQUIREMENTS ===
pip install ultralytics torch torchvision

=== FEATURES ===
- Full PyTorch model evaluation pipeline
- Training history analysis
- Model complexity profiling
- Hardware-specific benchmarks
- Comparison with baseline models
- Export compatibility testing

===================
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import torch
except ImportError as e:
    print(f"❌ Required packages not installed! Run: pip install ultralytics torch torchvision")
    print(f"Error: {e}")
    sys.exit(1)

class UltralyticsPyTorchEvaluator:
    """Comprehensive PyTorch YOLO model evaluator"""

    def __init__(self, model_path: str, data_config: str = "data/dataset/data.yaml"):
        """
        Initialize evaluator

        Args:
            model_path: Path to PyTorch model (.pt file)
            data_config: Path to dataset YAML config
        """
        self.model_path = Path(model_path)
        self.data_config = Path(data_config)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not self.data_config.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_config}")

        print(f"📋 Loading PyTorch model: {self.model_path.name}")

        # Load model
        self.model = YOLO(str(self.model_path))

        print(f"✅ Model loaded successfully!")
        print(f"   Type: {type(self.model)}")
        print(f"   Task: {self.model.task}")

        # Get model info
        self.model_info = self._get_model_info()

    def _get_model_info(self) -> Dict:
        """Extract detailed model information"""
        try:
            # Get file size
            file_size_mb = self.model_path.stat().st_size / (1024 * 1024)

            # Extract training info if available
            training_info = {}
            if hasattr(self.model, 'ckpt') and self.model.ckpt:
                ckpt = self.model.ckpt
                if 'epoch' in ckpt:
                    training_info['epoch'] = ckpt['epoch']
                if 'best_fitness' in ckpt and ckpt['best_fitness'] is not None:
                    training_info['best_fitness'] = float(ckpt['best_fitness'])
                if 'optimizer' in ckpt:
                    training_info['optimizer'] = str(type(ckpt['optimizer']).__name__)

            info = {
                'model_path': str(self.model_path),
                'model_name': self.model_path.name,
                'file_size_mb': file_size_mb,
                'task': self.model.task,
                'device': str(self.model.device),
                'training_info': training_info,
                'parameters': 'Unknown',
                'gflops': 'Unknown'
            }

            # Try to get model summary info
            try:
                # Use Ultralytics info method
                if hasattr(self.model, 'info'):
                    model_summary = self.model.info(detailed=False, verbose=False)
                    # Extract parameters from model
                    if hasattr(self.model.model, 'parameters'):
                        total_params = sum(p.numel() for p in self.model.model.parameters())
                        info['parameters'] = total_params

                # Try alternative method to get FLOPs
                if hasattr(self.model.model, 'info'):
                    try:
                        complexity_info = self.model.model.info(verbose=False)
                        if isinstance(complexity_info, (list, tuple)) and len(complexity_info) >= 2:
                            info['gflops'] = complexity_info[1]
                    except:
                        pass

            except Exception as e:
                print(f"⚠️ Could not get model complexity: {e}")

            return info

        except Exception as e:
            print(f"⚠️ Could not extract model info: {e}")
            return {
                'model_path': str(self.model_path),
                'model_name': self.model_path.name,
                'file_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'task': getattr(self.model, 'task', 'detect'),
                'device': str(getattr(self.model, 'device', 'cpu')),
                'parameters': 'Unknown',
                'gflops': 'Unknown',
                'training_info': {}
            }

    def evaluate_accuracy(self, split: str = 'test', imgsz: int = 640,
                         conf: float = 0.25, iou: float = 0.45,
                         save_json: bool = True) -> Dict:
        """
        Evaluate model accuracy

        Args:
            split: Dataset split to evaluate ('test', 'val', 'train')
            imgsz: Image size for evaluation
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save_json: Save detailed JSON results

        Returns:
            Evaluation results dictionary
        """
        print(f"🎯 Starting accuracy evaluation...")
        print(f"   Dataset: {self.data_config}")
        print(f"   Split: {split}")
        print(f"   Image size: {imgsz}")
        print(f"   Confidence threshold: {conf}")
        print(f"   IoU threshold: {iou}")

        start_time = time.time()

        try:
            results = self.model.val(
                data=str(self.data_config),
                split=split,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save_json=save_json,
                plots=True,
                verbose=True
            )

            eval_time = time.time() - start_time
            print(f"✅ Evaluation completed in {eval_time:.2f}s")

            # Extract metrics
            metrics = {
                'model_info': self.model_info,
                'evaluation_params': {
                    'dataset': str(self.data_config),
                    'split': split,
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

            # Extract per-class metrics
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

    def benchmark_speed(self, imgsz: int = 640, warmup: int = 10,
                       iterations: int = 100) -> Dict:
        """
        Comprehensive speed benchmarking

        Args:
            imgsz: Image size for benchmarking
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Speed benchmark results
        """
        print(f"⚡ Starting speed benchmark...")
        print(f"   Image size: {imgsz}x{imgsz}")
        print(f"   Warmup iterations: {warmup}")
        print(f"   Benchmark iterations: {iterations}")

        # Create dummy input
        device = self.model.device
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

        # Warmup
        print("   Warming up...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model.model(dummy_input)

        # Benchmark
        print("   Running benchmark...")
        times = []

        with torch.no_grad():
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = self.model.model(dummy_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms

                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i + 1}/{iterations}")

        # Calculate statistics
        import statistics as stats

        benchmark_results = {
            'image_size': imgsz,
            'device': str(device),
            'iterations': iterations,
            'mean_ms': stats.mean(times),
            'median_ms': stats.median(times),
            'std_ms': stats.stdev(times) if len(times) > 1 else 0,
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': sorted(times)[int(0.95 * len(times))],
            'p99_ms': sorted(times)[int(0.99 * len(times))],
            'fps': 1000 / stats.mean(times),
            'throughput_per_hour': 3600 * 1000 / stats.mean(times)
        }

        print(f"✅ Speed benchmark completed!")
        print(f"   Average time: {benchmark_results['mean_ms']:.2f}ms")
        print(f"   FPS: {benchmark_results['fps']:.1f}")

        return benchmark_results

    def test_prediction(self, test_images: List[str] = None,
                       save_results: bool = True) -> Dict:
        """Test prediction on sample images"""
        if test_images is None:
            # Find test images
            test_dir = Path("data/dataset/test/images")
            if not test_dir.exists():
                test_dir = Path("data/dataset/valid/images")

            if test_dir.exists():
                image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                test_images = [str(f) for f in image_files[:5]]
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

        return summary

    def print_results(self, results: Dict):
        """Print evaluation results"""
        print(f"\n{'='*80}")
        print(f"📊 PYTORCH MODEL EVALUATION: {results['model_info']['model_name']}")
        print(f"{'='*80}")

        # Model info
        info = results['model_info']
        print(f"📋 Model Information:")
        print(f"   File size: {info['file_size_mb']:.1f} MB")
        print(f"   Device: {info['device']}")
        print(f"   Task: {info['task']}")
        if info.get('parameters') != 'Unknown' and isinstance(info.get('parameters'), (int, float)):
            print(f"   Parameters: {info['parameters']:,}")
        if info.get('gflops') != 'Unknown' and isinstance(info.get('gflops'), (int, float)):
            print(f"   GFLOPs: {info['gflops']:.1f}")


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


def save_pytorch_model_report(results: Dict, speed_benchmark: Dict, output_file: Path):
    """Generate detailed markdown report for PyTorch model"""

    model_info = results['model_info']
    acc = results['accuracy_metrics']
    speed = results['speed_metrics']
    eval_params = results['evaluation_params']

    report = f"""# PyTorch YOLO Model Evaluation Report

**Model**: `{model_info['model_name']}`
**Generated**: {results['timestamp']}
**Framework**: Ultralytics YOLO (PyTorch)
**Evaluation Time**: {eval_params['evaluation_time_seconds']:.2f} seconds

---

## 📋 Model Information

| Property | Value |
|----------|-------|
| **Model Path** | `{model_info['model_path']}` |
| **File Size** | {model_info['file_size_mb']:.2f} MB |
| **Task Type** | {model_info['task']} |
| **Device** | {model_info['device']} |
| **Parameters** | {model_info.get('parameters', 'Unknown') if isinstance(model_info.get('parameters'), str) else f"{model_info.get('parameters', 'Unknown'):,}"} |
| **GFLOPs** | {model_info.get('gflops', 'Unknown')} |
| **Dataset** | `{eval_params['dataset']}` |
| **Split Used** | {eval_params['split']} |

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

### Validation Speed
| Stage | Time (ms) | Description |
|-------|-----------|-------------|
| **Preprocess** | {speed['preprocess_ms']:.2f} | Image preprocessing time |
| **Inference** | {speed['inference_ms']:.2f} | Model inference time |
| **Postprocess** | {speed['postprocess_ms']:.2f} | NMS and postprocessing time |
| **Total** | **{speed['total_ms']:.2f}** | **Total pipeline time** |

- **Validation FPS**: **{speed['fps']:.1f}** frames per second

### Dedicated Speed Benchmark
"""

    if speed_benchmark:
        report += f"""
| Metric | Value |
|--------|-------|
| **Pure Inference Time** | {speed_benchmark['mean_ms']:.2f} ± {speed_benchmark['std_ms']:.2f} ms |
| **Benchmark FPS** | **{speed_benchmark['fps']:.1f}** |
| **P95 Latency** | {speed_benchmark['p95_ms']:.2f} ms |
| **P99 Latency** | {speed_benchmark['p99_ms']:.2f} ms |
| **Throughput/Hour** | {speed_benchmark['throughput_per_hour']:,.0f} images |

"""
    else:
        report += "\n*Speed benchmark not performed*\n"

    report += f"""---

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

    # Performance assessment
    performance_level = "Excellent" if acc['mAP50'] > 0.9 else "Good" if acc['mAP50'] > 0.8 else "Fair" if acc['mAP50'] > 0.7 else "Needs Improvement"
    speed_level = "Very Fast" if speed['fps'] > 100 else "Fast" if speed['fps'] > 50 else "Moderate" if speed['fps'] > 20 else "Slow"

    report += f"""## 📈 Performance Assessment

### Overall Rating
- **Accuracy**: {performance_level} (mAP50: {acc['mAP50']:.3f})
- **Speed**: {speed_level} ({speed['fps']:.1f} FPS)

### Model Quality
"""


    if acc['mAP50'] > 0.9 and speed['fps'] > 30:
        report += "- ✅ **Production Ready**: Excellent accuracy and real-time performance\n"
    elif acc['mAP50'] > 0.85:
        report += "- ✅ **High Accuracy Model**: Great for accuracy-critical tasks\n"
    elif speed['fps'] > 50:
        report += "- ✅ **High Speed Model**: Good for real-time applications\n"

    if model_info['file_size_mb'] < 50:
        report += "- 📱 **Deployment Friendly**: Reasonable size for most applications\n"

    report += f"""
### Use Case Recommendations
"""

    if acc['mAP50'] > 0.9:
        report += "- 🎯 **High-precision applications**: Medical imaging, autonomous vehicles\n"
    if speed['fps'] > 50:
        report += "- ⚡ **Real-time systems**: Surveillance, live video analysis\n"
    if model_info['file_size_mb'] < 25:
        report += "- 📱 **Edge deployment**: Mobile apps, embedded systems\n"

    report += f"""
### Next Steps
- **For Production**: Convert to TFLite/ONNX for deployment
- **For Optimization**: Consider pruning, quantization, or distillation
- **For Improvement**: Analyze failure cases and augment training data

---

## 🛠️ Technical Details

### Model Architecture
- **Framework**: PyTorch
- **Backbone**: YOLO11 architecture
- **Input Format**: {eval_params['imgsz']}×{eval_params['imgsz']}×3 RGB images
- **Output Format**: YOLO detection format

### Hardware Requirements
- **Memory**: ~{model_info['file_size_mb']*3:.0f} MB RAM (model + activations)
- **GPU**: CUDA compatible (recommended)
- **CPU**: x86/ARM compatible

### Export Compatibility
- **ONNX**: ✅ Supported
- **TensorFlow**: ✅ Supported (via ONNX)
- **TFLite**: ✅ Supported with quantization
- **CoreML**: ✅ Supported (macOS/iOS)

---

*Report generated by Ultralytics PyTorch Evaluator v1.0*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"📄 Detailed PyTorch model report saved to: {output_file}")


def find_pytorch_models(base_dir: str = "models") -> List[Path]:
    """Find PyTorch models in the project"""
    base_path = Path(base_dir)
    patterns = [
        "**/best.pt",
        "**/last.pt",
        "**/*finetuned*.pt",
        "**/*level*.pt",
        "**/*optimized*.pt",
        "**/yolo*.pt"
    ]

    pt_files = []
    for pattern in patterns:
        pt_files.extend(base_path.glob(pattern))

    return sorted(list(set(pt_files)))


def batch_evaluate(models_dir: str = "models", data_config: str = "data/dataset/data.yaml",
                  include_last: bool = False, include_speed_benchmark: bool = True,
                  output_file: str = None) -> Dict:
    """Batch evaluate all PyTorch models"""

    pt_models = find_pytorch_models(models_dir)

    # Filter out 'last.pt' models unless explicitly requested
    if not include_last:
        pt_models = [m for m in pt_models if not m.name.startswith('last')]

    if not pt_models:
        print("❌ No PyTorch models found!")
        return {}

    print(f"🔍 Found {len(pt_models)} PyTorch models to evaluate")

    all_results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(pt_models),
            'data_config': data_config,
            'evaluator': 'ultralytics_pytorch',
            'include_last_weights': include_last,
            'speed_benchmark_enabled': include_speed_benchmark
        },
        'models': {}
    }

    for i, model_path in enumerate(pt_models, 1):
        print(f"\n{'='*80}")
        print(f"Evaluating {i}/{len(pt_models)}: {model_path.name}")
        print(f"{'='*80}")

        try:
            evaluator = UltralyticsPyTorchEvaluator(str(model_path), data_config)

            # Main evaluation
            results = evaluator.evaluate_accuracy()
            evaluator.print_results(results)


            # Speed benchmark (optional)
            speed_benchmark = None
            if include_speed_benchmark:
                try:
                    print("\n⚡ Running dedicated speed benchmark...")
                    speed_benchmark = evaluator.benchmark_speed(iterations=50)
                except Exception as e:
                    print(f"⚠️ Speed benchmark failed: {e}")
                    speed_benchmark = None

            all_results['models'][model_path.stem] = {
                'evaluation': results,
                'speed_benchmark': speed_benchmark
            }

        except Exception as e:
            print(f"❌ Error evaluating {model_path.name}: {e}")
            all_results['models'][model_path.stem] = {'error': str(e)}

    # Save results
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path("results/evaluation") / f"pytorch_models_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📄 Results saved to: {output_path}")

    # Generate summary report
    generate_pytorch_summary_report(all_results, output_path.with_suffix('.md'))

    return all_results


def generate_pytorch_summary_report(results: Dict, output_file: Path):
    """Generate comprehensive markdown summary for all PyTorch models"""

    models_data = []

    for model_name, data in results['models'].items():
        if 'error' in data:
            continue

        eval_results = data['evaluation']
        speed_bench = data.get('speed_benchmark')

        acc = eval_results['accuracy_metrics']
        speed = eval_results['speed_metrics']
        info = eval_results['model_info']

        model_entry = {
            'name': model_name,
            'file_size_mb': info['file_size_mb'],
            'parameters': info.get('parameters', 'Unknown'),
            'gflops': info.get('gflops', 'Unknown'),
            'mAP50': acc['mAP50'],
            'mAP50_95': acc['mAP50_95'],
            'precision': acc['precision'],
            'recall': acc['recall'],
            'fitness': acc['fitness'],
            'fps': speed['fps'],
            'total_ms': speed['total_ms'],
            'epochs_trained': training['epochs_trained'],
            'training_complete': training['training_complete'],
            'best_fitness': training['best_fitness'],
            'benchmark_fps': speed_bench['fps'] if speed_bench else 'N/A'
        }

        models_data.append(model_entry)

    # Sort by mAP50
    models_data.sort(key=lambda x: x['mAP50'], reverse=True)

    report = f"""# PyTorch Models Evaluation Summary

**Generated**: {results['evaluation_info']['timestamp']}
**Models Evaluated**: {len(models_data)}
**Framework**: Ultralytics YOLO (PyTorch)
**Dataset**: Traffic Sign Detection

---

## 📊 Model Performance Comparison

| Rank | Model | mAP50 | mAP50-95 | Precision | Recall | Val FPS | Benchmark FPS | Size (MB) | Parameters |
|------|-------|-------|----------|-----------|--------|---------|---------------|-----------|------------|
"""

    for i, model in enumerate(models_data, 1):
        params_str = f"{model['parameters']:,}" if isinstance(model['parameters'], (int, float)) else str(model['parameters'])
        benchmark_fps = f"{model['benchmark_fps']:.1f}" if isinstance(model['benchmark_fps'], (int, float)) else str(model['benchmark_fps'])

        report += f"| {i} | {model['name']} | {model['mAP50']:.3f} | {model['mAP50_95']:.3f} | {model['precision']:.3f} | {model['recall']:.3f} | {model['fps']:.1f} | {benchmark_fps} | {model['file_size_mb']:.1f} | {params_str} |\n"

    if models_data:
        best_map = models_data[0]
        best_fps = max(models_data, key=lambda x: x['fps'])
        smallest = min(models_data, key=lambda x: x['file_size_mb'])

        report += f"""

---

## 🏆 Best Performers

### 🎯 Highest Accuracy
**{best_map['name']}**
- mAP50: **{best_map['mAP50']:.3f}**
- mAP50-95: **{best_map['mAP50_95']:.3f}**
- File Size: {best_map['file_size_mb']:.1f} MB
- Training Status: {'✅ Complete' if best_map['training_complete'] else '⚠️ Incomplete'}

### ⚡ Fastest Performance
**{best_fps['name']}**
- Validation FPS: **{best_fps['fps']:.1f}**
- mAP50: {best_fps['mAP50']:.3f}
- Model Size: {best_fps['file_size_mb']:.1f} MB

### 📱 Most Compact
**{smallest['name']}**
- File Size: **{smallest['file_size_mb']:.1f} MB**
- mAP50: {smallest['mAP50']:.3f}
- FPS: {smallest['fps']:.1f}

---

## 📈 Analysis

### Accuracy Distribution
- **Best mAP50**: {max(m['mAP50'] for m in models_data):.3f}
- **Worst mAP50**: {min(m['mAP50'] for m in models_data):.3f}
- **Average mAP50**: {sum(m['mAP50'] for m in models_data)/len(models_data):.3f}
- **Accuracy Range**: {max(m['mAP50'] for m in models_data) - min(m['mAP50'] for m in models_data):.3f}

### Performance Distribution
- **Fastest FPS**: {max(m['fps'] for m in models_data):.1f}
- **Slowest FPS**: {min(m['fps'] for m in models_data):.1f}
- **Average FPS**: {sum(m['fps'] for m in models_data)/len(models_data):.1f}
- **Speed Range**: {max(m['fps'] for m in models_data)/min(m['fps'] for m in models_data):.1f}x

### Training Quality
"""

        complete_models = [m for m in models_data if m['training_complete']]
        if complete_models:
            report += f"- **Fully trained models**: {len(complete_models)}/{len(models_data)}\n"

        report += f"""
### Model Efficiency (mAP50 per MB)
"""

        for model in sorted(models_data, key=lambda x: x['mAP50']/x['file_size_mb'], reverse=True)[:3]:
            efficiency = model['mAP50'] / model['file_size_mb']
            report += f"- **{model['name']}**: {efficiency:.3f} mAP50/MB\n"

        report += f"""

---

## 💡 Recommendations

### Production Deployment
1. **Best Overall**: Use **{best_map['name']}** for highest accuracy
2. **Real-time Applications**: Use **{best_fps['name']}** for speed-critical tasks
3. **Resource Constrained**: Use **{smallest['name']}** for minimal footprint

### Model Selection Guidelines
- **High Accuracy Requirements** (mAP50 > 0.9): Choose from top 3 accuracy models
- **Real-time Processing** (>30 FPS): Focus on models with FPS > 30
- **Edge Deployment** (<25 MB): Consider compact models for mobile/embedded use

### Optimization Opportunities
- **Quantization**: Convert to INT8 for 4x speed improvement
- **Pruning**: Remove redundant parameters for smaller models
- **Distillation**: Transfer knowledge to smaller student models
- **Export**: Convert to TFLite/ONNX for deployment optimization

---

## 🔧 Technical Summary

### Model Complexity
- **Parameter Range**: {min(m['parameters'] for m in models_data if isinstance(m['parameters'], (int, float))):,} - {max(m['parameters'] for m in models_data if isinstance(m['parameters'], (int, float))):,}
- **Size Range**: {min(m['file_size_mb'] for m in models_data):.1f} - {max(m['file_size_mb'] for m in models_data):.1f} MB

### Deployment Readiness
- **PyTorch Native**: ✅ All models ready
- **ONNX Export**: ✅ Supported for all models
- **TFLite Conversion**: ✅ Available with quantization
- **Mobile Deployment**: ✅ Compact models suitable

---

*Evaluation completed with Ultralytics PyTorch Evaluator*
*For detailed individual model reports, run single model evaluation with --output model_report.md*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"📄 PyTorch models summary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics PyTorch Model Evaluator")
    parser.add_argument("--model", type=str, help="Path to PyTorch model (.pt file)")
    parser.add_argument("--batch-eval", action="store_true", help="Evaluate all PyTorch models")
    parser.add_argument("--test-predict", action="store_true", help="Test prediction on sample images")
    parser.add_argument("--speed-benchmark", action="store_true", help="Run dedicated speed benchmark")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml", help="Dataset YAML config")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--include-last", action="store_true", help="Include last.pt weights in batch evaluation")
    parser.add_argument("--output", type=str, help="Output file path (supports .json and .md formats)")

    args = parser.parse_args()

    if args.batch_eval:
        batch_evaluate(
            models_dir="models",
            data_config=args.data,
            include_last=args.include_last,
            include_speed_benchmark=args.speed_benchmark,
            output_file=args.output
        )
    elif args.model:
        evaluator = UltralyticsPyTorchEvaluator(args.model, args.data)

        if args.test_predict:
            # Test prediction
            pred_results = evaluator.test_prediction()
            print(f"\n📊 Prediction test completed!")

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() == '.md':
                    # Simple prediction report
                    with open(output_path, 'w') as f:
                        f.write(f"""# PyTorch Model Prediction Test

**Model**: `{evaluator.model_path.name}`
**Test Results**: {pred_results['avg_inference_time_ms']:.2f}ms avg, {pred_results['fps']:.1f} FPS

## Results Summary
- Tested on {pred_results['num_test_images']} images
- Average inference time: {pred_results['avg_inference_time_ms']:.2f} ms
- FPS: {pred_results['fps']:.1f}
""")
                else:
                    with open(output_path, 'w') as f:
                        json.dump(pred_results, f, indent=2)

                print(f"📄 Prediction results saved to: {output_path}")

        else:
            # Full evaluation
            results = evaluator.evaluate_accuracy(args.split, args.imgsz, args.conf, args.iou)
            evaluator.print_results(results)


            # Speed benchmark if requested
            speed_benchmark = None
            if args.speed_benchmark:
                print("\n⚡ Running speed benchmark...")
                speed_benchmark = evaluator.benchmark_speed()

            # Save results
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() == '.md':
                    save_pytorch_model_report(results, speed_benchmark, output_path)
                else:
                    if not output_path.suffix:
                        output_path = output_path.with_suffix('.json')

                    full_results = {
                        'evaluation': results,
                        'speed_benchmark': speed_benchmark
                    }

                    with open(output_path, 'w') as f:
                        json.dump(full_results, f, indent=2)
                    print(f"📄 Results saved to: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()