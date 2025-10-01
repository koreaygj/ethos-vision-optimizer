# Scripts Usage Guide

**프로젝트**: Ethos Vision Optimizer
**업데이트**: 2025-10-01
**총 스크립트 수**: 24개 (6개 카테고리)

이 가이드는 Ethos Vision Optimizer 프로젝트의 모든 스크립트 사용법을 상세히 설명합니다.

---

## 📁 **스크립트 구조 개요**

```
scripts/
├── training/           # NPU 최적화 훈련
├── evaluation/         # 모델 평가 및 성능 측정
├── convert/           # 모델 포맷 변환
├── analysis/          # 모델 분석 도구
├── validation/        # 검증 및 테스트
└── README_ORGANIZATION.md
```

---

## 🚀 **1. Training Scripts - NPU 최적화 훈련**

### 1.1 `npu_optimized_trainer.py` - 메인 훈련 스크립트

**위치**: [`scripts/training/npu_optimized_trainer.py`](../../scripts/training/npu_optimized_trainer.py)
**가장 중요한 스크립트**로, NPU 최적화를 위한 4단계 Level 훈련을 지원합니다.

#### **기본 사용법**
```bash
# Level 2 ReLU 훈련
python scripts/training/npu_optimized_trainer.py --level level2-relu

# Level 3 LeakyReLU 훈련
python scripts/training/npu_optimized_trainer.py --level level3-leaky

# Level 4 완전 최적화
python scripts/training/npu_optimized_trainer.py --level level4-relu
```

#### **상세 옵션**
```bash
# 전체 옵션 사용 예시
python scripts/training/npu_optimized_trainer.py \
  --level level3-relu \
  --epochs 100 \
  --batch-size 16 \
  --data data/dataset/data.yaml \
  --device 0 \
  --workers 8 \
  --project results/training \
  --name custom_experiment
```

#### **검사 전용 모드 (훈련 없이 모델 구조만 확인)**
```bash
# 모델 구조와 활성화 함수만 검사
python scripts/training/npu_optimized_trainer.py --level level2-relu --inspect

# 모든 레벨 모델 구조 검사
python scripts/training/npu_optimized_trainer.py --inspect-all
```

#### **지원되는 Level들**
| Level | 설명 | ReLU | LeakyReLU |
|-------|------|------|-----------|
| `level2-relu` | Backbone + Head C3k2 최적화 + ReLU | ✅ | ❌ |
| `level2-leaky` | Backbone + Head C3k2 최적화 + LeakyReLU | ❌ | ✅ |
| `level3-relu` | + C2PSA 최적화 + ReLU | ✅ | ❌ |
| `level3-leaky` | + C2PSA 최적화 + LeakyReLU | ❌ | ✅ |
| `level4-relu` | 완전 최적화 + ReLU | ✅ | ❌ |
| `level4-leaky` | 완전 최적화 + LeakyReLU | ❌ | ✅ |

#### **출력 결과**
```
results/training/
├── level2_relu_YYYYmmdd_HHMMSS/
│   ├── weights/
│   │   ├── best.pt              # 최고 성능 모델
│   │   └── last.pt              # 마지막 에포크 모델
│   ├── results.png              # 훈련 곡선
│   ├── confusion_matrix.png     # 혼동행렬
│   └── val_batch0_*.jpg         # 검증 샘플
```

---

## 📊 **2. Evaluation Scripts - 모델 평가**

### 2.1 `accuracy_analysis.py` - 정확도 상세 분석

**위치**: [`scripts/evaluation/accuracy_analysis.py`](../../scripts/evaluation/accuracy_analysis.py)
**용도**: 모델의 정확도를 Precision, Recall, F1-Score, IoU 등 다양한 메트릭으로 분석

```bash
# 기본 정확도 분석
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level2_relu/best.pt \
  --data data/dataset/data.yaml

# 임계값 변경하여 분석
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level3_leaky/best.pt \
  --data data/dataset/data.yaml \
  --conf-thres 0.5 \
  --iou-thres 0.6

# 상세 리포트 생성
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level4_relu/best.pt \
  --data data/dataset/data.yaml \
  --save-report results/evaluation/accuracy_report.md
```

**출력 예시**:
```
📊 Accuracy Analysis Results:
==================================================
   mAP@0.5: 0.847
   mAP@0.5:0.95: 0.623
   Precision: 0.856
   Recall: 0.789
   F1-Score: 0.821

📈 Per-Class Performance:
   Class 'car': mAP=0.92, Precision=0.89, Recall=0.85
   Class 'person': mAP=0.78, Precision=0.82, Recall=0.74
```

### 2.2 `yolo_model_evaluator.py` - 종합 모델 평가

**위치**: [`scripts/evaluation/yolo_model_evaluator.py`](../../scripts/evaluation/yolo_model_evaluator.py)
**용도**: 다중 모델 성능 비교, 추론 속도 측정

```bash
# 단일 모델 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level2_relu/best.pt \
  --data data/dataset/data.yaml

# 여러 모델 비교 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --models models/optimized_npu/level2_relu/best.pt models/optimized_npu/level3_relu/best.pt \
  --data data/dataset/data.yaml \
  --compare

# 속도 측정 포함
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --data data/dataset/data.yaml \
  --benchmark \
  --device cpu
```

### 2.3 `compare_all_models.py` - 전체 모델 비교

**위치**: [`scripts/evaluation/compare_all_models.py`](../../scripts/evaluation/compare_all_models.py)
**용도**: Original vs Optimized 모델 성능 한번에 비교

```bash
# 모든 최적화 레벨 비교
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/ \
  --data data/dataset/data.yaml

# 특정 메트릭만 비교
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/ \
  --data data/dataset/data.yaml \
  --metrics mAP precision recall
```

---

## 🔄 **3. Convert Scripts - 모델 포맷 변환**

### 3.1 `pt2onnx.py` - PyTorch → ONNX 변환

**위치**: [`scripts/convert/pt2onnx.py`](../../scripts/convert/pt2onnx.py)
**용도**: NPU 최적화를 위한 중간 포맷 변환

```bash
# 기본 변환
python scripts/convert/pt2onnx.py \
  --input models/optimized_npu/level3_relu/best.pt \
  --output models/pure/level3_relu.onnx

# 동적 입력 크기 지원
python scripts/convert/pt2onnx.py \
  --input models/optimized_npu/level4_leaky/best.pt \
  --output models/pure/level4_leaky.onnx \
  --dynamic-axes \
  --input-size 640 640

# 최적화 옵션 적용
python scripts/convert/pt2onnx.py \
  --input models/optimized_npu/level2_relu/best.pt \
  --output models/pure/level2_relu_optimized.onnx \
  --optimize \
  --opset-version 12
```

### 3.2 `pt2tflite_int8.py` - PyTorch → TFLite INT8 변환

**위치**: [`scripts/convert/pt2tflite_int8.py`](../../scripts/convert/pt2tflite_int8.py)
**용도**: 모바일/임베디드 배포용 경량화

```bash
# 기본 INT8 양자화 변환
python scripts/convert/pt2tflite_int8.py \
  --input models/optimized_npu/level3_relu/best.pt \
  --output models/pure/level3_relu_int8.tflite

# 캘리브레이션 데이터 사용
python scripts/convert/pt2tflite_int8.py \
  --input models/optimized_npu/level4_leaky/best.pt \
  --output models/pure/level4_leaky_int8.tflite \
  --calibration-data data/dataset/train/images \
  --num-calibration 100

# 검증 포함 변환
python scripts/convert/pt2tflite_int8.py \
  --input models/optimized_npu/level2_relu/best.pt \
  --output models/pure/level2_relu_int8.tflite \
  --validate \
  --test-data data/dataset/valid/images
```

### 3.3 `pt2tflite_fp16.py` - PyTorch → TFLite FP16 변환

**위치**: [`scripts/convert/pt2tflite_fp16.py`](../../scripts/convert/pt2tflite_fp16.py)
**용도**: GPU 가속 최적화

```bash
# FP16 변환
python scripts/convert/pt2tflite_fp16.py \
  --input models/optimized_npu/level3_leaky/best.pt \
  --output models/pure/level3_leaky_fp16.tflite

# 혼합 정밀도 지원
python scripts/convert/pt2tflite_fp16.py \
  --input models/optimized_npu/level4_relu/best.pt \
  --output models/pure/level4_relu_mixed.tflite \
  --mixed-precision
```

---

## 🔍 **4. Analysis Scripts - 모델 분석**

### 4.1 `primitive_operator_analyzer_v2.py` - NPU 호환성 분석

**위치**: [`scripts/analysis/primitive_operator_analyzer_v2.py`](../../scripts/analysis/primitive_operator_analyzer_v2.py)
**용도**: Primitive operator 분석 및 NPU 호환성 체크

```bash
# 기본 호환성 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level2_relu/best.pt

# 상세 분석 리포트
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level3_leaky/best.pt \
  --detailed \
  --output docs/analysis/level3_leaky_analysis.md

# 여러 모델 비교 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level2_relu/best.pt \
  models/optimized_npu/level3_relu/best.pt \
  models/optimized_npu/level4_relu/best.pt \
  --compare \
  --output docs/analysis/optimization_comparison.md
```

**출력 예시**:
```
🔍 NPU Compatibility Analysis
==================================================
✅ Supported Operators: 18/23 (78.3%)
❌ Unsupported Operators: 5/23 (21.7%)

📊 Detailed Analysis:
   ✅ Conv2d: 45 instances (100% compatible)
   ✅ ReLU: 23 instances (100% compatible)
   ❌ SiLU: 12 instances (0% compatible)
   ✅ MaxPool2d: 8 instances (100% compatible)

💡 Optimization Recommendations:
   1. Replace SiLU with ReLU/LeakyReLU for NPU compatibility
   2. Consider BatchNorm fusion for better performance
   3. Use depthwise convolutions where possible
```

### 4.2 `model_structure_analyzer.py` - 모델 구조 분석

**위치**: [`scripts/analysis/model_structure_analyzer.py`](../../scripts/analysis/model_structure_analyzer.py)
**용도**: 모델 아키텍처 및 파라미터 상세 분석

```bash
# 모델 구조 분석
python scripts/analysis/model_structure_analyzer.py \
  --model models/optimized_npu/level3_relu/best.pt \
  --output docs/analysis/level3_structure.md

# 파라미터 비교
python scripts/analysis/model_structure_analyzer.py \
  --models models/pure/yolov11n.pt models/optimized_npu/level4_leaky/best.pt \
  --compare-params \
  --output docs/analysis/parameter_comparison.md
```

---

## ✅ **5. Validation Scripts - 검증 및 테스트**

### 5.1 `activation_verifier.py` - 활성화 함수 검증

**위치**: [`scripts/validation/activation_verifier.py`](../../scripts/validation/activation_verifier.py)
**용도**: Glenn Jocher's GitHub #7296 이슈 해결을 위한 검증 도구

```bash
# 기본 검증
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level2_scales_backbone_relu.yaml

# 상세 분석
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level3_scales_backbone_head_leaked.yaml \
  --detailed

# 특정 activation과 비교
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_relu.yaml \
  --reference ReLU

# 리포트 생성
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_leaked.yaml \
  --output results/activation_report.md
```

**출력 예시**:
```
🔍 Verifying activation functions in model...
   ✅ Found ReLU at: model.0.act
   ✅ Found ReLU at: model.1.act
   ✅ Found ReLU at: model.2.cv1.act

📊 Activation Function Analysis Results:
==================================================
   ReLU: 112 instances
   ✅ ReLU verification: True

🌐 Global activation detected: ReLU
```

---

## 🔗 **6. 통합 워크플로우 예시**

### 6.1 **완전한 NPU 최적화 파이프라인**

```bash
# 1단계: 원본 모델 분석
python scripts/analysis/primitive_operator_analyzer_v2.py models/pure/yolov11n.pt

# 2단계: NPU 최적화 훈련 (Level 2 → Level 3 → Level 4)
python scripts/training/npu_optimized_trainer.py --level level2-relu --epochs 100
python scripts/training/npu_optimized_trainer.py --level level3-relu --epochs 100
python scripts/training/npu_optimized_trainer.py --level level4-relu --epochs 100

# 3단계: 모든 모델 성능 비교
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/

# 4단계: 최적화된 모델 분석
python scripts/analysis/primitive_operator_analyzer_v2.py models/optimized_npu/level4_relu/best.pt

# 5단계: 포맷 변환
python scripts/convert/pt2onnx.py --input models/optimized_npu/level4_relu/best.pt
python scripts/convert/pt2tflite_int8.py --input models/optimized_npu/level4_relu/best.pt

# 6단계: 활성화 함수 검증
python scripts/validation/activation_verifier.py --model models/train/npu_level4_activation_relu.yaml
```

### 6.2 **빠른 평가 워크플로우**

```bash
# 훈련된 모델들 빠른 성능 체크
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/ \
  --quick

# 상세 정확도 분석
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --save-report results/evaluation/final_accuracy.md

# NPU 호환성 최종 확인
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level4_leaky/best.pt \
  --output docs/analysis/final_compatibility.md
```

### 6.3 **문제 해결 워크플로우**

```bash
# 모델 구조만 검사 (문제 진단)
python scripts/training/npu_optimized_trainer.py --level level3-relu --inspect

# 활성화 함수 문제 확인
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level3_scales_backbone_head_relu.yaml \
  --detailed

# 변환 문제 확인
python scripts/convert/pt2onnx.py \
  --input models/optimized_npu/level2_relu/best.pt \
  --validate
```

---

## 📋 **스크립트 실행 전 체크리스트**

### ✅ **환경 준비**
- [ ] Python 3.9+ 설치됨
- [ ] requirements.txt 패키지 설치됨: `pip install -r requirements.txt`
- [ ] CUDA/MPS 설정 확인 (GPU 사용 시)
- [ ] 데이터셋 경로 확인: `data/dataset/data.yaml`

### ✅ **훈련 전 준비**
- [ ] 사전 훈련 모델 존재: `yolov11n.pt`
- [ ] 충분한 디스크 공간 (모델당 약 100MB)
- [ ] YAML 설정 파일 활성화 함수 확인

### ✅ **변환 전 준비**
- [ ] 입력 모델 파일 존재
- [ ] 출력 디렉토리 권한 확인
- [ ] 캘리브레이션 데이터 준비 (INT8 변환 시)

---

## 🚨 **자주 발생하는 문제 해결**

### **1. "Invalid padding string 'ReLU'" 오류**
```bash
# 원인: 개별 레이어에 activation 문자열 설정
# 해결: Global activation 설정 사용
activation: nn.ReLU()  # YAML 파일에서
```

### **2. "Model file not found" 오류**
```bash
# 파일명 확인
ls models/train/*level3*leak*

# 올바른 파일명 사용
python scripts/training/npu_optimized_trainer.py --level level3-leaky --inspect
```

### **3. Memory 부족 오류**
```bash
# 배치 크기 줄이기
python scripts/training/npu_optimized_trainer.py --level level4-relu --batch-size 8

# Workers 수 줄이기
python scripts/training/npu_optimized_trainer.py --level level4-relu --workers 4
```

### **4. CUDA Out of Memory**
```bash
# CPU 사용
python scripts/training/npu_optimized_trainer.py --level level3-relu --device cpu

# 또는 더 작은 모델 사용
python scripts/training/npu_optimized_trainer.py --level level2-relu
```

---

## 📚 **추가 참고 자료**

- **[Activation Function Fix Guide](activation-function-fix-documentation.md)**: GitHub #7296 이슈 해결 과정
- **[NPU Optimization Matrix](NPU_OPTIMIZATION_MATRIX.md)**: Level별 최적화 상세 설명
- **[Training Details](training-details.md)**: 훈련 과정 상세 가이드
- **[Scripts Organization](scripts/README_ORGANIZATION.md)**: 스크립트 구성 상세 설명

---

*이 문서는 Ethos Vision Optimizer 프로젝트의 모든 스크립트 사용법을 포괄적으로 다룹니다. 각 스크립트의 실행 예시와 출력 결과를 통해 효율적인 NPU 최적화 워크플로우를 구현할 수 있습니다.*