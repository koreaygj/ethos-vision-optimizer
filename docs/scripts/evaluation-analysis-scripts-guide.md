# Evaluation & Analysis Scripts Guide

**프로젝트**: Ethos Vision Optimizer
**업데이트**: 2025-10-01
**주요 카테고리**: Evaluation, Analysis, Validation

이 가이드는 모델 평가, 분석, 검증을 위한 스크립트들의 상세한 사용법을 설명합니다.

---

## 📊 **1. Evaluation Scripts - 모델 평가**

### **1.1 `accuracy_analysis.py` - 정확도 상세 분석**

**위치**: [`scripts/evaluation/accuracy_analysis.py`](../../scripts/evaluation/accuracy_analysis.py)
**목적**: 모델의 정확도를 다양한 메트릭으로 상세 분석

#### **기본 사용법**
```bash
# 단일 모델 정확도 분석
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level2_relu/best.pt \
  --data data/dataset/data.yaml

# 임계값 커스터마이징
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level3_leaky/best.pt \
  --data data/dataset/data.yaml \
  --conf-thres 0.25 \
  --iou-thres 0.45

# 상세 리포트 생성
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level4_relu/best.pt \
  --data data/dataset/data.yaml \
  --save-report results/evaluation/level4_accuracy.md \
  --plot-curves
```

#### **출력 예시**
```
🎯 정확도 분석 시작...
📊 데이터셋: 500 images, 1000 instances

📈 전체 성능 메트릭:
==================================================
   mAP@0.5: 0.847
   mAP@0.5:0.95: 0.623
   Precision: 0.856
   Recall: 0.789
   F1-Score: 0.821

📋 클래스별 성능:
==================================================
   Class 'car':
     - Instances: 450
     - mAP@0.5: 0.92
     - Precision: 0.89
     - Recall: 0.85
     - F1-Score: 0.87

   Class 'person':
     - Instances: 350
     - mAP@0.5: 0.78
     - Precision: 0.82
     - Recall: 0.74
     - F1-Score: 0.78

🎨 시각화 파일 저장:
   - Confusion Matrix: results/evaluation/confusion_matrix.png
   - PR Curve: results/evaluation/pr_curve.png
   - F1 Curve: results/evaluation/f1_curve.png
```

#### **고급 옵션**
```bash
# 클래스별 상세 분석
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level3_relu/best.pt \
  --data data/dataset/data.yaml \
  --per-class-analysis \
  --save-predictions results/evaluation/predictions.json

# 오류 분석 포함
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --data data/dataset/data.yaml \
  --error-analysis \
  --save-errors results/evaluation/error_samples/
```

### **1.2 `yolo_model_evaluator.py` - 종합 모델 평가**

**목적**: 다중 모델 성능 비교 및 추론 속도 측정

#### **단일 모델 평가**
```bash
# 기본 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level2_relu/best.pt \
  --data data/dataset/data.yaml

# 벤치마크 포함
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level3_leaky/best.pt \
  --data data/dataset/data.yaml \
  --benchmark \
  --runs 100 \
  --warmup 10
```

#### **다중 모델 비교**
```bash
# 여러 모델 동시 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --models \
    models/optimized_npu/level2_relu/best.pt \
    models/optimized_npu/level3_relu/best.pt \
    models/optimized_npu/level4_relu/best.pt \
  --data data/dataset/data.yaml \
  --compare \
  --output results/evaluation/model_comparison.csv

# GPU vs CPU 성능 비교
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --data data/dataset/data.yaml \
  --devices cuda cpu mps \
  --benchmark
```

#### **출력 예시**
```
🚀 모델 평가 시작...

📊 모델 성능 요약:
==================================================
Model: level2_relu/best.pt
   - Parameters: 3,498,256
   - Model Size: 13.34 MB
   - mAP@0.5: 0.847
   - Inference Time: 23.4ms (CPU)
   - FPS: 42.7

Model: level3_relu/best.pt
   - Parameters: 3,708,816
   - Model Size: 14.15 MB
   - mAP@0.5: 0.863
   - Inference Time: 26.1ms (CPU)
   - FPS: 38.3

⚡ 벤치마크 결과 (100 runs):
==================================================
   - 평균 추론 시간: 24.7ms ± 2.1ms
   - 최소 추론 시간: 22.1ms
   - 최대 추론 시간: 28.9ms
   - GPU 메모리 사용량: 1.2GB
```

### **1.3 `compare_all_models.py` - 전체 모델 비교**

**목적**: Original vs Optimized 모델 전체 성능 비교

#### **전체 비교 분석**
```bash
# 모든 최적화 레벨 비교
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/ \
  --data data/dataset/data.yaml \
  --output results/evaluation/full_comparison.md

# 빠른 비교 (속도 우선)
python scripts/evaluation/compare_all_models.py \
  --base-model models/pure/yolov11n.pt \
  --optimized-dir models/optimized_npu/ \
  --data data/dataset/data.yaml \
  --quick \
  --metrics mAP precision recall

# 특정 포맷 모델들 비교
python scripts/evaluation/compare_all_models.py \
  --models \
    models/pure/yolov11n.pt \
    models/pure/level2_relu.onnx \
    models/pure/level3_leaky_int8.tflite \
  --data data/dataset/data.yaml \
  --cross-format
```

#### **비교 리포트 예시**
```markdown
# 모델 성능 비교 리포트

## 📊 전체 성능 요약

| 모델 | mAP@0.5 | mAP@0.5:0.95 | 파라미터 | 크기 | 추론시간 | NPU 호환성 |
|------|---------|--------------|----------|------|----------|------------|
| **Original YOLOv11n** | 0.834 | 0.598 | 2.6M | 9.8MB | 18.2ms | 67% |
| **Level 2 ReLU** | 0.847 (+1.6%) | 0.623 (+4.2%) | 3.5M | 13.3MB | 23.4ms | 85% |
| **Level 3 LeakyReLU** | 0.863 (+3.5%) | 0.641 (+7.2%) | 3.7M | 14.2MB | 26.1ms | 92% |
| **Level 4 ReLU** | 0.856 (+2.6%) | 0.635 (+6.2%) | 3.7M | 14.2MB | 25.8ms | 95% |

## 🎯 최적화 효과 분석

### ✅ 성공 요인
- **활성화 함수 최적화**: SiLU → ReLU/LeakyReLU로 NPU 호환성 대폭 개선
- **구조적 최적화**: C3k2 → C2f 변환으로 효율성 향상
- **정확도 유지**: 모든 레벨에서 정확도 향상 또는 유지

### ⚠️ 트레이드오프
- **모델 크기 증가**: 최적화로 인한 약 40% 크기 증가
- **추론 시간 증가**: CPU에서 약 25-40% 시간 증가 (NPU에서는 개선 예상)
```

---

## 🔍 **2. Analysis Scripts - 모델 분석**

### **2.1 `primitive_operator_analyzer_v2.py` - NPU 호환성 분석**

**위치**: [`scripts/analysis/primitive_operator_analyzer_v2.py`](../../scripts/analysis/primitive_operator_analyzer_v2.py)
**목적**: Primitive operator 분석 및 Ethos-N NPU 호환성 체크

#### **기본 분석**
```bash
# 단일 모델 호환성 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level2_relu/best.pt

# 상세 분석 리포트
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level3_leaky/best.pt \
  --detailed \
  --output docs/analysis/level3_leaky_analysis.md

# JSON 형태 결과
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/optimized_npu/level4_relu/best.pt \
  --format json \
  --output results/analysis/level4_analysis.json
```

#### **비교 분석**
```bash
# 여러 모델 비교 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  models/pure/yolov11n.pt \
  models/optimized_npu/level2_relu/best.pt \
  models/optimized_npu/level3_relu/best.pt \
  models/optimized_npu/level4_relu/best.pt \
  --compare \
  --output docs/analysis/optimization_progression.md

# Before vs After 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  --before models/pure/yolov11n.pt \
  --after models/optimized_npu/level4_leaky/best.pt \
  --diff-analysis \
  --output docs/analysis/optimization_impact.md
```

#### **출력 예시**
```
🔍 NPU 호환성 분석 시작...
📄 모델: models/optimized_npu/level3_relu/best.pt

📊 Primitive Operator 분석:
==================================================
   총 Operator 수: 127개
   지원 Operator: 103개 (81.1%)
   미지원 Operator: 24개 (18.9%)

✅ 지원되는 Operators:
   - Conv2d: 45개 (100% NPU 호환)
   - ReLU: 38개 (100% NPU 호환)
   - MaxPool2d: 12개 (100% NPU 호환)
   - Add: 8개 (100% NPU 호환)
   - Concat: 6개 (100% NPU 호환)

❌ 미지원 Operators:
   - SiLU: 12개 (0% NPU 호환) → ReLU 대체 권장
   - LayerNorm: 4개 (0% NPU 호환) → BatchNorm2d 대체 권장
   - GELU: 3개 (0% NPU 호환) → ReLU 대체 권장

💡 최적화 권장사항:
==================================================
1. 🔄 활성화 함수 교체:
   - SiLU → ReLU: 12개 위치에서 적용 가능
   - GELU → LeakyReLU: 3개 위치에서 적용 가능

2. 🏗️ 구조적 최적화:
   - LayerNorm → BatchNorm2d: 메모리 효율성 개선
   - 복합 연산 분해: NPU 가속을 위한 단순화

3. 📈 예상 개선 효과:
   - NPU 호환성: 81.1% → 95%+ 예상
   - 추론 속도: CPU 대비 3-5배 향상 예상 (NPU)
   - 전력 효율: 60-80% 개선 예상
```

### **2.2 `model_structure_analyzer.py` - 모델 구조 분석**

**목적**: 모델 아키텍처 및 파라미터 상세 분석

#### **구조 분석**
```bash
# 기본 구조 분석
python scripts/analysis/model_structure_analyzer.py \
  --model models/optimized_npu/level3_relu/best.pt \
  --output docs/analysis/level3_structure.md

# 레이어별 상세 분석
python scripts/analysis/model_structure_analyzer.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --layer-wise \
  --memory-analysis \
  --output docs/analysis/level4_detailed.md

# 파라미터 비교
python scripts/analysis/model_structure_analyzer.py \
  --models \
    models/pure/yolov11n.pt \
    models/optimized_npu/level4_relu/best.pt \
  --compare-params \
  --visualize \
  --output docs/analysis/parameter_comparison.md
```

#### **출력 예시**
```
🏗️ 모델 구조 분석...

📐 모델 아키텍처 요약:
==================================================
   모델명: level3_relu/best.pt
   총 레이어 수: 157개
   총 파라미터: 3,708,816개
   학습 가능 파라미터: 3,708,800개
   모델 크기: 14.15 MB (FP32)

📊 레이어 타입 분포:
==================================================
   - Conv2d: 64개 (40.8%)
   - BatchNorm2d: 57개 (36.3%)
   - ReLU: 57개 (36.3%)
   - C2f: 9개 (5.7%)
   - Bottleneck: 9개 (5.7%)
   - Concat: 4개 (2.5%)
   - Upsample: 2개 (1.3%)

🧮 메모리 분석:
==================================================
   파라미터 메모리: 14.15 MB
   추론 메모리 (예상): 45.2 MB
   활성화 메모리: 31.1 MB
   총 메모리 요구량: 90.5 MB

📈 최적화 전후 비교:
==================================================
                  | Original  | Level3    | 변화율
   파라미터 수    | 2.6M      | 3.7M      | +42.3%
   모델 크기      | 9.8MB     | 14.2MB    | +44.9%
   NPU 호환성     | 67%       | 92%       | +37.3%
   mAP@0.5       | 0.834     | 0.863     | +3.5%
```

---

## ✅ **3. Validation Scripts - 검증 및 테스트**

### **3.1 `activation_verifier.py` - 활성화 함수 검증**

**위치**: [`scripts/validation/activation_verifier.py`](../../scripts/validation/activation_verifier.py)
**목적**: Glenn Jocher's GitHub #7296 이슈 해결을 위한 활성화 함수 검증

#### **기본 검증**
```bash
# YAML 설정 검증
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level2_scales_backbone_relu.yaml

# 상세 분석
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level3_scales_backbone_head_leaked.yaml \
  --detailed

# 특정 활성화 함수와 비교
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_relu.yaml \
  --reference ReLU \
  --count-expected 114
```

#### **리포트 생성**
```bash
# 검증 리포트 자동 생성
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_leaked.yaml \
  --output results/validation/activation_report.md \
  --include-recommendations

# 모든 레벨 검증
for level in level2 level3 level4; do
  for activation in relu leaked; do
    python scripts/validation/activation_verifier.py \
      --model models/train/npu_${level}_*_${activation}.yaml \
      --output results/validation/${level}_${activation}_report.md
  done
done
```

#### **검증 결과 예시**
```
🔍 활성화 함수 검증 시작...
📄 YAML 파일: npu_level3_scales_backbone_head_relu.yaml

✅ YAML 설정 검증:
==================================================
   전역 활성화 함수: nn.ReLU()
   설정 방식: @aleshem's solution ✅
   이전 방식 (act: 'ReLU') 사용 안함 ✅

🔍 모델 활성화 함수 분석:
==================================================
   ✅ Found ReLU at: model.0.act
   ✅ Found ReLU at: model.1.act
   ✅ Found ReLU at: model.2.cv1.act
   ... (총 114개 발견)

📊 검증 결과:
==================================================
   ReLU 인스턴스: 114개 ✅
   LeakyReLU 인스턴스: 0개 ✅
   SiLU 인스턴스: 0개 ✅ (문제 없음)
   기타 Identity: 4개 (정상)

✅ 검증 통과: ReLU verification = True

💡 권장사항:
==================================================
   ✅ 활성화 함수 설정이 올바르게 적용됨
   ✅ NPU 호환성 높음 (ReLU 100% 지원)
   ✅ GitHub #7296 이슈 해결됨
```

### **3.2 `model_compatibility_checker.py` - 모델 호환성 검사**

**목적**: 다양한 플랫폼에서의 모델 호환성 검사

#### **기본 호환성 검사**
```bash
# 전체 호환성 검사
python scripts/validation/model_compatibility_checker.py \
  --model models/optimized_npu/level3_relu/best.pt \
  --platforms npu onnx tflite \
  --output results/validation/compatibility_report.md

# NPU 전용 검사
python scripts/validation/model_compatibility_checker.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --platform npu \
  --detailed \
  --ethos-n-version N78
```

#### **변환 호환성 검사**
```bash
# 변환 전 호환성 예측
python scripts/validation/model_compatibility_checker.py \
  --model models/optimized_npu/level2_relu/best.pt \
  --predict-conversion onnx tflite \
  --output results/validation/conversion_readiness.md

# 실제 변환 테스트
python scripts/validation/model_compatibility_checker.py \
  --model models/optimized_npu/level4_relu/best.pt \
  --test-conversion \
  --formats pt onnx tflite \
  --validate-accuracy
```

---

## 🔄 **4. 통합 분석 워크플로우**

### **4.1 완전한 모델 평가 파이프라인**

```bash
#!/bin/bash
# complete_evaluation_pipeline.sh

MODEL_PATH="models/optimized_npu/level4_leaky/best.pt"
DATA_PATH="data/dataset/data.yaml"
OUTPUT_DIR="results/comprehensive_evaluation"

mkdir -p $OUTPUT_DIR

echo "🔍 1단계: NPU 호환성 분석..."
python scripts/analysis/primitive_operator_analyzer_v2.py \
  $MODEL_PATH \
  --detailed \
  --output $OUTPUT_DIR/npu_compatibility.md

echo "📊 2단계: 정확도 상세 분석..."
python scripts/evaluation/accuracy_analysis.py \
  --model $MODEL_PATH \
  --data $DATA_PATH \
  --save-report $OUTPUT_DIR/accuracy_analysis.md \
  --plot-curves

echo "⚡ 3단계: 성능 벤치마크..."
python scripts/evaluation/yolo_model_evaluator.py \
  --model $MODEL_PATH \
  --data $DATA_PATH \
  --benchmark \
  --devices cpu cuda mps \
  --output $OUTPUT_DIR/performance_benchmark.csv

echo "✅ 4단계: 활성화 함수 검증..."
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_leaked.yaml \
  --output $OUTPUT_DIR/activation_verification.md

echo "🔄 5단계: 변환 호환성 검사..."
python scripts/validation/model_compatibility_checker.py \
  --model $MODEL_PATH \
  --platforms npu onnx tflite \
  --output $OUTPUT_DIR/compatibility_check.md

echo "📈 6단계: 최종 리포트 생성..."
python scripts/evaluation/generate_final_report.py \
  --input-dir $OUTPUT_DIR \
  --output $OUTPUT_DIR/final_evaluation_report.md

echo "✅ 평가 완료! 결과: $OUTPUT_DIR/"
```

### **4.2 빠른 품질 검사 워크플로우**

```bash
#!/bin/bash
# quick_quality_check.sh

MODEL=$1
if [ -z "$MODEL" ]; then
  echo "사용법: $0 <model_path>"
  exit 1
fi

echo "🚀 빠른 품질 검사: $MODEL"

# 1. 활성화 함수 검증
echo "1️⃣ 활성화 함수 검증..."
python scripts/validation/activation_verifier.py \
  --model $(echo $MODEL | sed 's/best.pt/..\/..\/train\/npu_*yaml/g') \
  --quick

# 2. NPU 호환성 빠른 확인
echo "2️⃣ NPU 호환성 확인..."
python scripts/analysis/primitive_operator_analyzer_v2.py \
  $MODEL \
  --quick \
  --compatibility-only

# 3. 기본 성능 확인
echo "3️⃣ 기본 성능 확인..."
python scripts/evaluation/yolo_model_evaluator.py \
  --model $MODEL \
  --data data/dataset/data.yaml \
  --quick

echo "✅ 빠른 검사 완료!"
```

---

## 📈 **5. 결과 해석 및 최적화 가이드**

### **5.1 NPU 호환성 점수 해석**

| 호환성 점수 | 상태 | 권장 조치 |
|-------------|------|-----------|
| **95%+** | 🟢 우수 | 배포 준비 완료 |
| **85-94%** | 🟡 양호 | 미지원 연산자 최적화 권장 |
| **70-84%** | 🟠 보통 | 구조적 최적화 필요 |
| **70% 미만** | 🔴 부족 | 전면적인 아키텍처 재설계 필요 |

### **5.2 성능 메트릭 기준**

#### **정확도 기준**
- **mAP@0.5 ≥ 0.8**: 상용화 가능 수준
- **mAP@0.5:0.95 ≥ 0.6**: COCO 데이터셋 기준 우수
- **정확도 손실 < 5%**: 최적화 허용 범위

#### **속도 기준 (Ethos-N78 기준)**
- **추론 시간 < 30ms**: 실시간 처리 가능
- **FPS > 30**: 비디오 처리 적합
- **메모리 사용량 < 100MB**: 임베디드 환경 적합

### **5.3 최적화 우선순위**

1. **1단계**: 활성화 함수 최적화 (SiLU → ReLU/LeakyReLU)
2. **2단계**: 미지원 연산자 교체 (LayerNorm → BatchNorm2d)
3. **3단계**: 구조적 최적화 (C3k2 → C2f, C2PSA → C2f)
4. **4단계**: 양자화 및 프루닝
5. **5단계**: NPU 특화 최적화

---

## 📚 **6. 관련 문서 및 참고 자료**

- **[Activation Function Fix Documentation](activation-function-fix-documentation.md)**: GitHub #7296 해결 과정
- **[Training Scripts Guide](training-scripts-detailed-guide.md)**: 훈련 스크립트 상세 가이드
- **[Scripts Usage Guide](scripts-usage-guide.md)**: 전체 스크립트 사용법
- **[NPU Optimization Matrix](NPU_OPTIMIZATION_MATRIX.md)**: Level별 최적화 매트릭스

---

## 🔧 **7. 트러블슈팅**

### **자주 발생하는 문제들**

#### **1. "Model loading failed" 오류**
```bash
# 해결: 모델 파일 경로 및 형식 확인
python scripts/evaluation/accuracy_analysis.py \
  --model models/optimized_npu/level3_relu/best.pt \
  --validate-model-first
```

#### **2. "CUDA out of memory" 오류**
```bash
# 해결: CPU 사용 또는 배치 크기 감소
python scripts/evaluation/yolo_model_evaluator.py \
  --model models/optimized_npu/level4_leaky/best.pt \
  --device cpu \
  --batch-size 1
```

#### **3. "Activation verification failed" 오류 ([GitHub Issue #7296](https://github.com/ultralytics/ultralytics/issues/7296))**
**원인**: YAML에서 `act: 'ReLU'` 설정했지만 실제로는 SiLU 사용됨
**해결**: @aleshem의 해결책 적용 - `activation: nn.ReLU()` 형식 사용
```bash
# Glenn's 방법으로 활성화 함수 검증
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level2_scales_backbone_relu.yaml \
  --debug

# 상세 분석으로 문제 진단
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level2_scales_backbone_relu.yaml \
  --detailed --reference ReLU
```

---

*이 문서는 Ethos Vision Optimizer 프로젝트의 모든 평가, 분석, 검증 스크립트를 포괄적으로 다룹니다. 체계적인 모델 평가와 NPU 최적화 검증을 위한 완전한 가이드를 제공합니다.*