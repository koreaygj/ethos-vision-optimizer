# Level 3 LeakyReLU Training Results

**모델**: Level 3 - Full C2PSA 최적화 (LeakyReLU)
**훈련 완료**: 2025-10-01
**상태**: ✅ 100 에포크 완료

## 📋 훈련 설정

- **Level**: Level 3 LeakyReLU
- **Epochs**: 100
- **Batch Size**: 16
- **Device**: CUDA
- **Learning Rate**: 0.003
- **Activation Function**: LeakyReLU(0.1)
- **Pretrained**: yolov11n.pt

## 🎯 최적화 내용

- **구조 변경**:
  - Backbone + Head C3k2 → C2f 변환
  - C2PSA → C2f 변환 (Attention 최적화)
- **활성화 함수**: SiLU → LeakyReLU
- **예상 개선**: 35-40%
- **위험도**: Medium
- **NPU 호환성**: 92%

## 📊 최종 성능 메트릭

- **Precision**: 90.3%
- **Recall**: 81.8%
- **mAP@0.5**: 90.1%
- **mAP@0.5:0.95**: 76.9%
- **Fitness**: 0.769

## 📁 주요 파일

- `weights/best.pt`: 최고 성능 모델 (7.4MB)
- `weights/last.pt`: 마지막 에포크 모델 (7.4MB)
- `results.png`: 훈련 곡선 그래프
- `confusion_matrix.png`: 혼동 행렬
- `args.yaml`: 훈련 설정 백업

## 🔧 특징

- **C2PSA 최적화**: Attention 메커니즘을 C2f로 대체하여 NPU 친화적 구조로 변환
- **LeakyReLU 활성화**: ReLU 대비 더 나은 gradient flow로 안정적인 훈련
- **최고 호환성**: Level 3 단계에서 최고 NPU 호환성 (92%)
- **안정성**: Medium 위험도로 성능과 안정성의 균형

## 🔍 사용법

```bash
# 추론 실행
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level3_leaky_relu_full_optimization_100epochs/weights/best.pt \
  --data data/dataset/data.yaml

# 성능 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --model results/training/v3/training/level3_leaky_relu_full_optimization_100epochs/weights/best.pt \
  --data data/dataset/data.yaml

# NPU 호환성 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  results/training/v3/training/level3_leaky_relu_full_optimization_100epochs/weights/best.pt

# 활성화 함수 검증
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level3_scales_backbone_head_leaked.yaml \
  --reference LeakyReLU
```

---
*이 모델은 NPU 최적화 파이프라인의 Level 3 단계로, LeakyReLU 활성화 함수와 C2PSA Attention 최적화를 통해 최고 수준의 NPU 호환성과 안정적인 성능을 제공합니다.*