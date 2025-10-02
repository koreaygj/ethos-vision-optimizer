# Level 2 LeakyReLU Training Results

**모델**: Level 2 - Backbone + Head 최적화 (LeakyReLU)
**훈련 완료**: 2025-10-01
**상태**: ✅ 100 에포크 완료

## 📋 훈련 설정

- **Level**: Level 2 LeakyReLU
- **Epochs**: 100
- **Batch Size**: 16
- **Device**: CUDA
- **Learning Rate**: 0.003
- **Activation Function**: LeakyReLU(0.1)
- **Pretrained**: yolov11n.pt

## 🎯 최적화 내용

- **구조 변경**: Backbone + Head C3k2 → C2f 변환
- **활성화 함수**: SiLU → LeakyReLU
- **예상 개선**: 25-30%
- **위험도**: Low
- **NPU 호환성**: 87%

## 📊 최종 성능 메트릭

- **Precision**: 90.2%
- **Recall**: 81.8%
- **mAP@0.5**: 90.0%
- **mAP@0.5:0.95**: 76.8%
- **Fitness**: 0.768

## 📁 주요 파일

- `weights/best.pt`: 최고 성능 모델 (7.0MB)
- `weights/last.pt`: 마지막 에포크 모델 (7.0MB)
- `results.png`: 훈련 곡선 그래프
- `confusion_matrix.png`: 혼동 행렬
- `args.yaml`: 훈련 설정 백업

## 🔍 사용법

```bash
# 추론 실행
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level2_leaky_relu_backbone_head_optimized_100epochs/weights/best.pt \
  --data data/dataset/data.yaml

# 성능 평가
python scripts/evaluation/yolo_model_evaluator.py \
  --model results/training/v3/training/level2_leaky_relu_backbone_head_optimized_100epochs/weights/best.pt \
  --data data/dataset/data.yaml
```

---
*이 모델은 NPU 최적화 파이프라인의 Level 2 단계로, LeakyReLU 활성화 함수를 사용하여 기본적인 구조 최적화와 함께 NPU 호환성을 확보한 안정적인 모델입니다.*