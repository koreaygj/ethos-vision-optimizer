## yaml 종류

📁 생성된 YAML 파일들

  Level 1: npu_level1_scales.yaml
  - 적용: 2번 (모델 스케일 조정)
  - 위험도: ✅ 매우낮음 (95% 성공률)

  Level 2: npu_level2_scales_backbone.yaml
  - 적용: 2번 + 3번 (Backbone C3k2→C2f)
  - 위험도: ✅ 낮음 (85% 성공률)

  Level 3: npu_level3_scales_backbone_head.yaml
  - 적용: 2번 + 3번 + 4번 (Head C3k2→C2f)
  - 위험도: ⚠️ 중간 (70% 성공률)

  Level 4: npu_level4_full_optimization.yaml
  - 적용: 2,3,4번 + 5번(C2PSA→CSP) +
  6번(ConvTranspose2d)
  - 위험도: ⚠️⚠️ 높음 (60% 성공률)

## 호환성

| Level   | 계산량 감소 | NPU 호환성 | 정확도
  손실 |
  |---------|--------|---------|--------|
  | Level 1 | 25-30% | 45%     | 1-2%   |
  | Level 2 | 40-45% | 65%     | 3-5%   |
  | Level 3 | 50-55% | 75%     | 5-8%   |
  | Level 4 | 55-60% | 85%     | 8-12%  |