# NPU 최적화 모델 훈련 결과 리포트

**생성일**: 2025-10-01 10:46:30
**데이터셋**: Traffic Sign Detection (15 classes)
**총 훈련 레벨**: 1개
**사전 훈련 가중치**: yolov11n.pt

## 📊 훈련 결과 요약

| Level | 모델명 | 상태 | 훈련시간 | 학습률 | 예상개선 | 위험도 |
|-------|--------|------|----------|--------|----------|--------|
| LEVEL3-LEAKY | Level 3: + C2PSA 최적화 (LeakyReLU) | ✅ success | 1.4h 24.6m | 0.003 | 35-40% | Medium |

## 🎯 성공한 훈련 상세 정보

### LEVEL3-LEAKY: Level 3: + C2PSA 최적화 (LeakyReLU)

- **사전 훈련 가중치**: yolov11n.pt
- **훈련 시간**: 1.4h 24.6m
- **최고 모델**: `/lambda/nfs/yolo/results/training/level3-leaky_20251001_092155/weights/best.pt`


---
**💡 참고**: 이 리포트는 자동 생성되었습니다.
