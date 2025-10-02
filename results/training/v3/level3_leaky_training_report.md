# NPU 최적화 모델 훈련 결과 리포트

**생성일**: 2025-10-01 13:29:41
**데이터셋**: Traffic Sign Detection (15 classes)
**총 훈련 레벨**: 1개
**사전 훈련 가중치**: yolov11n.pt

## 📊 훈련 결과 요약

| Level | 모델명 | 상태 | 훈련시간 | 학습률 | 예상개선 | 위험도 |
|-------|--------|------|----------|--------|----------|--------|
| LEVEL4-RELU | Level 4: 완전 최적화 (ReLU) | ✅ success | 1.2h 9.5m | 0.003 | 40-45% | High |

## 🎯 성공한 훈련 상세 정보

### LEVEL4-RELU: Level 4: 완전 최적화 (ReLU)

- **사전 훈련 가중치**: yolov11n.pt
- **훈련 시간**: 1.2h 9.5m
- **최고 모델**: `/lambda/nfs/yolo/results/training/level4-relu_20251001_122011/weights/best.pt`


---
**💡 참고**: 이 리포트는 자동 생성되었습니다.
