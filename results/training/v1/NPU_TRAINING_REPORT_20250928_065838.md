# NPU 최적화 모델 훈련 결과 리포트 (사전 훈련 가중치 지원)

**생성일**: 2025-09-28 06:58:38
**데이터셋**: Traffic Sign Detection (15 classes)
**총 훈련 레벨**: 1개
**사전 훈련 가중치**: /home/ubuntu/yolo11n.pt

## 📊 훈련 결과 요약

| Level | 모델명 | 상태 | 훈련시간 | 학습률 | 예상개선 | 위험도 | 모델 경로 |
|-------|--------|------|----------|--------|----------|--------|-----------|
| LEVEL2 | Level 2: Backbone + Head 최적화 | ✅ success | 1.1h 3.0m | 0.003 | 25-30% | Low | best.pt |\n

## 🎯 성공한 훈련 상세 정보

### LEVEL2: Level 2: Backbone + Head 최적화

- **설명**: N/A
- **사전 훈련 가중치**: /home/ubuntu/yolo11n.pt
- **훈련 시간**: 1.1h 3.0m
- **학습률**: 0.003
- **완료 에폭**: 100
- **배치 크기**: 16
- **사용 디바이스**: cuda
- **최고 모델**: `/home/ubuntu/results/training/level2_20250928_055538/weights/best.pt`
- **최종 모델**: `/home/ubuntu/results/training/level2_20250928_055538/weights/last.pt`



## 🔧 훈련 설정

### NPU 최적화 훈련 파라미터
- **사전 훈련 가중치**: /home/ubuntu/yolo11n.pt
- **Optimizer**: SGD (양자화 친화적)
- **Learning Rate**: 0.003 (사전 훈련)
- **Momentum**: 0.9
- **Weight Decay**: 0.001
- **Mixed Precision**: False (양자화 준비)

### NPU 친화적 데이터 증강
- **복잡한 증강 비활성화**: Mosaic, MixUp, AutoAugment 등
- **단순 증강만 사용**: 좌우 반전, 최소한의 색상 변화
- **회전/원근 변환 비활성화**: NPU 효율성 고려

## 📈 다음 단계

### 성공한 모델들
1. **정확도 평가**: `scripts/01_evaluation/yolov11_pt_style_evaluator.py` 사용
2. **NPU 호환성 분석**: `scripts/04_npu_specific/analyze_convertible_operators.py` 실행
3. **Export 테스트**: ONNX 변환 및 NPU 툴체인 검증

### 실패한 모델들
1. **오류 분석**: 모델 구조 및 의존성 확인
2. **커스텀 모듈 구현**: CSP, C2f 등 필요 모듈 구현
3. **재훈련**: 문제 해결 후 재시도

---

**💡 참고**: 이 리포트는 자동 생성되었습니다. 상세한 훈련 로그는 각 모델의 결과 디렉토리를 확인하세요.
