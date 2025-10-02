# Ethos Vision Optimizer

**ARM Ethos-N NPU를 위한 YOLO 모델 최적화 파이프라인**

YOLO 모델을 ARM Ethos-N NPU에 최적화하여 NPU 호환성을 크게 향상시키는 종합적인 최적화 솔루션입니다.

[![NPU Compatibility](https://img.shields.io/badge/NPU%20Compatibility-Optimized-brightgreen)](docs/NPU_OPTIMIZATION_MATRIX.md)
[![Training Success](https://img.shields.io/badge/V3%20Training-6%2F6%20Success-success)](results/training/v3/TRAINING_RESULTS_SUMMARY.md)
[![Documentation](https://img.shields.io/badge/Docs-Comprehensive-blue)](docs/progress-tracking/PROJECT_PROGRESS_OVERVIEW.md)

---

## 🚀 **빠른 시작**

### **환경 설정**
```bash
# 기본 패키지 설치
pip install -r requirements.txt
```

### **NPU 최적화 모델 훈련**
```bash
# Level 2 ReLU 모델 
python scripts/training/npu_optimized_trainer.py --level level2-relu --epochs 100

# Level 4 LeakyReLU 모델 
python scripts/training/npu_optimized_trainer.py --level level4-leaky --epochs 100
```

### **훈련된 모델 사용하기**
```bash
# V3 훈련 완료 모델 직접 사용
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt \
  --data data/dataset/data.yaml
```

**💡 더 자세한 사용법**: [Scripts Usage Guide →](docs/scripts/scripts-usage-guide.md)

---

## 🎯 **진행 상황및 주요 성과**

### **✅ V3 훈련 세션 완료** (2025-10-01)
- **6개 모델 100% 성공**: Level 2-4, ReLU/LeakyReLU 각각
- **안정적인 성능**: 모든 모델에서 mAP@0.5 90%+ 달성

### **🔧 GitHub Issue #7296 해결**
- **Ultralytics YOLO 활성화 함수 문제** 완전 해결
- **@aleshem의 솔루션 적용**: `activation: nn.ReLU()` 형식
- **Glenn Jocher의 검증법 구현**: 실제 적용 여부 확인 시스템

### **📊 NPU 최적화 레벨**

| Level | Activation | 최적화 정도 | 위험도 | 사용 권장 시나리오 |
|-------|------------|------------|--------|-------------------|
| **Level 2** | ReLU/LeakyReLU | 기본 최적화 | Low | 초기 적용 및 검증 |
| **Level 3** | ReLU/LeakyReLU | 중급 최적화 | Medium | 실용적 배포 |
| **Level 4** | ReLU/LeakyReLU | 고급 최적화 | High | 프로덕션 최고 성능 |

**📈 상세 성능 분석**: [NPU Optimization Matrix →](docs/NPU_OPTIMIZATION_MATRIX.md)

**진행 상황**: [프로젝트 타임라인 🥇](docs/progress-tracking/timeline/PROJECT_TIMELINE.md)

---

## 🏗️ **프로젝트 구조**

```
ethos-vision-optimizer/
├── 📁 scripts/                    # 실행 스크립트
│   ├── training/                  # NPU 최적화 훈련
│   ├── evaluation/                # 모델 평가 및 성능 측정
│   ├── convert/                   # 모델 포맷 변환
│   ├── analysis/                  # 모델 분석 도구
│   └── validation/                # 검증 및 테스트
├── 📁 models/                     # 모델 저장소
│   ├── train/                     # 훈련 설정 (YAML)
│   └── pure/                      # 원본/변환 모델
├── 📁 results/                    # 실행 결과
│   ├── training/v3/               # ✅ V3 훈련 완료 (6개 모델)
│   └── evaluation/                # 평가 결과
├── 📁 docs/                       # 📚 포괄적 문서화
│   ├── scripts/                   # 스크립트 사용 가이드
│   ├── training/                  # 훈련 상세 가이드
│   ├── progress-tracking/         # 진행상황 추적
│   └── trouble-shoot/             # 문제 해결 가이드
├── 📁 docker/                     # 🐳 Docker 환경
│   ├── Dockerfile.ci-arm          # ARM Ethos-N 지원
│   └── ethos-n-driver-stack/      # NPU 드라이버
└── 📁 data/dataset/               # 데이터셋
```

**🔍 더 자세한 구조**: [Architecture Overview →](docs/architecture/pipeline-overview.md)

---

## 🛠️ **사용 방법**

### **1. 기존 V3 모델 사용 (권장)**

V3 훈련에서 완성된 6개 모델을 바로 사용할 수 있습니다.

```bash
# 안정성 우선: Level 2 ReLU (기본 NPU 최적화)
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level2_relu_backbone_head_optimized_100epochs/weights/best.pt

# 균형잡힌 성능: Level 3 LeakyReLU (중급 NPU 최적화)
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level3_leaky_relu_full_optimization_100epochs/weights/best.pt

# 최고 성능: Level 4 LeakyReLU (고급 NPU 최적화)
python scripts/evaluation/accuracy_analysis.py \
  --model results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt
```

**📖 각 모델 상세 정보**: [V3 Training Results →](results/training/v3/TRAINING_RESULTS_SUMMARY.md)

### **2. 새로운 모델 훈련**

```bash
# 활성화 함수 검증 (훈련 전 필수)
python scripts/validation/activation_verifier.py \
  --model models/train/npu_level4_activation_leaked.yaml --detailed

# NPU 최적화 훈련 실행
python scripts/training/npu_optimized_trainer.py --level level4-leaky --epochs 100

# 훈련 결과 검증
python scripts/training/npu_optimized_trainer.py --level level4-leaky --inspect
```

**📖 상세 훈련 가이드**: [Training Scripts Guide →](docs/training/training-scripts-detailed-guide.md)

### **3. 모델 분석 및 변환**

```bash
# NPU 호환성 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt

# PyTorch → ONNX 변환
python scripts/convert/pt2onnx.py \
  --input results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt

# PyTorch → TFLite 변환
python scripts/convert/pt2tflite_int8.py \
  --input results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt
```

**📖 전체 스크립트 사용법**: [Scripts Usage Guide →](docs/scripts/scripts-usage-guide.md)

---

## 🐳 **Docker 환경 (권장)**

복잡한 ARM Ethos-N 환경을 Docker로 간편하게 구축할 수 있습니다.

```bash
# Docker Hub에서 사전 빌드된 이미지 사용
docker pull koreaygj/tvm-ethosn-dev

# 컨테이너 실행
docker run -it \
  --name ethos-optimizer \
  -v $(pwd):/workspace \
  koreaygj/tvm-ethosn-dev \
  /bin/bash

# 컨테이너 내부에서 모델 분석
python scripts/analysis/primitive_operator_analyzer_v2.py \
  /workspace/results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt
```

**🐳 Docker 상세 가이드**: [Docker Guide →](docker/docker-guide.md)

---

## 🔍 **문제 해결**

### **자주 발생하는 문제**

#### **활성화 함수 적용 안됨** ([GitHub Issue #7296](https://github.com/ultralytics/ultralytics/issues/7296))
```bash
# 문제 확인
python scripts/validation/activation_verifier.py --model your_model.yaml --debug

# 해결방법: @aleshem 솔루션 적용
# YAML에서 act: 'ReLU' → activation: nn.ReLU() 로 변경
```

#### **메모리 부족 오류**
```bash
# GPU 메모리 부족 시 배치 크기 조정
python scripts/training/npu_optimized_trainer.py --level level4-leaky --batch-size 8
```

#### **파일명 불일치**
```bash
# 스크립트가 기대하는 파일명 확인
# leaky.yaml → leaked.yaml 으로 변경 필요
```

**🔧 종합 문제해결**: [Troubleshooting Guide →](docs/trouble-shoot/activation-function-fix-documentation.md)

---

## 📊 **성능 및 호환성**

### **NPU 최적화 성과**

| 항목 | 원본 YOLO11n | Level 2 | Level 3 | Level 4 LeakyReLU |
|------|--------------|---------|---------|------------------|
| **NPU 최적화** | 기본 | 향상됨 | 높음 | **최고** |
| **mAP@0.5** | 89.5% | 90.0% | 90.1% | 90.1% |
| **모델 크기** | 6.2MB | 7.0MB | 7.4MB | 7.4MB |
| **위험도** | - | Low | Medium | High |

### **지원 환경**
- **하드웨어**: ARM64 (Apple Silicon, ARM Cortex-A)
- **NPU**: ARM Ethos-N77, N78, N57, N37
- **OS**: Ubuntu 22.04, macOS (Apple Silicon)
- **Python**: 3.9+

**📈 상세 성능 분석**: [NPU Optimization Matrix →](docs/NPU_OPTIMIZATION_MATRIX.md)

---

## 📚 **문서 가이드**

### **시작하기**
- **[Scripts Usage Guide](docs/scripts/scripts-usage-guide.md)**: 모든 스크립트 사용법
- **[Training Guide](docs/training/training-scripts-detailed-guide.md)**: 상세 훈련 가이드
- **[Docker Guide](docker/docker-guide.md)**: Docker 환경 구축

### **결과 및 분석**
- **[V3 Training Results](results/training/v3/TRAINING_RESULTS_SUMMARY.md)**: 완료된 6개 모델 정보
- **[NPU Optimization Matrix](docs/NPU_OPTIMIZATION_MATRIX.md)**: Level별 최적화 전략
- **[Evaluation Guide](docs/scripts/evaluation-analysis-scripts-guide.md)**: 평가 및 분석 도구

### **문제 해결 및 개발**
- **[Activation Function Fix](docs/trouble-shoot/activation-function-fix-documentation.md)**: GitHub #7296 해결 과정
- **[Progress Tracking](docs/progress-tracking/PROJECT_PROGRESS_OVERVIEW.md)**: 프로젝트 진행상황
- **[Architecture Overview](docs/architecture/pipeline-overview.md)**: 시스템 아키텍처

---

## 🎯 **다음 단계**

### **현재 진행 중**
- **TVM 컴파일 파이프라인**: YOLO → TVM → Ethos-N 바이너리
- **성능 벤치마킹**: CPU vs NPU 성능 비교
- **실제 하드웨어 테스트**: 물리적 Ethos-N NPU 환경

### **기여하기**
이 프로젝트는 ARM Ethos-N NPU 생태계 발전에 기여하고 있습니다.

- **Issues**: 버그 리포트 및 기능 요청
- **Pull Requests**: 코드 기여 및 개선사항
- **Documentation**: 문서 개선 및 번역

**📞 연락처**: [GitHub Issues](https://github.com/your-repo/ethos-vision-optimizer/issues)

---

## 📄 **라이센스**

이 프로젝트는 오픈소스이며, 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.

- **Apache License 2.0**: 메인 코드
- **ARM 라이센스**: Ethos-N 드라이버 및 관련 도구
- **상업적 사용**: ARM 라이센스 확인 필요

---

*Ethos Vision Optimizer는 ARM Ethos-N NPU를 위한 YOLO 모델 최적화의 표준을 제시하며, 지속적으로 발전하고 있습니다. 최신 업데이트와 자세한 정보는 [프로젝트 진행상황](docs/progress-tracking/PROJECT_PROGRESS_OVERVIEW.md)에서 확인하실 수 있습니다.*