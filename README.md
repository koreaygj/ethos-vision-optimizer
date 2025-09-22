# EThes Vision Optimizer - NPU Simulation

TVM을 활용한 EThes-N78 NPU 시뮬레이션 환경

## 돌려야 하는 모델
- YOLO11n-32FP
- PoseNet-32FP

### 경량화 방식
- 양자화 (Quantization)
- 프루닝 (Pruning)
- 지식 증류 (Knowledge Distillation)

### TVM
Apache TVM을 사용한 모델 컴파일 및 최적화

### 보드
- YOCTO 5
- EThes-N78

### 도커 환경설정
- YOCTO 5 기반 환경

## Todo
- [ ] 검증을 위한 데이터 셋 확인 - https://github.com/bhaskrr/traffic-sign-detection-using-yolov11
- [ ] NPU가 활성화 되는지에 대한 결과를 얻기 위한 가상화 환경 구성
- [ ] 원본 모델 검증
- [ ] 원본 -> TFLite 형식 변환 (변환 모델 각각 검증 필요)
- [ ] TFLite 파일에 대한 TVM compiler로 엔진 빌드
- [ ] Compile 이후 보드에서 벤치마크

### 목표 성능
- 성능 저하율 기존 대비 5% 미만
- 경량화율 40% 이상

### 기한
11월 말 PoseNet, YOLOv11n → 필수 완료

### 역할
- 준, 경진: YOLOv11n
- 시호, 환홍: PoseNet

## 일정
- 9월 3째주: 원본 모델 검증 확인
- 9월 4째주: 원본 TFLite 파일로 변환
- 9월 5째주: TVM compiler로 엔진 빌드
- 10월 2째주: Compile 된 엔진 Dolphin 5에서 검증

---

## 빠른 시작

### 1. Docker Compose로 시뮬레이션 실행

```bash
# 시뮬레이션 실행
docker-compose up tvm-npu-simulator

# 벤치마크 분석
docker-compose up benchmark-analyzer

# Jupyter 노트북 (포트 8888)
docker-compose up jupyter
```

### 2. 개별 컨테이너 실행

```bash
# Docker 이미지 빌드
docker build -f docker/Dockerfile.tvm-simulation -t ethos-npu-sim .

# 시뮬레이션 실행
docker run -v $(pwd)/simulation:/workspace ethos-npu-sim
```

## 구조

```
├── docker/
│   ├── Dockerfile.tvm-simulation    # TVM + NPU 시뮬레이션 환경
│   └── dockerfile.simulate          # 기존 시뮬레이터
├── simulation/
│   ├── tvm_npu_simulation.py        # 메인 시뮬레이션 코드
│   └── benchmark_utils.py           # 벤치마크 분석 도구
├── models/                          # ONNX 모델 디렉토리
├── results/                         # 시뮬레이션 결과
└── docker-compose.yml               # Docker Compose 설정
```

## 사용법

### 모델 추가
`models/` 디렉토리에 ONNX 모델 파일을 추가하세요.

### 시뮬레이션 실행
```python
from tvm_npu_simulation import NPUSimulator

simulator = NPUSimulator()
model_id = simulator.load_model("models/your_model.onnx")
results = simulator.benchmark(model_id, num_runs=100)
```

### 결과 분석
```python
from benchmark_utils import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer("npu_simulation_results.json")
print(analyzer.generate_report())
analyzer.plot_performance()
```

## 특징

- **TVM 기반**: Apache TVM을 사용한 고성능 컴파일 및 최적화
- **NPU 시뮬레이션**: EThes-N78 NPU 하드웨어 특성 반영
- **벤치마크**: 추론 시간, FPS, 메모리 사용량 측정
- **Docker 환경**: 일관된 시뮬레이션 환경 제공
- **분석 도구**: 성능 리포트 및 차트 생성

## 요구사항

- Docker & Docker Compose
- ONNX 모델 파일
- EThes-N78 SDK (선택사항)