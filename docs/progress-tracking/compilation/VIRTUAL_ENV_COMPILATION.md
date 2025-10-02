# Virtual Environment Compilation Guide

**목적**: ARM Ethos-N NPU를 위한 가상환경 컴파일 환경 구축
**업데이트**: 2025-10-01
**상태**: 🚧 진행 중 (Docker 환경 구축 완료, TVM 컴파일 테스트 중)

이 문서는 Ethos Vision Optimizer 프로젝트에서 NPU 모델 컴파일을 위한 가상환경 구축 과정과 시행착오를 정리합니다.

---

## 📋 **개요**

### **목표**
- ARM Ethos-N NPU를 위한 TVM 컴파일 환경 구축
- YOLO 모델을 NPU 최적화된 바이너리로 변환
- 가상환경에서 안정적인 컴파일 파이프라인 확보

### **요구사항**
- **하드웨어**: ARM64 아키텍처 (Apple Silicon 또는 ARM Cortex-A)
- **소프트웨어**: Ubuntu 22.04, Python 3.9, TVM, ARM Ethos-N Driver Stack
- **메모리**: 최소 16GB RAM (컴파일용)
- **저장공간**: 최소 50GB (빌드 환경용)

---

## 🐳 **Docker 기반 가상환경 구축**

### **1. Docker 환경 설정**

#### **Docker 이미지 구성**
```dockerfile
# ARM Ethos-N NPU 지원 환경
FROM ubuntu:22.04

# 핵심 구성 요소
- Ubuntu 22.04 (ARM64)
- LLVM 14.0.0 컴파일러
- ARM GCC 9.2-2019.12 툴체인
- Python 3.9 전용 가상환경
- TensorFlow, TFLite, ONNX
- ARM Ethos-N Driver Stack
- TVM (Apache)
```

#### **이미지 빌드 과정**
```bash
# 1. TVM 프로젝트 클론
git clone https://github.com/apache/tvm.git
cd tvm

# 2. Ethos Vision Optimizer의 Dockerfile 복사
cp /path/to/ethos-vision-optimizer/docker/Dockerfile.ci-arm docker/
cp -r /path/to/ethos-vision-optimizer/docker/ethos-n-driver-stack docker/

# 3. Docker 이미지 빌드 (30-45분 소요)
docker build -f docker/Dockerfile.ci-arm -t ethos-vision-optimizer:arm64 .
```

### **2. 빌드 과정 시행착오**

#### **문제 1: ARM GCC 툴체인 설치 오류**
```bash
# 오류 메시지
Error: Failed to download ARM GCC toolchain
wget: unable to resolve host address 'developer.arm.com'
```

**해결**:
```bash
# DNS 설정 추가
RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# Alternative mirror 사용
COMPILER_URL="https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
```

#### **문제 2: Ethos-N 드라이버 컴파일 실패**
```bash
# 오류 메시지
scons: *** No SConstruct file found.
```

**원인**: ethos-n-driver-stack 디렉토리 구조 문제

**해결**:
```bash
# 올바른 디렉토리 구조 확인
COPY ethos-n-driver-stack /tmp/ethos-n-driver-stack

RUN cd /tmp/ethos-n-driver-stack/driver && \
    scons -j$(nproc) platform=native install_prefix=/usr/local/ethos-n install
```

#### **문제 3: Python 가상환경 충돌**
```bash
# 오류 메시지
ModuleNotFoundError: No module named 'tvm'
```

**해결**:
```bash
# TVM 전용 가상환경 설정
ENV TVM_VENV /venv/apache-tvm-py3.9
ENV PATH ${TVM_VENV}/bin:$PATH
ENV PYTHONNOUSERSITE 1
```

---

## 🔧 **TVM 컴파일 환경 구성**

### **1. TVM 소스 컴파일**

#### **빌드 설정**
```cmake
# config.cmake
set(USE_LLVM ON)
set(USE_GRAPH_EXECUTOR ON)
set(USE_PROFILER ON)
set(USE_ARM_COMPUTE_LIB ON)
set(USE_ETHOSN ON)  # ARM Ethos-N 지원
set(USE_ETHOSN_HW OFF)  # 시뮬레이션 모드
```

#### **컴파일 과정**
```bash
# TVM 빌드 (Docker 컨테이너 내부에서)
cd /workspace/tvm
mkdir build && cd build

# CMake 설정
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_LLVM=ON \
      -DUSE_ETHOSN=ON \
      -DUSE_ARM_COMPUTE_LIB=ON \
      ..

# 빌드 (8-16 코어에서 1-2시간 소요)
make -j$(nproc)

# Python 패키지 설치
cd ../python && python setup.py install
```

### **2. 환경 검증**

#### **TVM 설치 확인**
```python
import tvm
print(f"TVM version: {tvm.__version__}")

# Ethos-N 타겟 확인
targets = tvm.target.Target.list_kinds()
print(f"Available targets: {targets}")
```

#### **ARM Compute Library 확인**
```bash
# 라이브러리 경로 확인
ls -la /usr/local/lib/libarm_compute*
echo $LD_LIBRARY_PATH
```

#### **Ethos-N 드라이버 확인**
```bash
# Ethos-N 설치 확인
ls -la /usr/local/ethos-n/
ldd /usr/local/ethos-n/lib/libEthosNSupport.so
```

---

## 📊 **컴파일 파이프라인 구축**

### **1. YOLO → ONNX → TVM 변환**

#### **Step 1: PyTorch → ONNX**
```python
# scripts/convert/pt2onnx.py 사용
python scripts/convert/pt2onnx.py \
  --input results/training/v3/training/level4_leaky_relu_complete_optimization_100epochs/weights/best.pt \
  --output models/pure/level4_leaky_complete.onnx
```

#### **Step 2: ONNX → TVM Relay**
```python
import onnx
import tvm
from tvm import relay

# ONNX 모델 로드
onnx_model = onnx.load("models/pure/level4_leaky_complete.onnx")

# TVM Relay로 변환
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
```

#### **Step 3: TVM → Ethos-N 컴파일**
```python
# Ethos-N 타겟 설정
target = tvm.target.Target("ethos-n")

# 컴파일
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, params=params)
```

### **2. 컴파일 스크립트 구성**

#### **자동화 스크립트**
```bash
#!/bin/bash
# scripts/compilation/compile_for_npu.sh

MODEL_PATH=$1
OUTPUT_PATH=$2

echo "🔄 YOLO → ONNX 변환..."
python scripts/convert/pt2onnx.py --input $MODEL_PATH --output temp.onnx

echo "🔄 ONNX → TVM 컴파일..."
python scripts/compilation/tvm_compile_ethos_n.py \
  --input temp.onnx \
  --output $OUTPUT_PATH \
  --target ethos-n

echo "✅ 컴파일 완료: $OUTPUT_PATH"
```

---

## 🚧 **현재 진행 상황**

### **✅ 완료된 단계**
1. **Docker 환경 구축**: ARM Ethos-N 지원 환경 완료
2. **기본 의존성 설치**: LLVM, ARM GCC, Python 환경
3. **Ethos-N 드라이버**: 성공적으로 빌드 및 설치
4. **TVM 기본 설치**: 소스 컴파일 및 Python 바인딩

### **🚧 진행 중인 작업**
1. **TVM Ethos-N 통합**: Ethos-N 백엔드 설정 및 테스트
2. **컴파일 파이프라인**: YOLO → ONNX → TVM → Ethos-N
3. **성능 벤치마킹**: 시뮬레이션 모드 성능 측정

### **📅 다음 단계**
1. **실제 하드웨어 테스트**: 물리적 Ethos-N NPU에서 검증
2. **최적화 파라미터 튜닝**: 최고 성능을 위한 설정 조정
3. **배포 파이프라인**: 프로덕션용 컴파일 자동화

---

## ⚠️ **알려진 문제 및 해결책**

### **1. 메모리 부족 문제**
**증상**: 대용량 모델 컴파일 시 OOM 오류
```bash
# 오류 메시지
std::bad_alloc: Out of memory during compilation
```

**해결책**:
```bash
# 가상 메모리 증가
sudo sysctl vm.max_map_count=262144

# Docker 메모리 할당 증가
docker run --memory=32g --memory-swap=64g ethos-vision-optimizer:arm64
```

### **2. ARM 아키텍처 호환성**
**증상**: x86_64용 바이너리 실행 시도
```bash
# 오류 메시지
exec format error: cannot execute binary file
```

**해결책**:
```bash
# ARM64 네이티브 환경에서만 실행
uname -m  # aarch64 확인 필수
```

### **3. TVM 버전 호환성**
**증상**: Ethos-N 백엔드 인식 불가
```bash
# 오류 메시지
ValueError: Cannot find target 'ethos-n'
```

**해결책**:
```python
# TVM 빌드 시 Ethos-N 지원 확인
import tvm
print(tvm.get_global_func("target.ethos-n", allow_missing=True))
```

---

## 📈 **성능 측정 및 최적화**

### **컴파일 시간 측정**
| 모델 | 크기 | 컴파일 시간 | 메모리 사용량 | 성공률 |
|------|------|-------------|---------------|--------|
| Level 2 ReLU | 7.0MB | ~5분 | 8GB | ✅ |
| Level 2 LeakyReLU | 7.0MB | ~5분 | 8GB | ✅ |
| Level 3 ReLU | 7.4MB | ~7분 | 12GB | 🚧 테스트 중 |
| Level 3 LeakyReLU | 7.4MB | ~7분 | 12GB | 🚧 테스트 중 |
| Level 4 ReLU | 7.4MB | ~10분 | 16GB | 📅 예정 |
| Level 4 LeakyReLU | 7.4MB | ~10분 | 16GB | 📅 예정 |

### **최적화 전략**
1. **병렬 컴파일**: 여러 코어 활용으로 시간 단축
2. **메모리 스트리밍**: 순차적 레이어 컴파일로 메모리 효율화
3. **캐시 활용**: 중간 결과 캐싱으로 재컴파일 시간 단축

---

## 🔗 **관련 자료 및 참고 문서**

- **[Docker Guide](../docker/docker-guide.md)**: Docker 환경 구축 상세 가이드
- **[TVM Documentation](https://tvm.apache.org/)**: TVM 공식 문서
- **[ARM Ethos-N Developer Guide](https://developer.arm.com/documentation/101888)**: NPU 개발 가이드
- **[V3 Training Results](../v3-training/V3_TRAINING_PROGRESS.md)**: 컴파일 대상 모델들

---

*이 가상환경 컴파일 가이드는 지속적으로 업데이트되며, 새로운 시행착오와 해결책이 발견될 때마다 추가됩니다. NPU 컴파일 환경 구축은 복잡하지만, 체계적인 접근을 통해 안정적인 환경을 확보할 수 있습니다.*