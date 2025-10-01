# Docker Guide - Ethos Vision Optimizer

**프로젝트**: Ethos Vision Optimizer
**업데이트**: 2025-10-01
**목적**: ARM Ethos-N NPU 개발 환경 구축

이 가이드는 Ethos Vision Optimizer 프로젝트에서 ARM Ethos-N NPU 개발을 위한 Docker 환경 구축 방법을 설명합니다.

---

## 📋 **개요**

이 Docker 환경은 TVM(Tensor Virtual Machine)에서 ARM Ethos-N NPU를 활용하여 YOLO 모델 최적화 및 IR(Intermediate Representation) 모듈 분석을 위해 구성되었습니다.

### **주요 구성 요소**
- **기본 환경**: Ubuntu 22.04 (ARM64)
- **컴파일러**: LLVM 14.0.0, ARM GCC 9.2-2019.12
- **ML 프레임워크**: TensorFlow, TFLite, ONNX
- **NPU 지원**: ARM Ethos-N Driver Stack
- **개발 도구**: Python 3.9, Rust, CMake, GoogleTest

해당 도커 파일은 [docker hub](https://hub.docker.com/repository/docker/koreaygj/tvm-ethosn-dev/general)에서 간단하게 설치 가능합니다.

---

## 🐳 **Dockerfile 분석: [`Dockerfile.ci-arm`](Dockerfile.ci-arm)**

### **1. 기본 시스템 설정**

#### **베이스 이미지 및 시스템 업데이트**
```dockerfile
FROM ubuntu:22.04

# APT 패키지 관리 유틸리티 복사
COPY utils/apt-install-and-clear.sh /usr/local/bin/apt-install-and-clear

# APT 업데이트 시 날짜 검증 비활성화
RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" > /etc/apt/apt.conf.d/10no--check-valid-until
```

**설치 내용**:
- Ubuntu 22.04 LTS 기반 ARM64 환경
- APT 패키지 관리자 최적화 설정
- 패키지 설치 후 자동 캐시 정리를 위한 유틸리티

#### **보안 및 인증서 설정**
```dockerfile
RUN apt-install-and-clear -y ca-certificates gnupg2
```

**설치 내용**:
- SSL/TLS 인증서 관리
- GNU Privacy Guard (암호화 및 서명)

### **2. 기본 개발 환경**

#### **시간대 및 코어 패키지 설정**
```dockerfile
COPY install/ubuntu_setup_tz.sh /install/ubuntu_setup_tz.sh
RUN bash /install/ubuntu_setup_tz.sh

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh
```

**설치 내용**:
- 시스템 시간대 설정
- 기본 개발 도구 (gcc, g++, make, git, curl, wget 등)
- 시스템 라이브러리 및 헤더 파일

#### **CMake 빌드 시스템**
```dockerfile
COPY install/ubuntu_install_cmake_source.sh /install/ubuntu_install_cmake_source.sh
RUN bash /install/ubuntu_install_cmake_source.sh
```

**설치 내용**:
- 최신 CMake (소스 컴파일)
- C/C++ 프로젝트 빌드 시스템

#### **테스트 프레임워크**
```dockerfile
COPY install/ubuntu_install_googletest.sh /install/ubuntu_install_googletest.sh
RUN bash /install/ubuntu_install_googletest.sh
```

**설치 내용**:
- Google Test Framework
- C++ 단위 테스트 및 모킹 라이브러리

### **3. 컴파일러 환경**

#### **Rust 개발 환경**
```dockerfile
COPY install/ubuntu_install_rust.sh /install/ubuntu_install_rust.sh
RUN bash /install/ubuntu_install_rust.sh
ENV RUSTUP_HOME /opt/rust
ENV CARGO_HOME /opt/rust
ENV PATH $PATH:$CARGO_HOME/bin
```

**설치 내용**:
- Rust 컴파일러 및 Cargo 패키지 매니저
- 시스템 레벨 성능 최적화 도구 개발용

#### **컴파일 캐시 시스템**
```dockerfile
COPY install/ubuntu_install_sccache.sh /install/ubuntu_install_sccache.sh
RUN bash /install/ubuntu_install_sccache.sh
ENV PATH /opt/sccache:$PATH
```

**설치 내용**:
- sccache (공유 컴파일 캐시)
- 빌드 시간 단축을 위한 캐시 시스템

#### **LLVM 컴파일러**
```dockerfile
RUN apt-install-and-clear -y wget xz-utils && \
    LLVM_VERSION=14.0.0 && \
    LLVM_REL=clang+llvm-${LLVM_VERSION}-aarch64-linux-gnu && \
    LLVM_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_REL}.tar.xz && \
    wget --quiet ${LLVM_URL} && \
    tar -xJf ${LLVM_REL}.tar.xz && \
    mv ${LLVM_REL} /opt/llvm && \
    rm ${LLVM_REL}.tar.xz

ENV PATH=/opt/llvm/bin:$PATH
```

**설치 내용**:
- LLVM 14.0.0 (ARM64 네이티브 버전)
- Clang 컴파일러
- LLVM IR 생성 및 최적화 도구

#### **ARM GCC 크로스 컴파일러**
```dockerfile
RUN apt-get update && apt-get install -y wget && \
    COMPILER_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz" && \
    wget -q ${COMPILER_URL} -O /tmp/gcc-arm-compiler.tar.xz && \
    mkdir -p /opt/gcc-arm-compiler && \
    tar -xf /tmp/gcc-arm-compiler.tar.xz -C /opt/gcc-arm-compiler --strip-components=1 && \
    rm /tmp/gcc-arm-compiler.tar.xz

ENV PATH=/opt/gcc-arm-compiler/bin:$PATH
```

**설치 내용**:
- ARM GCC 9.2-2019.12 툴체인
- aarch64-none-linux-gnu 크로스 컴파일러
- ARM 아키텍처 특화 최적화 컴파일러

### **4. Python 및 ML 환경**

#### **Python 3.9 가상환경**
```dockerfile
ENV TVM_VENV /venv/apache-tvm-py3.9
COPY python/bootstrap/lockfiles /install/python/bootstrap/lockfiles
COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh 3.9
ENV PATH ${TVM_VENV}/bin:$PATH
ENV PYTHONNOUSERSITE 1  # Disable .local directory from affecting CI.
```

**설치 내용**:
- Python 3.9 전용 가상환경
- TVM 개발용 격리된 Python 환경
- CI/CD 환경에서 일관성 보장

#### **Python 패키지 생태계**
```dockerfile
COPY install/ubuntu_install_python_package.sh /install/ubuntu_install_python_package.sh
RUN bash /install/ubuntu_install_python_package.sh
```

**설치 내용**:
- NumPy, SciPy (과학 계산)
- Matplotlib, Pillow (시각화 및 이미지 처리)
- PyTorch, torchvision (딥러닝 프레임워크)
- Jupyter Notebook (개발 환경)

### **5. ML 프레임워크**

#### **TensorFlow (ARM64 특화)**
```dockerfile
COPY install/ubuntu_install_tensorflow_aarch64.sh /install/ubuntu_install_tensorflow_aarch64.sh
RUN bash /install/ubuntu_install_tensorflow_aarch64.sh
```

**설치 내용**:
- TensorFlow ARM64 최적화 버전
- ARM NEON 및 ARM64 특화 최적화
- GPU 가속 지원 (Mali GPU)

#### **TensorFlow Lite**
```dockerfile
COPY install/ubuntu_install_tflite.sh /install/ubuntu_install_tflite.sh
RUN bash /install/ubuntu_install_tflite.sh
```

**설치 내용**:
- TensorFlow Lite 런타임
- 모바일/임베디드 배포용 경량화 프레임워크
- ARM NPU 델리게이트 지원

#### **ONNX (Open Neural Network Exchange)**
```dockerfile
COPY install/ubuntu_install_onnx.sh /install/ubuntu_install_onnx.sh
RUN bash /install/ubuntu_install_onnx.sh
```

**설치 내용**:
- ONNX 런타임
- 다양한 ML 프레임워크 간 모델 교환 표준
- ONNX → TVM 변환 지원

### **6. 성능 최적화 도구**

#### **AutoTVM 의존성**
```dockerfile
COPY install/ubuntu_install_redis.sh /install/ubuntu_install_redis.sh
RUN bash /install/ubuntu_install_redis.sh
```

**설치 내용**:
- Redis 인메모리 데이터베이스
- AutoTVM 튜닝 결과 캐시 스토리지
- 분산 튜닝 환경 지원

#### **ARM Compute Library**
```dockerfile
COPY install/ubuntu_download_arm_compute_lib_binaries.sh /install/ubuntu_download_arm_compute_lib_binaries.sh
RUN bash /install/ubuntu_download_arm_compute_lib_binaries.sh
```

**설치 내용**:
- ARM Compute Library 바이너리
- ARM CPU/GPU 최적화된 연산 라이브러리
- NEON, Mali GPU 가속 지원

### **7. ARM Ethos-N NPU 지원**

#### **Ethos-N 드라이버 스택**
```dockerfile
RUN apt-get update && apt-get install -y scons
COPY ethos-n-driver-stack /tmp/ethos-n-driver-stack

RUN cd /tmp/ethos-n-driver-stack/driver && \
    scons -j$(nproc) platform=native install_prefix=/usr/local/ethos-n install && \
    cd / && \
    rm -rf /tmp/ethos-n-driver-stack

ENV LD_LIBRARY_PATH /usr/local/ethos-n/lib:$LD_LIBRARY_PATH
```

**설치 내용**:
- **SCons 빌드 시스템**: Ethos-N 드라이버 컴파일용
- **Ethos-N 드라이버**: NPU 하드웨어 추상화 레이어
- **Ethos-N 라이브러리**: NPU 연산 최적화 라이브러리
- **사용자 공간 드라이버**: 애플리케이션에서 NPU 접근
- **커널 모듈 인터페이스**: 시스템 레벨 NPU 제어

**NPU 최적화 기능**:
- INT8/INT16 양자화 연산
- 고효율 컨볼루션 연산
- 메모리 대역폭 최적화
- 저전력 추론 실행

---

## 🚀 **사용법**

### **1. Docker 이미지 빌드**

#### **사전 준비**
```bash
# TVM 프로젝트 클론
git clone https://github.com/apache/tvm.git
cd tvm

# Ethos Vision Optimizer의 Dockerfile 복사
cp /path/to/ethos-vision-optimizer/docker/Dockerfile.ci-arm docker/
cp -r /path/to/ethos-vision-optimizer/docker/ethos-n-driver-stack docker/
```

#### **Docker 이미지 빌드**
```bash
# TVM 프로젝트 루트에서 실행
docker build -f docker/Dockerfile.ci-arm -t ethos-vision-optimizer:arm64 .
```

**빌드 시간**: 약 30-45분 (하드웨어 성능에 따라)

### **2. 컨테이너 실행**

#### **기본 실행**
```bash
docker run -it \
  --name ethos-optimizer \
  -v $(pwd):/workspace \
  ethos-vision-optimizer:arm64 \
  /bin/bash
```

#### **개발 환경 실행**
```bash
docker run -it \
  --name ethos-dev \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -p 8888:8888 \
  ethos-vision-optimizer:arm64 \
  /bin/bash
```

### **3. 환경 검증**

#### **컴파일러 확인**
```bash
# 컨테이너 내부에서 실행
clang --version                    # LLVM 14.0.0 확인
aarch64-none-linux-gnu-gcc --version  # ARM GCC 9.2 확인
rustc --version                   # Rust 컴파일러 확인
```

#### **Python 환경 확인**
```bash
python --version                  # Python 3.9.x 확인
pip list | grep torch            # PyTorch 설치 확인
pip list | grep tensorflow       # TensorFlow 설치 확인
```

#### **NPU 지원 확인**
```bash
ls -la /usr/local/ethos-n/        # Ethos-N 설치 확인
echo $LD_LIBRARY_PATH            # 라이브러리 경로 확인
```

---

## 🔧 **개발 워크플로우**

### **1. 모델 최적화 파이프라인**

#### **YOLO 모델 → TVM IR 변환**
```bash
# 컨테이너 내부에서
cd /workspace
python scripts/convert/pt2onnx.py --input models/optimized_npu/level4_relu/best.pt
python scripts/analysis/tvm_ir_analyzer.py --model models/pure/level4_relu.onnx
```

#### **NPU 최적화 및 코드 생성**
```bash
# TVM을 통한 Ethos-N 컴파일
python tvm_compile_ethos_n.py \
  --model models/pure/level4_relu.onnx \
  --target ethos-n \
  --output models/compiled/level4_relu_ethos_n.tar
```

### **2. 성능 벤치마킹**

#### **NPU vs CPU 성능 비교**
```bash
# NPU 추론 성능 측정
python scripts/evaluation/npu_benchmark.py \
  --model models/compiled/level4_relu_ethos_n.tar \
  --data data/dataset/test.yaml

# CPU 추론 성능 비교
python scripts/evaluation/cpu_benchmark.py \
  --model models/optimized_npu/level4_relu/best.pt \
  --data data/dataset/test.yaml
```

### **3. 디버깅 및 분석**

#### **IR 모듈 분석**
```bash
# TVM IR 시각화
python scripts/analysis/visualize_ir.py \
  --input models/compiled/level4_relu_ethos_n.tar \
  --output docs/analysis/ir_visualization.html
```

---

## 📊 **성능 특성**

### **빌드된 이미지 사양**
- **이미지 크기**: 약 3.5-4.0 GB
- **메모리 요구사항**: 최소 8GB RAM (권장 16GB)
- **CPU**: ARM64 아키텍처 (Apple Silicon, ARM Cortex-A 시리즈)
- **디스크 공간**: 빌드용 15GB, 런타임용 8GB

### **지원 하드웨어**
- **Apple Silicon**: M1, M1 Pro, M1 Max, M2 시리즈
- **ARM Cortex-A**: A78, A710, A715, X1, X2, X3 코어
- **ARM Mali GPU**: G78, G710, G715 시리즈
- **ARM Ethos-N NPU**: N78, N57, N37, N77 시리즈

---

## ⚠️ **주의사항 및 제한사항**

### **하드웨어 제약**
1. **ARM64 전용**: x86_64 아키텍처에서는 실행 불가
2. **메모리 집약적**: 대용량 모델 컴파일 시 16GB+ RAM 필요
3. **NPU 하드웨어**: 실제 NPU 하드웨어 없이는 시뮬레이션 모드로만 동작

### **소프트웨어 호환성**
1. **TVM 버전**: Apache TVM의 특정 커밋과 연동
2. **Python 버전**: Python 3.9 고정 (다른 버전과 충돌 가능)
3. **CUDA 미지원**: ARM64 환경에서는 NVIDIA GPU 가속 불가

### **라이센스 고려사항**
- **Apache License 2.0**: TVM 및 관련 도구
- **ARM License**: Ethos-N 드라이버 및 Compute Library
- **상용 라이센스**: 일부 ARM 도구는 상용 용도 시 별도 라이센스 필요

---

## 📚 **관련 문서**

- **[TVM IR Guide](tvm-IR-guide.md)**: TVM IR 모듈 분석 가이드
- **[Scripts Usage Guide](../docs/scripts/scripts-usage-guide.md)**: 전체 스크립트 사용법
- **[NPU Optimization Matrix](../docs/NPU_OPTIMIZATION_MATRIX.md)**: NPU 최적화 전략
- **[Training Guide](../docs/training/training-scripts-detailed-guide.md)**: 모델 훈련 가이드

---

## 🔗 **외부 리소스**

- **[Apache TVM](https://tvm.apache.org/)**: TVM 공식 문서
- **[ARM Ethos-N](https://developer.arm.com/ip-products/processors/machine-learning/ethos-n)**: NPU 공식 문서
- **[ARM Compute Library](https://github.com/ARM-software/ComputeLibrary)**: ARM 최적화 라이브러리
- **[TensorFlow Lite](https://www.tensorflow.org/lite)**: 모바일 ML 프레임워크

---

*이 Docker 환경은 ARM Ethos-N NPU를 활용한 YOLO 모델 최적화를 위해 특별히 구성되었습니다. 모든 필수 도구와 라이브러리가 사전 설치되어 있어 즉시 개발을 시작할 수 있습니다.*
