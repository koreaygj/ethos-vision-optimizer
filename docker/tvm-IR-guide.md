# Docker TVM Ethos-N 사용 가이드

## 📋 개요

TVM과 Ethos-N NPU 백엔드를 사용하여 모델을 컴파일하고 NPU 분할을 확인하는 가이드입니다.

---

## 🐳 1단계: Docker 컨테이너 실행

아래 명령어를 실행하여 필요한 폴더들을 연결(마운트)하고 컨테이너 내부 쉘에 접속합니다.

```bash
# 아래 -v 옵션의 경로를 실제 모델 폴더 경로로 수정해주세요
docker run --rm -it \
  -v ${tvm 프로젝트 폴더}:/work/tvm \
  -v ${ai models 폴더}:/work/models \
  koreaygj/tvm-ethosn-dev:latest \
  /bin/bash
```

⚠️ **주의**: `-v /path/to/your/models:/work/models` 부분을 실제 모델이 있는 경로로 변경해주세요.

TVM 프로젝트 clone 시에 ethos-n을 지원하는 0.10.0 이상 브런치로 변경하세요.

이제 터미널은 컨테이너 내부를 보고 있는 상태입니다.

---

## 🔨 2단계: TVM 빌드 (컨테이너 내부에서 실행)

컨테이너 쉘에서 아래 명령어들을 순서대로 실행하여 TVM을 빌드합니다.

### 2.1 TVM 소스 코드로 이동
```bash
cd /work/tvm
```

### 2.2 빌드 디렉토리 생성 및 이동
```bash
# 기존 빌드 디렉토리가 있다면 삭제 후 새로 생성 (권장)
rm -rf build
mkdir -p build
cd build
```

### 2.3 config.cmake 파일 복사
```bash
cp ../cmake/config.cmake .
```

### 2.4 config.cmake 파일 수정
```bash
# USE_ETHOSN: 드라이버 경로 지정
set(USE_ETHOSN /usr/local/ethos-n)
# USE_LLVM: LLVM 경로 지정
set(USE_LLVM /opt/llvm/bin/llvm-config)
# USE_LIBBACKTRACE: OFF로 변경
set(USE_LIBBACKTRACE OFF)
```

실제 사용자의 tvm 디렉토리에 도커가 연결되어 있으므로, 사용자의 tvm 디렉토리에서 `build/cmake.config"에 직접 적용하는 것을 권장함.


### 2.5 CMake 실행 및 빌드
```bash
# CMake 설정
cmake ..

# 빌드 실행 (시간이 몇 분 소요됩니다)
make -j$(nproc)
```

### 서브 모델 오류 해결

```bash
# 서브모듈 초기화 및 업데이트
git submodule update --init --recursive

# dlpack.h 파일이 존재하는지 확인
ls -la 3rdparty/dlpack/include/dlpack/dlpack.h
```

✅ **성공 확인**: make 명령이 성공하면 Ethos-N 백엔드가 활성화된 TVM 라이브러리가 생성됩니다.

---

## 🔍 3단계: NPU 분할 결과 확인 (컨테이너 내부에서 실행)

이제 TVM을 사용하여 모델을 컴파일하고, NPU가 사용되도록 스케줄링되었는지 확인합니다.

### 3.1 확인용 Python 스크립트 생성 또는 복사

해당 model 폴더를 연결했다면, 로컬 pc에서 변경한 부분이 docker에 연동됩니다. 따라서 [python code](/scripts/tvm/check_partitioning.py) 파일을 해당 폴더에 복사하여서 실행할 수 있습니다.

```bash
# Python 경로 설정
export PYTHONPATH=/work/tvm/python:$PYTHONPATH

# 스크립트 실행
python3 check_partitioning.py
```

### 3.3 결과 해석

스크립트 실행 후, 터미널에 출력된 내용 중 `2. Partitioned Relay IR ...` 섹션을 확인하세요.

**✅ 성공 지표**:
- `@tvmgen_default_ethos_n_main_0` 과 같이 `ethos_n` 이름이 붙은 함수가 보임
- 함수 속성에 `Compiler = "ethos-n"` 라고 명시된 함수 블록이 존재

이것이 바로 모델의 일부가 NPU에서 실행되도록 성공적으로 분할 및 스케줄링되었음을 의미합니다.

original_ir_txt와 partitioned_ir_txt를 비교함으로써 tvm에서 ethos-n npu에 할당되었는지 아닌지를 확인할 수 있습니다.

---

## 📚 참고 자료

- **TVM 공식 문서**: https://tvm.apache.org/docs/
- **Ethos-N 백엔드**: https://tvm.apache.org/docs/deploy/ethos-n.html
- **Relay IR 가이드**: https://tvm.apache.org/docs/arch/relay_intro.html