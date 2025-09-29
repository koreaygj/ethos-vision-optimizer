#!/usr/bin/env python3
"""
YOLOv11 Primitive Operator Analyzer v2
복합 모듈(C2PSA, C2f 등)을 primitive operator로 분해하여 분석하고 Markdown 리포트 생성
"""

import torch
import torch.nn as nn
import argparse
from collections import Counter, defaultdict
from ultralytics import YOLO
import sys
import json
from pathlib import Path
from datetime import datetime

class EthosNCompatibilityChecker:
    """Ethos-N NPU 호환성 체크 클래스"""

    def __init__(self):
        self.ethos_n_data = self._load_ethos_n_specs()

    def _load_ethos_n_specs(self):
        """Ethos-N 지원 operator 정보 로드"""
        try:
            script_dir = Path(__file__).parent.parent.parent
            json_path = script_dir / "docs" / "ethos-n" / "ethos_n_supported_ops.json"

            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Ethos-N 스펙 파일 로드 실패: {e}")
            return None

    def is_supported(self, operator_name):
        """Operator가 Ethos-N에서 지원되는지 확인"""
        if not self.ethos_n_data:
            return "❓ 확인 불가"

        supported_ops = self.ethos_n_data.get("supported_operators", [])

        # PyTorch operator를 Ethos-N operator로 매핑
        pytorch_to_ethos_mapping = {
            'Conv2d': 'Conv2d',
            'Linear': 'Linear',
            'ReLU': 'ReLU',
            'LeakyReLU': 'LeakyReLU',
            'MaxPool2d': 'MaxPool2d',
            'AvgPool2d': 'AvgPool2d',
            'Add': 'Add',
            'Multiply': 'Mul',
            'Concatenate': 'Concat',
            'Split': 'Split',
            'Sigmoid': 'Sigmoid',
            'Tanh': 'Tanh',
            'Reshape': 'Reshape',
            'Flatten': 'Reshape',
            'Upsample': 'Resize',
            'Interpolate': 'Resize',
            'ConvTranspose2d': 'ConvTranspose2d',
            'Transpose': 'Transpose',
            # Functional 버전들
            'ReLU_F': 'ReLU',
            'LeakyReLU_F': 'LeakyReLU',
            'Conv2d_F': 'Conv2d',
            'MaxPool2d_F': 'MaxPool2d',
            'AvgPool2d_F': 'AvgPool2d'
        }

        ethos_op = pytorch_to_ethos_mapping.get(operator_name)

        if ethos_op in supported_ops:
            return "✅ 완전 지원"
        elif operator_name == 'BatchNorm2d':
            return "🔄 Conv 융합됨"
        elif operator_name in ['SiLU', 'GELU', 'Mish', 'Softmax', 'LayerNorm', 'GroupNorm', 'MatMul', 'SiLU_F', 'GELU_F', 'Softmax_F']:
            return "❌ 미지원"
        else:
            return "❓ 확인 필요"

    def get_operator_constraints(self, operator_name):
        """Operator의 제약 조건 반환"""
        if not self.ethos_n_data:
            return None

        constraints = self.ethos_n_data.get("operator_constraints", {})
        pytorch_to_ethos_mapping = {
            'Conv2d': 'Conv2d',
            'MaxPool2d': 'MaxPool2d',
            'AvgPool2d': 'AvgPool2d'
        }

        ethos_op = pytorch_to_ethos_mapping.get(operator_name)
        return constraints.get(ethos_op)

class PrimitiveOperatorExtractor:
    """모듈을 primitive operator로 분해하는 클래스"""

    def __init__(self):
        self.primitive_ops = Counter()
        self.module_breakdown = defaultdict(list)
        self.traced_ops = []
        self.compatibility_checker = EthosNCompatibilityChecker()

    def extract_primitives(self, module, prefix=""):
        """재귀적으로 모듈을 primitive operator로 분해"""

        # Primitive operators (더 이상 분해되지 않는 연산자들)
        primitive_types = {
            nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm,
            nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Mish,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
            nn.Dropout, nn.Dropout2d, nn.Upsample, nn.Sigmoid, nn.Tanh,
            nn.Softmax, nn.LogSoftmax, nn.Flatten, nn.Identity, nn.ConvTranspose2d
        }

        # Swish 활성화 함수 확인 (PyTorch 버전에 따라 다름)
        try:
            primitive_types.add(nn.Swish)
        except AttributeError:
            pass  # Swish가 없는 PyTorch 버전

        module_type = type(module).__name__

        # 현재 모듈이 primitive인지 확인
        if type(module) in primitive_types:
            self.primitive_ops[module_type] += 1
            self.module_breakdown[prefix].append(module_type)
            return

        # 특별한 경우: Functional operations
        if hasattr(module, 'forward'):
            # 모듈의 forward 메서드를 분석하여 functional ops 찾기
            self._analyze_forward_functional(module, prefix)

        # 하위 모듈 재귀 분석
        children = list(module.named_children())
        if children:
            for name, child in children:
                child_prefix = f"{prefix}.{name}" if prefix else name
                self.extract_primitives(child, child_prefix)
        else:
            # Leaf module이지만 primitive가 아닌 경우
            if module_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                self.primitive_ops[f"Unknown_{module_type}"] += 1
                self.module_breakdown[prefix].append(f"Unknown_{module_type}")

    def _analyze_forward_functional(self, module, prefix):
        """Forward 메서드에서 functional operations 분석"""
        import inspect

        try:
            source = inspect.getsource(module.forward)

            # 일반적인 functional operations 패턴 매칭
            functional_patterns = {
                'torch.cat': 'Concatenate',
                'torch.split': 'Split',
                'torch.chunk': 'Chunk',
                'torch.add': 'Add',
                'torch.mul': 'Multiply',
                'torch.matmul': 'MatMul',
                'F.relu': 'ReLU_F',
                'F.leaky_relu': 'LeakyReLU_F',
                'F.silu': 'SiLU_F',
                'F.gelu': 'GELU_F',
                'F.softmax': 'Softmax_F',
                'F.interpolate': 'Interpolate',
                'F.max_pool2d': 'MaxPool2d_F',
                'F.avg_pool2d': 'AvgPool2d_F',
                'F.conv2d': 'Conv2d_F',
                'F.linear': 'Linear_F'
            }

            for pattern, op_name in functional_patterns.items():
                if pattern in source:
                    self.primitive_ops[op_name] += source.count(pattern)
                    self.module_breakdown[prefix].append(op_name)

        except (OSError, TypeError):
            # Source code를 가져올 수 없는 경우
            pass

def analyze_primitive_operators(model_path, trace_forward=False, save_report=True):
    """모델의 primitive operator 구성을 분석하고 Markdown 리포트 생성"""

    print(f"🔬 Primitive Operator 분석 시작: {model_path}")
    print("=" * 70)

    # 리포트 내용을 저장할 리스트
    report_lines = []

    # 헤더 정보
    model_name = Path(model_path).name
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_lines.extend([
        f"# Primitive Operator Analysis Report",
        f"",
        f"**Model**: `{model_name}`",
        f"**Analysis Date**: {timestamp}",
        f"**Tool**: YOLOv11 Primitive Operator Analyzer v2",
        f"",
        f"---",
        f""
    ])

    try:
        # 모델 로드
        model = YOLO(model_path)
        pytorch_model = model.model

        # Primitive operator 추출기
        extractor = PrimitiveOperatorExtractor()

        # Ethos-N 호환성 정보 출력
        if extractor.compatibility_checker.ethos_n_data:
            print(f"✅ Ethos-N 스펙 파일 로드 완료")
            supported_ops = extractor.compatibility_checker.ethos_n_data.get("supported_operators", [])
            print(f"📋 지원되는 operator: {len(supported_ops)}개")
        else:
            print(f"⚠️ Ethos-N 스펙 파일 로드 실패 - 기본 호환성 정보 사용")

        print("📋 모듈별 Primitive Operator 분해:")
        print("-" * 70)

        # 모듈 분석 섹션 시작
        report_lines.extend([
            f"## 🔍 Module-Level Primitive Operator Breakdown",
            f"",
            f"| Module | Type | Conv2d | BatchNorm | Activations | Other Ops | Parameters |",
            f"|--------|------|--------|-----------|-------------|-----------|------------|"
        ])

        composite_modules_info = []

        # 모든 모듈을 primitive operator로 분해
        for name, module in pytorch_model.named_modules():
            if name:  # 루트 모듈 제외
                module_type = type(module).__name__

                # 해당 모듈만 분석
                module_extractor = PrimitiveOperatorExtractor()
                module_extractor.extract_primitives(module, name)

                # 파라미터 수 계산
                total_params = sum(p.numel() for p in module.parameters())

                if module_extractor.primitive_ops:
                    conv_count = module_extractor.primitive_ops.get('Conv2d', 0)
                    bn_count = module_extractor.primitive_ops.get('BatchNorm2d', 0)
                    activation_count = sum(module_extractor.primitive_ops.get(act, 0)
                                         for act in ['SiLU', 'ReLU', 'LeakyReLU', 'GELU'])
                    other_count = sum(v for k, v in module_extractor.primitive_ops.items()
                                    if k not in ['Conv2d', 'BatchNorm2d', 'SiLU', 'ReLU', 'LeakyReLU', 'GELU'])

                    # 복합 모듈만 상세 정보 저장
                    if module_type in ['C2f', 'C3k2', 'C2PSA', 'SPPF', 'Detect']:
                        composite_modules_info.append({
                            'name': name,
                            'type': module_type,
                            'ops': dict(module_extractor.primitive_ops),
                            'params': total_params
                        })

                    # 콘솔 출력
                    print(f"\n🔍 분석 중: {name} ({module_type})")
                    for op_type, count in module_extractor.primitive_ops.items():
                        print(f"  └─ {op_type}: {count}개")
                        extractor.primitive_ops[op_type] += count

                    # Markdown 테이블에 추가 (복합 모듈만)
                    if module_type in ['C2f', 'C3k2', 'C2PSA', 'SPPF', 'Detect']:
                        report_lines.append(
                            f"| `{name}` | {module_type} | {conv_count} | {bn_count} | {activation_count} | {other_count} | {total_params:,} |"
                        )

        # Forward pass tracing (선택적)
        if trace_forward:
            print(f"\n🔬 Forward Pass Tracing:")
            print("-" * 70)
            traced_ops = trace_forward_pass(pytorch_model)
            for op in traced_ops:
                print(f"  └─ {op}")
                extractor.primitive_ops[f"Traced_{op}"] += 1

        # 전체 통계
        print(f"\n📊 Primitive Operator 통계:")
        print("-" * 70)

        total_primitives = sum(extractor.primitive_ops.values())

        # 통계 테이블 생성
        report_lines.extend([
            f"",
            f"## 📊 Overall Primitive Operator Statistics",
            f"",
            f"**Total Primitive Operators**: {total_primitives:,}",
            f"",
            f"| Operator Type | Count | Percentage | NPU Support |",
            f"|---------------|-------|------------|-------------|"
        ])

        for op_type, count in extractor.primitive_ops.most_common():
            percentage = (count / total_primitives) * 100
            status = extractor.compatibility_checker.is_supported(op_type)
            print(f"{op_type:25} | {count:>4}개 | {percentage:>6.2f}% | {status}")

            # Markdown 테이블에 추가
            report_lines.append(f"| {op_type} | {count} | {percentage:.1f}% | {status} |")

        print(f"\n총 Primitive Operator 수: {total_primitives:,}개")

        # NPU 호환성 분석
        compatibility_info = analyze_primitive_npu_compatibility(extractor.primitive_ops, report_lines)

        # 복합 모듈 분석
        analyze_composite_modules_md(composite_modules_info, report_lines)

        # 리포트 저장
        if save_report:
            save_analysis_report(model_path, report_lines, extractor.primitive_ops, compatibility_info)

        return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_primitive_npu_compatibility(primitive_ops, report_lines):
    """Primitive operator 수준에서 NPU 호환성 분석 (Ethos-N JSON 기반)"""

    print(f"\n🎯 Primitive Operator NPU 호환성 (Ethos-N 기반):")
    print("-" * 70)

    # Ethos-N 호환성 체커 생성
    compatibility_checker = EthosNCompatibilityChecker()

    # Ethos-N 스펙 파일이 로드되었는지 확인
    if compatibility_checker.ethos_n_data:
        supported_ops = compatibility_checker.ethos_n_data.get("supported_operators", [])
        print(f"📋 Ethos-N 지원 operator 목록: {', '.join(supported_ops)}")
        print("-" * 70)

    compatible_count = 0
    conditional_count = 0
    incompatible_count = 0
    unknown_count = 0
    total_ops = sum(primitive_ops.values())

    # Markdown에 NPU 호환성 섹션 추가
    if compatibility_checker.ethos_n_data:
        device_info = compatibility_checker.ethos_n_data.get("npu_device", "Arm Ethos-N")
        tensor_reqs = compatibility_checker.ethos_n_data.get("tensor_requirements", {})

        report_lines.extend([
            f"",
            f"## 🎯 NPU Compatibility Analysis",
            f"",
            f"**NPU Device**: {device_info}",
            f"**Analysis Based On**: {compatibility_checker.ethos_n_data.get('description', 'Ethos-N supported operators')}",
            f"",
            f"### Tensor Requirements",
            f"- **Supported Data Types**: {', '.join(tensor_reqs.get('datatypes', []))}",
            f"- **Max Dimensions**: {tensor_reqs.get('max_dimensions', 'N/A')}",
            f"- **Max Batch Size**: {tensor_reqs.get('max_batch_size', 'N/A')}",
            f"- **Tensor Format**: {tensor_reqs.get('tensor_format', 'N/A')}",
            f"- **Quantization**: {tensor_reqs.get('quantization', 'N/A')}",
            f"",
            f"| Operator Type | Count | Percentage | NPU Support Status | Constraints |",
            f"|---------------|-------|------------|-------------------|-------------|"
        ])
    else:
        report_lines.extend([
            f"",
            f"## 🎯 NPU Compatibility Analysis",
            f"",
            f"⚠️ Ethos-N 스펙 파일을 로드할 수 없습니다.",
            f"",
            f"| Operator Type | Count | Percentage | NPU Support Status |",
            f"|---------------|-------|------------|-------------------|"
        ])

    for op_type, count in primitive_ops.items():
        status = compatibility_checker.is_supported(op_type)
        percentage = (count / total_ops) * 100
        constraints = compatibility_checker.get_operator_constraints(op_type)

        print(f"{op_type:25} | {count:>4}개 | {status}")

        # Markdown에 추가 (제약 조건 포함)
        if constraints:
            constraints_str = f"Kernel: {constraints.get('kernel_sizes', 'N/A')}, Stride: {constraints.get('strides', 'N/A')}"
            report_lines.append(f"| {op_type} | {count} | {percentage:.1f}% | {status} | {constraints_str} |")
        else:
            if compatibility_checker.ethos_n_data:
                report_lines.append(f"| {op_type} | {count} | {percentage:.1f}% | {status} | - |")
            else:
                report_lines.append(f"| {op_type} | {count} | {percentage:.1f}% | {status} |")

        if status.startswith('✅'):
            compatible_count += count
        elif status.startswith('🔄'):
            conditional_count += count
        elif status.startswith('❌'):
            incompatible_count += count
        else:
            unknown_count += count

    # 호환성 요약
    compatibility_rate = compatible_count / total_ops * 100
    conditional_rate = conditional_count / total_ops * 100
    incompatible_rate = incompatible_count / total_ops * 100
    overall_compatibility = (compatible_count + conditional_count * 0.5) / total_ops * 100

    print(f"\n📊 Primitive Level NPU 호환성:")
    print(f"✅ 완전 지원:     {compatible_count:>4}개 ({compatibility_rate:>5.1f}%)")
    print(f"⚠️ 조건부 지원:   {conditional_count:>4}개 ({conditional_rate:>5.1f}%)")
    print(f"❌ 미지원:       {incompatible_count:>4}개 ({incompatible_rate:>5.1f}%)")
    if unknown_count > 0:
        print(f"❓ 확인 필요:     {unknown_count:>4}개 ({(unknown_count/total_ops)*100:>5.1f}%)")
    print(f"\n🎯 전체 호환성: {overall_compatibility:.1f}%")

    # Markdown에 요약 추가
    report_lines.extend([
        f"",
        f"### Compatibility Summary",
        f"",
        f"| Support Level | Count | Percentage |",
        f"|---------------|-------|------------|",
        f"| ✅ Fully Supported | {compatible_count} | {compatibility_rate:.1f}% |",
        f"| ⚠️ Conditional Support | {conditional_count} | {conditional_rate:.1f}% |",
        f"| ❌ Not Supported | {incompatible_count} | {incompatible_rate:.1f}% |"
    ])

    if unknown_count > 0:
        report_lines.append(f"| ❓ Unknown | {unknown_count} | {(unknown_count/total_ops)*100:.1f}% |")

    report_lines.extend([
        f"",
        f"**Overall NPU Compatibility Score**: {overall_compatibility:.1f}%",
        f""
    ])

    return {
        'compatible': compatible_count,
        'conditional': conditional_count,
        'incompatible': incompatible_count,
        'unknown': unknown_count,
        'overall_score': overall_compatibility
    }

def analyze_composite_modules_md(modules_info, report_lines):
    """복합 모듈 분석을 Markdown에 추가"""

    report_lines.extend([
        f"## 🏗️ Composite Module Analysis",
        f"",
        f"Detailed breakdown of complex modules:",
        f""
    ])

    for module_info in modules_info:
        name = module_info['name']
        module_type = module_info['type']
        ops = module_info['ops']
        params = module_info['params']
        memory_mb = params * 4 / (1024**2)  # 4 bytes per float32

        report_lines.extend([
            f"### {name} ({module_type})",
            f"",
            f"- **Parameters**: {params:,}",
            f"- **Memory Usage**: {memory_mb:.2f} MB",
            f"",
            f"**Primitive Operators**:",
            f""
        ])

        for op_type, count in ops.items():
            report_lines.append(f"- {op_type}: {count}")

        report_lines.append(f"")

def save_analysis_report(model_path, report_lines, primitive_ops, compatibility_info):
    """분석 리포트를 Markdown 파일로 저장"""

    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # docs 디렉토리 확인 및 생성
    docs_dir = Path('docs/analysis')
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성
    report_filename = f"primitive_operator_analysis_{model_name}_{timestamp}.md"
    report_path = docs_dir / report_filename

    # 추가 요약 정보
    report_lines.extend([
        f"## 📈 Optimization Recommendations",
        f"",
        f"Based on the analysis, here are the key optimization recommendations:",
        f""
    ])

    # SiLU 최적화 권장사항
    silu_count = primitive_ops.get('SiLU', 0)
    if silu_count > 0:
        total_ops = sum(primitive_ops.values())
        silu_percentage = (silu_count / total_ops) * 100
        report_lines.extend([
            f"### 🔥 Priority 1: Replace SiLU Activations",
            f"",
            f"- **Impact**: {silu_count} SiLU operators ({silu_percentage:.1f}% of total)",
            f"- **Solution**: Replace with `LeakyReLU(negative_slope=0.1)`",
            f"- **Expected Improvement**: +{silu_percentage:.1f}% NPU compatibility",
            f""
        ])

    # BatchNorm 융합 권장사항
    bn_count = primitive_ops.get('BatchNorm2d', 0)
    conv_count = primitive_ops.get('Conv2d', 0)
    if bn_count > 0:
        report_lines.extend([
            f"### 🔄 Priority 2: BatchNorm-Conv Fusion",
            f"",
            f"- **Impact**: {bn_count} BatchNorm2d operators can be fused with Conv2d",
            f"- **Solution**: Enable BatchNorm fusion during model optimization",
            f"- **Expected Improvement**: Reduced memory usage and faster inference",
            f""
        ])

    # 전체 결론
    report_lines.extend([
        f"## 🎯 Conclusion",
        f"",
        f"- **Total Primitive Operators**: {sum(primitive_ops.values()):,}",
        f"- **NPU Compatibility Score**: {compatibility_info['overall_score']:.1f}%",
        f"- **Primary Optimization Target**: SiLU activation functions",
        f"- **Secondary Optimization**: BatchNorm-Conv fusion",
        f"",
        f"---",
        f"",
        f"*Analysis generated by YOLOv11 Primitive Operator Analyzer v2*"
    ])

    # 파일 저장
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n📄 분석 리포트 저장 완료: {report_path}")
        return report_path

    except Exception as e:
        print(f"❌ 리포트 저장 실패: {e}")
        return None

def trace_forward_pass(model):
    """Forward pass를 trace하여 실제 실행되는 operations 추적"""
    traced_ops = []

    # Hook을 사용한 operation tracing
    def trace_hook(module, input, output):
        op_name = type(module).__name__
        if op_name not in ['Sequential', 'ModuleList']:
            traced_ops.append(op_name)

    # 모든 모듈에 hook 등록
    hooks = []
    for module in model.modules():
        hook = module.register_forward_hook(trace_hook)
        hooks.append(hook)

    try:
        # Dummy input으로 forward pass 실행
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"⚠️ Forward pass tracing 실패: {e}")
    finally:
        # Hook 제거
        for hook in hooks:
            hook.remove()

    return traced_ops

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Primitive Operator Analyzer v2')
    parser.add_argument('model_path', help='Path to .pt model file')
    parser.add_argument('--trace', action='store_true', help='Enable forward pass tracing')
    parser.add_argument('--no-save', action='store_true', help='Disable report saving')

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model_path}")
        sys.exit(1)

    success = analyze_primitive_operators(
        args.model_path,
        trace_forward=args.trace,
        save_report=not args.no_save
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()