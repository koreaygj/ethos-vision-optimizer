"""
TVM Ethos-N Partitioning Analysis Tool
========================================

이 도구는 TFLite 모델을 분석하여 Ethos-N NPU로 오프로드 가능한 연산을 식별합니다.

사용법:
------
# 기본 분석
python check_partitioning.py model.tflite

# 상세 IR을 콘솔에도 출력
python check_partitioning.py model.tflite --verbose

# 결과 파일 이름 지정
python check_partitioning.py model.tflite --output my_analysis.txt

# CPU 전용 모드 (Ethos-N 파티셔닝 비활성화)
python check_partitioning.py model.tflite --skip-ethosn

# Ethos-N 변형 및 설정 지정
python check_partitioning.py model.tflite --ethosn-variant n78 --ethosn-tops 4

출력 파일:
---------
- result.txt                  : 분석 리포트 (통계, 요약)
- result_summary.json         : JSON 형식 요약
- result_original_ir.txt      : 원본 Relay IR
- result_partitioned_ir.txt   : 파티셔닝된 IR (Ethos-N 어노테이션 포함)

주의사항:
--------
- 이 도구는 분석 전용이며, 실제 Ethos-N 바이너리를 생성하지 않습니다
- 실제 NPU 실행은 Ethos-N 하드웨어 또는 ARM 공식 툴체인이 필요합니다
- 파티셔닝 결과는 "이론적으로 오프로드 가능한 연산"을 보여줍니다
"""

import argparse
import tvm
from tvm import relay
from tvm.contrib import utils
import tflite.Model
from datetime import datetime
import os
import json
import numpy as np
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('autotvm').setLevel(logging.ERROR)


def check_ethosn_support():
    """Ethos-N 타겟 지원 여부 확인"""
    try:
        tvm.target.Target("llvm")  # 기본 확인
        # Ethos-N 타겟은 에뮬레이터에서 직접 사용 불가
        return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='TVM Ethos-N 파티셔닝 및 컴파일 도구'
    )
    parser.add_argument('model_path', type=str, help='TFLite 모델 경로')
    parser.add_argument('--verbose', action='store_true', help='상세 IR 출력')
    parser.add_argument('--output', type=str, default='result.txt', 
                       help='결과 저장 경로')

    
    # Ethos-N 설정
    parser.add_argument('--ethosn-variant', type=str, default='n78',
                       choices=['n77', 'n78', 'n57', 'n37'],
                       help='Ethos-N variant')
    parser.add_argument('--ethosn-tops', type=int, default=4,
                       help='Ethos-N TOPS')
    parser.add_argument('--ethosn-ple-ratio', type=int, default=4,
                       help='Ethos-N PLE ratio')
    parser.add_argument('--skip-ethosn', action='store_true',
                       help='Ethos-N 파티셔닝 건너뛰기')
    
    args = parser.parse_args()
    
    output_file = open(args.output, 'w', encoding='utf-8')
    
    def print_and_save(message):
        print(message)
        output_file.write(message + '\n')
        output_file.flush()
    
    # --- 헤더 ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_and_save("=" * 80)
    print_and_save(" TVM + Ethos-N Partitioning & Compilation Report")
    print_and_save(f" Generated: {timestamp}")
    print_and_save(f" Model: {args.model_path}")
    
    if not args.skip_ethosn:
        print_and_save(f" Ethos-N Config: {args.ethosn_variant}, "
                      f"TOPS={args.ethosn_tops}, PLE={args.ethosn_ple_ratio}")
    else:
        print_and_save(" Mode: CPU only (Ethos-N disabled)")
    
    print_and_save("=" * 80)
    print_and_save("")
    
    # --- 1. 모델 로드 ---
    print_and_save("[1/3] 모델 로딩...")
    
    try:
        with open(args.model_path, 'rb') as f:
            tflite_model_buf = f.read()

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
        subgraph = tflite_model.Subgraphs(0)
        input_idx = subgraph.Inputs(0)
        input_tensor = subgraph.Tensors(input_idx)
        input_name = input_tensor.Name().decode('utf-8')
        input_shape = tuple(input_tensor.ShapeAsNumpy())
        input_type = input_tensor.Type()
        
        type_mapping = {
            0: "float32", 1: "float16", 2: "int32",
            3: "uint8", 4: "int64", 9: "int8"
        }
        dtype = type_mapping.get(input_type, "float32")
        
        shape_dict = {input_name: input_shape}
        dtype_dict = {input_name: dtype}
        
        print_and_save(f"  입력 이름: {input_name}")
        print_and_save(f"  입력 shape: {input_shape}")
        print_and_save(f"  데이터 타입: {dtype}")

        mod, params = relay.frontend.from_tflite(
            tflite_model, 
            shape_dict=shape_dict, 
            dtype_dict=dtype_dict
        )
        print_and_save("  ✓ 모델 로드 성공\n")

    except Exception as e:
        print_and_save(f"  ✗ 모델 로드 실패: {e}")
        import traceback
        print_and_save(traceback.format_exc())
        output_file.close()
        return

    # Original IR 저장
    try:
        original_ir = mod.astext(show_meta_data=False)
        ir_output = args.output.replace('.txt', '_original_ir.txt')
        with open(ir_output, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" Original Relay IR\n")
            f.write("=" * 80 + "\n\n")
            f.write(original_ir)
        print_and_save(f"  Original IR 저장: {ir_output}")
    except Exception as e:
        print_and_save(f"  ! IR 저장 실패 (무시됨): {e}")
    
    if args.verbose:
        print_and_save("\n" + "=" * 80)
        print_and_save(" Original Relay IR ")
        print_and_save("=" * 80)
        print_and_save(original_ir)
        print_and_save("")

    # --- 2. 파티셔닝 ---
    print_and_save("[2/3] 최적화 및 파티셔닝...")
    
    partitioning_success = False
    ethos_n_count = 0
    
    try:
        with tvm.transform.PassContext(opt_level=3):
            mod = relay.transform.InferType()(mod)
            
            if not args.skip_ethosn:
                print_and_save("  Ethos-N 파티셔닝 시도 중...")
                
                try:
                    mod = relay.transform.AnnotateTarget(
                        "ethos-n",
                        include_non_call_ops=False
                    )(mod)
                    mod = relay.transform.MergeCompilerRegions()(mod)
                    mod = relay.transform.PartitionGraph()(mod)
                    
                    partitioned_ir = mod.astext(show_meta_data=False)
                    ethos_n_count = partitioned_ir.count('Compiler="ethos-n"')
                    
                    if ethos_n_count > 0:
                        print_and_save(f"  ✓ 파티셔닝 성공 ({ethos_n_count}개 함수가 Ethos-N으로 오프로드됨)")
                        partitioning_success = True
                    else:
                        print_and_save("  ! 파티셔닝 실패: Ethos-N으로 오프로드된 함수 없음")
                        args.skip_ethosn = True
                    
                except Exception as e:
                    print_and_save(f"  ! 파티셔닝 실패: {str(e)[:150]}")
                    print_and_save("  ! CPU 전용 모드로 전환")
                    args.skip_ethosn = True
            else:
                print_and_save("  CPU 전용 모드 (파티셔닝 건너뜀)")
        
        print_and_save("  ✓ 최적화 완료\n")
        
    except Exception as e:
        print_and_save(f"  ✗ 최적화 실패: {e}")
        import traceback
        print_and_save(traceback.format_exc())
        output_file.close()
        return

    # --- 3. 파티셔닝 통계 ---
    if partitioning_success:
        print_and_save("=" * 80)
        print_and_save(" Partitioning Statistics ")
        print_and_save("=" * 80)
        
        partitioned_ir = mod.astext(show_meta_data=False)
        
        print_and_save(f"Ethos-N 함수 개수: {ethos_n_count}")
        
        operations = {
            'qnn.conv2d': partitioned_ir.count('qnn.conv2d'),
            'qnn.add': partitioned_ir.count('qnn.add'),
            'qnn.concatenate': partitioned_ir.count('qnn.concatenate'),
            'reshape': partitioned_ir.count('reshape('),
            'nn.max_pool2d': partitioned_ir.count('nn.max_pool2d'),
        }
        
        print_and_save("\n주요 연산 개수:")
        for op_name, count in operations.items():
            if count > 0:
                print_and_save(f"  - {op_name}: {count}")
        
        # Partitioned IR 저장
        try:
            partitioned_ir_output = args.output.replace('.txt', '_partitioned_ir.txt')
            with open(partitioned_ir_output, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(" Partitioned Relay IR (with Ethos-N annotations)\n")
                f.write("=" * 80 + "\n\n")
                f.write(partitioned_ir)
            print_and_save(f"\n  Partitioned IR 저장: {partitioned_ir_output}")
        except Exception as e:
            print_and_save(f"\n  ! Partitioned IR 저장 실패 (무시됨): {e}")
        
        if args.verbose:
            print_and_save("\n" + "=" * 80)
            print_and_save(" Partitioned Relay IR ")
            print_and_save("=" * 80)
            print_and_save(partitioned_ir)
        
        print_and_save("")

    # --- 4. JSON 요약 ---
    try:
        json_output = args.output.replace('.txt', '_summary.json')
        summary = {
            'timestamp': timestamp,
            'model_path': args.model_path,
            'input_info': {
                'name': input_name,
                'shape': [int(x) for x in input_shape],
                'dtype': dtype
            },
            'partitioning': {
                'enabled': not args.skip_ethosn,
                'success': partitioning_success,
                'ethos_n_functions': ethos_n_count if partitioning_success else 0
            }
        }
        
        with open(json_output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print_and_save(f"\n  JSON 요약: {json_output}")
        
    except Exception as e:
        print_and_save(f"\nJSON 저장 오류 (무시됨): {e}")

    # --- 완료 ---
    print_and_save("\n" + "=" * 80)
    print_and_save("[3/3] 완료!")
    print_and_save("=" * 80)
    print_and_save(f"\n결과 파일:")
    print_and_save(f"  - 분석 리포트: {args.output}")
    print_and_save(f"  - JSON 요약: {json_output}")
    
    if partitioning_success:
        print_and_save(f"  - Original IR: {ir_output}")
        print_and_save(f"  - Partitioned IR: {partitioned_ir_output}")
        print_and_save(f"\n✓ {ethos_n_count}개 함수가 Ethos-N으로 오프로드 가능")
        print_and_save("  (실제 실행은 Ethos-N 하드웨어 또는 ARM 공식 툴체인 필요)")
    else:
        print_and_save(f"  - Original IR: {ir_output}")
    
    print_and_save("")
    
    output_file.close()


if __name__ == '__main__':
    main()