#!/usr/bin/env python3
"""
NPU 최적화 YOLO 모델 훈련 스크립트 (모델 구조 확인 기능 추가)
data/dataset을 사용한 단계별 NPU 최적화 모델 훈련
"""

import os
import sys
import torch
import torch.nn as nn
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import json
import argparse
import yaml
from collections import Counter, defaultdict

class NPUOptimizedTrainer:
    def __init__(self, pretrained_weights=None, inspect_only=False):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_yaml = self.project_root / "data" / "dataset" / "data.yaml"
        self.models_dir = self.project_root / "models" / "train"
        self.results_dir = self.project_root / "results" / "training"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 사전 훈련 가중치 설정
        self.pretrained_weights = pretrained_weights or 'yolov11n.pt'
        
        # 검사 전용 모드
        self.inspect_only = inspect_only

        # 훈련 결과 저장용
        self.training_results = {}

        # Activation 검증용
        self.activation_stats = defaultdict(int)

        # 디바이스 설정 (MPS 지원)
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("🚀 MPS (Apple Silicon GPU) 가속 사용")
            torch.mps.set_per_process_memory_fraction(0.8)
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("🚀 CUDA GPU 가속 사용")
        else:
            self.device = 'cpu'
            print("⚠️ CPU만 사용 (느린 훈련)")

    def get_npu_models(self):
        """NPU 최적화 모델 목록 반환"""
        npu_models = {
            'level1': {
                'path': self.models_dir / "npu_level1_scales.yaml",
                'name': 'Level 1: Backbone C3k2 최적화',
                'description': 'Backbone C3k2 → C2f 변환',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '15-20%',
                'risk': 'Low'
            },
            'level2-relu': {
                'path': self.models_dir / "npu_level2_scales_backbone_relu.yaml",
                'name': 'Level 2: Backbone + Head 최적화 (ReLU)',
                'description': 'Backbone + Head C3k2 → C2f 변환 + ReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '25-30%',
                'risk': 'Low'
            },
            'level3-relu': {
                'path': self.models_dir / "npu_level3_scales_backbone_head_relu.yaml",
                'name': 'Level 3: + C2PSA 최적화 (ReLU)',
                'description': 'Backbone + Head + C2PSA → CSP + ReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '35-40%',
                'risk': 'Medium'
            },
            'level4-relu': {
                'path': self.models_dir / "npu_level4_activation_relu.yaml",
                'name': 'Level 4: 완전 최적화 (ReLU)',
                'description': 'All optimizations + ReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '40-45%',
                'risk': 'High'
            },
            'level2-leaky': {
                'path': self.models_dir / "npu_level2_scales_backbone_leaky.yaml",
                'name': 'Level 2: Backbone + Head 최적화 (LeakyReLU)',
                'description': 'Backbone + Head C3k2 → C2f 변환 + LeakyReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '25-30%',
                'risk': 'Low'
            },
            'level3-leaky': {
                'path': self.models_dir / "npu_level3_scales_backbone_head_leaked.yaml",
                'name': 'Level 3: + C2PSA 최적화 (LeakyReLU)',
                'description': 'Backbone + Head + C2PSA → CSP + LeakyReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '35-40%',
                'risk': 'Medium'
            },
            'level4-leaky': {
                'path': self.models_dir / "npu_level4_activation_leaked.yaml",
                'name': 'Level 4: 완전 최적화 (LeakyReLU)',
                'description': 'All optimizations + LeakyReLU',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '40-45%',
                'risk': 'High'
            }
        }
        return npu_models

    def inspect_yaml_config(self, yaml_path):
        """YAML 파일 내용 검사 및 출력"""
        print(f"\n{'='*80}")
        print(f"📄 YAML 설정 파일 검사: {yaml_path.name}")
        print(f"{'='*80}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"\n✅ YAML 파일 로드 성공")
            
            # 기본 정보
            print(f"\n🔍 기본 설정:")
            print(f"   - 클래스 수: {config.get('nc', 'N/A')}")
            if 'scales' in config:
                print(f"   - 스케일: {list(config['scales'].keys())}")
            
            # Backbone 검사
            if 'backbone' in config:
                print(f"\n🧱 Backbone 구조 ({len(config['backbone'])} 레이어):")
                self._print_layers(config['backbone'], 'Backbone')
            
            # Head 검사
            if 'head' in config:
                print(f"\n🎯 Head 구조 ({len(config['head'])} 레이어):")
                self._print_layers(config['head'], 'Head')
            
            # Activation 함수 분석
            self._analyze_activations(config)
            
            return config
            
        except Exception as e:
            print(f"❌ YAML 파일 로드 실패: {e}")
            return None

    def _print_layers(self, layers, section_name):
        """레이어 정보 출력"""
        activation_counts = {}
        module_counts = {}
        
        for idx, layer in enumerate(layers):
            # 딕셔너리 형식과 리스트 형식 모두 지원
            if isinstance(layer, dict):
                from_val = layer.get('from', 'N/A')
                module = layer.get('module', 'N/A')
                args = layer.get('args', [])
                activation = layer.get('activation', 'Default')
                repeats = layer.get('repeats', 1)
            else:
                # 리스트 형식 [from, repeats, module, args, ...]
                from_val = layer[0] if len(layer) > 0 else 'N/A'
                repeats = layer[1] if len(layer) > 1 else 1
                module = layer[2] if len(layer) > 2 else 'N/A'
                args = layer[3] if len(layer) > 3 else []
                activation = layer[4] if len(layer) > 4 else 'Default'
            
            # 통계 집계
            activation_counts[activation] = activation_counts.get(activation, 0) + 1
            module_counts[module] = module_counts.get(module, 0) + 1
            
            # 레이어 정보 출력 (처음 5개와 마지막 2개만)
            if idx < 5 or idx >= len(layers) - 2:
                print(f"   [{idx:2d}] {module:15s} | from: {str(from_val):8s} | "
                      f"repeats: {repeats} | activation: {activation}")
            elif idx == 5:
                print(f"   ... ({len(layers) - 7} more layers) ...")
        
        # 통계 출력
        print(f"\n   📊 {section_name} 통계:")
        print(f"      모듈 분포: {dict(module_counts)}")
        print(f"      활성화 함수: {dict(activation_counts)}")

    def _analyze_activations(self, config):
        """활성화 함수 사용 분석"""
        print(f"\n🔥 활성화 함수 분석:")
        
        all_activations = []
        
        # Backbone 활성화 함수 수집
        if 'backbone' in config:
            for layer in config['backbone']:
                if isinstance(layer, dict) and 'activation' in layer:
                    all_activations.append(layer['activation'])
                elif isinstance(layer, list) and len(layer) > 4:
                    all_activations.append(layer[4])
        
        # Head 활성화 함수 수집
        if 'head' in config:
            for layer in config['head']:
                if isinstance(layer, dict) and 'activation' in layer:
                    all_activations.append(layer['activation'])
                elif isinstance(layer, list) and len(layer) > 4:
                    all_activations.append(layer[4])
        
        # 통계 출력
        activation_stats = {}
        for act in all_activations:
            activation_stats[act] = activation_stats.get(act, 0) + 1
        
        total = len(all_activations)
        if total > 0:
            for act, count in sorted(activation_stats.items(), key=lambda x: -x[1]):
                percentage = (count / total) * 100
                print(f"   - {act}: {count}개 ({percentage:.1f}%)")
        else:
            print(f"   ⚠️ 명시적 활성화 함수 없음 (모듈 기본값 사용)")

    def inspect_model_structure(self, model, level_name):
        """초기화된 모델 구조 검사 및 출력"""
        print(f"\n{'='*80}")
        print(f"🔍 모델 구조 검사: {level_name}")
        print(f"{'='*80}")
        
        try:
            print(f"\n📐 모델 아키텍처:")
            print(model.model)
            
            print(f"\n📊 모델 통계:")
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            print(f"   - 총 파라미터: {total_params:,}")
            print(f"   - 학습 가능 파라미터: {trainable_params:,}")
            print(f"   - 모델 크기: {total_params * 4 / 1024 / 1024:.2f} MB (FP32 기준)")
            
            # Glenn의 방식으로 활성화 함수 검증
            activation_verification = self.verify_model_activations(model)

            print(f"\n🔥 활성화 함수 검증 결과:")
            if activation_verification['total_found'] > 0:
                print(f"   ✅ 총 {activation_verification['total_found']}개의 활성화 함수 발견")
                for act_type, count in activation_verification['stats'].items():
                    print(f"   - {act_type}: {count}개")
            else:
                print("   ⚠️ 명시적 활성화 함수를 찾지 못했습니다.")
                print("   전역 활성화 설정이 사용 중일 수 있습니다.")
            
            # 레이어 타입 통계
            print(f"\n📦 레이어 타입 분포:")
            layer_types = {}
            for name, module in model.model.named_modules():
                module_type = type(module).__name__
                if module_type != 'Sequential' and module_type != 'ModuleList':
                    layer_types[module_type] = layer_types.get(module_type, 0) + 1
            
            for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1])[:10]:
                print(f"   - {layer_type}: {count}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 구조 검사 실패: {e}")
            return False

    def train_model(self, model_config, level_name):
        """개별 모델 훈련 (사전 훈련 가중치 지원)"""
        print(f"\n🎯 {model_config['name']} 훈련 시작")
        print(f"📝 설명: {model_config['description']}")
        print(f"⚡ 예상 개선: {model_config['expected_improvement']}")
        print(f"⚠️ 위험도: {model_config['risk']}")
        print(f"🔄 사전 훈련 가중치: {self.pretrained_weights if self.pretrained_weights else '없음 (랜덤 초기화)'}")
        print("=" * 60)

        # 1. YAML 설정 검사
        yaml_config = self.inspect_yaml_config(model_config['path'])
        if not yaml_config:
            return None

        # 모델 초기화 및 가중치 로드
        try:
            # 2. 커스텀 아키텍처 로드
            model = YOLO(model_config['path'])
            print(f"\n✅ 커스텀 구조 로드: {model_config['path'].name}")

            # 3. 모델 구조 검사
            self.inspect_model_structure(model, level_name)

            # 검사 전용 모드면 여기서 종료
            if self.inspect_only:
                print(f"\n✅ 모델 검사 완료 (훈련 스킵)")
                return {
                    'level': level_name,
                    'model_name': model_config['name'],
                    'status': 'inspected',
                    'yaml_config': yaml_config
                }

            # 4. 사전 훈련 가중치 로드 시도
            if self.pretrained_weights:
                try:
                    pretrained_path = Path(self.pretrained_weights)

                    if not pretrained_path.is_absolute():
                        if pretrained_path.suffix == '.pt' and len(pretrained_path.parts) == 1:
                            model.load(self.pretrained_weights)
                        else:
                            pretrained_path = self.project_root / self.pretrained_weights
                            if pretrained_path.exists():
                                model.load(str(pretrained_path))
                            else:
                                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
                    else:
                        if pretrained_path.exists():
                            model.load(str(pretrained_path))
                        else:
                            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

                    print(f"\n✅ 사전 훈련 가중치 로드 성공: {self.pretrained_weights}")
                    print(f"   📈 수렴 속도 향상 및 더 나은 초기 성능 기대")

                except Exception as e:
                    print(f"\n⚠️ 사전 훈련 가중치 로드 실패: {e}")
                    print(f"   📝 랜덤 초기화로 진행 (훈련 시간이 더 오래 걸릴 수 있음)")
            else:
                print(f"\n📝 사전 훈련 가중치 없이 랜덤 초기화로 진행")

        except Exception as e:
            print(f"\n❌ 모델 초기화 실패: {e}")
            return None

        # 사전 훈련 가중치 사용 시 학습률 조정
        lr0 = 0.003 if self.pretrained_weights else 0.005

        # 훈련 파라미터 설정
        train_args = {
            'data': str(self.data_yaml),
            'epochs': model_config['epochs'],
            'batch': model_config['batch'],
            'imgsz': 640,
            'device': self.device,
            'project': str(self.results_dir),
            'name': f"{level_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,

            # NPU 친화적 훈련 설정
            'optimizer': 'SGD',
            'lr0': lr0,
            'lrf': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.001,
            'warmup_epochs': 5.0,
            'patience': 25,
            'amp': False,
            'deterministic': True,

            # NPU 호환 데이터 증강
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 0.0,
            'translate': 0.05,
            'scale': 0.2,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': '',
            'erasing': 0.0,
        }

        print(f"\n🏋️ 훈련 시작...")
        print(f"   📊 Epochs: {train_args['epochs']}")
        print(f"   📦 Batch: {train_args['batch']}")
        print(f"   🖥️ Device: {train_args['device']}")
        print(f"   📈 Learning Rate: {lr0} ({'Pretrained' if self.pretrained_weights else 'Random Init'})")

        start_time = time.time()

        try:
            results = model.train(**train_args)
            training_time = time.time() - start_time

            result_data = {
                'level': level_name,
                'model_name': model_config['name'],
                'model_path': str(model_config['path']),
                'pretrained_weights': self.pretrained_weights,
                'status': 'success',
                'training_time_seconds': training_time,
                'training_time_formatted': f"{training_time/3600:.1f}h {(training_time%3600)/60:.1f}m",
                'epochs_completed': train_args['epochs'],
                'batch_size': train_args['batch'],
                'device_used': train_args['device'],
                'learning_rate': lr0,
                'expected_improvement': model_config['expected_improvement'],
                'risk_level': model_config['risk'],
                'best_model_path': str(results.save_dir / 'weights' / 'best.pt') if hasattr(results, 'save_dir') else None,
                'last_model_path': str(results.save_dir / 'weights' / 'last.pt') if hasattr(results, 'save_dir') else None,
                'metrics': self.extract_metrics(results) if results else None,
                'yaml_config': yaml_config
            }

            print(f"\n✅ {level_name} 훈련 완료!")
            print(f"⏱️ 훈련 시간: {result_data['training_time_formatted']}")
            if result_data['best_model_path']:
                print(f"💾 최고 모델: {result_data['best_model_path']}")

            return result_data

        except Exception as e:
            print(f"\n❌ {level_name} 훈련 실패: {e}")
            return {
                'level': level_name,
                'model_name': model_config['name'],
                'pretrained_weights': self.pretrained_weights,
                'status': 'failed',
                'error': str(e),
                'training_time_seconds': time.time() - start_time,
                'yaml_config': yaml_config
            }

    def extract_metrics(self, results):
        """훈련 결과에서 메트릭 추출"""
        try:
            if hasattr(results, 'results_dict'):
                return results.results_dict
            elif hasattr(results, 'metrics'):
                return results.metrics
            else:
                return {"note": "Metrics extraction not available"}
        except:
            return {"note": "Metrics extraction failed"}

    def inspect_all_models(self):
        """모든 모델 구조만 검사 (훈련 없이)"""
        npu_models = self.get_npu_models()
        
        print(f"\n{'='*80}")
        print(f"🔍 모든 NPU 최적화 모델 검사 모드")
        print(f"{'='*80}")
        
        inspection_results = {}
        
        for level, model_config in npu_models.items():
            if not model_config['path'].exists():
                print(f"\n❌ 모델 파일 없음: {model_config['path']}")
                continue
            
            try:
                # YAML 검사
                yaml_config = self.inspect_yaml_config(model_config['path'])
                
                # 모델 초기화 및 구조 검사
                model = YOLO(model_config['path'])
                self.inspect_model_structure(model, level)
                
                inspection_results[level] = {
                    'name': model_config['name'],
                    'status': 'success',
                    'yaml_config': yaml_config
                }
                
            except Exception as e:
                print(f"\n❌ {level} 검사 실패: {e}")
                inspection_results[level] = {
                    'name': model_config['name'],
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 검사 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.results_dir / f"model_inspection_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(inspection_results, f, indent=2, default=str)
        
        print(f"\n\n📊 검사 결과 저장: {json_path}")
        return inspection_results

    def train_all_levels(self, levels=None):
        """모든 레벨 또는 지정된 레벨들 훈련"""
        npu_models = self.get_npu_models()

        if levels is None:
            levels = list(npu_models.keys())

        print(f"\n🚀 NPU 최적화 모델 훈련 시작")
        print(f"📂 데이터셋: {self.data_yaml}")
        print(f"🎯 훈련 레벨: {', '.join(levels)}")
        print(f"🔄 사전 훈련 가중치: {self.pretrained_weights if self.pretrained_weights else '없음'}")
        print(f"📊 결과 저장: {self.results_dir}")
        print(f"🔍 검사 전용 모드: {'예' if self.inspect_only else '아니오'}")
        print("=" * 80)

        for level in levels:
            if level not in npu_models:
                print(f"\n⚠️ 알 수 없는 레벨: {level}")
                continue

            model_config = npu_models[level]

            if not model_config['path'].exists():
                print(f"\n❌ 모델 파일 없음: {model_config['path']}")
                continue

            result = self.train_model(model_config, level)
            if result:
                self.training_results[level] = result

        if not self.inspect_only:
            self.save_training_summary()

    def train_single_level(self, level):
        """단일 레벨만 훈련"""
        self.train_all_levels(levels=[level])

    def save_training_summary(self):
        """훈련 결과 요약 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        json_path = self.results_dir / f"npu_training_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)

        md_path = self.results_dir / f"NPU_TRAINING_REPORT_{timestamp}.md"
        self.generate_markdown_report(md_path)

        print(f"\n📊 훈련 결과 저장:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📝 Report: {md_path}")

    def generate_markdown_report(self, output_path):
        """마크다운 훈련 리포트 생성"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# NPU 최적화 모델 훈련 결과 리포트

**생성일**: {timestamp}
**데이터셋**: Traffic Sign Detection (15 classes)
**총 훈련 레벨**: {len(self.training_results)}개
**사전 훈련 가중치**: {self.pretrained_weights if self.pretrained_weights else '없음 (랜덤 초기화)'}

## 📊 훈련 결과 요약

| Level | 모델명 | 상태 | 훈련시간 | 학습률 | 예상개선 | 위험도 |
|-------|--------|------|----------|--------|----------|--------|
"""

        for level, result in self.training_results.items():
            status_emoji = "✅" if result['status'] == 'success' else ("🔍" if result['status'] == 'inspected' else "❌")
            training_time = result.get('training_time_formatted', 'N/A')
            learning_rate = result.get('learning_rate', 'N/A')
            expected_improvement = result.get('expected_improvement', 'N/A')
            risk = result.get('risk_level', 'N/A')

            report += f"| {level.upper()} | {result['model_name']} | {status_emoji} {result['status']} | {training_time} | {learning_rate} | {expected_improvement} | {risk} |\n"

        successful_trainings = {k: v for k, v in self.training_results.items() if v['status'] == 'success'}

        if successful_trainings:
            report += f"\n## 🎯 성공한 훈련 상세 정보\n"
            for level, result in successful_trainings.items():
                report += f"\n### {level.upper()}: {result['model_name']}\n\n"
                report += f"- **사전 훈련 가중치**: {result.get('pretrained_weights', '없음')}\n"
                report += f"- **훈련 시간**: {result['training_time_formatted']}\n"
                report += f"- **최고 모델**: `{result.get('best_model_path', 'N/A')}`\n\n"

        report += "\n---\n**💡 참고**: 이 리포트는 자동 생성되었습니다.\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    def get_activation_name(self, activation) -> str:
        """Get human-readable activation function name"""
        if activation is None:
            return "None"

        activation_names = {
            nn.ReLU: "ReLU",
            nn.LeakyReLU: "LeakyReLU",
            nn.ELU: "ELU",
            nn.ReLU6: "ReLU6",
            nn.PReLU: "PReLU",
            nn.GELU: "GELU",
            nn.SiLU: "SiLU",
            nn.Sigmoid: "Sigmoid",
            nn.Tanh: "Tanh",
            nn.Hardswish: "Hardswish",
            nn.Mish: "Mish"
        }

        for act_type, name in activation_names.items():
            if isinstance(activation, act_type):
                return name

        return str(type(activation).__name__)

    def verify_model_activations(self, model) -> dict:
        """
        Verify activation functions in a loaded model
        Implementation of Glenn's suggestion from GitHub issue
        """
        print("🔍 Verifying activation functions in model...")

        # Reset stats
        self.activation_stats.clear()
        activation_layers = []

        def analyze_module(module, name="", level=0):
            """Recursively analyze activation functions"""
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                # Direct activation function detection
                if isinstance(child_module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.ReLU6,
                                           nn.PReLU, nn.GELU, nn.SiLU, nn.Sigmoid,
                                           nn.Tanh, nn.Hardswish, nn.Mish)):
                    activation_name = self.get_activation_name(child_module)
                    self.activation_stats[activation_name] += 1
                    activation_layers.append({
                        'name': full_name,
                        'type': activation_name,
                        'module': str(type(child_module).__name__),
                        'level': level
                    })

                    # Glenn's debug statement implementation
                    print(f"   ✅ Found {activation_name} at: {full_name}")

                # Check modules with embedded activations
                elif hasattr(child_module, 'act') or hasattr(child_module, 'activation'):
                    act_func = getattr(child_module, 'act', None) or getattr(child_module, 'activation', None)
                    if act_func is not None:
                        activation_name = self.get_activation_name(act_func)
                        self.activation_stats[activation_name] += 1
                        activation_layers.append({
                            'name': full_name,
                            'type': f"{str(type(child_module).__name__)}({activation_name})",
                            'module': str(type(child_module).__name__),
                            'level': level,
                            'embedded': True
                        })
                        print(f"   ✅ Found embedded {activation_name} in: {full_name}")

                # Recursively check children
                if len(list(child_module.children())) > 0:
                    analyze_module(child_module, full_name, level + 1)

        # Analyze the model
        analyze_module(model.model)

        # Print summary (Glenn's suggestion)
        print("\n📊 Activation Function Analysis Results:")
        print("=" * 50)

        if self.activation_stats:
            for activation, count in sorted(self.activation_stats.items()):
                print(f"   {activation}: {count} instances")

                # Glenn's verification check
                if activation == "ReLU":
                    print(f"   ✅ ReLU verification: {count > 0}")
                elif activation == "LeakyReLU":
                    print(f"   ✅ LeakyReLU verification: {count > 0}")
        else:
            print("   ⚠️ No explicit activation functions detected!")
            print("   This might indicate global activation setting is used.")

        # Check for global activation setting
        try:
            if hasattr(model, 'cfg') and model.cfg:
                if isinstance(model.cfg, dict) and 'act' in model.cfg:
                    global_act = model.cfg['act']
                    print(f"   🌐 Global activation detected: {global_act}")
                    self.activation_stats[f"Global-{global_act}"] += 1
        except Exception as e:
            print(f"   ⚠️ Could not check global activation: {e}")

        return {
            'stats': dict(self.activation_stats),
            'layers': activation_layers,
            'total_found': sum(self.activation_stats.values())
        }

    def compare_activation_training_results(self, results_dict: dict):
        """
        Compare training results to detect if different activations yield identical results
        (Glenn's main concern from the GitHub issue)
        """
        print("\n🔄 Comparing activation function training results...")

        # Group results by activation type
        activation_groups = defaultdict(list)

        for level, result in results_dict.items():
            if 'activation_verification' in result:
                primary_activation = max(result['activation_verification']['stats'].items(),
                                       key=lambda x: x[1], default=('Unknown', 0))[0]
                activation_groups[primary_activation].append((level, result))

        # Check for identical results (Glenn's issue)
        print("📊 Results by activation function:")
        for activation, level_results in activation_groups.items():
            print(f"\n   {activation}:")
            for level, result in level_results:
                status = result.get('status', 'unknown')
                training_time = result.get('training_time_formatted', 'N/A')
                print(f"     - {level}: {status} (time: {training_time})")

        # Warning for identical results
        if len(activation_groups) > 1:
            print("\n⚠️ Multiple activation types detected.")
            print("   If training results are identical, this might indicate:")
            print("   1. Global activation setting overriding individual settings")
            print("   2. Model configuration not properly applied")
            print("   3. Dataset characteristics don't emphasize activation differences")
            print("   4. Other hyperparameters dominating the training process")

def main():
    parser = argparse.ArgumentParser(description='NPU 최적화 YOLO 모델 훈련 (모델 검사 기능 추가)')
    parser.add_argument('--level', type=str, 
                       choices=['level1', 'level2-relu', 'level3-relu', 'level4-relu', 
                               'level2-leaky', 'level3-leaky', 'level4-leaky'],
                       help='특정 레벨만 훈련')
    parser.add_argument('--levels', type=str, nargs='+',
                       choices=['level1', 'level2-relu', 'level3-relu', 'level4-relu',
                               'level2-leaky', 'level3-leaky', 'level4-leaky'],
                       help='여러 레벨 선택')
    parser.add_argument('--pretrained', type=str, default='yolov11n.pt',
                       help='사전 훈련 가중치 경로 (기본값: yolov11n.pt)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='사전 훈련 가중치 사용하지 않음')
    parser.add_argument('--inspect', action='store_true',
                       help='모델 구조만 검사 (훈련 안함)')
    parser.add_argument('--inspect-all', action='store_true',
                       help='모든 모델 구조 검사 (훈련 안함)')

    args = parser.parse_args()

    # 사전 훈련 가중치 설정
    pretrained_weights = None if args.no_pretrained else args.pretrained
    
    # 검사 전용 모드 설정
    inspect_only = args.inspect or args.inspect_all

    trainer = NPUOptimizedTrainer(pretrained_weights=pretrained_weights, inspect_only=inspect_only)

    # 모드별 실행
    if args.inspect_all:
        print("🔍 모든 모델 검사 모드")
        trainer.inspect_all_models()
    elif args.inspect:
        if args.level:
            print(f"🔍 {args.level} 모델 검사 모드")
            trainer.train_single_level(args.level)
        elif args.levels:
            print(f"🔍 선택된 모델들 검사 모드: {', '.join(args.levels)}")
            trainer.train_all_levels(levels=args.levels)
        else:
            print("⚠️ --inspect 사용 시 --level 또는 --levels 지정 필요")
            print("   또는 --inspect-all 사용")
    else:
        # 일반 훈련 모드
        if pretrained_weights:
            print(f"📦 사전 훈련 가중치: {pretrained_weights}")
            print(f"   ⚡ 빠른 수렴 및 더 나은 초기 성능 기대")
        else:
            print(f"⚠️ 랜덤 초기화 모드 (훈련 시간이 더 오래 걸릴 수 있음)")

        if args.level:
            print(f"🎯 단일 레벨 훈련: {args.level}")
            trainer.train_single_level(args.level)
        elif args.levels:
            print(f"🎯 선택된 레벨들 훈련: {', '.join(args.levels)}")
            trainer.train_all_levels(levels=args.levels)
        else:
            print("🎯 모든 레벨 순차 훈련")
            trainer.train_all_levels()

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()