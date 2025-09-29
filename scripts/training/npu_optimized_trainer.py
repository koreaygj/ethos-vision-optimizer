#!/usr/bin/env python3
"""
NPU 최적화 YOLO 모델 훈련 스크립트 (사전 훈련 가중치 지원 버전)
data/dataset을 사용한 단계별 NPU 최적화 모델 훈련
"""

import os
import sys
import torch
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import json
import argparse

class NPUOptimizedTrainer:
    def __init__(self, pretrained_weights=None):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_yaml = self.project_root / "data" / "dataset" / "data.yaml"
        self.models_dir = self.project_root / "models" / "train"
        self.results_dir = self.project_root / "results" / "training"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 사전 훈련 가중치 설정
        self.pretrained_weights = pretrained_weights or 'yolov11n.pt'

        # 훈련 결과 저장용
        self.training_results = {}

        # 디바이스 설정 (MPS 지원)
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("🚀 MPS (Apple Silicon GPU) 가속 사용")
            # MPS 메모리 최적화
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
            'level2': {
                'path': self.models_dir / "npu_level2_scales_backbone.yaml",
                'name': 'Level 2: Backbone + Head 최적화',
                'description': 'Backbone + Head C3k2 → C2f 변환',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '25-30%',
                'risk': 'Low'
            },
            'level3': {
                'path': self.models_dir / "npu_level3_scales_backbone_head.yaml",
                'name': 'Level 3: + C2PSA 최적화',
                'description': 'Backbone + Head + C2PSA → CSP',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '35-40%',
                'risk': 'Medium'
            },
            'level4': {
                'path': self.models_dir / "npu_level4_full_optimization.yaml",
                'name': 'Level 4: 완전 최적화',
                'description': 'All optimizations + ConvTranspose2d',
                'epochs': 100,
                'batch': 16,
                'expected_improvement': '40-45%',
                'risk': 'High'
            }
        }
        return npu_models

    def train_model(self, model_config, level_name):
        """개별 모델 훈련 (사전 훈련 가중치 지원)"""
        print(f"\\n🎯 {model_config['name']} 훈련 시작")
        print(f"📝 설명: {model_config['description']}")
        print(f"⚡ 예상 개선: {model_config['expected_improvement']}")
        print(f"⚠️ 위험도: {model_config['risk']}")
        print(f"🔄 사전 훈련 가중치: {self.pretrained_weights if self.pretrained_weights else '없음 (랜덤 초기화)'}")
        print("=" * 60)

        # 모델 초기화 및 가중치 로드
        try:
            # 1. 커스텀 아키텍처 로드
            model = YOLO(model_config['path'])
            print(f"✅ 커스텀 구조 로드: {model_config['path'].name}")

            # 2. 사전 훈련 가중치 로드 시도
            if self.pretrained_weights:
                try:
                    pretrained_path = Path(self.pretrained_weights)

                    # 상대 경로인 경우 절대 경로로 변환
                    if not pretrained_path.is_absolute():
                        # Ultralytics 기본 모델들 (yolov11n.pt 등) 또는 프로젝트 내 경로
                        if pretrained_path.suffix == '.pt' and len(pretrained_path.parts) == 1:
                            # Ultralytics 기본 모델 (자동 다운로드)
                            model.load(self.pretrained_weights)
                        else:
                            # 프로젝트 내 상대 경로
                            pretrained_path = self.project_root / self.pretrained_weights
                            if pretrained_path.exists():
                                model.load(str(pretrained_path))
                            else:
                                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
                    else:
                        # 절대 경로
                        if pretrained_path.exists():
                            model.load(str(pretrained_path))
                        else:
                            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

                    print(f"✅ 사전 훈련 가중치 로드 성공: {self.pretrained_weights}")
                    print(f"   📈 수렴 속도 향상 및 더 나은 초기 성능 기대")

                except Exception as e:
                    print(f"⚠️ 사전 훈련 가중치 로드 실패: {e}")
                    print(f"   📝 랜덤 초기화로 진행 (훈련 시간이 더 오래 걸릴 수 있음)")
            else:
                print(f"📝 사전 훈련 가중치 없이 랜덤 초기화로 진행")

        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
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

            # NPU 친화적 훈련 설정 (사전 훈련 가중치 고려)
            'optimizer': 'SGD',          # 양자화 친화적
            'lr0': lr0,                  # 사전 훈련 시 낮은 학습률
            'lrf': 0.1,                  # 최종 학습률
            'momentum': 0.9,             # SGD 최적 모멘텀
            'weight_decay': 0.001,       # 강한 정규화
            'warmup_epochs': 5.0,        # 충분한 워밍업
            'patience': 25,              # 조기 종료
            'amp': False,                # Mixed Precision OFF (양자화 준비)
            'deterministic': True,       # 재현 가능한 결과

            # NPU 호환 데이터 증강
            'hsv_h': 0.01,       # 최소 Hue 변화
            'hsv_s': 0.5,        # 적당한 채도 변화
            'hsv_v': 0.3,        # 최소 밝기 변화
            'degrees': 0.0,      # 회전 비활성화
            'translate': 0.05,   # 최소 이동
            'scale': 0.2,        # 최소 스케일 변화
            'shear': 0.0,        # 전단 비활성화
            'perspective': 0.0,  # 원근 비활성화
            'flipud': 0.0,       # 상하반전 비활성화
            'fliplr': 0.5,       # 좌우반전만 유지
            'mosaic': 0.0,       # 모자이크 OFF
            'mixup': 0.0,        # MixUp OFF
            'copy_paste': 0.0,   # Copy-Paste OFF
            'auto_augment': '',  # 자동증강 OFF
            'erasing': 0.0,      # Random Erasing OFF
        }

        print(f"🏋️ 훈련 시작...")
        print(f"   📊 Epochs: {train_args['epochs']}")
        print(f"   📦 Batch: {train_args['batch']}")
        print(f"   🖥️ Device: {train_args['device']}")
        print(f"   📈 Learning Rate: {lr0} ({'Pretrained' if self.pretrained_weights else 'Random Init'})")

        start_time = time.time()

        try:
            # 훈련 실행
            results = model.train(**train_args)

            training_time = time.time() - start_time

            # 결과 저장
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
                'metrics': self.extract_metrics(results) if results else None
            }

            print(f"✅ {level_name} 훈련 완료!")
            print(f"⏱️ 훈련 시간: {result_data['training_time_formatted']}")
            if result_data['best_model_path']:
                print(f"💾 최고 모델: {result_data['best_model_path']}")

            return result_data

        except Exception as e:
            print(f"❌ {level_name} 훈련 실패: {e}")
            return {
                'level': level_name,
                'model_name': model_config['name'],
                'pretrained_weights': self.pretrained_weights,
                'status': 'failed',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
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

    def train_all_levels(self, levels=None):
        """모든 레벨 또는 지정된 레벨들 훈련"""
        npu_models = self.get_npu_models()

        if levels is None:
            levels = list(npu_models.keys())

        print(f"🚀 NPU 최적화 모델 훈련 시작")
        print(f"📂 데이터셋: {self.data_yaml}")
        print(f"🎯 훈련 레벨: {', '.join(levels)}")
        print(f"🔄 사전 훈련 가중치: {self.pretrained_weights if self.pretrained_weights else '없음'}")
        print(f"📊 결과 저장: {self.results_dir}")
        print("=" * 80)

        for level in levels:
            if level not in npu_models:
                print(f"⚠️ 알 수 없는 레벨: {level}")
                continue

            model_config = npu_models[level]

            # 모델 파일 존재 확인
            if not model_config['path'].exists():
                print(f"❌ 모델 파일 없음: {model_config['path']}")
                continue

            # 훈련 실행
            result = self.train_model(model_config, level)
            if result:
                self.training_results[level] = result

        # 전체 결과 저장
        self.save_training_summary()

    def train_single_level(self, level):
        """단일 레벨만 훈련"""
        self.train_all_levels(levels=[level])

    def save_training_summary(self):
        """훈련 결과 요약 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON 결과 저장
        json_path = self.results_dir / f"npu_training_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)

        # 마크다운 리포트 생성
        md_path = self.results_dir / f"NPU_TRAINING_REPORT_{timestamp}.md"
        self.generate_markdown_report(md_path)

        print(f"\\n📊 훈련 결과 저장:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📝 Report: {md_path}")

    def generate_markdown_report(self, output_path):
        """마크다운 훈련 리포트 생성"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# NPU 최적화 모델 훈련 결과 리포트 (사전 훈련 가중치 지원)

**생성일**: {timestamp}
**데이터셋**: Traffic Sign Detection (15 classes)
**총 훈련 레벨**: {len(self.training_results)}개
**사전 훈련 가중치**: {self.pretrained_weights if self.pretrained_weights else '없음 (랜덤 초기화)'}

## 📊 훈련 결과 요약

| Level | 모델명 | 상태 | 훈련시간 | 학습률 | 예상개선 | 위험도 | 모델 경로 |
|-------|--------|------|----------|--------|----------|--------|-----------|
"""

        for level, result in self.training_results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            training_time = result.get('training_time_formatted', 'N/A')
            learning_rate = result.get('learning_rate', 'N/A')
            expected_improvement = result.get('expected_improvement', 'N/A')
            risk = result.get('risk_level', 'N/A')
            model_path = Path(result.get('best_model_path', 'N/A')).name if result.get('best_model_path') else 'N/A'

            report += f"| {level.upper()} | {result['model_name']} | {status_emoji} {result['status']} | {training_time} | {learning_rate} | {expected_improvement} | {risk} | {model_path} |\\n"

        # 성공한 훈련들의 상세 정보
        successful_trainings = {k: v for k, v in self.training_results.items() if v['status'] == 'success'}

        if successful_trainings:
            report += f"""

## 🎯 성공한 훈련 상세 정보

"""
            for level, result in successful_trainings.items():
                report += f"""### {level.upper()}: {result['model_name']}

- **설명**: {result.get('description', 'N/A')}
- **사전 훈련 가중치**: {result.get('pretrained_weights', '없음')}
- **훈련 시간**: {result['training_time_formatted']}
- **학습률**: {result.get('learning_rate', 'N/A')}
- **완료 에폭**: {result['epochs_completed']}
- **배치 크기**: {result['batch_size']}
- **사용 디바이스**: {result['device_used']}
- **최고 모델**: `{result.get('best_model_path', 'N/A')}`
- **최종 모델**: `{result.get('last_model_path', 'N/A')}`

"""

        # 실패한 훈련들
        failed_trainings = {k: v for k, v in self.training_results.items() if v['status'] == 'failed'}

        if failed_trainings:
            report += f"""

## ❌ 실패한 훈련 정보

"""
            for level, result in failed_trainings.items():
                report += f"""### {level.upper()}: {result['model_name']}

- **오류**: {result.get('error', 'Unknown error')}
- **소요 시간**: {result['training_time_seconds']:.1f}초

"""

        report += f"""

## 🔧 훈련 설정

### NPU 최적화 훈련 파라미터
- **사전 훈련 가중치**: {self.pretrained_weights if self.pretrained_weights else '없음 (랜덤 초기화)'}
- **Optimizer**: SGD (양자화 친화적)
- **Learning Rate**: {'0.003 (사전 훈련)' if self.pretrained_weights else '0.005 (랜덤 초기화)'}
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
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    parser = argparse.ArgumentParser(description='NPU 최적화 YOLO 모델 훈련 (사전 훈련 가중치 지원)')
    parser.add_argument('--level', type=str, choices=['level1', 'level2', 'level3', 'level4'],
                       help='특정 레벨만 훈련 (기본값: 모든 레벨)')
    parser.add_argument('--levels', type=str, nargs='+',
                       choices=['level1', 'level2', 'level3', 'level4'],
                       help='여러 레벨 선택 (예: --levels level1 level2)')
    parser.add_argument('--pretrained', type=str, default='yolov11n.pt',
                       help='사전 훈련 가중치 경로 (기본값: yolov11n.pt)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='사전 훈련 가중치 사용하지 않음 (완전 랜덤 초기화)')

    args = parser.parse_args()

    # 사전 훈련 가중치 설정
    pretrained_weights = None if args.no_pretrained else args.pretrained

    trainer = NPUOptimizedTrainer(pretrained_weights=pretrained_weights)

    # 설정 정보 출력
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

    print("\\n🎉 훈련 완료!")

if __name__ == "__main__":
    main()