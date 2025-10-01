#!/usr/bin/env python3
"""
Activation Function Verification Tool
=====================================

This script verifies that activation functions are properly applied in YOLO models
and provides detailed analysis of the model architecture.

Usage:
    python scripts/validation/activation_verifier.py --model models/train/npu_level2_scales_backbone_relu.yaml
    python scripts/validation/activation_verifier.py --model models/train/npu_level3_scales_backbone_head_leaky.yaml --detailed
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import *
except ImportError:
    print("❌ Ultralytics not found. Please install: pip install ultralytics")
    sys.exit(1)


class ActivationVerifier:
    """Verify and analyze activation functions in YOLO models"""

    def __init__(self, model_config: str):
        """
        Initialize the verifier

        Args:
            model_config: Path to YAML model configuration
        """
        self.model_config = Path(model_config)
        self.model = None
        self.activation_stats = defaultdict(int)
        self.layer_analysis = []

    def load_model(self) -> bool:
        """Load YOLO model from configuration"""
        try:
            print(f"🔄 Loading model from: {self.model_config}")

            # Create model from YAML config
            self.model = YOLO(str(self.model_config))

            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {type(self.model.model)}")
            print(f"   Task: {self.model.task}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

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
            nn.Swish: "Swish",
            nn.Sigmoid: "Sigmoid",
            nn.Tanh: "Tanh",
            nn.Hardswish: "Hardswish",
            nn.Mish: "Mish"
        }

        for act_type, name in activation_names.items():
            if isinstance(activation, act_type):
                return name

        return str(type(activation).__name__)

    def analyze_layer_activations(self, module, name="", level=0):
        """Recursively analyze activation functions in model layers"""

        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is an activation function
            if isinstance(child_module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.ReLU6,
                                       nn.PReLU, nn.GELU, nn.SiLU, nn.Sigmoid,
                                       nn.Tanh, nn.Hardswish)):
                activation_name = self.get_activation_name(child_module)
                self.activation_stats[activation_name] += 1

                self.layer_analysis.append({
                    'name': full_name,
                    'type': 'Activation',
                    'activation': activation_name,
                    'level': level,
                    'module': str(type(child_module).__name__)
                })

            # Check for modules that contain activation functions
            elif hasattr(child_module, 'act') or hasattr(child_module, 'activation'):
                act_func = getattr(child_module, 'act', None) or getattr(child_module, 'activation', None)
                if act_func is not None:
                    activation_name = self.get_activation_name(act_func)
                    self.activation_stats[activation_name] += 1

                    self.layer_analysis.append({
                        'name': full_name,
                        'type': str(type(child_module).__name__),
                        'activation': activation_name,
                        'level': level,
                        'module': str(type(child_module).__name__)
                    })

            # Check Conv layers specifically
            elif isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                # Conv layers might not have explicit activation but check parent container
                parent_activation = "Inherited"

                self.layer_analysis.append({
                    'name': full_name,
                    'type': 'Conv',
                    'activation': parent_activation,
                    'level': level,
                    'module': str(type(child_module).__name__),
                    'in_channels': getattr(child_module, 'in_channels', 'N/A'),
                    'out_channels': getattr(child_module, 'out_channels', 'N/A'),
                    'kernel_size': getattr(child_module, 'kernel_size', 'N/A')
                })

            # Recursively analyze child modules
            if len(list(child_module.children())) > 0:
                self.analyze_layer_activations(child_module, full_name, level + 1)

    def verify_activations(self, detailed: bool = False) -> dict:
        """
        Verify activation functions in the model

        Args:
            detailed: Whether to show detailed layer-by-layer analysis

        Returns:
            Dictionary with verification results
        """
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return {}

        print("\n🔍 Analyzing model activations...")

        # Reset analysis
        self.activation_stats.clear()
        self.layer_analysis.clear()

        # Analyze the model
        self.analyze_layer_activations(self.model.model)

        # Print summary
        print("\n📊 Activation Function Summary:")
        print("=" * 50)

        if self.activation_stats:
            for activation, count in sorted(self.activation_stats.items()):
                print(f"   {activation}: {count} instances")
        else:
            print("   ⚠️ No explicit activation functions found!")
            print("   This might indicate:")
            print("   - Global activation setting is being used")
            print("   - Activation functions are embedded differently")

        # Detailed analysis
        if detailed and self.layer_analysis:
            print("\n🔍 Detailed Layer Analysis:")
            print("=" * 80)
            print(f"{'Layer Name':<30} {'Type':<15} {'Activation':<15} {'Details'}")
            print("-" * 80)

            for layer in self.layer_analysis:
                details = ""
                if 'in_channels' in layer:
                    details = f"{layer['in_channels']}→{layer['out_channels']}"

                print(f"{layer['name']:<30} {layer['type']:<15} {layer['activation']:<15} {details}")

        # Global activation check
        self._check_global_activation()

        return {
            'activation_stats': dict(self.activation_stats),
            'layer_analysis': self.layer_analysis,
            'total_layers': len(self.layer_analysis)
        }

    def _check_global_activation(self):
        """Check for global activation settings"""
        try:
            # Try to access model configuration
            if hasattr(self.model, 'cfg') and self.model.cfg:
                cfg = self.model.cfg
                if isinstance(cfg, dict):
                    if 'act' in cfg:
                        print(f"\n🌐 Global activation detected: {cfg['act']}")
                    elif 'activation' in cfg:
                        print(f"\n🌐 Global activation detected: {cfg['activation']}")

        except Exception as e:
            print(f"\n⚠️ Could not check global activation: {e}")

    def compare_with_reference(self, reference_activation: str):
        """
        Compare current model activations with reference

        Args:
            reference_activation: Expected activation function name
        """
        print(f"\n🔄 Comparing with reference activation: {reference_activation}")

        if reference_activation in self.activation_stats:
            count = self.activation_stats[reference_activation]
            print(f"✅ Found {count} instances of {reference_activation}")
        else:
            print(f"❌ No instances of {reference_activation} found!")
            print("   Available activations:", list(self.activation_stats.keys()))

    def generate_report(self, output_file: str = None):
        """Generate detailed verification report"""
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(f"# Activation Function Verification Report\n\n")
                f.write(f"**Model Config**: {self.model_config}\n")
                f.write(f"**Analysis Date**: {pd.Timestamp.now()}\n\n")

                f.write("## Activation Summary\n\n")
                for activation, count in sorted(self.activation_stats.items()):
                    f.write(f"- **{activation}**: {count} instances\n")

                f.write("\n## Detailed Layer Analysis\n\n")
                f.write("| Layer Name | Type | Activation | Details |\n")
                f.write("|------------|------|------------|----------|\n")

                for layer in self.layer_analysis:
                    details = ""
                    if 'in_channels' in layer:
                        details = f"{layer['in_channels']}→{layer['out_channels']}"

                    f.write(f"| {layer['name']} | {layer['type']} | {layer['activation']} | {details} |\n")

            print(f"📄 Report saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Verify activation functions in YOLO models")
    parser.add_argument("--model", type=str, required=True, help="Path to YAML model configuration")
    parser.add_argument("--detailed", action="store_true", help="Show detailed layer analysis")
    parser.add_argument("--reference", type=str, help="Reference activation to compare with (e.g., 'ReLU')")
    parser.add_argument("--output", type=str, help="Output report file path")

    args = parser.parse_args()

    # Verify model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    print("🚀 Activation Function Verification Tool")
    print("=" * 50)

    # Create verifier and analyze
    verifier = ActivationVerifier(args.model)

    if not verifier.load_model():
        sys.exit(1)

    # Perform verification
    results = verifier.verify_activations(detailed=args.detailed)

    # Compare with reference if provided
    if args.reference:
        verifier.compare_with_reference(args.reference)

    # Generate report if requested
    if args.output:
        verifier.generate_report(args.output)

    print("\n✅ Verification completed!")


if __name__ == "__main__":
    main()