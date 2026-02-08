# SafeLoRA for Quantized Models
# ECE 285 Project - Safety Alignment of Quantized Models

from .safe_lora_quantized import SafeLoRAQuantized
from .config import SafeLoRAQuantizedConfig
from .quantization_utils import (
    load_quantized_model,
    dequantize_layer_weights,
    get_quantization_config,
)
from .evaluation import SafetyEvaluator
from .experiment_config import (
    BASELINE_CONFIG,
    SAFELORA_CONFIG,
    COMPARISON_CONFIG,
    BaselineExperimentConfig,
    SafeLoRAExperimentConfig,
    ComparisonConfig,
    create_baseline_config,
    create_safelora_config,
)

__version__ = "0.1.0"
__all__ = [
    # Core SafeLoRA
    "SafeLoRAQuantized",
    "SafeLoRAQuantizedConfig", 
    "SafetyEvaluator",
    # Quantization utilities
    "load_quantized_model",
    "dequantize_layer_weights",
    "get_quantization_config",
    # Experiment configurations
    "BASELINE_CONFIG",
    "SAFELORA_CONFIG",
    "COMPARISON_CONFIG",
    "BaselineExperimentConfig",
    "SafeLoRAExperimentConfig",
    "ComparisonConfig",
    "create_baseline_config",
    "create_safelora_config",
]

