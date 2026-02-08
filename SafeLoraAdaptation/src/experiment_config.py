# Copyright 2024 SafeLoRA Quantized Project
# Centralized experiment configuration using Pydantic

"""
Centralized Configuration for All Experiments

This file contains all configuration variables for the project.
Modify the values here or create instances with custom values to run experiments.

Usage:
    # Option 1: Use default configurations
    from src.experiment_config import BASELINE_CONFIG, SAFELORA_CONFIG, COMPARISON_CONFIG
    
    # Option 2: Create custom configuration
    from src.experiment_config import BaselineExperimentConfig
    config = BaselineExperimentConfig(
        base_model="meta-llama/Llama-2-7b-chat-hf",
        harmful_prompts_path="data/my_prompts.csv",
    )
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from pathlib import Path


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class ModelConfig(BaseModel):
    """Configuration for model loading."""
    
    base_model: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="Path to base model (HuggingFace hub or local path)"
    )
    aligned_model: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="Path to safety-aligned model (for SafeLoRA)"
    )
    peft_model: Optional[str] = Field(
        default=None,
        description="Path to PEFT/LoRA adapter (optional)"
    )
    
    class Config:
        extra = "forbid"


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""
    
    load_in_4bit: bool = Field(
        default=True,
        description="Load model in 4-bit quantization"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit quantization"
    )
    bnb_4bit_compute_dtype: str = Field(
        default="float16",
        description="Compute dtype for 4-bit (float16 or bfloat16)"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Quantization type (nf4 or fp4)"
    )
    use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization"
    )
    
    @model_validator(mode='after')
    def check_exclusive_quantization(self):
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit to True")
        return self
    
    @property
    def quantization_method(self) -> str:
        """Get the quantization method string."""
        if self.load_in_4bit:
            return "bitsandbytes_4bit"
        elif self.load_in_8bit:
            return "bitsandbytes_8bit"
        return "none"


class EvaluationConfig(BaseModel):
    """Configuration for safety evaluation."""
    
    harmful_prompts_path: str = Field(
        default="data/advbench_harmful_behaviors.csv",
        description="Path to harmful prompts file"
    )
    num_samples: Optional[int] = Field(
        default=None,
        description="Number of samples to evaluate (None = all)"
    )
    use_llama_guard: bool = Field(
        default=False,
        description="Use Llama Guard for safety evaluation"
    )
    use_gpt4: bool = Field(
        default=False,
        description="Use GPT-4 as judge for safety evaluation"
    )
    max_new_tokens: int = Field(
        default=512,
        description="Maximum tokens to generate for responses"
    )


class OutputConfig(BaseModel):
    """Configuration for output and saving."""
    
    output_dir: str = Field(
        default="results",
        description="Directory to save results"
    )
    save_adapter: bool = Field(
        default=False,
        description="Save the projected adapter (SafeLoRA only)"
    )
    verbose: bool = Field(
        default=True,
        description="Print detailed logs"
    )


# =============================================================================
# SAFELORA SPECIFIC CONFIGURATION
# =============================================================================

class SafeLoRAProjectionConfig(BaseModel):
    """Configuration for SafeLoRA projection."""
    
    select_layers_type: Literal["threshold", "number", "all", "auto"] = Field(
        default="number",
        description="Layer selection method"
    )
    num_proj_layers: int = Field(
        default=10,
        description="Number of layers to project (for 'number' selection)"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine threshold (for 'threshold' selection)"
    )
    projection_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Projection strength (0.0 to 1.0)"
    )
    compute_in_fp: bool = Field(
        default=True,
        description="Compute safety subspace in full precision"
    )
    cache_subspace_path: Optional[str] = Field(
        default=None,
        description="Path to cache/load safety subspace matrices"
    )


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

class BaselineExperimentConfig(BaseModel):
    """
    Complete configuration for baseline experiment.
    
    Example:
        config = BaselineExperimentConfig(
            model=ModelConfig(base_model="meta-llama/Llama-2-7b-chat-hf"),
            quantization=QuantizationConfig(load_in_4bit=True),
            evaluation=EvaluationConfig(num_samples=100),
            output=OutputConfig(output_dir="results/baseline"),
        )
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=lambda: OutputConfig(output_dir="results/baseline"))
    
    # Convenience properties for flat access
    @property
    def base_model(self) -> str:
        return self.model.base_model
    
    @property
    def peft_model(self) -> Optional[str]:
        return self.model.peft_model
    
    @property
    def load_in_4bit(self) -> bool:
        return self.quantization.load_in_4bit
    
    @property
    def load_in_8bit(self) -> bool:
        return self.quantization.load_in_8bit
    
    @property
    def harmful_prompts_path(self) -> str:
        return self.evaluation.harmful_prompts_path
    
    @property
    def num_samples(self) -> Optional[int]:
        return self.evaluation.num_samples
    
    @property
    def output_dir(self) -> str:
        return self.output.output_dir


class SafeLoRAExperimentConfig(BaseModel):
    """
    Complete configuration for SafeLoRA experiment.
    
    Example:
        config = SafeLoRAExperimentConfig(
            model=ModelConfig(
                base_model="meta-llama/Llama-2-7b-hf",
                aligned_model="meta-llama/Llama-2-7b-chat-hf",
                peft_model="path/to/adapter",
            ),
            projection=SafeLoRAProjectionConfig(num_proj_layers=15),
        )
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    projection: SafeLoRAProjectionConfig = Field(default_factory=SafeLoRAProjectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=lambda: OutputConfig(output_dir="results/safelora"))
    
    # Convenience properties
    @property
    def base_model(self) -> str:
        return self.model.base_model
    
    @property
    def aligned_model(self) -> str:
        return self.model.aligned_model
    
    @property
    def peft_model(self) -> Optional[str]:
        return self.model.peft_model


class ComparisonConfig(BaseModel):
    """
    Configuration for comparing baseline and SafeLoRA results.
    
    Example:
        config = ComparisonConfig(
            baseline_results=["results/baseline/*.json"],
            safelora_results=["results/safelora/*.json"],
        )
    """
    
    baseline_results: List[str] = Field(
        default=["results/baseline/*.json"],
        description="Glob patterns for baseline result files"
    )
    safelora_results: List[str] = Field(
        default=["results/safelora/*.json"],
        description="Glob patterns for SafeLoRA result files"
    )
    output_dir: str = Field(
        default="results/comparison",
        description="Directory to save comparison results"
    )


# =============================================================================
# DEFAULT CONFIGURATIONS - MODIFY THESE FOR YOUR EXPERIMENTS
# =============================================================================

# Default baseline experiment configuration
BASELINE_CONFIG = BaselineExperimentConfig(
    model=ModelConfig(
        base_model="meta-llama/Llama-2-7b-chat-hf",
        peft_model=None,  # Set to your LoRA adapter path
    ),
    quantization=QuantizationConfig(
        load_in_4bit=True,
        load_in_8bit=False,
    ),
    evaluation=EvaluationConfig(
        harmful_prompts_path="data/advbench_harmful_behaviors.csv",
        num_samples=100,  # Set to None for all samples
        use_llama_guard=False,
        use_gpt4=False,
        max_new_tokens=512,
    ),
    output=OutputConfig(
        output_dir="results/baseline",
        verbose=True,
    ),
)

# Default SafeLoRA experiment configuration
SAFELORA_CONFIG = SafeLoRAExperimentConfig(
    model=ModelConfig(
        base_model="meta-llama/Llama-2-7b-hf",  # Unaligned base
        aligned_model="meta-llama/Llama-2-7b-chat-hf",  # Safety-aligned
        peft_model=None,  # Set to your LoRA adapter path
    ),
    quantization=QuantizationConfig(
        load_in_4bit=True,
        load_in_8bit=False,
    ),
    projection=SafeLoRAProjectionConfig(
        select_layers_type="number",
        num_proj_layers=10,
        threshold=0.5,
        projection_strength=1.0,
        compute_in_fp=True,
        cache_subspace_path=None,
    ),
    evaluation=EvaluationConfig(
        harmful_prompts_path="data/advbench_harmful_behaviors.csv",
        num_samples=100,
        use_llama_guard=False,
        use_gpt4=False,
        max_new_tokens=512,
    ),
    output=OutputConfig(
        output_dir="results/safelora",
        save_adapter=False,
        verbose=True,
    ),
)

# Default comparison configuration
COMPARISON_CONFIG = ComparisonConfig(
    baseline_results=["results/baseline/*.json"],
    safelora_results=["results/safelora/*.json"],
    output_dir="results/comparison",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_baseline_config(**kwargs) -> BaselineExperimentConfig:
    """
    Create a baseline config with custom overrides.
    
    Example:
        config = create_baseline_config(
            base_model="my-model",
            num_samples=50,
            load_in_4bit=True,
        )
    """
    # Start with defaults
    model_kwargs = {}
    quant_kwargs = {}
    eval_kwargs = {}
    output_kwargs = {}
    
    # Map flat kwargs to nested structure
    model_fields = {'base_model', 'aligned_model', 'peft_model'}
    quant_fields = {'load_in_4bit', 'load_in_8bit', 'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'use_double_quant'}
    eval_fields = {'harmful_prompts_path', 'num_samples', 'use_llama_guard', 'use_gpt4', 'max_new_tokens'}
    output_fields = {'output_dir', 'save_adapter', 'verbose'}
    
    for key, value in kwargs.items():
        if key in model_fields:
            model_kwargs[key] = value
        elif key in quant_fields:
            quant_kwargs[key] = value
        elif key in eval_fields:
            eval_kwargs[key] = value
        elif key in output_fields:
            output_kwargs[key] = value
    
    return BaselineExperimentConfig(
        model=ModelConfig(**{**BASELINE_CONFIG.model.model_dump(), **model_kwargs}),
        quantization=QuantizationConfig(**{**BASELINE_CONFIG.quantization.model_dump(), **quant_kwargs}),
        evaluation=EvaluationConfig(**{**BASELINE_CONFIG.evaluation.model_dump(), **eval_kwargs}),
        output=OutputConfig(**{**BASELINE_CONFIG.output.model_dump(), **output_kwargs}),
    )


def create_safelora_config(**kwargs) -> SafeLoRAExperimentConfig:
    """
    Create a SafeLoRA config with custom overrides.
    
    Example:
        config = create_safelora_config(
            base_model="meta-llama/Llama-2-7b-hf",
            aligned_model="meta-llama/Llama-2-7b-chat-hf",
            num_proj_layers=15,
        )
    """
    model_kwargs = {}
    quant_kwargs = {}
    proj_kwargs = {}
    eval_kwargs = {}
    output_kwargs = {}
    
    model_fields = {'base_model', 'aligned_model', 'peft_model'}
    quant_fields = {'load_in_4bit', 'load_in_8bit', 'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type', 'use_double_quant'}
    proj_fields = {'select_layers_type', 'num_proj_layers', 'threshold', 'projection_strength', 'compute_in_fp', 'cache_subspace_path'}
    eval_fields = {'harmful_prompts_path', 'num_samples', 'use_llama_guard', 'use_gpt4', 'max_new_tokens'}
    output_fields = {'output_dir', 'save_adapter', 'verbose'}
    
    for key, value in kwargs.items():
        if key in model_fields:
            model_kwargs[key] = value
        elif key in quant_fields:
            quant_kwargs[key] = value
        elif key in proj_fields:
            proj_kwargs[key] = value
        elif key in eval_fields:
            eval_kwargs[key] = value
        elif key in output_fields:
            output_kwargs[key] = value
    
    return SafeLoRAExperimentConfig(
        model=ModelConfig(**{**SAFELORA_CONFIG.model.model_dump(), **model_kwargs}),
        quantization=QuantizationConfig(**{**SAFELORA_CONFIG.quantization.model_dump(), **quant_kwargs}),
        projection=SafeLoRAProjectionConfig(**{**SAFELORA_CONFIG.projection.model_dump(), **proj_kwargs}),
        evaluation=EvaluationConfig(**{**SAFELORA_CONFIG.evaluation.model_dump(), **eval_kwargs}),
        output=OutputConfig(**{**SAFELORA_CONFIG.output.model_dump(), **output_kwargs}),
    )

