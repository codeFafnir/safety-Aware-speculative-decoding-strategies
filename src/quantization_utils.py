# Copyright 2024 SafeLoRA Quantized Project
# Utilities for loading and handling quantized models

import torch
from typing import Optional, Dict, Any, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings


def get_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytes quantization configuration.
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype (float16 or bfloat16)
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        use_double_quant: Whether to use nested quantization
    
    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def get_quantization_config(config) -> Optional[BitsAndBytesConfig]:
    """
    Create quantization config from SafeLoRAQuantizedConfig.
    
    Args:
        config: SafeLoRAQuantizedConfig instance
    
    Returns:
        BitsAndBytesConfig or None
    """
    if config.quantization_method == "bitsandbytes_4bit":
        return get_bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            use_double_quant=config.use_double_quant,
        )
    elif config.quantization_method == "bitsandbytes_8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def load_quantized_model(
    model_path: str,
    quantization_method: str = "bitsandbytes_4bit",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a quantized model with the specified quantization method.
    
    Args:
        model_path: Path to model (local or HuggingFace hub)
        quantization_method: One of "none", "bitsandbytes_4bit", "bitsandbytes_8bit", "gptq", "awq"
        device_map: Device mapping strategy
        torch_dtype: Default dtype for non-quantized weights
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **kwargs
    }
    
    if quantization_method == "bitsandbytes_4bit":
        bnb_config = get_bnb_config(load_in_4bit=True)
        model_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
    elif quantization_method == "bitsandbytes_8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
    elif quantization_method == "gptq":
        # GPTQ models are pre-quantized, load directly
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
    elif quantization_method == "awq":
        # AWQ models are pre-quantized, load directly
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
    elif quantization_method == "none":
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method}")
    
    return model, tokenizer


def dequantize_layer_weights(
    layer: torch.nn.Module,
    layer_name: str,
    quantization_method: str
) -> torch.Tensor:
    """
    Dequantize layer weights to full precision for safety subspace computation.
    
    Args:
        layer: The quantized layer module
        layer_name: Name of the layer (for logging)
        quantization_method: Quantization method used
    
    Returns:
        Dequantized weight tensor in float32
    """
    if quantization_method in ["bitsandbytes_4bit", "bitsandbytes_8bit"]:
        return _dequantize_bnb(layer, quantization_method)
    elif quantization_method == "gptq":
        return _dequantize_gptq(layer)
    elif quantization_method == "awq":
        return _dequantize_awq(layer)
    elif quantization_method == "none":
        return layer.weight.data.float()
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method}")


def _dequantize_bnb(layer: torch.nn.Module, method: str) -> torch.Tensor:
    """
    Dequantize bitsandbytes quantized weights.
    
    For bitsandbytes, we can use the dequantize method if available,
    or reconstruct from the quantized representation.
    """
    import bitsandbytes as bnb
    
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'CB'):
        # 4-bit NF4 quantization
        weight = layer.weight
        
        # Dequantize using bitsandbytes
        if method == "bitsandbytes_4bit":
            # For Linear4bit layers
            if hasattr(bnb.functional, 'dequantize_4bit'):
                dequant_weight = bnb.functional.dequantize_4bit(
                    weight.data,
                    weight.quant_state,
                )
                return dequant_weight.float()
            else:
                # Fallback: use forward pass trick
                warnings.warn("Using forward pass for dequantization - may be slow")
                with torch.no_grad():
                    eye = torch.eye(weight.shape[1], device=weight.device, dtype=torch.float16)
                    return layer(eye).T.float()
        
        elif method == "bitsandbytes_8bit":
            # For 8-bit layers
            if hasattr(layer.weight, 'SCB'):
                # Int8 dequantization
                return (layer.weight.data.float() * layer.weight.SCB.float() / 127.0)
            
    # Fallback for non-quantized or unknown format
    if hasattr(layer, 'weight'):
        return layer.weight.data.float()
    
    raise ValueError(f"Cannot dequantize layer: {type(layer)}")


def _dequantize_gptq(layer: torch.nn.Module) -> torch.Tensor:
    """
    Dequantize GPTQ quantized weights.
    
    GPTQ stores:
    - qweight: quantized weights
    - scales: scaling factors
    - qzeros: zero points (for asymmetric quantization)
    - g_idx: group indices
    """
    if hasattr(layer, 'qweight'):
        # Reconstruct weights from GPTQ format
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros if hasattr(layer, 'qzeros') else None
        g_idx = layer.g_idx if hasattr(layer, 'g_idx') else None
        
        # This is a simplified dequantization
        # Full implementation depends on GPTQ bit-width and group size
        bits = layer.bits if hasattr(layer, 'bits') else 4
        
        # Unpack and dequantize
        weight = _unpack_gptq_weights(qweight, scales, qzeros, g_idx, bits)
        return weight.float()
    
    return layer.weight.data.float()


def _dequantize_awq(layer: torch.nn.Module) -> torch.Tensor:
    """
    Dequantize AWQ quantized weights.
    
    AWQ (Activation-aware Weight Quantization) stores weights similarly to GPTQ.
    """
    if hasattr(layer, 'qweight'):
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros if hasattr(layer, 'qzeros') else None
        
        # AWQ typically uses 4-bit quantization
        weight = _unpack_awq_weights(qweight, scales, qzeros)
        return weight.float()
    
    return layer.weight.data.float()


def _unpack_gptq_weights(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    bits: int = 4
) -> torch.Tensor:
    """
    Unpack GPTQ quantized weights to full precision.
    
    This is a reference implementation - actual unpacking depends on:
    - Bit-width (usually 4-bit)
    - Group size
    - Whether symmetric or asymmetric quantization is used
    """
    # Number of weights packed per int32
    pack_factor = 32 // bits
    
    # Get dimensions
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * pack_factor
    
    # Initialize output tensor
    weight = torch.zeros((out_features, in_features), device=qweight.device, dtype=torch.float32)
    
    # Unpack weights
    mask = (1 << bits) - 1
    for i in range(pack_factor):
        shift = i * bits
        unpacked = ((qweight >> shift) & mask).float()
        
        # Apply scales and zero points
        if qzeros is not None:
            zeros = ((qzeros >> shift) & mask).float()
        else:
            zeros = 2 ** (bits - 1)  # Symmetric quantization center
        
        # Dequantize: w = (q - zero) * scale
        col_start = i * qweight.shape[1]
        col_end = (i + 1) * qweight.shape[1]
        
        if g_idx is not None:
            # Group-wise dequantization
            for j in range(qweight.shape[1]):
                group = g_idx[col_start + j].item() if col_start + j < len(g_idx) else j // (in_features // scales.shape[1])
                weight[:, col_start + j] = (unpacked[:, j] - zeros[:, group]) * scales[:, group]
        else:
            # Per-channel or uniform dequantization
            if scales.shape[1] == 1:
                weight[:, col_start:col_end] = (unpacked - zeros) * scales
            else:
                # Assuming group size divides evenly
                group_size = in_features // scales.shape[1]
                for g in range(scales.shape[1]):
                    g_start = g * group_size
                    g_end = min((g + 1) * group_size, col_end) - col_start
                    if g_start < col_end - col_start:
                        weight[:, col_start + g_start:col_start + g_end] = \
                            (unpacked[:, g_start:g_end] - zeros[:, g:g+1]) * scales[:, g:g+1]
    
    return weight


def _unpack_awq_weights(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Unpack AWQ quantized weights to full precision.
    
    AWQ typically uses 4-bit quantization with group-wise scales.
    """
    # AWQ uses similar packing to GPTQ
    return _unpack_gptq_weights(qweight, scales, qzeros, g_idx=None, bits=4)


def get_layer_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """
    Get a layer from a model by its full name (dot-separated path).
    
    Args:
        model: The model to search
        layer_name: Dot-separated path to the layer (e.g., "model.layers.0.self_attn.q_proj")
    
    Returns:
        The requested layer module
    """
    parts = layer_name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def is_quantized_layer(layer: torch.nn.Module) -> bool:
    """
    Check if a layer is quantized.
    
    Args:
        layer: The layer to check
    
    Returns:
        True if the layer appears to be quantized
    """
    # Check for bitsandbytes quantization
    if hasattr(layer, 'weight'):
        weight = layer.weight
        if hasattr(weight, 'CB') or hasattr(weight, 'SCB'):
            return True
    
    # Check for GPTQ/AWQ quantization
    if hasattr(layer, 'qweight'):
        return True
    
    return False


def estimate_memory_usage(
    model_path: str,
    quantization_method: str = "none",
) -> Dict[str, float]:
    """
    Estimate memory usage for different quantization methods.
    
    Args:
        model_path: Path to model
        quantization_method: Quantization method
    
    Returns:
        Dictionary with memory estimates in GB
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Estimate parameter count
    if hasattr(config, 'num_parameters'):
        num_params = config.num_parameters
    else:
        # Rough estimation for transformer models
        hidden_size = getattr(config, 'hidden_size', 4096)
        num_layers = getattr(config, 'num_hidden_layers', 32)
        vocab_size = getattr(config, 'vocab_size', 32000)
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
        
        # Attention params: Q, K, V, O projections
        attn_params = 4 * hidden_size * hidden_size * num_layers
        # FFN params
        ffn_params = 3 * hidden_size * intermediate_size * num_layers
        # Embeddings
        embed_params = vocab_size * hidden_size
        # LM head
        lm_head_params = vocab_size * hidden_size
        
        num_params = attn_params + ffn_params + embed_params + lm_head_params
    
    # Calculate memory based on quantization
    bytes_per_param = {
        "none": 2.0,  # FP16
        "bitsandbytes_8bit": 1.0,
        "bitsandbytes_4bit": 0.5,
        "gptq": 0.5,
        "awq": 0.5,
    }
    
    bpp = bytes_per_param.get(quantization_method, 2.0)
    model_memory_gb = (num_params * bpp) / (1024 ** 3)
    
    # Add overhead for activations and optimizer states (rough estimate)
    activation_memory_gb = model_memory_gb * 0.5
    
    return {
        "model_memory_gb": model_memory_gb,
        "estimated_activation_memory_gb": activation_memory_gb,
        "total_estimated_gb": model_memory_gb + activation_memory_gb,
        "num_parameters": num_params,
    }

