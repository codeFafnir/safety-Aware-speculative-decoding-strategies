"""Model loading and vanilla generation for SSD + AASD."""
import os
import gc
import time
import torch
from typing import Tuple

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def download_model(model_name: str, models_dir: str) -> str:
    safe_name = model_name.replace("/", "_")
    local_path = os.path.join(models_dir, safe_name)
    if os.path.exists(local_path) and os.listdir(local_path):
        has_config = os.path.exists(os.path.join(local_path, "config.json"))
        has_model = any(
            f.endswith((".safetensors", ".bin")) for f in os.listdir(local_path)
        )
        if has_config and has_model:
            print(f"checkmark {model_name} already at {local_path}")
            return local_path
    print(f"Downloading {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )
    print(f"Downloaded to {local_path}")
    return local_path


def get_local_model_path(model_name: str, models_dir: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    safe_name = model_name.replace("/", "_")
    local_path = os.path.join(models_dir, safe_name)
    if not os.path.exists(local_path) or not os.listdir(local_path):
        return download_model(model_name, models_dir)
    return local_path


def unload_model(model, tokenizer=None):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Model unloaded.")


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool,
    models_dir: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    local_path = get_local_model_path(model_name, models_dir)
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(local_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


@torch.no_grad()
def vanilla_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> Tuple[str, float, int]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    ilen = inputs["input_ids"].shape[1]
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    lat = time.time() - t0
    text = tokenizer.decode(out[0][ilen:], skip_special_tokens=True).strip()
    n_gen = out.shape[1] - ilen
    return text, lat, n_gen
