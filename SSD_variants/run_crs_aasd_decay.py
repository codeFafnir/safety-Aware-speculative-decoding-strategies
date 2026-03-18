#!/usr/bin/env python3
"""
Run SSD-CRS-AASD with decaying lambda across 2 GPUs.
  GPU 0 → draft model (Qwen2.5-1.5B-Instruct)
  GPU 1 → target model (Qwen2.5-7B-Instruct)

Usage:
  python run_crs_aasd_decay.py [--n_harmful 35] [--n_benign 15]
                                [--lambda_align 0.3] [--lambda_decay 0.92]
                                [--responses_dir /workspace/results/crs_aasd_decay]
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ssd_experiments import (
    Config, config as _base_config,
    save_responses, load_responses,
    build_datasets,
    SSDDecoderCRSAASD, Qwen3Guard,
    evaluate,
)

def load_model_on_device(model_name: str, device: str, use_4bit: bool = False):
    """Load a model pinned to a specific GPU."""
    local = os.path.join("/workspace/downloaded_models",
                         model_name.replace("/", "_"))
    src = local if os.path.isdir(local) else model_name
    print(f"  Loading {model_name} → {device} ...")

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": device,         # pin to specific GPU
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_harmful",      type=int,   default=35)
    parser.add_argument("--n_benign",       type=int,   default=15)
    parser.add_argument("--lambda_align",   type=float, default=0.3)
    parser.add_argument("--lambda_decay",   type=float, default=0.92)
    parser.add_argument("--tau",            type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--responses_dir",  type=str,
                        default="/workspace/results/crs_aasd_decay")
    parser.add_argument("--skip_eval",      action="store_true")
    args = parser.parse_args()

    os.makedirs(args.responses_dir, exist_ok=True)
    cfg = _base_config
    cfg.max_new_tokens = args.max_new_tokens

    print(f"\nConfig: lambda_align={args.lambda_align}, "
          f"lambda_decay={args.lambda_decay}, tau={args.tau}")
    print(f"Output: {args.responses_dir}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    harmful, benign = build_datasets()
    harmful = harmful[:args.n_harmful]
    benign  = benign[:args.n_benign]
    print(f"Dataset: {len(harmful)} harmful, {len(benign)} benign\n")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    local_draft = os.path.join("/workspace/downloaded_models",
                               cfg.draft_model.replace("/", "_"))
    src_tok = local_draft if os.path.isdir(local_draft) else cfg.draft_model
    tok = AutoTokenizer.from_pretrained(src_tok, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ── Load models: draft on GPU 0, target on GPU 1 ─────────────────────────
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    draft  = load_model_on_device(cfg.draft_model,  "cuda:0", use_4bit=False)
    target = load_model_on_device(cfg.target_model, "cuda:1", use_4bit=False)

    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    print(f"\nGPU memory after load:\n{result.stdout.strip()}\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    h_path = os.path.join(args.responses_dir, "crs_aasd_decay_harmful.json")
    b_path = os.path.join(args.responses_dir, "crs_aasd_decay_benign.json")

    decoder = SSDDecoderCRSAASD(
        draft, target, tok, cfg,
        lambda_align=args.lambda_align,
        tau=args.tau,
        lambda_decay=args.lambda_decay,
    )

    if not os.path.exists(h_path):
        print("="*60)
        print(f"GENERATING: SSD-CRS-AASD (λ_0={args.lambda_align}, γ={args.lambda_decay})")
        print("="*60)
        results_h = []
        for item in tqdm(harmful, desc="crs_aasd_decay harmful"):
            resp, lat, stats = decoder.generate(item["prompt"], cfg.max_new_tokens)
            results_h.append({**item, "response": resp, "latency": lat,
                               "method": "ssd_crs_aasd_decay", **stats})
        save_responses(results_h, h_path)

        mean_lam = sum(r.get("mean_lambda", 0) for r in results_h) / len(results_h)
        mean_len = sum(len(r["response"].split()) for r in results_h) / len(results_h)
        print(f"  mean_lambda (avg over responses): {mean_lam:.4f}")
        print(f"  mean response length: {mean_len:.0f} words")
    else:
        print(f"  Harmful responses exist — skipping generation")

    if not os.path.exists(b_path):
        results_b = []
        for item in tqdm(benign, desc="crs_aasd_decay benign"):
            resp, lat, stats = decoder.generate(item["prompt"], cfg.max_new_tokens)
            results_b.append({**item, "response": resp, "latency": lat,
                               "method": "ssd_crs_aasd_decay", **stats})
        save_responses(results_b, b_path)
    else:
        print(f"  Benign responses exist — skipping generation")

    del decoder, draft, target
    torch.cuda.empty_cache()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.skip_eval:
        print("\nSkipping eval (--skip_eval).")
        return

    print("\n" + "="*60)
    print("EVALUATION (Qwen3Guard)")
    print("="*60)
    guard = Qwen3Guard(cfg.guard_model, use_4bit=False)
    guard.load()

    m_h, r_h = evaluate(load_responses(h_path), guard, is_harmful=True)
    m_b, r_b = evaluate(load_responses(b_path), guard, is_harmful=False)
    guard.unload()

    save_responses(r_h, os.path.join(args.responses_dir, "crs_aasd_decay_harmful_eval.json"))
    save_responses(r_b, os.path.join(args.responses_dir, "crs_aasd_decay_benign_eval.json"))

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  ASR:           {m_h['asr_strict']:.1f}%")
    print(f"  Refusal:       {m_h['refusal_rate']:.1f}%")
    print(f"  Over-refusal:  {m_b['over_refusal_pct']:.1f}%")
    print(f"  Avg resp len:  {m_b['avg_response_len']:.0f}")

    # Compare against frozen-lambda baseline from existing results
    import json
    baseline_path = "/workspace/results/metrics.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        if "ssd_crs_aasd" in baseline:
            b_asr = baseline["ssd_crs_aasd"].get("asr_strict",
                    baseline["ssd_crs_aasd"].get("asr", "?"))
            print(f"\n  vs frozen-λ AASD ASR: {b_asr}%  →  Δ = {m_h['asr_strict'] - float(b_asr):.1f} pp")

    import json
    out = {
        "method": "ssd_crs_aasd_decay",
        "lambda_align": args.lambda_align,
        "lambda_decay": args.lambda_decay,
        "tau": args.tau,
        "harmful": m_h,
        "benign": m_b,
    }
    out_path = os.path.join(args.responses_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
