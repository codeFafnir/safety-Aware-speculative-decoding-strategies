#!/usr/bin/env python3
"""
CRS Component Ablation Study
=============================
Systematically ablates each term in the Composite Risk Score:

  r_t = w1*(1-match) + w2*KL(p_m||p_M) + w3*ΔH + w4*refusal_mass

Ablations run sequentially but share a single model load, making the
study much cheaper than re-loading models for every variant.

Usage:
  python ssd_crs_ablation.py [--max_new_tokens 256] [--no_4bit]
                              [--n_harmful 40] [--n_benign 20]
                              [--responses_dir ./results/ablation]

The n_harmful/n_benign flags let you run a smaller subset for speed.
Default: full dataset (70 harmful + 30 benign).
"""

import os, sys, json, time, gc, math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import argparse

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

# ── reuse everything from ssd_experiments ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from ssd_experiments import (
    Config, config as _base_config,
    load_model, unload, save_responses, load_responses,
    build_datasets,
    SSDDecoderCRS, Qwen3Guard,
    evaluate, _response_len, _is_refusal,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION CONFIGS
# Each entry: (name, display_label, w1, w2, w3, w4)
#   w1 = match mismatch
#   w2 = KL divergence
#   w3 = entropy gap ΔH
#   w4 = refusal token mass
# ══════════════════════════════════════════════════════════════════════════════

ABLATIONS = [
    # name             label                       w1    w2    w3    w4
    ("full",           "Full CRS (all)",            0.3,  0.3,  0.1,  0.3),
    ("no_match",       "CRS − match",               0.0,  0.3,  0.1,  0.3),
    ("no_kl",          "CRS − KL",                  0.3,  0.0,  0.1,  0.3),
    ("no_dh",          "CRS − ΔH",                  0.3,  0.3,  0.0,  0.3),
    ("no_refusal",     "CRS − refusal_mass",         0.3,  0.3,  0.1,  0.0),
    ("match_only",     "match only",                0.3,  0.0,  0.0,  0.0),
    ("kl_only",        "KL only",                   0.0,  0.3,  0.0,  0.0),
    ("refusal_only",   "refusal_mass only",          0.0,  0.0,  0.0,  0.3),
    ("kl_refusal",     "KL + refusal_mass",          0.0,  0.3,  0.0,  0.3),
]


def make_config(w1, w2, w3, w4, base: Config) -> Config:
    """Return a shallow copy of base config with overridden CRS weights."""
    import copy
    cfg = copy.copy(base)
    cfg.crs_w1 = w1
    cfg.crs_w2 = w2
    cfg.crs_w3 = w3
    cfg.crs_w4 = w4
    return cfg


def run_ablation_variant(
    name: str, cfg: Config,
    draft, target, tok,
    harmful: List[Dict], benign: List[Dict],
    responses_dir: str,
    batch_size: int = 8,  # unused, kept for API compat
) -> Tuple[str, str]:
    """Run a single CRS variant. Returns (harmful_path, benign_path)."""
    decoder = SSDDecoderCRS(draft, target, tok, cfg)
    method = f"ssd_crs_{name}"

    results_h = []
    for item in tqdm(harmful, desc=f"{method} harmful"):
        resp, lat, stats = decoder.generate(item["prompt"], cfg.max_new_tokens)
        results_h.append({**item, "response": resp, "latency": lat, "method": method, **stats})
    h_path = os.path.join(responses_dir, f"{method}_harmful.json")
    save_responses(results_h, h_path)

    results_b = []
    for item in tqdm(benign, desc=f"{method} benign"):
        resp, lat, stats = decoder.generate(item["prompt"], cfg.max_new_tokens)
        results_b.append({**item, "response": resp, "latency": lat, "method": method, **stats})
    b_path = os.path.join(responses_dir, f"{method}_benign.json")
    save_responses(results_b, b_path)

    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h_path, b_path


def evaluate_all(
    ablations: List[tuple],
    responses_dir: str,
    guard_model: str,
    use_4bit: bool,
) -> Dict:
    """Run guard evaluation on all ablation variants."""
    guard = Qwen3Guard(guard_model, use_4bit)
    guard.load()

    all_metrics = {}
    for name, label, *_ in ablations:
        method = f"ssd_crs_{name}"
        h_path = os.path.join(responses_dir, f"{method}_harmful.json")
        b_path = os.path.join(responses_dir, f"{method}_benign.json")
        if not os.path.exists(h_path) or not os.path.exists(b_path):
            print(f"  Skipping {method} (response files not found)")
            continue
        print(f"  Evaluating {method} ...")
        m_h, r_h = evaluate(load_responses(h_path), guard, is_harmful=True)
        m_b, r_b = evaluate(load_responses(b_path), guard, is_harmful=False)
        all_metrics[name] = {"label": label, "harmful": m_h, "benign": m_b}
        save_responses(r_h, os.path.join(responses_dir, f"{method}_harmful_eval.json"))
        save_responses(r_b, os.path.join(responses_dir, f"{method}_benign_eval.json"))

    guard.unload()
    return all_metrics


def print_ablation_table(all_metrics: Dict, ablations: List[tuple], responses_dir: str):
    rows = []
    for name, label, w1, w2, w3, w4 in ablations:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        mh, mb = m["harmful"], m["benign"]

        method = f"ssd_crs_{name}"
        h_path = os.path.join(responses_dir, f"{method}_harmful.json")
        b_path = os.path.join(responses_dir, f"{method}_benign.json")
        union_frac_h = union_frac_b = match_h = match_b = mean_kl_h = mean_ref_h = float("nan")
        if os.path.exists(h_path):
            rh = load_responses(h_path)
            if rh:
                union_frac_h = np.mean([r.get("union_tokens", 0) / max(r.get("total_steps", 1), 1) for r in rh]) * 100
                match_h      = np.mean([r.get("match_ratio", float("nan")) for r in rh if "match_ratio" in r])
                mean_kl_h    = np.mean([r.get("mean_kl_norm", float("nan")) for r in rh if "mean_kl_norm" in r])
                mean_ref_h   = np.mean([r.get("mean_refusal_mass", float("nan")) for r in rh if "mean_refusal_mass" in r])
        if os.path.exists(b_path):
            rb = load_responses(b_path)
            if rb:
                union_frac_b = np.mean([r.get("union_tokens", 0) / max(r.get("total_steps", 1), 1) for r in rb]) * 100
                match_b      = np.mean([r.get("match_ratio", float("nan")) for r in rb if "match_ratio" in r])

        rows.append({
            "Ablation":      label,
            "w1 w2 w3 w4":  f"{w1} {w2} {w3} {w4}",
            "ASR%↓":         f"{mh['asr_strict']:.1f}",
            "Refusal%↑":     f"{mh['refusal_rate']:.1f}",
            "Controv%":      f"{mh['controversial_pct']:.1f}",
            "Over-ref%↓":    f"{mb['over_refusal_pct']:.1f}",
            "Resp len":      f"{mb['avg_response_len']:.0f}",
            "Union%":        f"{union_frac_h:.1f}" if not math.isnan(union_frac_h) else "-",
            "Match(h)":      f"{match_h:.3f}"      if not math.isnan(match_h)      else "-",
            "KL(h)":         f"{mean_kl_h:.3f}"    if not math.isnan(mean_kl_h)    else "-",
            "RefMass(h)":    f"{mean_ref_h:.4f}"   if not math.isnan(mean_ref_h)   else "-",
        })

    print("\n" + "="*100)
    print("CRS COMPONENT ABLATION RESULTS")
    print("="*100)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save to JSON and CSV
    out_json = os.path.join(responses_dir, "ablation_metrics.json")
    out_csv  = os.path.join(responses_dir, "ablation_results.csv")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"\nMetrics → {out_json}")
    print(f"CSV     → {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="CRS component ablation study")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_4bit",        action="store_true")
    parser.add_argument("--n_harmful",      type=int, default=None,
                        help="Subset of harmful prompts (default: all 70)")
    parser.add_argument("--n_benign",       type=int, default=None,
                        help="Subset of benign prompts (default: all 30)")
    parser.add_argument("--responses_dir",  type=str, default="./results/crs_ablation")
    parser.add_argument("--eval_only",      action="store_true",
                        help="Skip generation; only re-run guard evaluation on existing responses")
    parser.add_argument("--skip_eval",      action="store_true",
                        help="Skip guard evaluation after generation (useful for parallel runs)")
    parser.add_argument("--skip_names",     nargs="*", default=[],
                        help="Ablation names to skip (e.g. full no_dh)")
    parser.add_argument("--only_names",     nargs="*", default=None,
                        help="Run only these ablation names (e.g. full no_kl refusal_only)")
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Number of prompts to process simultaneously (default: 8)")
    args = parser.parse_args()

    cfg = _base_config
    if args.no_4bit:
        cfg.use_4bit = False
    cfg.max_new_tokens = args.max_new_tokens
    os.makedirs(args.responses_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    harmful, benign = build_datasets()
    if args.n_harmful:
        harmful = harmful[:args.n_harmful]
    if args.n_benign:
        benign = benign[:args.n_benign]
    print(f"\nAblation dataset: {len(harmful)} harmful, {len(benign)} benign")
    print(f"Output directory: {args.responses_dir}")

    to_run = [(n, lbl, w1, w2, w3, w4)
              for (n, lbl, w1, w2, w3, w4) in ABLATIONS
              if n not in args.skip_names
              and (args.only_names is None or n in args.only_names)]

    # ── Generation pass (shared model load) ───────────────────────────────
    if not args.eval_only:
        print("\n" + "="*60)
        print("LOADING MODELS (shared across all ablations)")
        print("="*60)
        draft, tok = load_model(cfg.draft_model,  cfg.use_4bit)
        target, _  = load_model(cfg.target_model, cfg.use_4bit)

        for name, label, w1, w2, w3, w4 in to_run:
            method = f"ssd_crs_{name}"
            h_path = os.path.join(args.responses_dir, f"{method}_harmful.json")
            b_path = os.path.join(args.responses_dir, f"{method}_benign.json")
            if os.path.exists(h_path) and os.path.exists(b_path):
                print(f"\n  [{name}] responses already exist — skipping generation")
                continue

            print(f"\n{'='*60}")
            print(f"ABLATION: {label}  (w1={w1} w2={w2} w3={w3} w4={w4})")
            print("="*60)
            variant_cfg = make_config(w1, w2, w3, w4, cfg)
            run_ablation_variant(name, variant_cfg, draft, target, tok,
                                 harmful, benign, args.responses_dir,
                                 batch_size=args.batch_size)

        unload(draft)
        unload(target, tok)
        print("\nAll generation passes complete.")

    # ── Evaluation pass ───────────────────────────────────────────────────
    if not args.skip_eval:
        print("\n" + "="*60)
        print("EVALUATION (Qwen3Guard)")
        print("="*60)
        all_metrics = evaluate_all(ABLATIONS, args.responses_dir,
                                    cfg.guard_model, cfg.use_4bit)
        print_ablation_table(all_metrics, ABLATIONS, args.responses_dir)
    else:
        print("\nSkipping eval (--skip_eval). Run with --eval_only when all generation passes complete.")


if __name__ == "__main__":
    main()
