#!/usr/bin/env python3
"""
PPL-Gate Ablation Study
=======================
Ablates the perplexity gate in SSD-CRS by varying:
  - use_ppl_gate=False          (gate disabled)
  - ppl_threshold=50            (aggressive: fires on more prompts)
  - ppl_threshold=150           (default)
  - ppl_threshold=500           (conservative: rarely fires)

Base decoder: SSDDecoderCRS (all other hyperparameters fixed).

Usage:
  python ssd_pplgate_ablation.py [--n_harmful 35] [--n_benign 15]
                                  [--responses_dir ./results/pplgate_ablation]
"""

import os, sys, json, gc, copy, math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from ssd_experiments import (
    Config, config as _base_config,
    load_model, unload, save_responses, load_responses,
    build_datasets,
    SSDDecoderCRS, Qwen3Guard,
    evaluate, _is_refusal,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Ablation variants ─────────────────────────────────────────────────────────
# (name, label, use_ppl_gate, ppl_threshold)
ABLATIONS = [
    ("no_gate",    "No PPL gate",          False, 150.0),
    ("ppl_50",     "PPL gate τ=50",        True,  50.0),
    ("ppl_150",    "PPL gate τ=150 (def)", True,  150.0),
    ("ppl_500",    "PPL gate τ=500",       True,  500.0),
]


def make_config(use_ppl_gate: bool, ppl_threshold: float, base: Config) -> Config:
    cfg = copy.copy(base)
    cfg.use_ppl_gate  = use_ppl_gate
    cfg.ppl_threshold = ppl_threshold
    return cfg


def run_variant(
    name: str, cfg: Config,
    draft, target, tok,
    harmful: List[Dict], benign: List[Dict],
    responses_dir: str,
) -> Tuple[str, str]:
    """Generate responses for one PPL-gate variant. Returns (harmful_path, benign_path)."""
    decoder = SSDDecoderCRS(draft, target, tok, cfg)
    method  = f"pplgate_{name}"

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


def evaluate_all(ablations, responses_dir, guard_model, use_4bit) -> Dict:
    guard = Qwen3Guard(guard_model, use_4bit)
    guard.load()
    all_metrics = {}
    for name, label, *_ in ablations:
        method = f"pplgate_{name}"
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


def print_table(all_metrics, ablations, responses_dir):
    rows = []
    for name, label, use_gate, threshold in ablations:
        if name not in all_metrics:
            continue
        m  = all_metrics[name]
        mh = m["harmful"]
        mb = m["benign"]

        # Compute PPL-forced-union stats from raw response files
        method = f"pplgate_{name}"
        h_path = os.path.join(responses_dir, f"{method}_harmful.json")
        b_path = os.path.join(responses_dir, f"{method}_benign.json")
        ppl_forced_h = ppl_forced_b = union_pct = float("nan")
        if os.path.exists(h_path):
            rh = load_responses(h_path)
            if rh:
                ppl_forced_h = sum(r.get("forced_union_by_ppl", False) for r in rh)
                union_pct    = np.mean(
                    [r.get("union_tokens", 0) / max(r.get("total_steps", 1), 1) for r in rh]
                ) * 100
        if os.path.exists(b_path):
            rb = load_responses(b_path)
            if rb:
                ppl_forced_b = sum(r.get("forced_union_by_ppl", False) for r in rb)

        gate_str = f"τ={threshold:.0f}" if use_gate else "off"
        rows.append({
            "Variant":        label,
            "Gate":           gate_str,
            "ASR%↓":          f"{mh['asr_strict']:.1f}",
            "Refusal%↑":      f"{mh['refusal_rate']:.1f}",
            "Over-ref%↓":     f"{mb['over_refusal_pct']:.1f}",
            "Resp len":        f"{mb['avg_response_len']:.0f}",
            "Union%":         f"{union_pct:.1f}" if not math.isnan(union_pct) else "-",
            "PPL-forced(H)":  f"{int(ppl_forced_h)}" if not math.isnan(ppl_forced_h) else "-",
            "PPL-forced(B)":  f"{int(ppl_forced_b)}" if not math.isnan(ppl_forced_b) else "-",
        })

    print("\n" + "=" * 90)
    print("PPL-GATE ABLATION RESULTS  (base: SSD-CRS)")
    print("=" * 90)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save
    out_json = os.path.join(responses_dir, "pplgate_ablation_results.json")
    out_csv  = os.path.join(responses_dir, "pplgate_ablation_results.csv")

    # Build compact JSON matching crs_ablation_results.json format
    compact = {
        "dataset": "35 harmful (DI-20 + JBB-15) + 15 benign (XSTest)",
        "target":  _base_config.target_model,
        "draft":   _base_config.draft_model,
        "guard":   _base_config.guard_model,
        "note":    "PPL gate ablation on SSD-CRS base; varying use_ppl_gate and ppl_threshold",
        "ablations": {},
    }
    for name, label, use_gate, threshold in ablations:
        if name not in all_metrics:
            continue
        mh = all_metrics[name]["harmful"]
        mb = all_metrics[name]["benign"]

        method = f"pplgate_{name}"
        h_path = os.path.join(responses_dir, f"{method}_harmful.json")
        union_pct_v = ppl_forced_v = float("nan")
        if os.path.exists(h_path):
            rh = load_responses(h_path)
            if rh:
                union_pct_v  = round(np.mean([r.get("union_tokens",0)/max(r.get("total_steps",1),1) for r in rh])*100, 1)
                ppl_forced_v = int(sum(r.get("forced_union_by_ppl", False) for r in rh))

        compact["ablations"][name] = {
            "label":           label,
            "use_ppl_gate":    use_gate,
            "ppl_threshold":   threshold,
            "asr_strict":      mh["asr_strict"],
            "refusal_rate":    mh["refusal_rate"],
            "over_refusal_pct": mb["over_refusal_pct"],
            "avg_response_len": mb["avg_response_len"],
            "union_pct":       union_pct_v,
            "ppl_forced_harmful": ppl_forced_v,
        }

    with open(out_json, "w") as f:
        json.dump(compact, f, indent=2)
    df.to_csv(out_csv, index=False)
    print(f"\nJSON → {out_json}")
    print(f"CSV  → {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="PPL gate ablation study")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_4bit",        action="store_true")
    parser.add_argument("--n_harmful",      type=int, default=35)
    parser.add_argument("--n_benign",       type=int, default=15)
    parser.add_argument("--responses_dir",  type=str, default="./results/pplgate_ablation")
    parser.add_argument("--eval_only",      action="store_true")
    parser.add_argument("--skip_eval",      action="store_true")
    parser.add_argument("--only_names",     nargs="*", default=None)
    args = parser.parse_args()

    cfg = _base_config
    if args.no_4bit:
        cfg.use_4bit = False
    cfg.max_new_tokens = args.max_new_tokens
    os.makedirs(args.responses_dir, exist_ok=True)

    harmful, benign = build_datasets()
    harmful = harmful[:args.n_harmful]
    benign  = benign[:args.n_benign]
    print(f"\nPPL-gate ablation: {len(harmful)} harmful, {len(benign)} benign")
    print(f"Output directory:  {args.responses_dir}")

    to_run = [(n, lbl, g, t)
              for (n, lbl, g, t) in ABLATIONS
              if args.only_names is None or n in args.only_names]

    # ── Generation ────────────────────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "="*60)
        print("LOADING MODELS (shared across all variants)")
        print("="*60)
        draft,  tok = load_model(cfg.draft_model,  cfg.use_4bit)
        target, _   = load_model(cfg.target_model, cfg.use_4bit)

        for name, label, use_gate, threshold in to_run:
            method = f"pplgate_{name}"
            h_path = os.path.join(args.responses_dir, f"{method}_harmful.json")
            b_path = os.path.join(args.responses_dir, f"{method}_benign.json")
            if os.path.exists(h_path) and os.path.exists(b_path):
                print(f"\n  [{name}] responses exist — skipping")
                continue

            print(f"\n{'='*60}")
            gate_str = f"τ={threshold:.0f}" if use_gate else "disabled"
            print(f"VARIANT: {label}  (gate={gate_str})")
            print("="*60)
            variant_cfg = make_config(use_gate, threshold, cfg)
            run_variant(name, variant_cfg, draft, target, tok,
                        harmful, benign, args.responses_dir)

        unload(draft)
        unload(target, tok)
        print("\nAll generation passes complete.")

    # ── Evaluation ────────────────────────────────────────────────────────
    if not args.skip_eval:
        print("\n" + "="*60)
        print("EVALUATION (Qwen3Guard)")
        print("="*60)
        all_metrics = evaluate_all(ABLATIONS, args.responses_dir,
                                   cfg.guard_model, cfg.use_4bit)
        print_table(all_metrics, ABLATIONS, args.responses_dir)
    else:
        print("\nSkipping eval (--skip_eval). Rerun with --eval_only when ready.")


if __name__ == "__main__":
    main()
