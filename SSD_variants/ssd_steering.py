#!/usr/bin/env python3
"""
SSD-Steering: Hidden-state steering extension for SSD-CRS.

Inspired by: "Steering Pretrained Drafters during Speculative Decoding"
             arXiv: 2511.09844 (AAAI 2026)

How it works:
  1. During the target verification pass (already happening in SSD-CRS), extract
     the verifier's last-token hidden state h_t from the final transformer layer.
  2. A lightweight linear projection maps h_t → g_t in the draft model's hidden space.
  3. g_t is injected as an additive bias into all MLP layers of the draft model
     via PyTorch forward hooks, continuously conditioning the drafter toward safe outputs.
  4. Steering magnitude is scaled by the current CRS risk score:
       scale = base_magnitude * (1 + amplify * smoothed_crs)
     → High-risk steps (jailbreak detected) apply stronger steering.
     → Low-risk (benign) steps apply minimal steering to preserve utility.

Two usage modes:
  a) Zero-shot (default): random-init projection, validates the mechanism
  b) Trained: run pretrain_projection() on harmful prompts before evaluation

Usage:
    # Zero-shot (no training)
    python ssd_steering.py --phases steering eval

    # With optional projection pretraining
    python ssd_steering.py --phases pretrain steering eval

    # Load existing checkpoint
    python ssd_steering.py --phases steering eval --ckpt steering_ckpts/proj.pt
"""

import gc, math, os, json, sys, time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ── Import base classes from sibling module ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from ssd_experiments import (
    Config, SSDDecoderCRS, load_model, unload,
    save_responses, load_responses, evaluate, Qwen3Guard,
    build_datasets, config, print_results,
)


# ══════════════════════════════════════════════════════════════════════════════
# STEERING CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SteeringConfig:
    # Injection magnitude
    base_magnitude: float = 0.05   # steering scale at zero CRS risk
    amplify:        float = 2.0    # multiplier at maximum CRS risk
    # scale = base_magnitude * (1 + amplify * smoothed_crs)
    # e.g. at crs=0.5 → scale = 0.05 * 2.0 = 0.10

    # Which layers to steer (fraction of total MLP layers, from the top)
    layer_frac: float = 1.0        # 1.0 = all layers; 0.5 = top half only

    # Checkpoint path (for saving/loading trained projection)
    ckpt_dir: str = "./steering_ckpts"

    # Training hyperparameters (pretrain_projection)
    train_lr:    float = 1e-3
    train_epochs: int  = 5
    train_batch:  int  = 4


# ══════════════════════════════════════════════════════════════════════════════
# STEERING PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

class SteeringProjection(nn.Module):
    """
    Linear projection: verifier_hidden_dim → draft_hidden_dim.
    Produces steering vector g_t injected into draft MLP layers.

    Initialized with small normal weights so random-init steering is a
    gentle perturbation rather than destructive noise.
    """

    def __init__(self, target_hidden: int, draft_hidden: int):
        super().__init__()
        self.proj = nn.Linear(target_hidden, draft_hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"  Steering projection saved → {path}")

    @classmethod
    def load(cls, path: str, target_hidden: int, draft_hidden: int) -> "SteeringProjection":
        proj = cls(target_hidden, draft_hidden)
        proj.load_state_dict(torch.load(path, map_location="cpu"))
        proj.eval()
        print(f"  Steering projection loaded ← {path}")
        return proj


# ══════════════════════════════════════════════════════════════════════════════
# HOOK-BASED MLP INJECTOR
# ══════════════════════════════════════════════════════════════════════════════

class SteeringInjector:
    """
    Registers persistent forward hooks on the draft model's MLP layers.
    On each forward pass, adds g_t * scale as an additive bias to the MLP output.

    Call set(g, scale) to update the steering vector before each draft step.
    Call off() to zero steering (e.g. between prompts or on benign steps).
    Call remove() on cleanup.
    """

    def __init__(self, draft_model, layer_frac: float = 1.0):
        self._g: Optional[torch.Tensor] = None
        self._scale: float = 0.0
        self._handles: List = []

        layers = self._get_layers(draft_model)
        n_steer = max(1, int(len(layers) * layer_frac))
        # Steer the last n layers (closer to output → stronger effect on logits)
        for layer in layers[-n_steer:]:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                h = mlp.register_forward_hook(self._hook)
                self._handles.append(h)

        print(f"  SteeringInjector: {len(self._handles)} MLP hooks registered "
              f"(layer_frac={layer_frac:.1f}, total_layers={len(layers)})")

    @staticmethod
    def _get_layers(model) -> list:
        m = getattr(model, "model", model)
        return list(getattr(m, "layers", []))

    def _hook(self, module, input, output):
        if self._g is None or self._scale < 1e-9:
            return output
        g = self._g.to(device=output.device, dtype=output.dtype)
        return output + g.view(1, 1, -1) * self._scale

    def set(self, g: torch.Tensor, scale: float):
        """Update steering vector and scale for the next forward pass."""
        self._g = g.detach()
        self._scale = scale

    def off(self):
        """Disable steering (scale → 0)."""
        self._g = None
        self._scale = 0.0

    def remove(self):
        """Remove all hooks (call on cleanup)."""
        for h in self._handles:
            h.remove()
        self._handles = []


# ══════════════════════════════════════════════════════════════════════════════
# SSD-STEERING DECODER
# ══════════════════════════════════════════════════════════════════════════════

class SSDDecoderSteering(SSDDecoderCRS):
    """
    SSD + CRS + Hidden-State Steering.

    Extends SSDDecoderCRS by:
      - Extracting target model's last-token hidden state at each verification step
        (output_hidden_states=True on the target pass, which is already happening)
      - Projecting it to the draft model's hidden space via SteeringProjection
      - Injecting the result as an additive bias into all draft MLP layers
      - Scaling the injection magnitude proportional to the current CRS risk score:
          scale = base_mag * (1 + amplify * smoothed_crs)

    Token selection, mode switching, and CRS computation are inherited unchanged
    from SSDDecoderCRS. The only change is that the draft model's internals are
    continuously nudged toward safe outputs by the verifier's hidden representation.
    """

    def __init__(self, draft, target, tokenizer, cfg: Config,
                 scfg: SteeringConfig,
                 proj: Optional[SteeringProjection] = None):
        super().__init__(draft, target, tokenizer, cfg)
        self.scfg = scfg

        # Discover hidden dims from model configs
        t_hidden = getattr(target.config, "hidden_size", 3584)
        d_hidden = getattr(draft.config,  "hidden_size", 1536)
        print(f"  Steering dims: target={t_hidden}, draft={d_hidden}")

        self.proj = proj if proj is not None else SteeringProjection(t_hidden, d_hidden)
        self.proj.eval()

        self.injector = SteeringInjector(draft, scfg.layer_frac)

    # ── KV helpers that also return hidden states ──────────────────────────

    @torch.no_grad()
    def _init_kv_h(self, model, ids) -> Tuple[torch.Tensor, object, torch.Tensor]:
        """Full forward pass: returns (last_logit, past_kv, last_hidden)."""
        dev = next(model.parameters()).device
        out = model(input_ids=ids.to(dev), use_cache=True, output_hidden_states=True)
        logit  = out.logits[0, -1, :].float().cpu()
        hidden = out.hidden_states[-1][0, -1, :].float().cpu()
        return logit, out.past_key_values, hidden

    @torch.no_grad()
    def _forward_batch_h(self, model, token_ids: list, past_kv
                         ) -> Tuple[torch.Tensor, object, Optional[torch.Tensor]]:
        """Batch KV step: returns (logits, past_kv, last_hidden)."""
        if not token_ids:
            return torch.empty(0), past_kv, None
        dev = next(model.parameters()).device
        inp = torch.tensor([token_ids], dtype=torch.long, device=dev)
        out = model(input_ids=inp, past_key_values=past_kv, use_cache=True,
                    output_hidden_states=True)
        logits = out.logits[0].float().cpu()
        hidden = out.hidden_states[-1][0, -1, :].float().cpu()
        return logits, out.past_key_values, hidden

    def _apply_steering(self, t_hidden: torch.Tensor, smoothed_crs: float):
        """Project target hidden state → g_t, scale by CRS risk, set injector."""
        proj_dev = next(self.proj.parameters()).device
        with torch.no_grad():
            g_t = self.proj(t_hidden.to(proj_dev)).cpu()
        scale = self.scfg.base_magnitude * (1.0 + self.scfg.amplify * smoothed_crs)
        self.injector.set(g_t, scale)

    # ── Main generate loop ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, user_msg: str, max_new_tokens: int = 256) -> Tuple[str, float, Dict]:
        draft_prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": self.cfg.draft_system_prompt},
             {"role": "user",   "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
        target_prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": self.cfg.target_system_prompt},
             {"role": "user",   "content": user_msg}],
            tokenize=False, add_generation_prompt=True)

        d_ids = self.tok(draft_prompt,  return_tensors="pt")["input_ids"]
        t_ids = self.tok(target_prompt, return_tensors="pt")["input_ids"]

        forced_union = False
        if self.cfg.use_ppl_gate:
            ppl = self._prompt_ppl(d_ids)
            forced_union = ppl > self.cfg.ppl_threshold

        mode     = 2 if forced_union else 1
        alpha_I  = self.cfg.alpha_I
        C        = self.cfg.sample_space_c
        T        = self.cfg.lookahead_T
        risk_buf: deque = deque(maxlen=self.cfg.crs_window)
        smoothed_crs = 0.0

        generated  = []
        risk_trace = []
        union_tok  = inter_tok = mode_sw = 0
        t0 = time.time()

        # ── Init KV caches; get initial target hidden state ───────────────
        d_last_logits, d_kv = self._init_kv(self.draft, d_ids)
        t_last_logits, t_kv, t_hidden = self._init_kv_h(self.target, t_ids)

        # Apply initial steering (smoothed_crs=0 → base_magnitude only)
        self._apply_steering(t_hidden, smoothed_crs)

        steps = 0
        while steps < max_new_tokens:
            # ── Draft T tokens greedily (hooks inject steering on every step) ──
            expert_draft_probs  = [torch.softmax(d_last_logits, dim=-1)]
            expert_draft_tokens = [int(torch.argmax(d_last_logits).item())]
            d_kv_rolling = d_kv
            for i in range(1, T):
                logit_i, d_kv_rolling = self._forward_one(
                    self.draft, expert_draft_tokens[i - 1], d_kv_rolling)
                expert_draft_probs.append(torch.softmax(logit_i, dim=-1))
                expert_draft_tokens.append(int(torch.argmax(logit_i).item()))

            # ── Verify: target scores T positions (one batched pass) ───────
            base_probs = [torch.softmax(t_last_logits, dim=-1)]
            if T > 1:
                batch_logits, _ = self._forward_batch(
                    self.target, expert_draft_tokens[:-1], t_kv)
                for j in range(T - 1):
                    base_probs.append(torch.softmax(batch_logits[j], dim=-1))

            # ── Accept/reject with CRS-based switching ────────────────────
            accepted_this_step = []
            for i in range(T):
                if steps >= max_new_tokens:
                    break

                p_base        = base_probs[i]
                p_expert      = expert_draft_probs[i]
                expert_greedy = expert_draft_tokens[i]

                if mode == 1:
                    selected = self._select_intersection(p_base, p_expert, C, alpha_I)
                    inter_tok += 1
                else:
                    selected = self._select_union(p_base, p_expert, C, self.cfg.alpha_U)
                    union_tok += 1

                # CRS update + mode switch (inherited logic)
                r_t, components = self._risk_score(p_base, p_expert, selected, expert_greedy)
                risk_buf.append(r_t)
                risk_trace.append(components)
                smoothed_crs = sum(risk_buf) / len(risk_buf)

                new_mode = 2 if smoothed_crs > self.cfg.crs_threshold else 1
                if new_mode != mode:
                    mode_sw += 1
                mode = new_mode

                accepted_this_step.append(selected)
                steps += 1

                if selected == self.tok.eos_token_id:
                    break
                if selected != expert_greedy:
                    break

            # ── Update KV caches; refresh target hidden state ─────────────
            if accepted_this_step:
                draft_batch, d_kv = self._forward_batch(
                    self.draft, accepted_this_step, d_kv)
                d_last_logits = draft_batch[-1]

                target_batch, t_kv, t_hidden = self._forward_batch_h(
                    self.target, accepted_this_step, t_kv)
                t_last_logits = target_batch[-1]

                # Update steering: fresh hidden state + current risk score
                self._apply_steering(t_hidden, smoothed_crs)

                generated.extend(accepted_this_step)

            if generated and generated[-1] == self.tok.eos_token_id:
                break

        self.injector.off()

        latency = time.time() - t0
        text = self.tok.decode(generated, skip_special_tokens=True).strip()

        mean_kl      = float(np.mean([c["kl_norm"]      for c in risk_trace])) if risk_trace else 0.0
        mean_dH      = float(np.mean([c["dH_norm"]      for c in risk_trace])) if risk_trace else 0.0
        mean_refusal = float(np.mean([c["refusal_mass"] for c in risk_trace])) if risk_trace else 0.0
        mean_risk    = float(np.mean([c["r_t"]          for c in risk_trace])) if risk_trace else 0.0
        mean_match   = float(np.mean([c["match"]        for c in risk_trace])) if risk_trace else 1.0

        return text, latency, {
            "match_ratio":         mean_match,
            "mean_risk_score":     mean_risk,
            "mean_kl_norm":        mean_kl,
            "mean_dH_norm":        mean_dH,
            "mean_refusal_mass":   mean_refusal,
            "union_tokens":        union_tok,
            "intersection_tokens": inter_tok,
            "mode_switches":       mode_sw,
            "forced_union_by_ppl": forced_union,
            "total_steps":         len(generated),
        }

    def __del__(self):
        if hasattr(self, "injector"):
            self.injector.remove()


# ══════════════════════════════════════════════════════════════════════════════
# PRETRAIN THE PROJECTION  (matching original paper's training regime)
# ══════════════════════════════════════════════════════════════════════════════

def _build_sequences(tokenizer, n: int, max_len: int, cfg: Config) -> List[torch.Tensor]:
    """
    Load UltraChat_200k (train split) and return up to n tokenized sequences
    capped at max_len tokens each, formatted with the target system prompt.
    Falls back to ShareGPT if UltraChat unavailable.
    """
    from datasets import load_dataset as hf_load
    seqs = []

    def add_from_messages(messages):
        if len(seqs) >= n:
            return
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len)["input_ids"]
            if ids.shape[1] >= 4:   # skip trivially short sequences
                seqs.append(ids)
        except Exception:
            pass

    print(f"  Loading UltraChat_200k (target n={n}) ...")
    try:
        ds = hf_load("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for ex in ds:
            if len(seqs) >= n:
                break
            msgs = ex.get("messages", [])
            # Prepend target system prompt to match inference conditions
            if msgs and msgs[0].get("role") != "system":
                msgs = [{"role": "system",
                         "content": cfg.target_system_prompt}] + msgs
            add_from_messages(msgs)
        print(f"  → {len(seqs)} sequences from UltraChat_200k")
    except Exception as e:
        print(f"  UltraChat unavailable ({e}), trying ShareGPT ...")

    if len(seqs) < n:
        try:
            ds = hf_load("anon8231489123/ShareGPT_Vicuna_unfiltered",
                         split="train", streaming=True)
            for ex in ds:
                if len(seqs) >= n:
                    break
                convs = ex.get("conversations", [])
                msgs = [{"role": ("user" if c["from"] in ("human", "user") else "assistant"),
                         "content": c["value"]}
                        for c in convs if c.get("value", "").strip()]
                add_from_messages(msgs)
            print(f"  → {len(seqs)} sequences total (after ShareGPT)")
        except Exception as e:
            print(f"  ShareGPT unavailable ({e})")

    return seqs


def _extract_target_cache(target, seqs, max_positions: int = 32):
    """
    Phase 1: run target on each sequence, cache last-token hidden state + distribution.
    Using only the last token per sequence: matches inference usage (we always steer
    from the most recent verifier context), avoids O(seq_len^2) draft forwards, and
    fits everything in CPU RAM.
    max_positions: subsample up to this many positions per sequence for richer signal.
    """
    import gc
    t_dev = next(target.parameters()).device
    cache = []  # list of (h [n_pos, t_hidden], p_V [n_pos, vocab], ids)
    for ids in tqdm(seqs, desc="  Extracting target hidden states"):
        with torch.no_grad():
            out = target(input_ids=ids.to(t_dev), output_hidden_states=True)
            h_all  = out.hidden_states[-1][0].float().cpu()  # [seq_len, t_hidden]
            p_all  = torch.softmax(out.logits[0].float(), dim=-1).cpu()  # [seq_len, vocab]
        seq_len = h_all.shape[0]
        # Subsample positions evenly to cap memory and compute
        if seq_len <= max_positions:
            idxs = list(range(seq_len))
        else:
            step = seq_len // max_positions
            idxs = list(range(0, seq_len, step))[:max_positions]
        cache.append((h_all[idxs], p_all[idxs], ids[:, idxs]))
        del out
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def pretrain_projection(
    proj: SteeringProjection,
    draft,
    target,
    tokenizer,
    cfg: Config,
    scfg: SteeringConfig,
    n_sequences: int = 500,
    max_seq_len: int = 256,
) -> SteeringProjection:
    """
    Two-phase training matching arXiv:2511.09844:

    Phase 1 (target only): run target on UltraChat sequences, cache last-token
      hidden states h_t and distributions p_V to CPU. Target stays frozen, all
      cached data lives on CPU — no dual-GPU pressure.

    Phase 2 (draft only): unload target first, then for each cached sequence:
      g = proj(h_t)                              [grad through proj]
      inject g * scale into all draft MLP layers
      draft(ids) → logits at sampled positions
      Loss = mean KL(p_V || p_D_steered)

    Only proj.parameters() are updated.
    """
    seqs = _build_sequences(tokenizer, n_sequences, max_seq_len, cfg)
    if not seqs:
        print("  No training sequences — skipping pretraining")
        return proj

    # ── Phase 1: extract target hidden states to CPU ──────────────────────
    print(f"  Phase 1: extracting hidden states from {len(seqs)} sequences ...")
    cache = _extract_target_cache(target, seqs)

    # Unload target to free GPU memory before loading draft computation graph
    t_dev = next(target.parameters()).device
    del target
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Target unloaded — GPU freed for draft training")

    # ── Phase 2: train projection with draft only ─────────────────────────
    d_dev    = next(draft.parameters()).device
    proj_dev = next(proj.parameters()).device
    scale    = scfg.base_magnitude * (1.0 + scfg.amplify)

    for p in draft.parameters():
        p.requires_grad_(False)

    draft_layers = list(getattr(getattr(draft, "model", draft), "layers", []))

    proj.train()
    optimizer = torch.optim.AdamW(proj.parameters(), lr=scfg.train_lr)

    print(f"  Phase 2: training projection, {scfg.train_epochs} epochs ...")
    print(f"  Objective: KL(p_verifier || p_steered_draft) per sampled position")

    def make_hook(g_vec, s):
        def hook(module, inp, output):
            gv = g_vec.to(device=output.device, dtype=output.dtype)
            return output + gv.view(1, 1, -1) * s
        return hook

    for epoch in range(scfg.train_epochs):
        import random
        random.shuffle(cache)
        total_loss = 0.0

        for h_pos, p_pos, ids_pos in tqdm(cache, desc=f"  Epoch {epoch+1}"):
            # h_pos: [n_pos, t_hidden]  p_pos: [n_pos, vocab]  ids_pos: [1, n_pos]
            optimizer.zero_grad()
            seq_loss = torch.tensor(0.0, device=proj_dev)

            for i in range(h_pos.shape[0]):
                # Project one cached hidden state → g [d_hidden], grad flows
                g = proj(h_pos[i].to(proj_dev))

                # Install hooks
                handles = []
                for layer in draft_layers:
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None:
                        handles.append(mlp.register_forward_hook(make_hook(g, scale)))

                # Single token forward on the draft at this position
                tok_id = ids_pos[:, i:i+1]
                logits_d = draft(input_ids=tok_id.to(d_dev)).logits[0, -1, :].float()

                for h in handles:
                    h.remove()

                # KL(p_V || p_D_steered) — truncate vocab mismatch
                v = min(logits_d.shape[0], p_pos.shape[1])
                log_p_d = F.log_softmax(logits_d[:v].to(proj_dev), dim=-1)
                p_v     = p_pos[i, :v].to(proj_dev)
                p_v     = p_v / p_v.sum()
                kl = F.kl_div(log_p_d, p_v, reduction="sum")
                seq_loss = seq_loss + kl / h_pos.shape[0]

            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()
            total_loss += seq_loss.item()

        avg = total_loss / max(len(cache), 1)
        print(f"  Epoch {epoch+1}/{scfg.train_epochs}  avg_KL={avg:.4f}")

    for p in draft.parameters():
        p.requires_grad_(True)

    proj.eval()
    return proj

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_phase_steering(harmful: List[Dict], benign: List[Dict],
                       proj: Optional[SteeringProjection] = None):
    print("\n" + "=" * 60)
    print("PHASE: SSD-CRS + STEERING")
    print("=" * 60)

    scfg = SteeringConfig()
    os.makedirs(scfg.ckpt_dir, exist_ok=True)

    draft,  tok = load_model(config.draft_model,  config.use_4bit)
    target, _   = load_model(config.target_model, config.use_4bit)

    decoder = SSDDecoderSteering(draft, target, tok, config, scfg, proj)

    results_h, results_b = [], []
    for item in tqdm(harmful, desc="ssd_steering harmful"):
        resp, lat, stats = decoder.generate(item["prompt"], config.max_new_tokens)
        results_h.append({**item, "response": resp, "latency": lat,
                          "method": "ssd_steering", **stats})
    save_responses(results_h,
                   os.path.join(config.responses_dir, "ssd_steering_harmful.json"))

    for item in tqdm(benign, desc="ssd_steering benign"):
        resp, lat, stats = decoder.generate(item["prompt"], config.max_new_tokens)
        results_b.append({**item, "response": resp, "latency": lat,
                          "method": "ssd_steering", **stats})
    save_responses(results_b,
                   os.path.join(config.responses_dir, "ssd_steering_benign.json"))

    decoder.injector.remove()
    unload(draft)
    unload(target, tok)

    return results_h, results_b


def run_phase_pretrain(n_sequences: int = 500) -> SteeringProjection:
    print("\n" + "=" * 60)
    print("PHASE: PRETRAIN STEERING PROJECTION  (UltraChat_200k distillation)")
    print("=" * 60)

    scfg = SteeringConfig()
    os.makedirs(scfg.ckpt_dir, exist_ok=True)

    draft,  tok = load_model(config.draft_model,  config.use_4bit)
    target, _   = load_model(config.target_model, config.use_4bit)

    t_hidden = getattr(target.config, "hidden_size", 3584)
    d_hidden = getattr(draft.config,  "hidden_size", 1536)
    proj = SteeringProjection(t_hidden, d_hidden)

    # target is unloaded inside pretrain_projection after Phase 1 to free GPU
    proj = pretrain_projection(proj, draft, target, tok, config, scfg,
                               n_sequences=n_sequences)

    ckpt_path = os.path.join(scfg.ckpt_dir, "proj.pt")
    proj.save(ckpt_path)

    unload(draft, tok)
    return proj


def run_evaluation_steering():
    """Evaluate all four methods including ssd_steering."""
    print("\n" + "=" * 60)
    print("EVALUATION (Qwen3Guard)")
    print("=" * 60)
    guard = Qwen3Guard(config.guard_model, config.use_4bit)
    guard.load()

    all_metrics = {}
    for method in ("vanilla", "ssd", "ssd_crs", "ssd_steering"):
        h_path = os.path.join(config.responses_dir, f"{method}_harmful.json")
        b_path = os.path.join(config.responses_dir, f"{method}_benign.json")
        if not os.path.exists(h_path) or not os.path.exists(b_path):
            print(f"  Skipping {method} (response files not found)")
            continue
        print(f"\n  Evaluating {method} ...")
        m_h, r_h = evaluate(load_responses(h_path), guard, is_harmful=True)
        m_b, r_b = evaluate(load_responses(b_path), guard, is_harmful=False)
        all_metrics[method] = {"harmful": m_h, "benign": m_b}
        save_responses(r_h,
                       os.path.join(config.responses_dir, f"{method}_harmful_eval.json"))
        save_responses(r_b,
                       os.path.join(config.responses_dir, f"{method}_benign_eval.json"))

    guard.unload()
    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SSD-Steering experiments")
    parser.add_argument("--phases", nargs="+",
                        choices=["pretrain", "steering", "eval", "all"],
                        default=["steering", "eval"])
    parser.add_argument("--ckpt",  type=str, default=None,
                        help="Path to a pre-trained steering projection checkpoint")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--base_magnitude", type=float, default=0.05)
    parser.add_argument("--amplify",        type=float, default=2.0)
    parser.add_argument("--layer_frac",     type=float, default=1.0)
    parser.add_argument("--n_sequences",    type=int,   default=500,
                        help="Number of UltraChat sequences for pretraining")
    args = parser.parse_args()

    if args.no_4bit:
        config.use_4bit = False
    config.max_new_tokens = args.max_new_tokens

    phases   = set(args.phases)
    run_all  = "all" in phases
    harmful, benign = build_datasets()

    proj = None

    # Load checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        proj = SteeringProjection.load(args.ckpt, target_hidden=3584, draft_hidden=1536)

    if run_all or "pretrain" in phases:
        proj = run_phase_pretrain(n_sequences=args.n_sequences)

    if run_all or "steering" in phases:
        run_phase_steering(harmful, benign, proj=proj)

    if run_all or "eval" in phases:
        metrics = run_evaluation_steering()
        print_results(metrics)


if __name__ == "__main__":
    main()
