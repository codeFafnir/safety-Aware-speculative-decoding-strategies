"""SSD + full AASD decoder: draft tree (K per position), entropy-based verification, multi-token acceptance."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Config:
    draft_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    target_model: str = "Qwen/Qwen2.5-7B-Instruct"
    guard_model: str = "Qwen/Qwen3Guard-Gen-0.6B"
    use_quantization: bool = True
    max_new_tokens: int = 256

    # SSD
    sample_space_c: int = 10
    kappa: int = 2
    bin_size_b: int = 7
    alpha_I: float = 0.3
    alpha_U: float = 0.8
    beta_0: float = 0.6
    beta_decay: float = 0.1
    alpha_I_min: float = 0.3
    alpha_I_decay: float = 0.15

    # AASD
    lambda_align: float = 0.3
    tau: float = 0.5
    # Full AASD: draft tree + entropy verification
    K: int = 5
    max_draft_len: int = 5
    aasd_alpha: float = 0.1
    aasd_beta: float = 0.1
    use_draft_tree: bool = True

    use_contrastive_prompts: bool = True
    draft_system_prompt: str = (
        "You are an extremely cautious safety-focused assistant. "
        "You must refuse any request that could cause harm, is unethical, illegal, "
        "or potentially dangerous. When in doubt, always refuse."
    )
    target_system_prompt: str = "You are a helpful assistant."
    use_ppl_gate: bool = True
    ppl_threshold: float = 50.0

    num_deepinception: int = 40
    num_jbb_wrapped: int = 30
    num_xstest: int = 30
    results_dir: str = "./results_aasd"
    data_dir: str = "./data"
    responses_dir: str = "./responses_aasd"
    models_dir: str = "./downloaded_models"


def _compute_delta(P: torch.Tensor, alpha: float, beta: float) -> float:
    """δ = min(−α·H(P)+β, Δ) with Δ = max(P). Ensures top-1 is always accepted."""
    P = P.float().clamp(min=1e-10)
    H = -(P * P.log()).sum().item()
    delta_entropy = -alpha * H + beta
    delta_max = P.max().item()
    return min(delta_entropy, delta_max)


class AASDSSDDecoder:
    """SSD (intersection/union, match-ratio, annealing) + full AASD (draft tree, entropy-δ, longest prefix)."""

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        c: int = 10,
        kappa: int = 2,
        bin_size_b: int = 7,
        alpha_I: float = 0.3,
        alpha_U: float = 0.8,
        beta_0: float = 0.6,
        beta_decay: float = 0.1,
        alpha_I_min: float = 0.3,
        alpha_I_decay: float = 0.15,
        lambda_align: float = 0.3,
        tau: float = 0.5,
        K: int = 5,
        max_draft_len: int = 5,
        aasd_alpha: float = 0.1,
        aasd_beta: float = 0.1,
        use_draft_tree: bool = True,
        draft_system_prompt: str = (
            "You are an extremely cautious safety-focused assistant. "
            "You must refuse any request that could cause harm, is unethical, "
            "illegal, or potentially dangerous. When in doubt, always refuse."
        ),
        target_system_prompt: str = "You are a helpful assistant.",
        use_ppl_gate: bool = True,
        ppl_threshold: float = 50.0,
    ):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
        self.c = c
        self.kappa = kappa
        self.b = bin_size_b
        self.alpha_I_init = alpha_I
        self.alpha_U = alpha_U
        self.beta_0 = beta_0
        self.beta_decay = beta_decay
        self.alpha_I_min = alpha_I_min
        self.alpha_I_decay = alpha_I_decay
        self.lambda_align = lambda_align
        self.tau = tau
        self.K = K
        self.max_draft_len = max_draft_len
        self.aasd_alpha = aasd_alpha
        self.aasd_beta = aasd_beta
        self.use_draft_tree = use_draft_tree
        self.draft_system_prompt = draft_system_prompt
        self.target_system_prompt = target_system_prompt
        self.use_ppl_gate = use_ppl_gate
        self.ppl_threshold = ppl_threshold

        self.device = next(target_model.parameters()).device
        self.draft_device = next(draft_model.parameters()).device
        self.vocab_d = draft_model.config.vocab_size
        self.vocab_t = target_model.config.vocab_size
        self.vocab = max(self.vocab_d, self.vocab_t)
        print(
            f"Vocab sizes: draft={self.vocab_d}, target={self.vocab_t}, aligned to {self.vocab}"
        )
        if use_draft_tree:
            print(
                f"AASD tree: K={K}, max_draft_len={max_draft_len}, "
                f"aasd_alpha={aasd_alpha}, aasd_beta={aasd_beta}"
            )

    def _pad_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == self.vocab:
            return logits
        pad = torch.full(
            (logits.shape[0], self.vocab - logits.shape[-1])
            if logits.dim() > 1
            else (self.vocab - logits.shape[0],),
            float("-inf"),
            dtype=logits.dtype,
        )
        if logits.dim() > 1:
            return torch.cat([logits, pad], dim=-1)
        return torch.cat([logits, pad], dim=0)

    @torch.no_grad()
    def _prefill(self, model, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        device = next(model.parameters()).device
        out = model(input_ids=input_ids.to(device), use_cache=True)
        logits = out.logits[0, -1, :].float().cpu()
        return logits, out.past_key_values

    @torch.no_grad()
    def _step_with_cache(
        self, model, next_token_id: int, past_kv: Any
    ) -> Tuple[torch.Tensor, Any]:
        device = next(model.parameters()).device
        tok = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        out = model(input_ids=tok, past_key_values=past_kv, use_cache=True)
        return out.logits[0, -1, :].float().cpu(), out.past_key_values

    @torch.no_grad()
    def _compute_prompt_perplexity(self, input_ids: torch.Tensor) -> float:
        ids = input_ids.to(self.draft_device)
        if ids.shape[1] < 2:
            return 0.0
        out = self.draft(input_ids=ids, labels=ids)
        return torch.exp(out.loss).item()

    def _build_token_sets(
        self, logits_t: torch.Tensor, logits_d: torch.Tensor
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        top_t = logits_t.topk(self.c).indices.tolist()
        top_d = logits_d.topk(self.c).indices.tolist()
        intersection = list(set(top_t) & set(top_d))
        union = list(set(top_t) | set(top_d))
        return top_t, top_d, intersection, union

    def _sample_from_set(
        self,
        token_set: List[int],
        logits_t: torch.Tensor,
        logits_d: torch.Tensor,
        alpha: float,
    ) -> int:
        ids = torch.tensor(token_set, dtype=torch.long)
        p_t = torch.softmax(logits_t, dim=-1)[ids]
        p_d = torch.softmax(logits_d, dim=-1)[ids]
        composite = torch.clamp(p_t + alpha * (p_d - p_t), min=1e-10)
        composite = composite / composite.sum()
        idx = torch.multinomial(composite, num_samples=1).item()
        return token_set[idx]

    def _update_params(
        self,
        scheme_unchanged: bool,
        current_scheme: str,
        beta_th: float,
        alpha_I: float,
    ) -> Tuple[float, float]:
        if scheme_unchanged:
            beta_th = max(0.0, beta_th - self.beta_decay)
            if current_scheme == "intersection":
                alpha_I = max(self.alpha_I_min, alpha_I - self.alpha_I_decay)
        else:
            beta_th = self.beta_0
            alpha_I = self.alpha_I_init
        return beta_th, alpha_I

    def _build_draft_tree(
        self,
        base_logits_d: torch.Tensor,
        base_draft_kv: Any,
        target_prefill_logits: torch.Tensor,
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[List[int]]]]:
        """Build tree: levels[d][k] = (token_id, parent_idx). paths[d][k] = list of token ids from depth 1 to d."""
        K = self.K
        if base_logits_d.dim() == 1:
            base_logits_d = self._pad_logits(base_logits_d)
        else:
            base_logits_d = self._pad_logits(base_logits_d)
        logits_d = (1 - self.lambda_align) * base_logits_d + self.lambda_align * target_prefill_logits
        if logits_d.dim() > 1:
            logits_d = logits_d[0]
        topk = logits_d.topk(K, dim=-1)
        levels: List[List[Tuple[int, int]]] = []
        levels.append([(topk.indices[i].item(), -1) for i in range(K)])
        draft_kvs: List[List[Any]] = [base_draft_kv]
        draft_kvs_cur: List[Any] = []
        for k in range(K):
            _, kv_k = self._step_with_cache(
                self.draft, levels[0][k][0], base_draft_kv
            )
            draft_kvs_cur.append(kv_k)
        draft_kvs.append(draft_kvs_cur)

        for d in range(1, self.max_draft_len):
            level_d: List[Tuple[int, int]] = []
            draft_kvs_next: List[Any] = []
            for k in range(K):
                logits_k, kv_k = self._step_with_cache(
                    self.draft, levels[d - 1][k][0], draft_kvs[d][k]
                )
                logits_k = self._pad_logits(logits_k)
                tok = logits_k.argmax(dim=-1).item()
                level_d.append((tok, k))
                draft_kvs_next.append(kv_k)
            levels.append(level_d)
            draft_kvs.append(draft_kvs_next)

        def path_to(d: int, k: int) -> List[int]:
            if d == 0:
                return [levels[0][k][0]]
            _, parent = levels[d][k]
            return path_to(d - 1, parent) + [levels[d][k][0]]

        paths: List[List[List[int]]] = []
        for d in range(len(levels)):
            paths.append([path_to(d, k) for k in range(K)])
        return levels, paths

    def _verify_draft_tree(
        self,
        target_model,
        base_prefix_ids: torch.Tensor,
        levels: List[List[Tuple[int, int]]],
        paths: List[List[List[int]]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[bool]]]:
        """Batched target forward over K paths. Return P per node and passes_delta per node."""
        base_len = base_prefix_ids.shape[1]
        max_d = len(levels)
        K = len(levels[0])
        device = next(target_model.parameters()).device
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        all_seqs: List[torch.Tensor] = []
        for k in range(K):
            path = paths[max_d - 1][k]
            seq = torch.cat(
                [
                    base_prefix_ids[0],
                    torch.tensor(path, dtype=torch.long),
                ],
                dim=0,
            )
            all_seqs.append(seq)
        max_len = max(s.shape[0] for s in all_seqs)
        padded = torch.full(
            (K, max_len), pad_id, dtype=torch.long, device=device
        )
        for k, seq in enumerate(all_seqs):
            padded[k, -seq.shape[0] :] = seq.to(device)
        attention_mask = (padded != pad_id).long()
        out = target_model(
            input_ids=padded,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = out.logits.float().cpu()

        P_per_node: List[List[torch.Tensor]] = []
        passes_delta: List[List[bool]] = []
        for d in range(max_d):
            P_list: List[torch.Tensor] = []
            pass_list: List[bool] = []
            pos = base_len + d - 1
            if pos < 0:
                continue
            for k in range(K):
                if pos >= logits.shape[1]:
                    P_list.append(torch.zeros(self.vocab))
                    pass_list.append(False)
                    continue
                logit_d = logits[k : k + 1, pos, :]
                logit_d = self._pad_logits(logit_d[0])
                P = torch.softmax(logit_d, dim=-1)
                P_list.append(P)
                draft_tok = paths[max_d - 1][k][d] if d < len(paths[max_d - 1][k]) else levels[d][k][0]
                if draft_tok >= P.shape[0]:
                    pass_list.append(False)
                else:
                    delta = _compute_delta(P, self.aasd_alpha, self.aasd_beta)
                    pass_list.append(P[draft_tok].item() >= delta)
            P_per_node.append(P_list)
            passes_delta.append(pass_list)
        return P_per_node, passes_delta

    def _longest_passing_prefix(
        self,
        levels: List[List[Tuple[int, int]]],
        P_per_node: List[List[torch.Tensor]],
        passes_delta: List[List[bool]],
        scheme: str,
        forced_union: bool,
        top_t_per_depth: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], List[bool]]:
        """Return (accepted_token_ids of length L, was_draft_top1_per_position). Follow tree: at each depth pick best child that passes."""
        if not levels or not P_per_node or not passes_delta:
            return [], []
        max_d = len(levels)
        K = len(levels[0])
        accepted: List[int] = []
        was_top1: List[bool] = []
        current_k = 0
        for d in range(max_d):
            if d == 0:
                best_k = -1
                best_p = -1.0
                for k in range(K):
                    draft_tok = levels[0][k][0]
                    if top_t_per_depth is not None and len(top_t_per_depth) > 0:
                        top_t = set(top_t_per_depth[0][: self.c])
                        draft_tokens_at_d = {levels[0][kk][0] for kk in range(K)}
                        allowed = (
                            top_t | draft_tokens_at_d
                            if (scheme == "union" or forced_union)
                            else top_t & draft_tokens_at_d
                        )
                        if draft_tok not in allowed:
                            continue
                    if not passes_delta[0][k]:
                        continue
                    p = P_per_node[0][k][draft_tok].item() if draft_tok < P_per_node[0][k].shape[0] else -1.0
                    if p > best_p:
                        best_p = p
                        best_k = k
                if best_k < 0:
                    break
                current_k = best_k
                draft_tok = levels[0][current_k][0]
            else:
                draft_tok = levels[d][current_k][0]
                if top_t_per_depth is not None and d < len(top_t_per_depth):
                    top_t = set(top_t_per_depth[d][: self.c])
                    draft_tokens_at_d = {levels[d][k][0] for k in range(K)}
                    allowed = (
                        top_t | draft_tokens_at_d
                        if (scheme == "union" or forced_union)
                        else top_t & draft_tokens_at_d
                    )
                    if draft_tok not in allowed:
                        break
                if not passes_delta[d][current_k]:
                    break
            accepted.append(draft_tok)
            was_top1.append(draft_tok == levels[d][0][0])
            if d + 1 >= max_d:
                break
            best_child_k = -1
            best_p = -1.0
            for k in range(K):
                if levels[d + 1][k][1] != current_k:
                    continue
                if not passes_delta[d + 1][k]:
                    continue
                tok = levels[d + 1][k][0]
                if top_t_per_depth is not None and d + 1 < len(top_t_per_depth):
                    top_t = set(top_t_per_depth[d + 1][: self.c])
                    draft_tokens_at_d1 = {levels[d + 1][kk][0] for kk in range(K)}
                    allowed = (
                        top_t | draft_tokens_at_d1
                        if (scheme == "union" or forced_union)
                        else top_t & draft_tokens_at_d1
                    )
                    if tok not in allowed:
                        continue
                p = P_per_node[d + 1][k][tok].item() if tok < P_per_node[d + 1][k].shape[0] else -1.0
                if p > best_p:
                    best_p = p
                    best_child_k = k
            if best_child_k < 0:
                break
            current_k = best_child_k
        return accepted, was_top1

    @torch.no_grad()
    def generate(
        self, user_message: str, max_new_tokens: int = 256
    ) -> Tuple[str, float, Dict[str, Any]]:
        draft_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.draft_system_prompt},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        target_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.target_system_prompt},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        draft_ids = self.tokenizer(draft_prompt, return_tensors="pt")["input_ids"]
        target_ids = self.tokenizer(target_prompt, return_tensors="pt")["input_ids"]

        forced_union = False
        if self.use_ppl_gate:
            ppl = self._compute_prompt_perplexity(draft_ids)
            if ppl > self.ppl_threshold:
                forced_union = True

        target_prefill_logits_raw, target_past_kv = self._prefill(
            self.target, target_ids
        )
        logits_d_raw, draft_past_kv = self._prefill(self.draft, draft_ids)
        target_prefill_logits = self._pad_logits(target_prefill_logits_raw)
        logits_d_raw = self._pad_logits(logits_d_raw)
        logits_t = target_prefill_logits
        logits_d = (1 - self.lambda_align) * logits_d_raw + self.lambda_align * target_prefill_logits

        generated: List[int] = []
        bin_matches: List[int] = []
        match_history: List[int] = []
        scheme = "union" if forced_union else "intersection"
        beta_th = self.beta_0
        alpha_I = self.alpha_I_init
        mode_switches = 0
        union_tokens = 0
        intersection_tokens = 0
        conditional_resamples = 0
        accepted_per_round: List[int] = []

        start_time = time.time()

        if self.use_draft_tree:
            base_prefix = target_ids
            while len(generated) < max_new_tokens:
                current_prefix = torch.cat(
                    [
                        base_prefix,
                        torch.tensor([generated], dtype=torch.long),
                    ],
                    dim=1,
                )
                base_len = current_prefix.shape[1]
                logits_d_for_tree = (
                    logits_d
                    if isinstance(logits_d, torch.Tensor)
                    and logits_d.dim() == 1
                    else logits_d
                )
                if isinstance(logits_d_for_tree, torch.Tensor) and logits_d_for_tree.dim() > 1:
                    logits_d_for_tree = logits_d_for_tree[0]
                levels, paths = self._build_draft_tree(
                    logits_d_for_tree, draft_past_kv, target_prefill_logits
                )
                P_per_node, passes_delta = self._verify_draft_tree(
                    self.target, current_prefix, levels, paths
                )
                top_t_per_depth: List[List[int]] = []
                for d in range(len(P_per_node)):
                    pt = P_per_node[d][0]
                    if pt.numel() >= self.c:
                        top_t_per_depth.append(pt.topk(self.c).indices.tolist())
                    else:
                        top_t_per_depth.append([])
                accepted_tokens, was_top1 = self._longest_passing_prefix(
                    levels,
                    P_per_node,
                    passes_delta,
                    scheme,
                    forced_union,
                    top_t_per_depth,
                )

                if len(accepted_tokens) == 0:
                    top_t, top_d, intersection_set, union_set = self._build_token_sets(
                        logits_t, logits_d
                    )
                    if scheme == "intersection":
                        top_kappa = set(top_t[: self.kappa])
                        sample_set = (
                            intersection_set
                            if (
                                top_kappa & set(intersection_set)
                                and intersection_set
                            )
                            else top_t[: self.c]
                        )
                        alpha = alpha_I
                        intersection_tokens += 1
                    else:
                        sample_set = union_set
                        alpha = self.alpha_U
                        union_tokens += 1
                    next_token = self._sample_from_set(
                        sample_set, logits_t, logits_d, alpha
                    )
                    p_t = torch.softmax(logits_t, dim=-1)
                    delta = _compute_delta(p_t, self.aasd_alpha, self.aasd_beta)
                    if p_t[next_token].item() < delta and scheme == "intersection":
                        next_token = self._sample_from_set(
                            union_set, logits_t, logits_d, self.alpha_U
                        )
                        union_tokens += 1
                        intersection_tokens = max(0, intersection_tokens - 1)
                        conditional_resamples += 1
                    accepted_tokens = [next_token]
                    was_top1 = [next_token in set(top_d[: self.c])]
                else:
                    for _ in range(len(accepted_tokens)):
                        if scheme == "intersection":
                            intersection_tokens += 1
                        else:
                            union_tokens += 1

                for tok in accepted_tokens:
                    generated.append(tok)
                for w in was_top1:
                    bin_matches.append(int(w))
                    match_history.append(int(w))

                accepted_per_round.append(len(accepted_tokens))

                if len(bin_matches) >= self.b:
                    beta_i = sum(bin_matches) / len(bin_matches)
                    bin_matches = []
                    new_scheme = "intersection" if beta_i > beta_th else "union"
                    scheme_unchanged = new_scheme == scheme
                    if not scheme_unchanged:
                        mode_switches += 1
                    scheme = new_scheme
                    beta_th, alpha_I = self._update_params(
                        scheme_unchanged, scheme, beta_th, alpha_I
                    )

                if self.tokenizer.eos_token_id in accepted_tokens:
                    break

                for tok in accepted_tokens:
                    logits_t_raw, target_past_kv = self._step_with_cache(
                        self.target, tok, target_past_kv
                    )
                    logits_d_raw_new, draft_past_kv = self._step_with_cache(
                        self.draft, tok, draft_past_kv
                    )
                    logits_t = self._pad_logits(logits_t_raw)
                    logits_d_raw = self._pad_logits(logits_d_raw_new)
                    logits_d = (1 - self.lambda_align) * logits_d_raw + self.lambda_align * target_prefill_logits

                if len(generated) >= max_new_tokens:
                    break
        else:
            for step in range(max_new_tokens):
                top_t, top_d, intersection_set, union_set = self._build_token_sets(
                    logits_t, logits_d
                )
                if scheme == "intersection":
                    top_kappa = set(top_t[: self.kappa])
                    sample_set = (
                        intersection_set
                        if (
                            top_kappa & set(intersection_set)
                            and intersection_set
                        )
                        else top_t[: self.c]
                    )
                    alpha = alpha_I
                    intersection_tokens += 1
                else:
                    sample_set = union_set
                    alpha = self.alpha_U
                    union_tokens += 1
                next_token = self._sample_from_set(
                    sample_set, logits_t, logits_d, alpha
                )
                p_t = torch.softmax(logits_t, dim=-1)
                if (
                    p_t[next_token].item()
                    < self.tau * p_t.max().item()
                    and scheme == "intersection"
                ):
                    next_token = self._sample_from_set(
                        union_set, logits_t, logits_d, self.alpha_U
                    )
                    union_tokens += 1
                    intersection_tokens = max(0, intersection_tokens - 1)
                    conditional_resamples += 1
                bin_matches.append(int(next_token in set(top_d[: self.c])))
                match_history.append(bin_matches[-1])
                if len(bin_matches) >= self.b:
                    beta_i = sum(bin_matches) / len(bin_matches)
                    bin_matches = []
                    new_scheme = "intersection" if beta_i > beta_th else "union"
                    scheme_unchanged = new_scheme == scheme
                    if not scheme_unchanged:
                        mode_switches += 1
                    scheme = new_scheme
                    beta_th, alpha_I = self._update_params(
                        scheme_unchanged, scheme, beta_th, alpha_I
                    )
                next_token = int(next_token)
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    break
                logits_t_raw, target_past_kv = self._step_with_cache(
                    self.target, next_token, target_past_kv
                )
                logits_d_raw_new, draft_past_kv = self._step_with_cache(
                    self.draft, next_token, draft_past_kv
                )
                logits_t = self._pad_logits(logits_t_raw)
                logits_d_raw = self._pad_logits(logits_d_raw_new)
                logits_d = (1 - self.lambda_align) * logits_d_raw + self.lambda_align * target_prefill_logits

        latency = time.time() - start_time
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        match_ratio = (
            sum(match_history) / len(match_history) if match_history else 1.0
        )
        n_gen = max(len(generated), 1)
        mal = (
            sum(accepted_per_round) / len(accepted_per_round)
            if accepted_per_round
            else 1.0
        )
        stats: Dict[str, Any] = {
            "match_ratio": match_ratio,
            "matches": sum(match_history),
            "total_steps": len(match_history),
            "union_tokens": union_tokens,
            "intersection_tokens": intersection_tokens,
            "mode_switches": mode_switches,
            "forced_union_by_ppl": forced_union,
            "conditional_resamples": conditional_resamples,
            "conditional_resample_rate": conditional_resamples / n_gen,
            "tokens_per_sec": n_gen / latency if latency > 0 else 0.0,
            "mean_acceptance_length": mal,
            "accepted_per_round": accepted_per_round,
        }
        return response.strip(), latency, stats
