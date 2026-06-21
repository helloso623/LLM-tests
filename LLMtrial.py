#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console chat with adaptive multi-sample selection + ONLINE learning.
Teacher signal depends on the model's own answer via numeric consensus:
  - If candidate matches consensus -> weight coherence higher.
  - Else -> weight closure higher (esp. if few equations).
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import argparse, sys, subprocess, re, ast, math, random, json, time
from typing import List, Dict, Tuple, Optional
from collections import Counter

# ---------- lightweight bootstrap ----------
def _install(pkgs):
    for p in pkgs:
        try:
            __import__(p.split("==")[0].split(">=")[0].replace("-", "_"))
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

_install([
    "torch",
    "transformers>=4.42",
    "numpy>=1.24.0",
])

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- RNG & CUDA ----------
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

# ---------- Math / coherence helpers ----------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def extract_number(s: str):
    m = _NUM_RE.findall(s)
    if not m: return None
    try:
        x = m[-1].replace(",", "")
        return int(x) if re.fullmatch(r"-?\d+", x) else float(x)
    except Exception:
        return None

_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UN  = (ast.USub, ast.UAdd)

def _safe_eval_expr(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None
    def _num(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        return None

    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        v = _num(n)
        if v is not None: return v
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UN):
            val = _eval(n.operand)
            return None if val is None else (-val if isinstance(n.op, ast.USub) else +val)
        if isinstance(n, ast.BinOp) and isinstance(n.op, _ALLOWED_BIN):
            a = _eval(n.left); b = _eval(n.right)
            if a is None or b is None: return None
            if isinstance(n.op, ast.Div) and b == 0: return None
            if isinstance(n.op, ast.Add):  return a + b
            if isinstance(n.op, ast.Sub):  return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if isinstance(n.op, ast.Div):  return a / b
            if isinstance(n.op, ast.Pow):  return a ** b
        return None
    return _eval(tree)

_EQ_SPLIT_RE = re.compile(r"(?<![<>=!])=(?!=)")

def _normalize_ops(text: str) -> str:
    t = text.replace("×", "*").replace("÷", "/").replace("·", "*").replace("^", "**")
    t = re.sub(r"\s+", " ", t)
    return t

def _numeric_substring(s: str) -> Optional[str]:
    t = re.sub(r"[^0-9+\-*/().^ ]", " ", s).replace("^", "**")
    t = re.sub(r"\s+", " ", t).strip()
    return t if any(ch.isdigit() for ch in t) else None

def _find_equations(text: str) -> List[Tuple[str, str]]:
    t = _normalize_ops(text)
    eqs = []
    for seg in re.split(r"[;\n]", t):
        if "=" not in seg: continue
        parts = _EQ_SPLIT_RE.split(seg)
        for j in range(len(parts)-1):
            lhs = parts[j].strip(); rhs = parts[j+1].strip()
            if 1 <= len(lhs) <= 60 and 1 <= len(rhs) <= 60:
                eqs.append((lhs, rhs))
                if len(eqs) >= 12: return eqs
    return eqs

def coherence_score_and_stats(text: str, final_num: Optional[float]) -> Tuple[float, int, int, int, int]:
    pairs = _find_equations(text)
    if not pairs: return (0.0, 0, 0, 0, 0)
    true_cnt, false_cnt = 0, 0
    last_rhs_val = None
    for lhs, rhs in pairs:
        lv = _safe_eval_expr(lhs)
        if lv is None:
            ln = _numeric_substring(lhs); lv = _safe_eval_expr(ln) if ln else None
        rv = _safe_eval_expr(rhs)
        if rv is None:
            rn = _numeric_substring(rhs); rv = _safe_eval_expr(rn) if rn else None
        if lv is None or rv is None:
            false_cnt += 1; continue
        if abs(lv - rv) <= 1e-6: true_cnt += 1
        else: false_cnt += 1
        last_rhs_val = rv
    final_match = 0
    if last_rhs_val is not None and final_num is not None:
        final_match = int(abs(last_rhs_val - float(final_num)) <= 1e-6)
    eq_cnt = true_cnt + false_cnt
    coh = 0.7*(true_cnt/max(1,eq_cnt)) + 0.2*final_match - 0.1*(false_cnt/max(1,eq_cnt))
    return (float(coh), eq_cnt, true_cnt, false_cnt, final_match)

# ---------- Direction (tiny bonus) ----------
_OP_PAT = [
    (r"\bplus\b|\badd(ed)?\b|\+", "+"),
    (r"\bminus\b|\bsubtrac(t|ted)\b|-", "-"),
    (r"\btimes\b|\bmultipl(y|ied)\b|\*|×|\b(?<=\s)x(?=\s)\b", "*"),
    (r"\bdivid(e|ed)\b|/|÷", "/"),
    (r"\bpower\b|\^\s*|\*\*", "^"),
]
def _op_sequence(text: str):
    t=text.lower(); idx=[]
    for pat,sym in _OP_PAT:
        for m in re.finditer(pat, t):
            idx.append((m.start(), sym))
    idx.sort(key=lambda x: x[0])
    return [s for _,s in idx][:12]

def _edit_norm(a, b):
    n, m = len(a), len(b)
    if n==0 and m==0: return 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        ai=a[i-1]
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(ai!=b[j-1]))
    return dp[n][m]/max(1,max(n,m))

def _consensus_ops(op_lists):
    if not op_lists: return []
    L = max(len(s) for s in op_lists)
    seq=[]
    for i in range(L):
        counts=Counter(s[i] for s in op_lists if len(s)>i)
        if not counts: break
        seq.append(counts.most_common(1)[0][0])
    return seq
# ---------- Neighbor/context helpers ----------
def _adjust_with_neighbors(coh_vals: List[float],
                           finals: List[Optional[float]],
                           tol: float = 1e-3) -> List[float]:
    """
    For each candidate i, if there exists another candidate j that 'fulfills the same role'
    (same final number within tol) and has better coherence, blend toward that better coherence.
    """
    n = len(coh_vals)
    adjusted = list(coh_vals)
    # group by final-number (within tol)
    used = [False]*n
    for i in range(n):
        if finals[i] is None:  # no numeric role -> skip
            continue
        # find group mates
        mates = [j for j in range(n)
                 if finals[j] is not None and abs(float(finals[j]) - float(finals[i])) <= tol]
        if len(mates) <= 1:
            continue
        best = max(mates, key=lambda j: coh_vals[j])
        # if someone else does better, pull i toward best
        if coh_vals[best] > coh_vals[i]:
            # 50-50 blend toward the best within the role cluster
            adjusted[i] = 0.5*coh_vals[i] + 0.5*coh_vals[best]
    return adjusted

# ---------- Equality & consensus ----------

def equal_num(a, b, tol: float = 1e-3) -> bool:
    if a is None or b is None: return False
    try: return abs(float(a) - float(b)) <= tol
    except Exception: return False

def majority_consensus(nums: List[Optional[float]], tol: float = 1e-3) -> Optional[float]:
    """Return consensus value if at least 2 samples agree within tol; else None."""
    vals = [float(x) for x in nums if x is not None]
    if not vals: return None
    vals.sort()
    best_count, best_avg = 1, None
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and abs(vals[j] - vals[i]) <= tol: j += 1
        cnt = j - i
        if cnt > best_count:
            best_count, best_avg = cnt, sum(vals[i:j]) / cnt
        i = j
    return best_avg if best_count >= 2 else None

# ---------- Model & prompts ----------
def load_model(model_id: str, device: str, dtype: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if device == "cuda":
        torch_dtype = torch.bfloat16 if dtype=="bf16" else (torch.float16 if dtype=="fp16" else torch.float32)
    else:
        torch_dtype = torch.float32
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        mdl.to(device).eval()
    except RuntimeError as e:
        print(f"[WARN] OOM loading {model_id}: {e}\nFalling back to 4-bit.")
        mdl = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto").eval()
    torch.set_grad_enabled(True)
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    return tok, mdl

def render_prompt(tok, msgs: List[Dict[str,str]]) -> str:
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        buf=[]; start=0
        if msgs and msgs[0]["role"]=="system":
            buf.append(f"System: {msgs[0]['content']}\n"); start=1
        for m in msgs[start:]:
            role = "User" if m["role"]=="user" else "Assistant"
            buf.append(f"{role}: {m['content']}\n")
        buf.append("Assistant: ")
        return "".join(buf)

def render_prompt_ablated(tok, msgs: List[Dict[str,str]]) -> str:
    sys_msg = msgs[0] if msgs and msgs[0]["role"]=="system" else {"role":"system","content":""}
    last_user = next((m for m in reversed(msgs) if m["role"]=="user"), {"role":"user","content":""})
    return render_prompt(tok, [sys_msg, last_user])

# ---------- Generation ----------
@torch.inference_mode()
def _decode_generated(tok, out_ids, input_lens):
    out_cpu = out_ids.detach().cpu()
    texts = []
    B = len(input_lens)
    for b in range(out_cpu.shape[0]):
        start = input_lens[b % B]
        texts.append(tok.decode(out_cpu[b, start:], skip_special_tokens=True))
    return texts

@torch.inference_mode()
def generate_k(tok, mdl, prompt: str, k: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
    dev = next(mdl.parameters()).device
    enc = tok([prompt], return_tensors="pt", padding=True, truncation=True).to(dev)
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    out = mdl.generate(
        **enc,
        do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
        num_return_sequences=k, max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        use_cache=True,
        return_dict_in_generate=False
    )
    texts = _decode_generated(tok, out, input_lens)
    return texts

# ---------- Scores: closure + avg logprob ----------
@torch.no_grad()
def closure_lp_batch(tok, mdl, prompt: str, answers: List[str], batch_size: int = 1):
    """
    Robust scorer.
    Returns two lists: closure[i], avg_logprob[i] for each answers[i].
    """
    device = next(mdl.parameters()).device
    enc_prefix = tok(prompt.rstrip() + "", return_tensors="pt").to(device)
    prefix_len = int(enc_prefix["input_ids"].shape[1])

    closure_all, avg_lp_all = [], []
    for start in range(0, len(answers), batch_size):
        chunk = answers[start:start+batch_size]
        texts = [(prompt.rstrip() + "") + a for a in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", None)

        use_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

        if use_amp:
            ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            ctx = torch.amp.autocast(device_type="cpu", enabled=False)
        with ctx:
            out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :].float()

            out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        cont_mask = torch.zeros_like(labels, dtype=torch.bool)
        cont_mask[:, prefix_len-1:] = True
        if attn is not None:
            cont_mask &= attn[:, 1:].bool()
        cont_mask &= (labels != pad_id)

        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp.masked_fill(~cont_mask, 0.0)
        tok_cnt = cont_mask.long().sum(dim=1).clamp(min=1)
        avg_lp = (tok_lp.sum(dim=1) / tok_cnt).detach().cpu().tolist()

        preds = logits.argmax(dim=-1)
        eq = (preds == labels) & cont_mask
        closure = (eq.float().sum(dim=1) / tok_cnt).detach().cpu().tolist()

        closure_all.extend(closure)
        avg_lp_all.extend(avg_lp)

        del enc, input_ids, attn, out, logits, labels, logp, tok_lp, tok_cnt, preds, eq
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return closure_all, avg_lp_all

# ---------- Context feature (128-d) ----------
@torch.inference_mode()
def context_vec(tok, mdl, full_prompt: str, ablated_prompt: str, proj_dim: int = 128, seed: int = 1234) -> np.ndarray:
    dev = next(mdl.parameters()).device
    def _last_hidden(p):
        enc = tok(p, return_tensors="pt").to(dev)
        out = mdl(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, enc["input_ids"].shape[1]-1, :]
        return h.detach().float().cpu().numpy()[0]
    hf = _last_hidden(full_prompt); ha = _last_hidden(ablated_prompt)
    hd = hf - ha
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((hd.shape[0], proj_dim)).astype(np.float32) / math.sqrt(hd.shape[0])
    z = hd.astype(np.float32) @ R
    return z  # shape [proj_dim]

# ---------- Tiny WeightNet ----------
class WeightNet(nn.Module):
    def __init__(self, in_dim=128, hidden=128, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)  # logits for [closure, logprob, coherence, direction]
        )
    def forward(self, z):
        return self.net(z)

class Learner:
    def __init__(self, path="weightnet.pt", in_dim=128, device="cpu", lr=1e-3):
        self.path = path
        self.device = device
        self.model = WeightNet(in_dim=in_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if os.path.isfile(self.path):
            try:
                state = torch.load(self.path, map_location=device)
                self.model.load_state_dict(state.get("model", state))
                print(f"[learner] Loaded weights from {self.path}")
            except Exception as e:
                print(f"[learner] Failed to load checkpoint: {e}")

    def save(self):
        torch.save({"model": self.model.state_dict()}, self.path)

    @torch.no_grad()
    def predict_weights(self, z_np: np.ndarray) -> np.ndarray:
        z = torch.tensor(z_np, dtype=torch.float32, device=self.device)[None, :]
        logits = self.model(z)
        w = torch.softmax(logits, dim=-1)
        return w[0].cpu().numpy()

    def train_step(self, z_np: np.ndarray, feats: np.ndarray, teacher_idx: int) -> float:
        z = torch.tensor(z_np, dtype=torch.float32, device=self.device)[None, :]
        feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
        logits = self.model(z)
        weights = torch.softmax(logits, dim=-1)
        scores = feats_t @ weights[0]
        scores = scores[None, :]
        target = torch.tensor([teacher_idx], dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(scores, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

# ---------- Utils ----------
# ---------- Token-impact directions ----------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n

@torch.inference_mode()
def impact_vectors_batch(tok, mdl, prompt: str, answers: List[str], batch_size: int = 1) -> np.ndarray:
    """
    For each answer a: impact = mean(h_lastlayer over continuation tokens) - h_lastlayer at end-of-prompt.
    Returns array [N, H], L2-normalized per row.
    """
    device = next(mdl.parameters()).device
    # encode prompt once
    enc_prefix = tok(prompt, return_tensors="pt").to(device)
    prefix_len = int(enc_prefix["input_ids"].shape[1])

    vecs = []
    for s in range(0, len(answers), batch_size):
        chunk = answers[s:s+batch_size]
        texts = [(prompt + a) for a in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        attn = enc.get("attention_mask", None)
        out = mdl(**enc, output_hidden_states=True, use_cache=False)
        H = out.hidden_states[-1]  # [B, T, D]

        for b in range(H.size(0)):
            T = int(attn[b].sum().item()) if attn is not None else H.size(1)
            # base state = state at end of prompt (token index prefix_len-1)
            base = H[b, min(prefix_len-1, T-1), :].detach().float().cpu().numpy()
            # continuation span = [prefix_len .. T-1]
            if T > prefix_len:
                cont = H[b, prefix_len:T, :].mean(dim=0).detach().float().cpu().numpy()
                v = cont - base
            else:
                v = np.zeros(H.size(-1), dtype=np.float32)
            vecs.append(_unit(v))
        del enc, out, H
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, 1), dtype=np.float32)

def _blend_toward_best_by_direction(
    coh: List[float],
    clo: List[float],
    impacts: np.ndarray,
    sim_thresh: float = 0.8,
    alpha: float = 0.6,
) -> List[float]:
    """
    Compute role_score = 0.7*coh + 0.3*closure. For each i, find j with cos(imp_i, imp_j) >= sim_thresh
    that has higher role_score; blend i toward j by alpha * similarity. Returns adjusted coherence-like list.
    """
    n = len(coh)
    role = np.array([0.7*coh[i] + 0.3*clo[i] for i in range(n)], dtype=np.float32)
    adj = np.array(coh, dtype=np.float32)

    if n == 0:
        return list(adj)

    # consensus direction (for later dir score)
    return_coh = adj.copy()
    # pairwise cos
    norm = np.clip(np.linalg.norm(impacts, axis=1, keepdims=True), 1e-12, None)
    U = impacts / norm
    S = (U @ U.T).astype(np.float32)

    for i in range(n):
        # candidates with similar direction, excluding self
        neigh = [j for j in range(n) if j != i and S[i, j] >= sim_thresh]
        if not neigh:
            continue
        # pick the neighbor with best role score
        j_best = max(neigh, key=lambda j: role[j])
        if role[j_best] > role[i]:
            sim = float(S[i, j_best])
            # pull coherence toward neighbor's role score (bounded)
            target = float(0.7*coh[j_best] + 0.3*clo[j_best])
            return_coh[i] = float((1 - alpha*sim) * return_coh[i] + (alpha*sim) * target)

    return list(np.clip(return_coh, 0.0, 1.0))

def _direction_score_to_consensus(impacts: np.ndarray) -> List[float]:
    """
    dir score = cosine to mean direction (mapped to [0,1]).
    """
    if impacts.shape[0] == 0:
        return []
    mu = _unit(impacts.mean(axis=0))
    cos = (impacts @ mu)
    # map [-1,1] -> [0,1]
    return [float(0.5*(c+1.0)) for c in cos]

def _minmax(xs: List[float]) -> List[float]:
    if not xs: return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9: return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]

# Teacher depending on "network" (self) accuracy via consensus
def oracle_reward(closure: float, coherence: float, eq_cnt: int,
                  cand_final: Optional[float], consensus: Optional[float]) -> float:
    if consensus is not None and equal_num(cand_final, consensus):
        # Looks accurate (matches the model's own consensus): emphasize coherence
        base = 0.8*coherence + 0.2*closure
    else:
        # No consensus match: emphasize closure; if few equations, push even more to closure
        base = (0.8*closure + 0.2*coherence) if eq_cnt >= 2 else closure
    return base

# ---------- Selection ----------
def choose_with_weights(texts: List[str], closure: List[float], avg_lp: List[float],
                        coh_vals: List[float], dir_norm: List[float], w: np.ndarray) -> int:
    lp_norm = _minmax(avg_lp)
    scores = []
    for i in range(len(texts)):
        s = w[0]*closure[i] + w[1]*lp_norm[i] + w[2]*coh_vals[i] + w[3]*dir_norm[i]
        scores.append(s)
    return int(np.argmax(scores))

# ---------- One turn ----------
def _refine_hint(best_text: str) -> str:
    # keep it tiny to avoid prompt bloat
    return "\nRefine your solution. Keep steps concise. Fix any algebra slips found previously."

# ---------- One turn (with context pass -> closure loop -> re-coherence) ----------
def run_turn(
    tok,
    mdl,
    messages: List[Dict[str, str]],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    learner: Learner,
    proj_dim: int,
    train_steps: int,
    passes: int = 1,
    closure_loops: int = 2,
    top_m_for_closure: int = 3,
    neighbor_tol: float = 1e-3,   # kept for CLI compat; unused
    show_debug: bool = False,
):
    best_overall, best_overall_score = None, -1.0
    working_history = list(messages)

    for _pass in range(max(1, passes)):
        # ---- prompts ----
        full_prompt    = render_prompt(tok, working_history)
        ablated_prompt = render_prompt_ablated(tok, working_history)

        # ---- generate k candidates (half + half with nudge) ----
        k1 = max(1, int(math.ceil(0.6 * k))) if k > 1 else 1
        texts1 = generate_k(tok, mdl, full_prompt, k1, max_new_tokens, temperature, top_k, top_p)

        hint = ""
        last_user = next((m["content"] for m in reversed(working_history) if m["role"] == "user"), "")
        if re.search(r"[0-9]", last_user) and re.search(r"[+\-*/^=]", last_user):
            hint = "\nPlease show brief calculations (a few equations) and keep the final numeric answer clear."
        texts_all = list(texts1)
        if k1 < k:
            texts2 = generate_k(tok, mdl, full_prompt + hint, k - k1, max_new_tokens, temperature, top_k, top_p)
            texts_all.extend(texts2)

        # ---- score (closure + avg logprob) ----
        closure, avg_lp = closure_lp_batch(tok, mdl, full_prompt, texts_all)

        # ---- initial coherence, eq counts, finals ----
        coh_vals, eq_cnts, finals = [], [], []
        for t in texts_all:
            final_num = extract_number(t)
            coh, eq_cnt, *_ = coherence_score_and_stats(t, final_num)
            coh_vals.append(coh); eq_cnts.append(eq_cnt); finals.append(final_num)

        # ---- TOKEN-IMPACT DIRECTIONS ----
        impacts = impact_vectors_batch(tok, mdl, full_prompt, texts_all, batch_size=1)
        dir_norm = _direction_score_to_consensus(impacts)

        # ---- consensus final number for teacher (unchanged) ----
        consensus_val = majority_consensus(finals, tol=1e-3)
        rewards = [oracle_reward(closure[i], coh_vals[i], eq_cnts[i], finals[i], consensus_val)
                   for i in range(len(texts_all))]
        teacher_idx = int(np.argmax(rewards))

        # ---- context vector + quick learning (unchanged) ----
        z = context_vec(tok, mdl, full_prompt, ablated_prompt, proj_dim=proj_dim)
        lp_norm = _minmax(avg_lp)
        feats = np.stack([np.array(closure), np.array(lp_norm), np.array(coh_vals), np.array(dir_norm)], axis=1)
        for _ in range(max(1, train_steps)):
            _ = learner.train_step(z, feats, teacher_idx)
        w = learner.predict_weights(z)

        # ===================== CONTEXT PASS (directional) =====================
        # Blend coherence toward better neighbors that move in the *same impact direction*.
        coh_dir = _blend_toward_best_by_direction(coh_vals, closure, impacts, sim_thresh=0.8, alpha=0.6)

        # Recompute pick with direction-adjusted coherence
        pick_idx = choose_with_weights(texts_all, closure, avg_lp, coh_dir, dir_norm, w)
        pick_text = texts_all[pick_idx]
        pick_score = coh_dir[pick_idx]

        # Track the best
        if pick_score > best_overall_score:
            best_overall, best_overall_score = pick_text, pick_score

        # ===================== CLOSURE LOOP =====================
        combined_scores = []
        for i in range(len(texts_all)):
            s = w[0]*closure[i] + w[1]*lp_norm[i] + w[2]*coh_dir[i] + w[3]*dir_norm[i]
            combined_scores.append((s, i))
        combined_scores.sort(reverse=True)
        top_idxs = [idx for _, idx in combined_scores[:max(1, min(top_m_for_closure, len(texts_all)))]]

        refined = [texts_all[i] for i in top_idxs]
        refine_prompt_base = full_prompt + "\n\nRefine your solution. Re-check each equation for algebraic accuracy and keep the reasoning concise. Keep the final numeric answer explicit."

        for _loop in range(max(0, closure_loops)):
            refined_inputs = [(refine_prompt_base + "\n\nPrevious attempt:\n" + r) for r in refined]
            new_batch = []
            for rp in refined_inputs:
                out = generate_k(tok, mdl, rp, 1, max_new_tokens, max(0.3, temperature*0.8), top_k, top_p)
                new_batch.extend(out)
            refined = new_batch

            # Re-score on refined
            r_closure, r_avg_lp = closure_lp_batch(tok, mdl, full_prompt, refined)
            r_coh_vals, r_finals = [], []
            for t in refined:
                f = extract_number(t)
                c, _, *_ = coherence_score_and_stats(t, f)
                r_coh_vals.append(c); r_finals.append(f)

            # Recompute impacts & directional neighbor blend on refined set
            r_impacts = impact_vectors_batch(tok, mdl, full_prompt, refined, batch_size=1)
            r_dir_norm = _direction_score_to_consensus(r_impacts)
            r_lp_norm = _minmax(r_avg_lp)
            r_coh_dir = _blend_toward_best_by_direction(r_coh_vals, r_closure, r_impacts, sim_thresh=0.8, alpha=0.6)

            best_r_idx = choose_with_weights(refined, r_closure, r_avg_lp, r_coh_dir, r_dir_norm, w)
            best_r_text  = refined[best_r_idx]
            best_r_score = r_coh_dir[best_r_idx]

            if best_r_score > best_overall_score:
                best_overall, best_overall_score = best_r_text, best_r_score

        # ===================== reseed next pass =====================
        working_history = list(messages)
        working_history.append({"role": "assistant", "content": best_overall})
        working_history.append({"role": "user", "content": "\nRefine your solution. Keep steps concise. Fix any algebra slips found previously."})

        if show_debug:
            print(f"[pass {_pass+1}] pick_score={pick_score:.3f} best_overall_score={best_overall_score:.3f}")

    learner.save()
    return best_overall




# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Adaptive chat (self-consensus teacher; coherence vs closure).")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model id or local path (use local path to avoid downloads).")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--system", type=str, default="You are a concise assistant you try to get every problem right to the best of your abilities ( also you always answer and show all your working briefly) ")
    ap.add_argument("--show_debug", action="store_true")
    ap.add_argument("--proj_dim", type=int, default=128, help="context vector dim")
    ap.add_argument("--learn_file", type=str, default="weightnet.pt")
    ap.add_argument("--learn_lr", type=float, default=1e-3)
    ap.add_argument("--learn_steps", type=int, default=5, help="GD steps per turn")
    ap.add_argument("--passes", type=int, default=1, help="number of iterative re-generation passes per turn")

    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ({args.dtype})…")
    tok, mdl = load_model(args.model, args.device, args.dtype)
    dev = next(mdl.parameters()).device
    if args.device == "cuda":
        try:
            print(f"GPU: {torch.cuda.get_device_name()}  VRAM≈{round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
        except Exception:
            pass

    learner = Learner(path=args.learn_file, in_dim=args.proj_dim, device=dev, lr=args.learn_lr)

    print("Chat ready. Commands: /reset, /exit\n")
    history: List[Dict[str,str]] = [{"role":"system","content": args.system}]

    try:
        while True:
            user = input("you> ").strip()
            if not user: continue
            if user.lower() in ("/exit","/quit","/q"): break
            if user.lower() == "/reset":
                history = [{"role":"system","content": args.system}]
                print("(history cleared)")
                continue

            history.append({"role":"user","content": user})
            print("ai> (thinking…)", flush=True)
            t0 = time.time()
            reply = run_turn(
    tok, mdl, history,
    k=args.k, max_new_tokens=args.max_new_tokens,
    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    learner=learner, proj_dim=args.proj_dim, train_steps=args.learn_steps,
    passes=args.passes,               # <-- added (was implicitly referenced before)
    closure_loops=2,                  # <-- you can tune
    top_m_for_closure=3,              # <-- you can tune
    neighbor_tol=1e-3,                # <-- same-role tolerance
    show_debug=args.show_debug
)

            dt = time.time() - t0
            print(reply.strip() + f"\n— [{args.k} cand, {dt:.2f}s]", flush=True)
            history.append({"role":"assistant","content": reply})
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()
