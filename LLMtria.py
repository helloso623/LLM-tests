#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised, self-growing chat agent — upgraded:
- No intents. Concepts/variables emerge from embeddings.
- ANN retrieval (FAISS if available) + sparse growth.
- Energy ENSEMBLE learns constraints from data (no equations).
- Gradient-based imputation + Laplace-ish uncertainty (Hessian diag approx).
- SORA-style rater with exponentiated fitness (COH/CLS/LP/CTX/U/SAFE + novelty).
- Emotion shaping, Hebbian memory.
- Meta-controller: refine if energy ↑ ; targeted ask if uncertainty high.
- Prune/Merge by usage EMA & cosine.
"""

import os
os.environ["TRANSFORMERS_NO_TF"]="1"; os.environ["TRANSFORMERS_NO_TORCHVISION"]="1"; os.environ["TRANSFORMERS_NO_JAX"]="1"

import argparse, sys, subprocess, re, ast, math, random, json, time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

def _install(pkgs):
    for p in pkgs:
        try: __import__(p.split("==")[0].split(">=")[0].replace("-", "_"))
        except Exception: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
_install(["torch","transformers>=4.42","numpy>=1.24.0","sympy"])
# try ANN
try:
    _install(["faiss-cpu"])
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sympy import symbols, Eq, sympify, solve as sym_solve, N  # (kept for future equation paths)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception: pass

# ---------------- Basic text math helpers ----------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UN  = (ast.USub, ast.UAdd)

def extract_number(s: str):
    m = _NUM_RE.findall(s or "")
    if not m: return None
    x = m[-1].replace(",", "")
    try: return int(x) if re.fullmatch(r"-?\d+", x) else float(x)
    except: return None

def _safe_eval_expr(expr: str) -> Optional[float]:
    try: tree = ast.parse(expr, mode="eval")
    except: return None
    def _num(n):
        if isinstance(n, ast.Constant) and isinstance(n.value,(int,float)): return float(n.value)
        return None
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        v=_num(n)
        if v is not None: return v
        if isinstance(n, ast.UnaryOp) and isinstance(n.op,_ALLOWED_UN):
            val=_eval(n.operand); 
            return None if val is None else (-val if isinstance(n.op, ast.USub) else +val)
        if isinstance(n, ast.BinOp) and isinstance(n.op,_ALLOWED_BIN):
            a=_eval(n.left); b=_eval(n.right)
            if a is None or b is None: return None
            if isinstance(n.op, ast.Div) and b==0: return None
            if isinstance(n.op, ast.Add):  return a+b
            if isinstance(n.op, ast.Sub):  return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div):  return a/b
            if isinstance(n.op, ast.Pow):  return a**b
        return None
    return _eval(tree)

def find_equations(text: str) -> List[Tuple[str,str]]:
    t = (text or "").replace("×","*").replace("÷","/").replace("·","*").replace("^","**")
    t = re.sub(r"\s+"," ",t)
    out=[]
    for seg in re.split(r"[;\n]", t):
        if "=" not in seg: continue
        parts = re.split(r"(?<![<>=!])=(?!=)", seg)
        for j in range(len(parts)-1):
            lhs=parts[j].strip(); rhs=parts[j+1].strip()
            if 1<=len(lhs)<=80 and 1<=len(rhs)<=80:
                out.append((lhs,rhs))
                if len(out)>=16: return out
    return out

def equation_coherence(text: str, final_num: Optional[float]) -> float:
    pairs = find_equations(text)
    if not pairs: return 0.0
    ok=0; bad=0; last=None
    for lhs,rhs in pairs:
        lv=_safe_eval_expr(lhs); rv=_safe_eval_expr(rhs)
        if lv is None or rv is None: bad+=1; continue
        if abs(lv-rv)<=1e-6: ok+=1
        else: bad+=1
        last=rv
    final_match = 1 if (last is not None and final_num is not None and abs(last-final_num)<=1e-6) else 0
    eq_cnt = ok+bad
    return 0.7*(ok/max(1,eq_cnt))+0.2*final_match-0.1*(bad/max(1,eq_cnt))

def affect_from_text(s: str) -> float:
    s = (s or "").lower()
    pos = any(w in s for w in ["great","good","awesome","thanks","nice","love","perfect","cool"])
    neg = any(w in s for w in ["angry","mad","annoyed","bad","hate","worst","ugh","frustrated","terrible"])
    if pos and not neg: return +0.6
    if neg and not pos: return -0.6
    return 0.0

# ---------------- LM ----------------
# --- Unresolved-variable detection (math targets) ---
_MATH_CHUNK_RE = re.compile(r"[0-9\.\,\s\+\-\*\/\^\(\)]+")  # digits + ops

def parse_math_targets(user_text: str) -> List[float]:
    """
    Extract simple math expressions from the user's message and evaluate them.
    Returns numeric values we expect to see resolved in the answer.
    """
    t = (user_text or "").strip()
    outs: List[float] = []
    for m in _MATH_CHUNK_RE.finditer(t):
        chunk = m.group(0)
        if any(op in chunk for op in "+-*/^"):
            expr = chunk.replace("^", "**").replace(",", "")
            val = _safe_eval_expr(expr)
            if val is not None and np.isfinite(val):
                v = float(val)
                if not any(abs(v-u) <= 1e-6 for u in outs):
                    outs.append(v)
    return outs

def unresolved_penalty(candidate: str, targets: List[float]) -> Tuple[float, int]:
    """
    Add penalty if candidate doesn't resolve expected targets.
    """
    pen = 0.0
    solved = 0
    cand = candidate or ""
    # numbers in candidate
    nums = []
    for m in _NUM_RE.findall(cand):
        try: nums.append(float(m.replace(",", "")))
        except: pass

    # target satisfaction
    for t in targets:
        if any(abs(t - n) <= 1e-6 for n in nums):
            solved += 1
    if targets:
        pen += 0.60 * (len(targets) - solved)   # penalty per unsolved
        if solved == 0 and (cand.strip().endswith("?") or "Quick check" in cand):
            pen += 0.50
        if extract_number(cand) is None:
            pen += 0.40
    if cand.strip().endswith("..."):
        pen += 0.30

    return pen, solved
# --- Load model, last hidden, generate k ---
def load_model(model_id: str, dtype: str="bf16"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if DEVICE=="cuda":
        torch_dtype = torch.bfloat16 if dtype=="bf16" else (torch.float16 if dtype=="fp16" else torch.float32)
    else:
        torch_dtype = torch.float32
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        mdl.to(DEVICE).eval()
    except RuntimeError as e:
        print(f"[WARN] OOM loading {model_id}: {e}\nFalling back to 4-bit.")
        mdl = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto").eval()
    try: torch.set_float32_matmul_precision("high")
    except: pass
    return tok, mdl

@torch.inference_mode()
def last_hidden(tok, mdl, text: str) -> torch.Tensor:
    enc = tok(text, return_tensors="pt").to(DEVICE)
    out = mdl(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][:, enc["input_ids"].shape[1]-1, :].float()  # [1,D]

@torch.inference_mode()
def generate_k(tok, mdl, prompt: str, k: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
    enc = tok([prompt], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    out = mdl.generate(
        **enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
        num_return_sequences=k, max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True
    )
    out_cpu = out.detach().cpu()
    texts=[]
    for b in range(out_cpu.shape[0]):
        start = input_lens[b % len(input_lens)]
        texts.append(tok.decode(out_cpu[b, start:], skip_special_tokens=True))
    return texts

# ---------------- Optional Safety Gate (cheap) ----------------
BAD_PAT = re.compile(r"(credit card|ssn|social security|violent act|build a bomb|make a bomb|harm yourself|suicide)", re.I)
def safety_gate(text: str) -> bool:
    if not text: return True
    return BAD_PAT.search(text) is None

# ---------------- Concept Bank with ANN ----------------
class ConceptBank:
    """
    Self-growing concepts with ANN retrieval.
    Each concept i has key k_i (D), scalar value v_i, usage u_i.
    """
    def __init__(self, dim: int, k_retrieve: int = 16, novelty=0.86):
        self.dim = dim
        self.k = k_retrieve
        self.novelty = float(novelty)
        self.keys = torch.empty(0, dim, device=DEVICE)
        self.values = torch.empty(0, 1, device=DEVICE)
        self.usage = torch.empty(0, 1, device=DEVICE)
        self._faiss = None
        if FAISS_OK:
            self._faiss = faiss.IndexFlatIP(dim)

    def _rebuild_faiss(self):
        if not self._faiss: return
        self._faiss.reset()
        if self.keys.shape[0] > 0:
            K = F.normalize(self.keys, dim=-1).detach().cpu().numpy().astype('float32')
            self._faiss.add(K)

    @torch.no_grad()
    def retrieve(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys.shape[0]==0:
            return torch.empty(0, dtype=torch.long, device=DEVICE), torch.empty(0, device=DEVICE)
        qn = F.normalize(q, dim=-1)  # [1,D]
        if self._faiss and self.keys.shape[0] >= 32:
            D = self.dim
            k = min(self.k, self.keys.shape[0])
            q_np = qn.detach().cpu().numpy().astype('float32')
            sims, idx = self._faiss.search(q_np, k)   # [1,k]
            idx = torch.tensor(idx[0], device=DEVICE, dtype=torch.long)
            sims = torch.tensor(sims[0], device=DEVICE, dtype=qn.dtype)
            return idx, sims
        # torch fallback
        kn = F.normalize(self.keys, dim=-1)
        sims = torch.matmul(kn, qn.t()).squeeze(-1)     # [N]
        topk = min(self.k, sims.numel())
        vals, idx = torch.topk(sims, topk)
        return idx, vals

    @torch.no_grad()
    def maybe_add(self, q: torch.Tensor):
        # add new concept if max sim < novelty
        if self.keys.shape[0]==0:
            self.keys = q.clone()
            self.values = torch.zeros(1,1, device=DEVICE)
            self.usage  = torch.zeros(1,1, device=DEVICE)
            self._rebuild_faiss()
            return
        idx, sims = self.retrieve(q)
        if sims.numel()==0 or float(sims.max()) < self.novelty:
            self.keys = torch.cat([self.keys, q], dim=0)
            self.values = torch.cat([self.values, torch.zeros(1,1, device=DEVICE)], dim=0)
            self.usage  = torch.cat([self.usage, torch.zeros(1,1, device=DEVICE)], dim=0)
            self._rebuild_faiss()

    @torch.no_grad()
    def hebbian(self, idx: torch.Tensor, reward: float):
        if idx.numel()==0: return
        self.usage[idx,0] = 0.98*self.usage[idx,0] + 0.18*(1.0+reward)
        self.usage.clamp_(0, 10)

    @torch.no_grad()
    def prune_and_merge(self, cos_thresh=0.985, usage_floor=0.02):
        if self.keys.shape[0] < 4: return
        # prune underused
        keep = (self.usage.squeeze(-1) >= usage_floor)
        if keep.float().mean().item() < 1.0:
            self.keys = self.keys[keep]; self.values = self.values[keep]; self.usage = self.usage[keep]
        # merge near-duplicates
        kn = F.normalize(self.keys, dim=-1)
        sims = kn @ kn.t()
        N = sims.shape[0]
        merged = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        new_keys=[]; new_vals=[]; new_usg=[]
        for i in range(N):
            if merged[i]: continue
            group = (sims[i] > cos_thresh) & (~merged)
            idxs = torch.where(group)[0]
            k_avg = self.keys[idxs].mean(dim=0)
            v_avg = self.values[idxs].mean(dim=0)
            u_max = self.usage[idxs].max(dim=0).values
            new_keys.append(k_avg); new_vals.append(v_avg); new_usg.append(u_max)
            merged[idxs]=True
        self.keys = torch.stack(new_keys, dim=0)
        self.values = torch.stack(new_vals, dim=0)
        self.usage = torch.stack(new_usg, dim=0)
        self._rebuild_faiss()

# ---------------- Energy Ensemble (emergent constraints) ----------------
class EnergyHead(nn.Module):
    def __init__(self, dim_ctx: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_ctx+8),
            nn.Linear(dim_ctx+8, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x8: torch.Tensor, ctx: torch.Tensor):
        # x8: [8] slice; ctx: [1,dim_ctx]
        return self.net(torch.cat([x8, ctx.squeeze(0)], dim=-1)).squeeze()

class PairwiseEnergy(nn.Module):
    def __init__(self, dim_ctx: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))  # starts inert
        self.proj = nn.Linear(dim_ctx, 8)
    def forward(self, x8: torch.Tensor, ctx: torch.Tensor):
        # simple pairwise cohesion under ctx projection
        p = self.proj(ctx)  # [1,8]
        return (self.w * ((x8 - p.squeeze(0))**2).mean()).squeeze()

class EnergyEnsemble(nn.Module):
    def __init__(self, dim_ctx: int, heads=4, hidden=128):
        super().__init__()
        self.heads = nn.ModuleList([EnergyHead(dim_ctx, hidden) for _ in range(heads)])
        self.pair = PairwiseEnergy(dim_ctx)
    def _to8(self, x: torch.Tensor):
        v = x
        if v.numel()>8:
            chunks = torch.chunk(v, 8, dim=0)
            v = torch.stack([c.mean() for c in chunks], dim=0)
        elif v.numel()<8:
            v = F.pad(v, (0, 8-v.numel()))
        return v
    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        x8 = self._to8(x)
        e = 0.0
        for h in self.heads:
            e = e + (h(x8, ctx)**2)
        e = e + self.pair(x8, ctx)
        return e

# ---------------- Imputation + Uncertainty (Laplace-ish) ----------------
def impute_with_uncertainty(values: torch.Tensor, known_mask: torch.Tensor, energy: EnergyEnsemble, ctx: torch.Tensor,
                            steps=80, lr=0.05) -> Tuple[torch.Tensor, float, float]:
    x = values.clone().detach().requires_grad_(True)
    y = values.detach()
    lam = 6.0
    opt = torch.optim.SGD([x], lr=lr)
    E_last = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        e = energy(x.squeeze(-1), ctx)
        data = lam*((x - y)[known_mask]**2).mean() if known_mask.any() else 0.0
        loss = e + (data if isinstance(data, torch.Tensor) else 0.0)
        loss.backward(); opt.step()
        E_last = float(e.detach().item())
    # Uncertainty via finite-difference Hessian diag approx on energy
    with torch.no_grad():
        x0 = x.squeeze(-1)
        eps = 1e-3
        H_diag = []
        for i in range(x0.numel()):
            xi = x0.clone()
            xi[i] += eps
            e1 = energy(xi, ctx).item()
            xi = x0.clone()
            xi[i] -= eps
            e2 = energy(xi, ctx).item()
            e0 = energy(x0, ctx).item()
            h = max(0.0, (e1 - 2*e0 + e2) / (eps*eps))
            H_diag.append(h)
        # uncertainty ~ 1 / (1 + mean(H_diag))
        unc = 1.0 / (1.0 + (sum(H_diag)/max(1,len(H_diag))))
    return x.detach(), float(E_last if E_last is not None else 0.0), float(unc)

# --- normalized edit distance for string similarity ---
def _edit_norm(a: str, b: str) -> float:
    """
    Normalized Levenshtein distance in [0,1].
    0.0 = identical, 1.0 = completely different.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + (ai != b[j-1])  # substitution
            )
    return dp[n][m] / max(1, max(n, m))

# ---------------- Fitness ----------------

def fitness(scores: Dict[str,float], emo: float=0.0, novelty: float=0.0) -> float:
    if not scores.get("SAFE", True): return -1e9
    # exp fusion with small novelty bump
    a=b=c=d=e=f=1.0
    base = math.exp(a*scores.get("COH",0)+b*scores.get("CLS",0)+c*scores.get("LP",0)
                    + d*scores.get("CTX",0)+e*scores.get("U",0)+f*scores.get("CONS",0))
    return base * (1.0 + 0.35*emo + 0.15*novelty)

# ---------------- The Agent ----------------
class Agent:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", dtype="bf16"):
        self.tok, self.mdl = load_model(model_id, dtype)
        with torch.no_grad():
            probe = last_hidden(self.tok, self.mdl, "probe")
        self.hdim = int(probe.shape[-1])
        self.ctx_dim = 128
        rng = np.random.default_rng(1234)
        W = rng.standard_normal((self.hdim, self.ctx_dim)).astype(np.float32)/math.sqrt(self.hdim)
        self.proj = torch.tensor(W, device=DEVICE)
        self.bank = ConceptBank(dim=self.hdim, k_retrieve=16, novelty=0.86)
        self.energy = EnergyEnsemble(dim_ctx=self.ctx_dim, heads=6, hidden=160).to(DEVICE)
        self.optE = torch.optim.Adam(self.energy.parameters(), lr=1e-3)

    @torch.no_grad()
    def context_vec(self, full: str, ablated: str) -> torch.Tensor:
        hf = last_hidden(self.tok, self.mdl, full)
        ha = last_hidden(self.tok, self.mdl, ablated)
        hd = (hf - ha)  # [1,D]
        return (hd @ self.proj).float()  # [1,128]

    def render_prompt(self, messages: List[Dict[str,str]]):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            buf=[]
            for m in messages:
                role = m["role"].capitalize()
                buf.append(f"{role}: {m['content']}\n")
            buf.append("Assistant: ")
            return "".join(buf)

    def observe(self, user_text: str, history_text: str):
        sys_stub = "System: be concise."
        full = f"{sys_stub}\nHistory:{history_text}\nUser:{user_text}\nAssistant:"
        abla = f"{sys_stub}\nUser:{user_text}\nAssistant:"
        ctx = self.context_vec(full, abla)  # [1,128]
        q = last_hidden(self.tok, self.mdl, user_text)   # [1,D]
        self.bank.maybe_add(q)
        idx, sims = self.bank.retrieve(q)
        return ctx, q, idx, sims

    def train_energy_online(self, x_hat: torch.Tensor, known: torch.Tensor, ctx: torch.Tensor):
        if x_hat.numel()==0: return
        x_param = x_hat.clone().detach().requires_grad_(True)
        lam=2.0
        opt = torch.optim.Adam([x_param], lr=0.03)
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            e = self.energy(x_param.squeeze(-1), ctx)
            data = lam*((x_param - x_hat)[known]**2).mean() if known.any() else 0.0
            loss = e + (data if isinstance(data, torch.Tensor) else 0.0)
            loss.backward(); opt.step()
        # encourage low energy on current slice
        self.optE.zero_grad(set_to_none=True)
        e_real = self.energy(x_hat.squeeze(-1), ctx)
        e_real.backward(); self.optE.step()

    @torch.no_grad()
    def closure_logprob(self, prompt: str, cont: str) -> Tuple[float,float]:
        enc = self.tok([prompt+cont], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = self.mdl(**enc)
        logits = out.logits[:, :-1, :].float()
        labels = enc["input_ids"][:, 1:]
        # use entire continuation; prompt part already in sequence
        mask = torch.ones_like(labels, dtype=torch.bool, device=DEVICE)
        logp = F.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        cnt = mask.long().sum(dim=1).clamp(min=1)
        avg_lp = (tok_lp.sum(dim=1)/cnt).item()
        preds = logits.argmax(dim=-1)
        eq = (preds==labels) & mask
        closure = (eq.float().sum(dim=1)/cnt).item()
        return float(closure), float(avg_lp)

    def score_candidates(self, prompt: str, cands: List[str], ctx_signal: float, novelty: float,
                     user_text: str) -> Tuple[int, List[float]]:
        targets = parse_math_targets(user_text)  # expected numeric results
        scores=[]
        for t in cands:
            num = extract_number(t)
            coh = equation_coherence(t, num)
            clo, lp = self.closure_logprob(prompt, t)
            safe = safety_gate(t)
            util = 1.0 if num is not None else 0.25

            sc = fitness({"COH":coh,"CLS":clo,"LP":lp,"CTX":ctx_signal,"U":util,"CONS":0.0,"SAFE":safe},
                     emo=0.0, novelty=novelty)

            # NEW: unresolved-variable penalty
            pen, solved = unresolved_penalty(t, targets)
            sc -= pen

            # bonus if shows equations for math queries
            if targets and find_equations(t):
                sc += 0.15

            scores.append(sc if safe else -1e9)
        best = int(np.argmax(scores)) if scores else 0
        return best, scores


    def respond(self, history: List[Dict[str,str]], k=8, max_new_tokens=4096, temperature=0.7, top_p=0.9, top_k=50):
        # ---- inputs & context ----
        user_text = next((m["content"] for m in reversed(history) if m["role"]=="user"), "")
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-8:]])

        # observe latent-vars / ANN
        ctx, q, idx, sims = self.observe(user_text, history_text)
        m = idx.numel()

        # math-like targets from user (e.g., "2+2" -> [4.0])
        targets = parse_math_targets(user_text)
        demand_numeric = bool(targets or re.search(r"[0-9].*[\+\-\*/\^]|[\+\-\*/\^].*[0-9]", user_text))

        # ---- init variable slice ----
        vals = torch.zeros(m,1, device=DEVICE)
        known = torch.zeros(m,1, dtype=torch.bool, device=DEVICE)
        if m>0:
            vals = self.bank.values[idx,:].clone()

        # map observed scalar to first slot if present
        obs_scalar = extract_number(user_text)
        if m>0 and obs_scalar is not None:
            vals[0,0] = float(obs_scalar); known[0,0]=True

        # ---- impute (no questions; adapt + credit assignment) ----
        E_before = 0.0; unc = 0.0
        if m>0:
            x_hat, E_before, unc = impute_with_uncertainty(vals, known, self.energy, ctx, steps=64, lr=0.05)
            with torch.no_grad():
                self.bank.values[idx,0] = 0.8*self.bank.values[idx,0] + 0.2*x_hat.squeeze(-1)
            self.train_energy_online(x_hat, known, ctx)
            self.bank.hebbian(idx, reward=(-0.05 if unc > 0.55 else (+0.25 if E_before < 0.25 else 0.0)))

        # ---- prompt (decisive hints when needed) ----
        hints = []
        if demand_numeric:
            hints.append("If the question contains calculations, compute them and state the final numeric result explicitly.")
        if m>0 and E_before > 0.35:
            hints.append("Avoid clarifying questions. Provide a direct solution with minimal steps.")
        plan = "Use concise steps. " + (" ".join(hints) if hints else "If a number helps, compute it clearly.")

        msgs = [{"role":"system","content":"You are concise, accurate, and verify reasoning briefly."}]
        msgs += history
        msgs.append({"role":"user","content": plan})
        prompt = self.render_prompt(msgs)

        # ---- context alignment & novelty ----
        ctx_signal = 0.5
        novelty = 0.0
        if m>0:
            qn = F.normalize(q, dim=-1); kn = F.normalize(self.bank.keys[idx], dim=-1)
            cs = (kn @ qn.t()).squeeze(-1)
            ctx_signal = float(cs.mean().clamp(-1,1).item()*0.5 + 0.5)
            novelty = float((1.0 - cs.max().clamp(0,1).item()) * 0.8)  # newer-topic boost

        # ---- self-consistency rounds (diverse) ----
        # gather candidates across temperatures; dedupe near-duplicates
        ROUNDS = 3
        temps = [temperature, max(0.5, temperature*0.85), 0.4]
        all_cands: List[str] = []
        for r in range(ROUNDS):
            batch = generate_k(self.tok, self.mdl, prompt, k=max(2, k//ROUNDS),
                               max_new_tokens=max_new_tokens, temperature=temps[min(r, len(temps)-1)],
                               top_p=top_p, top_k=top_k)
            # anti-duplicate by edit distance on raw text
            for cand in batch:
                if not any(_edit_norm(cand, prev) < 0.002 for prev in all_cands):
                    all_cands.append(cand)
        cands = all_cands if all_cands else ["..."]

        # ---- optional pre-ranking coherence loops (cheap nudge) ----
        COH_LOOPS = 2
        for _ in range(COH_LOOPS):
            best_i_tmp, _ = self.score_candidates(prompt, cands, ctx_signal, novelty, user_text)
            peek = cands[best_i_tmp]
            if equation_coherence(peek, extract_number(peek)) >= 0.10:
                break
            # low coherence -> regenerate smaller, cooler batch
            cands = generate_k(self.tok, self.mdl, prompt, k=max(2, k//2), max_new_tokens=max_new_tokens,
                               temperature=max(0.35, temperature*0.7), top_p=top_p, top_k=top_k)

        # ---- rank with unresolved-variable penalties ----
        best_i, scores = self.score_candidates(prompt, cands, ctx_signal, novelty, user_text)
        ans = cands[best_i].strip() if cands else "..."

        # safety hard gate
        if not safety_gate(ans):
            return "I can’t help with that. Want me to summarize safe options instead?"

        # ---- progress check on numeric targets ----
        def solved_targets(text: str) -> int:
            nums = []
            for mnum in _NUM_RE.findall(text or ""):
                try: nums.append(float(mnum.replace(",", "")))
                except: pass
            return sum(1 for t in targets if any(abs(t - n) <= 1e-6 for n in nums))

        need_force = False
        if demand_numeric:
            if solved_targets(ans) < len(targets) if targets else (extract_number(ans) is None):
                need_force = True

        # ---- multi-iteration refine (context/coherence/closure) ----
        MAX_REFINE   = 10
        MIN_GAIN     = 0.2
        PATIENCE     = 10
        LOW_COH_THR  = 0.9
        FORCE_MATH   = bool(targets)
        ANTI_META    = ["Quick check", "prefer a precise", "do you prefer", "?"] if demand_numeric else []

        # current fitness for acceptance
        _, s_cur = self.score_candidates(prompt, [ans], ctx_signal, novelty, user_text)
        best_fit = s_cur[0]
        no_improve = 0

        for _iter in range(MAX_REFINE):
            if not (need_force or equation_coherence(ans, extract_number(ans)) < LOW_COH_THR):
                break

            refine_msgs = msgs[:-1] + [{
                "role":"user",
                "content":"Answer directly. If calculations exist, show one short line and give the final numeric result. No clarifying questions and get it correctly ."
            }]
            refine_prompt = self.render_prompt(refine_msgs)

            alt = generate_k(self.tok, self.mdl, refine_prompt, k=2, max_new_tokens=max_new_tokens,
                             temperature=0.4, top_p=0.9, top_k=top_k)
            if not alt:
                break

            # filter out meta / repeats aggressively
            alt = [t for t in alt
                   if not any(s in t for s in ANTI_META)
                   and _edit_norm(t, ans) > 0.05]

            if not alt:
                no_improve += 1
                if no_improve > PATIENCE: break
                continue

            best2, _ = self.score_candidates(refine_prompt, alt, ctx_signal, novelty, user_text)
            alt_ans = alt[best2].strip()

            if not safety_gate(alt_ans):
                no_improve += 1
                if no_improve > PATIENCE: break
                continue

            # accept only with meaningful improvement
            _, s_new = self.score_candidates(refine_prompt, [alt_ans], ctx_signal, novelty, user_text)
            gain = s_new[0] - best_fit
            if gain >= MIN_GAIN:
                ans = alt_ans
                best_fit = s_new[0]
                no_improve = 0
                # early stop if math targets solved
                if FORCE_MATH and solved_targets(ans) >= len(targets):
                    break
            else:
                no_improve += 1
                if no_improve > PATIENCE:
                    break

            # if still needs number but none given, keep forcing
            if demand_numeric:
                need_force = (targets and solved_targets(ans) < len(targets)) or (not targets and extract_number(ans) is None)
            else:
                need_force = False

        # ---- maintenance ----
        self.bank.prune_and_merge()
        return ans



# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Unsupervised energy-chat agent (ANN, ensemble, uncertainty, meta-control).")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    tok, mdl = None, None  # just to show load msg
    print(f"Loading {args.model} on {DEVICE} …")
    agent = Agent(model_id=args.model, dtype=args.dtype)
    print("Chat ready. Commands: /reset, /exit\n")
    history: List[Dict[str,str]] = []

    try:
        while True:
            user = input("you> ").strip()
            if not user: continue
            if user.lower() in ("/exit","/quit","/q"): break
            if user.lower()=="/reset":
                history=[]; print("(history cleared)"); continue
            history.append({"role":"user","content": user})
            print("ai> (thinking…)", flush=True)
            t0=time.time()
            reply = agent.respond(history, k=args.k, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            dt = time.time()-t0
            print(reply + f"\n— [{args.k} cand, {dt:.2f}s]")
            history.append({"role":"assistant","content": reply})
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()
