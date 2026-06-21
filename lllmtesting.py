#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K evaluator with adaptive multi-sample selection + ONLINE learning (WeightNet).
- Generates k candidates (two-phase: k1 + k2 refine)
- Features: closure, avg logprob, coherence (equation checks), direction (op-seq consensus)
- Trains a tiny MLP (WeightNet) online per item to weight those features (LLM stays frozen)
- Persists WeightNet so it remembers across items (--learn_file)
- Compares against baselines: First, Best-LP, Self-Consistency (same total k)

Defaults are tuned to run on CPU or small GPUs. Use --limit for quick smoke tests.
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import argparse, re, csv, json, math, random, time, sys, subprocess
from collections import Counter
import ast

from typing import List, Tuple, Optional, Dict, Any

def _install(pkgs):
    for p in pkgs:
        try:
            __import__(p.split("==")[0].split(">=")[0].replace("-", "_"))
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

_install([
    "torch",
    "transformers>=4.42",
    "datasets>=2.19.0",
    "numpy>=1.24.0",
    "tqdm",
])

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -------------------- RNG & CUDA knobs --------------------
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

# -------------------- Math / coherence helpers --------------------
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def extract_number(s: str):
    m = _NUM_RE.findall(s)
    if not m: return None
    try:
        x = m[-1].replace(",", "")
        return int(x) if re.fullmatch(r"-?\d+", x) else float(x)
    except Exception:
        return None

def parse_gsm8k_gold(ans: str):
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", ans)
    if not m: return None
    x = m.group(1).replace(",", "")
    return int(x) if re.fullmatch(r"-?\d+", x) else float(x)

_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UN  = (ast.USub, ast.UAdd)
import ast

def _safe_eval_expr(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None
    def _num(n):
        if isinstance(n, ast.Num): return float(n.n)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        return None
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        v = _num(n)
        if v is not None: return v
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UN):
            val = _eval(n.operand);  return None if val is None else (-val if isinstance(n.op, ast.USub) else +val)
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

def coherence_score_and_stats(text: str, final_num: Optional[float]) -> Tuple[float, int]:
    pairs = _find_equations(text)
    if not pairs: return (0.0, 0)
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
    return (float(coh), eq_cnt)

# -------------------- Direction (tiny bonus) --------------------
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

# -------------------- Model & prompts --------------------
def load_model(model_id: str, device: str, dtype: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if device == "cuda":
        torch_dtype = torch.bfloat16 if dtype=="bf16" else (torch.float16 if dtype=="fp16" else torch.float32)
    else:
        torch_dtype = torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    mdl.to(device).eval()
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    return tok, mdl

def render_prompt(question: str) -> str:
    return (
        "Solve the grade-school math problem.\n"
        f"Question: {question.strip()}\n"
        "Show brief calculations (a few equations). On the last line write: Answer: <number>."
    )

def render_prompt_ablated(question: str) -> str:
    return (
        "Solve the grade-school math problem.\n"
        f"Question: {question.strip()}\n"
        "On the last line write: Answer: <number>."
    )

# -------------------- Generation --------------------
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

# -------------------- Scores: closure + avg logprob (chunked) --------------------
@torch.no_grad()
def closure_lp_batch(tok, mdl, prompt: str, answers: List[str], batch_size: int = 1):
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
        with (torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else torch.autocast("cpu", dtype=torch.float32, enabled=False)):
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

# -------------------- Context feature (projection of hidden diff) --------------------
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
    return z  # [proj_dim]

# -------------------- WeightNet (tiny learned scorer) --------------------
class WeightNet(nn.Module):
    def __init__(self, in_dim=128, hidden=128, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)   # logits for [closure, logprob, coherence, direction]
        )
    def forward(self, z):  # z: [B,in_dim]
        return self.net(z)

class Learner:
    def __init__(self, path="weightnet.pt", in_dim=128, device="cpu", lr=1e-3, reset=False):
        self.path = path
        self.device = device
        self.model = WeightNet(in_dim=in_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        if os.path.isfile(self.path) and not reset:
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
        weights = torch.softmax(logits, dim=-1)   # [1,4]
        scores = feats_t @ weights[0]             # [N]
        scores = scores[None, :]                  # [1,N]
        target = torch.tensor([teacher_idx], dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(scores, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

# -------------------- Util --------------------
def _minmax(xs: List[float]) -> List[float]:
    if not xs: return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9: return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]

def oracle_reward(closure: float, coherence: float, eq_cnt: int) -> float:
    if eq_cnt >= 2:
        return 0.5*closure + 0.5*coherence
    return closure

def equal_num(a, b):
    if a is None or b is None: return False
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-3
    return int(a) == int(b)

# -------------------- Per-item logic --------------------
def evaluate_item(tok, mdl, question: str, gold: float,
                  k: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                  learner: Learner, proj_dim: int, learn_steps: int) -> Dict[str, Any]:

    prompt = render_prompt(question)
    ablated = render_prompt_ablated(question)

    # Two-phase sampling
    k1 = max(1, int(math.ceil(0.6*k))) if k>1 else 1
    texts1 = generate_k(tok, mdl, prompt, k1, max_new_tokens, temperature, top_k, top_p)

    hint = "\nPlease show brief calculations (a few equations) and keep the final numeric answer clear."
    texts_all = list(texts1)
    if k1 < k:
        texts2 = generate_k(tok, mdl, prompt + hint, k - k1, max_new_tokens, temperature, top_k, top_p)
        texts_all.extend(texts2)

    # Features
    closure, avg_lp = closure_lp_batch(tok, mdl, prompt, texts_all, batch_size=1)
    coh_vals, eq_cnts, nums = [], [], []
    for t in texts_all:
        num = extract_number(t)
        nums.append(num)
        coh, eq_cnt = coherence_score_and_stats(t, num)
        coh_vals.append(coh); eq_cnts.append(eq_cnt)
    op_lists = [_op_sequence(t) for t in texts_all]
    cons_ops  = _consensus_ops(op_lists)
    dir_sim   = [1.0 - _edit_norm(ops, cons_ops) if cons_ops else 0.0 for ops in op_lists]
    dir_norm  = _minmax(dir_sim)
    lp_norm   = _minmax(avg_lp)

    # Teacher (oracle) index
    rewards = [oracle_reward(closure[i], coh_vals[i], eq_cnts[i]) for i in range(len(texts_all))]
    if k1 < len(texts_all):
        for i in range(k1, len(texts_all)):
            rewards[i] += 0.01
    teacher_idx = int(np.argmax(rewards))

    # Context feature + online learning
    z = context_vec(tok, mdl, prompt, ablated, proj_dim=proj_dim)
    feats = np.stack([
        np.array(closure, dtype=np.float32),
        np.array(lp_norm, dtype=np.float32),
        np.array(coh_vals, dtype=np.float32),
        np.array(dir_norm, dtype=np.float32),
    ], axis=1)  # [N,4]

    for _ in range(max(1, learn_steps)):
        _ = learner.train_step(z, feats, teacher_idx)

    w = learner.predict_weights(z)
    scores = feats @ w
    pick_idx = int(np.argmax(scores))
    pick_text = texts_all[pick_idx]  # <-- chosen raw answer text

    pick_num = nums[pick_idx]
    ok_adapt = equal_num(pick_num, gold)

    # Baselines (same pool)
    ok_first = equal_num(nums[0], gold)
    idx_lp   = int(np.argmax(avg_lp)); ok_lp = equal_num(nums[idx_lp], gold)
    freq = Counter([n for n in nums if n is not None])
    pick_sc = None
    if len(freq) > 0:
        bc = max(freq.values()); tied = [n for n,c in freq.items() if c==bc]
        best_lp = -1e9; best_num_lp = None
        for i, n in enumerate(nums):
            if n in tied and avg_lp[i] > best_lp:
                best_lp = avg_lp[i]; best_num_lp = n
        pick_sc = tied[0] if len(tied)==1 else best_num_lp
    ok_sc = equal_num(pick_sc, gold)

    return {
    
        "ok_first": int(ok_first),
        "ok_lp": int(ok_lp),
        "ok_sc": int(ok_sc),
        "ok_adapt": int(ok_adapt),
        "pick_first": nums[0],
        "pick_lp": nums[idx_lp],
        "pick_sc": pick_sc,
        "pick_adapt": pick_num,
        "weights": w.tolist(),
        "answer_text": pick_text,        # NEW: chosen answer text
        "candidates": texts_all,         # NEW: all raw candidates
        "features": {                    # NEW: per-candidate features (aligned with candidates)
            "closure": closure,
            "avg_logprob": avg_lp,
            "coherence": coh_vals,
            "direction": dir_norm,
            "lp_norm": lp_norm,
            "numbers": nums
        }
    }

    

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = full GSM8K test (~1319).")
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--learn_file", type=str, default="weightnet.pt")
    ap.add_argument("--learn_lr", type=float, default=1e-3)
    ap.add_argument("--learn_steps", type=int, default=3, help="GD steps per item (default 3)")
    ap.add_argument("--reset_memory", action="store_true", help="ignore existing learn_file on start")
    ap.add_argument("--out_csv", type=str, default="gsm8k_weightnet_results.csv")
    args = ap.parse_args()

    # Load model
    print(f"Loading {args.model} on {args.device} ({args.dtype})…")
    tok, mdl = load_model(args.model, args.device, args.dtype)
    if args.device == "cuda":
        try:
            print(f"GPU: {torch.cuda.get_device_name()}  VRAM≈{round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB")
        except Exception:
            pass

    # Load data
    print("Loading GSM8K test split…")
    ds = load_dataset("gsm8k", "main")["test"]
    items = []
    for ex in ds:
        g = parse_gsm8k_gold(ex["answer"])
        if g is None: continue
        items.append((ex["question"], g))
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    N = len(items)
    print(f"Eval items: {N} | k={args.k} | learn_steps={args.learn_steps} | proj_dim={args.proj_dim}")

    learner = Learner(path=args.learn_file, in_dim=args.proj_dim,
                      device=next(mdl.parameters()).device, lr=args.learn_lr,
                      reset=args.reset_memory)

    acc = {"first":0, "lp":0, "sc":0, "adapt":0}
    rows = []
    t0 = time.time()

    for i, (q, gold) in tqdm(list(enumerate(items, 1)), total=N, dynamic_ncols=True, desc="Items"):
        res = evaluate_item(
            tok, mdl, q, gold,
            k=args.k, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            learner=learner, proj_dim=args.proj_dim, learn_steps=args.learn_steps
        )
        for k in acc: acc[k] += res[f"ok_{k}"]
        rows.append([
    q, gold,
    res["pick_first"], res["pick_lp"], res["pick_sc"], res["pick_adapt"],
    res["ok_first"], res["ok_lp"], res["ok_sc"], res["ok_adapt"],
    json.dumps(res["weights"]),
    res["answer_text"]  # NEW
])


        # Save memory occasionally
        if i % 50 == 0:
            learner.save()

    learner.save()
    dt = time.time() - t0

    print("\n=== GSM8K (same total k) ===")
    for key,label in [("first","First"),("lp","Best-LP"),("sc","Self-Consistency"),("adapt","ADAPTIVE (WeightNet)")]:
        print(f"Acc ({label:>16}): {acc[key]/N:.3f}")
    print(f"Processed {N} items in {dt:.1f}s")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
    "question","gold",
    "pick_first","pick_lp","pick_sc","pick_adapt",
    "ok_first","ok_lp","ok_sc","ok_adapt",
    "weights",
    "answer_text"  # NEW
])

        w.writerows(rows)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
