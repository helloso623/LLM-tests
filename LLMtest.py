#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import argparse, re, time, csv, sys, subprocess, json, random, math, ast
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

def pip_install(pkgs):
    for p in pkgs:
        try:
            __import__(p.split("==")[0].split(">=")[0].replace("-", "_"))
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

pip_install([
    "transformers>=4.42.0",
    "datasets>=2.19.0",
    "tqdm",
    "numpy>=1.24.0"
])
try:
    import torch
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# seeds & CUDA perf
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

# ---------------- Robust number + Answer: parsing ----------------
NUM_RE   = re.compile(r"-?\d[\d,]*\.?\d*")
FRAC_RE  = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
ANS_LINE = re.compile(r"(?i)^\s*answer\s*:\s*(.+?)\s*$")

def _to_num(token: str) -> Optional[float]:
    if token is None: return None
    s = token.strip().rstrip(".").replace(",", "")
    m = FRAC_RE.match(s)
    if m:
        d = int(m.group(2))
        if d == 0: return None
        return int(m.group(1))/d
    try:
        return int(s) if re.fullmatch(r"-?\d+", s) else float(s)
    except Exception:
        return None

def truncate_after_answer(text: str) -> str:
    lines = (text or "").splitlines()
    last = -1
    for i, ln in enumerate(lines):
        if ANS_LINE.match(ln): last = i
    return "\n".join(lines[:last+1]) if last >= 0 else text

def get_final_answer(text: str) -> Optional[float]:
    if not text: return None
    lines = text.splitlines()
    for ln in reversed(lines):
        m = ANS_LINE.match(ln)
        if not m: continue
        payload = re.sub(r"[^\d\-\./, ]", " ", m.group(1))
        payload = re.sub(r"\s+", " ", payload).strip()
        nums = NUM_RE.findall(payload)
        if nums: return _to_num(nums[-1])
        if FRAC_RE.match(payload): return _to_num(payload)
    # fallback: last numeric anywhere
    m = NUM_RE.findall(text)
    return _to_num(m[-1]) if m else None

def all_numbers(text: str) -> List[float]:
    if not text: return []
    nums = []
    for s in NUM_RE.findall(text):
        v = _to_num(s)
        if v is not None: nums.append(float(v))
    return nums

def parse_gsm8k_gold(ans: str):
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", ans)
    if not m: return None
    return _to_num(m.group(1))

def equal_num(a, b, eps=1e-3):
    if a is None or b is None: return False
    af = float(a); bf = float(b)
    # prefer exact int equality if both near ints
    if abs(af - round(af)) < 1e-9 and abs(bf - round(bf)) < 1e-9:
        return int(round(af)) == int(round(bf))
    return abs(af - bf) <= eps

def text_contains_gold(text: str, gold: float, eps=1e-3) -> bool:
    """Counts as correct if the gold number appears ANYWHERE in the text (tolerant)."""
    g = float(gold)
    for v in all_numbers(text or ""):
        if abs(float(v) - g) <= eps:
            return True
    return False

def fmt_float(x):
    try: return f"{x:.3f}"
    except: return "0.000"

# ---------------- Safe equation eval ----------------
_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UN  = (ast.USub, ast.UAdd)

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

def normalize_ops(text: str) -> str:
    t = text.replace("×", "*").replace("÷", "/").replace("·", "*").replace("^", "**")
    t = re.sub(r"\s+", " ", t)
    return t

def _numeric_substring(s: str) -> Optional[str]:
    t = re.sub(r"[^0-9+\-*/().^ ]", " ", s).replace("^", "**")
    t = re.sub(r"\s+", " ", t).strip()
    return t if any(ch.isdigit() for ch in t) else None

def find_equations(text: str) -> List[Tuple[str, str]]:
    t = normalize_ops(text)
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

def coherence_score_and_stats(text: str, final_num: Optional[float]) -> Tuple[float, int, int, int, int, set]:
    pairs = find_equations(text)
    if not pairs: return (0.0, 0, 0, 0, 0, set())
    true_cnt, false_cnt = 0, 0
    last_rhs_val = None
    true_set = set()
    for lhs, rhs in pairs:
        lv = _safe_eval_expr(lhs) or (_safe_eval_expr(_numeric_substring(lhs)) if _numeric_substring(lhs) else None)
        rv = _safe_eval_expr(rhs) or (_safe_eval_expr(_numeric_substring(rhs)) if _numeric_substring(rhs) else None)
        if lv is None or rv is None:
            false_cnt += 1; continue
        if abs(lv - rv) <= 1e-6:
            true_cnt += 1
            true_set.add(round(rv, 6))
        else:
            false_cnt += 1
        last_rhs_val = rv
    final_match = 0
    if last_rhs_val is not None and final_num is not None:
        final_match = int(abs(last_rhs_val - float(final_num)) <= 1e-6)
    eq_cnt = true_cnt + false_cnt
    coh = 0.7*(true_cnt/max(1,eq_cnt)) + 0.2*final_match - 0.1*(false_cnt/max(1,eq_cnt))
    return (float(coh), eq_cnt, true_cnt, false_cnt, final_match, true_set)

# ---------------- Direction ----------------
_OP_MAP = [
    (r"\bplus\b|\badd(ed)?\b|\+", "+"),
    (r"\bminus\b|\bsubtrac(t|ted)\b|-", "-"),
    (r"\btimes\b|\bmultipl(y|ied)\b|\*|×|\b(?<=\s)x(?=\s)\b", "*"),
    (r"\bdivid(e|ed)\b|/|÷", "/"),
    (r"\bpower\b|\^\s*|\*\*", "^"),
]

def op_sequence(text: str) -> List[str]:
    t = text.lower()
    idxs = []
    for pat, sym in _OP_MAP:
        for m in re.finditer(pat, t):
            idxs.append((m.start(), sym))
    idxs.sort(key=lambda x: x[0])
    return [sym for _, sym in idxs][:12]

def edit_distance_norm(a: List[str], b: List[str]) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0: return 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        ai = a[i-1]
        for j in range(1,m+1):
            cost = 0 if ai==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / max(1, max(n, m))

def ngram_jaccard(a: str, b: str, n: int=3) -> float:
    def grams(s):
        toks = s.lower().split()
        return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1)) if len(toks)>=n else set([tuple(toks)]) if toks else set()
    A, B = grams(a), grams(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

_GOAL_WORDS = ["total","sum","price","cost","time","minutes","hours","remaining","area","perimeter","distance","ways","probability","mean","average","rate","speed","volume","apples","students"]

def extract_goal_tag(question: str) -> Optional[str]:
    q = question.lower()
    for w in _GOAL_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", q): return w
    return None

def cand_has_goal(text: str, goal: Optional[str]) -> bool:
    if not goal: return True
    return bool(re.search(rf"\b{re.escape(goal)}\b", text.lower()))

# ---------------- Context model ----------------
class ContextModel:
    def __init__(self):
        self.uni = Counter(); self.bi = Counter(); self.V = set()
    def fit(self, seqs: List[List[str]]):
        for s in seqs:
            prev = "<s>"; self.uni[prev]+=1; self.V.add(prev)
            for tok in s + ["</s>"]:
                self.uni[tok]+=1; self.bi[(prev,tok)]+=1; self.V.add(tok); prev = tok
    def prob(self, prev: str, tok: str) -> float:
        V = max(1, len(self.V))
        return (self.bi.get((prev, tok), 0) + 1) / (self.uni.get(prev, 0) + V)
    def cross_entropy(self, s: List[str]) -> float:
        prev = "<s>"; H=0.0; T=0
        for tok in s + ["</s>"]:
            p = self.prob(prev, tok)
            H += -math.log(max(p, 1e-12)); T += 1; prev = tok
        return H/max(1,T)
    def next_argmax(self, prev: str) -> str:
        best, bestp = None, -1.0
        for t in self.V:
            if t=="<s>": continue
            p = self.prob(prev, t)
            if p > bestp: best, bestp = t, p
        return best if best else "</s>"

def cm_quality(cm: ContextModel, eval_seqs: List[List[str]]) -> float:
    if not eval_seqs: return 0.0
    uni = Counter()
    for s in eval_seqs:
        uni["<s>"] += 1
        for tok in s + ["</s>"]:
            uni[tok] += 1
    Z = sum(uni.values()) + len(uni)
    def Hb_of(s):
        H=0.0
        for tok in s+["</s>"]:
            p = (uni.get(tok,0)+1)/Z
            H += -math.log(p)
        return H/max(1,len(s)+1)
    H_deltas, agree_hits, agree_tot = [], 0, 0
    for s in eval_seqs:
        Hb = Hb_of(s); Hg = cm.cross_entropy(s)
        delta = max(0.0, min(1.0, (Hb - Hg)/max(1e-9, Hb)))
        H_deltas.append(delta)
        prev = "<s>"
        for tok in s + ["</s>"]:
            pred = cm.next_argmax(prev)
            if pred == tok: agree_hits += 1
            agree_tot += 1
            prev = tok
    q_op = float(np.mean(H_deltas)); agree = (agree_hits / max(1, agree_tot))
    return float(0.6*q_op + 0.4*agree)

# ---------------- Generation (truncate after Answer:) ----------------
@torch.inference_mode()
def decode_generated(tok, out_ids, input_lens):
    out_cpu = out_ids.detach().cpu()
    texts=[]
    B=len(input_lens)
    for b in range(out_cpu.size(0)):
        start = input_lens[b % B]
        t = tok.decode(out_cpu[b, start:], skip_special_tokens=True)
        texts.append(truncate_after_answer(t))
    return texts

@torch.inference_mode()
def generate_k(tok, mdl, prompts: List[str], k: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
    dev = next(mdl.parameters()).device
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    out = mdl.generate(
        **enc,
        do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature,
        num_return_sequences=k, max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        return_dict_in_generate=False
    )
    texts = decode_generated(tok, out, input_lens)
    B = len(prompts)
    groups = [texts[i*k:(i+1)*k] for i in range(B)]
    # lengths (approx; not used critically)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    out_np = out.detach().cpu().numpy()
    gen_lens = []
    for b in range(out_np.shape[0]):
        base = input_lens[b % B]
        row = out_np[b]
        nonpad = int((row != pad_id).sum())
        gen_lens.append(max(0, nonpad - base))
    return groups, gen_lens

# ---------------- Data & model ----------------
def load_gsm8k(split="test"):
    ds = load_dataset("gsm8k", "main")[split]
    items = []
    for ex in ds:
        q = ex["question"].strip()
        gold = parse_gsm8k_gold(ex["answer"])
        if gold is None: continue
        prompt = (
            "Solve the grade-school math problem.\n"
            f"Question: {q}\n"
            "Show brief calculations (a few equations). On the last line write: Answer: <number>."
        )
        items.append((prompt, gold, q))
    return items

def load_model(name: str, device: str, dtype: str):
    def _torch_dtype(device_, d):
        return torch.float32 if device_!="cuda" else (torch.bfloat16 if d=="bf16" else (torch.float16 if d=="fp16" else torch.float32))
    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=_torch_dtype(device, dtype), device_map=None)
        mdl.to(device).eval()
        return tok, mdl
    except RuntimeError as e:
        print(f"[WARN] OOM loading {name}: {e}. Falling back to Qwen/Qwen2-1.5B-Instruct.")
        fallback = "Qwen/Qwen2-1.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(fallback, use_fast=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(fallback, torch_dtype=_torch_dtype(device, "fp16" if device=="cuda" else "fp32"))
        mdl.to(device).eval()
        return tok, mdl

# --- batched closure + avg logprob (OOM-safe) ---
@torch.inference_mode()
def closure_lp_batch(tok, mdl, prompt: str, answers: List[str], batch_size: int = 8) -> Tuple[List[float], List[float]]:
    dev = next(mdl.parameters()).device
    enc_prefix = tok(prompt.rstrip()+"\n", return_tensors="pt").to(dev)
    prefix_len = int(enc_prefix["input_ids"].shape[1])
    all_cl, all_lp = [], []
    for s in range(0, len(answers), batch_size):
        chunk = answers[s:s+batch_size]
        enc = tok([prompt.rstrip()+"\n"+a for a in chunk], return_tensors="pt", padding=True, truncation=True).to(dev)
        logits = mdl(**enc, use_cache=False).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        pad_id = tok.pad_token_id or 0
        cont = torch.zeros_like(labels, dtype=torch.bool)
        cont[:, prefix_len-1:] = True
        attn = enc.get("attention_mask", None)
        if attn is not None: cont &= attn[:, 1:].bool()
        cont &= (labels != pad_id)
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).masked_fill(~cont, 0.0)
        cnt = cont.long().sum(dim=1).clamp(min=1)
        all_lp += (tok_lp.sum(dim=1)/cnt).detach().cpu().tolist()
        preds = logits.argmax(dim=-1)
        eq = (preds == labels) & cont
        all_cl += (eq.float().sum(dim=1)/cnt).detach().cpu().tolist()
        del enc, logits, labels
        if dev.type=="cuda": torch.cuda.empty_cache()
    return all_cl, all_lp

# ---------------- K split ----------------
def split_k(k: int, max_loops: int=5) -> List[int]:
    parts=[]; k_rem=k
    for r in range(max_loops):
        if k_rem<=0: break
        a = max(1, int(math.ceil(0.6*k))) if r==0 else (max(1, int(math.ceil(0.3*k))) if r==1 else k_rem)
        a = min(a, k_rem); parts.append(a); k_rem -= a
        if k_rem==0: break
    return parts[:max_loops]

# ---------------- Adaptive per item (your logic kept) ----------------
def build_consensus_opseq(ops_lists: List[List[str]]) -> List[str]:
    cm = ContextModel(); cm.fit(ops_lists)
    seq=[]; prev="<s>"
    for _ in range(8):
        nxt = cm.next_argmax(prev)
        if nxt == "</s>": break
        seq.append(nxt); prev = nxt
    return seq

def direction_score(ops: List[str], cons_ops: List[str]) -> float:
    return 1.0 - edit_distance_norm(ops, cons_ops)

def adaptive_pick_for_item(
    tok, mdl, prompt: str, gold: float, question: str, k: int, max_new_tokens: int,
    temperature: float, top_k: int, top_p: float,
    max_loops: int=5, cmq_thresh: float=0.6,
    dual_anchor: bool=False, ablate_coh: bool=False, ablate_cl: bool=False,
    lp_batch: int = 8
) -> Dict[str, Any]:

    goal_tag = extract_goal_tag(question)
    pool = []; total_gen_tokens = 0
    last_best_num = None; consec_low_parse = 0

    score_cache: Dict[str, Tuple[float, float]] = {}  # text -> (closure, lp)

    def score_texts(texts: List[str]) -> Tuple[List[float], List[float]]:
        missing = [t for t in texts if t not in score_cache]
        if missing:
            cl_new, lp_new = closure_lp_batch(tok, mdl, prompt, missing, batch_size=lp_batch)
            for t, c, l in zip(missing, cl_new, lp_new):
                score_cache[t] = (c, l)
        cl = [score_cache[t][0] for t in texts]
        lp = [score_cache[t][1] for t in texts]
        return cl, lp

    def add_candidates(texts: List[str], gen_lens: List[int]):
        nonlocal total_gen_tokens
        for t, glen in zip(texts, gen_lens):
            num = get_final_answer(t)
            ops = op_sequence(t)
            coh, eqn_cnt, t_cnt, f_cnt, fmatch, true_eqs = coherence_score_and_stats(t, num)
            pool.append({
                "text": t, "num": num, "ops": ops,
                "coh": coh, "coh_stats": (eqn_cnt, t_cnt, f_cnt, fmatch),
                "true_eqs": true_eqs,
                "closure": None, "lp": None
            })
            total_gen_tokens += int(glen)

    def immediate_score_last(n_new: int):
        if n_new <= 0: return
        start = len(pool) - n_new
        idxs = list(range(start, len(pool)))
        cl, lp = score_texts([pool[i]["text"] for i in idxs])
        for j, i in enumerate(idxs):
            pool[i]["closure"] = cl[j]; pool[i]["lp"] = lp[j]

    # generation budget split
    k_parts = split_k(k, max_loops=max_loops)
    cm_first = ContextModel(); cmq_smoothed = None; loops_used = 0

    # Round 1
    texts1, lens1 = generate_k(tok, mdl, [prompt], k_parts[0], max_new_tokens, temperature, top_k, top_p)
    add_candidates(texts1[0], lens1); immediate_score_last(len(texts1[0]))
    cm_first.fit([c["ops"] for c in pool])
    cons_ops = build_consensus_opseq([c["ops"] for c in pool])
    dir_scores = [direction_score(c["ops"], cons_ops) for c in pool]

    # anchors (dual hedge optional)
    order = np.argsort(-np.array(dir_scores))
    anchors = [int(order[0])]
    for idx in order[1:]:
        if edit_distance_norm(pool[int(idx)]["ops"], pool[anchors[0]]["ops"]) >= 0.4:
            anchors.append(int(idx)); break
    if not dual_anchor: anchors = anchors[:1]

    def rank_for_anchor(anchor_i: int, use_closure_primary: bool):
        anchor_ops = pool[anchor_i]["ops"]; anchor_true_eqs = pool[anchor_i]["true_eqs"]
        # neighborhood: ops + soft goal + eq overlap, staged widening
        neigh_idx = []
        # pass 1: goal + tight
        for i,c in enumerate(pool):
            if goal_tag and not cand_has_goal(c["text"], goal_tag): continue
            if edit_distance_norm(c["ops"], anchor_ops) <= 0.25: neigh_idx.append(i)
        # pass 2: goal + medium
        if len(neigh_idx) < 3:
            for i,c in enumerate(pool):
                if goal_tag and not cand_has_goal(c["text"], goal_tag): continue
                if edit_distance_norm(c["ops"], anchor_ops) <= 0.35: neigh_idx.append(i)
        # pass 3: drop goal + wider
        if len(neigh_idx) < 3:
            for i,c in enumerate(pool):
                if edit_distance_norm(c["ops"], anchor_ops) <= 0.45: neigh_idx.append(i)
        # pass 4: directional shortlist
        if not neigh_idx:
            ranked = sorted(((i, edit_distance_norm(pool[i]["ops"], anchor_ops)) for i in range(len(pool))), key=lambda x: x[1])
            take = max(5, min(10, len(ranked))); neigh_idx = [i for i,_ in ranked[:take]]
        # eq-overlap if possible
        if anchor_true_eqs:
            overlap = [i for i in neigh_idx if pool[i]["true_eqs"] & anchor_true_eqs]
            if overlap: neigh_idx = overlap

        texts = [pool[i]["text"] for i in neigh_idx]
        cl, lp = score_texts(texts)
        for j, i in enumerate(neigh_idx):
            pool[i]["closure"] = cl[j]; pool[i]["lp"] = lp[j]

        def coh_for_rank(i):
            if ablate_coh: return -1e9
            eq_cnt = pool[i]["coh_stats"][0]
            return pool[i]["coh"] if eq_cnt >= 2 else -1e9
        def cl_for_rank(i):
            return -1e9 if ablate_cl else (pool[i]["closure"] if pool[i]["closure"] is not None else -1e9)

        if use_closure_primary:
            neigh_idx.sort(key=lambda i: (-cl_for_rank(i), -coh_for_rank(i), -(pool[i]["lp"] or -1e9), len(pool[i]["text"])))
        else:
            neigh_idx.sort(key=lambda i: (-coh_for_rank(i), -cl_for_rank(i), -(pool[i]["lp"] or -1e9), len(pool[i]["text"])))
        best_i = neigh_idx[0]
        return best_i, neigh_idx, anchor_ops

    # Round 2 for CMQ (minibatch, EMA; guard ≥3)
    if len(k_parts)>1 and k_parts[1]>0:
        hint = (" Keep operation order: " + " ".join(cons_ops[:4]) + ".") if cons_ops else ""
        refine = f"{prompt}\n{hint} Recompute carefully. On the last line write: Answer: <number>."
        texts2, lens2 = generate_k(tok, mdl, [refine], k_parts[1], max_new_tokens, temperature, top_k, top_p)
        add_candidates(texts2[0], lens2); immediate_score_last(len(texts2[0]))
        recent_ops = [c["ops"] for c in pool[-min(5, k_parts[1]):]]
        if len(recent_ops) >= 3:
            cmq_now = cm_quality(cm_first, recent_ops)
            cmq_smoothed = cmq_now if cmq_smoothed is None else (0.7*cmq_smoothed + 0.3*cmq_now)

    # Iterate ≤5
    for r in range(len(k_parts)):
        loops_used = r+1
        use_closure_primary = bool(cmq_smoothed is not None and cmq_smoothed >= cmq_thresh)
        if consec_low_parse >= 2: use_closure_primary = True

        # rank for each anchor, pick better by primary
        bests = []
        for a in anchors:
            best_i, neigh_idx, anchor_ops = rank_for_anchor(a, use_closure_primary)
            prim = pool[best_i]["closure"] if use_closure_primary and not ablate_cl else (pool[best_i]["coh"] if not ablate_coh else -1e9)
            bests.append((prim, best_i, neigh_idx, anchor_ops))
        bests.sort(key=lambda x: -x[0])
        _, best_i, neigh_idx, chosen_anchor_ops = bests[0]
        best_text = pool[best_i]["text"]; best_num = pool[best_i]["num"]

        # minimal final verify (prefer coherent top)
        def _ok(text):
            n = get_final_answer(text)
            coh, eqc, *_ = coherence_score_and_stats(text, n)
            return (eqc >= 2) and (coh >= 0.45)
        cands = [best_text] + [pool[i]["text"] for i in neigh_idx[1:3]]
        for t in cands:
            if _ok(t):
                best_text = t; best_num = get_final_answer(t); break

        # parse signal for fallback
        eq_parsed_now = sum(pool[i]["coh_stats"][0] for i in neigh_idx[:min(5, len(neigh_idx))])
        consec_low_parse = (consec_low_parse + 1) if eq_parsed_now < 2 else 0

        # convergence: text sim, answer consensus, direction sim, number stability
        top3 = [pool[i]["text"] for i in neigh_idx[:3]] if len(neigh_idx)>=3 else [best_text]
        sim = 0.0
        if len(top3) >= 2:
            sims = []
            for ia in range(len(top3)):
                for ib in range(ia+1, len(top3)):
                    sims.append( ngram_jaccard(top3[ia], top3[ib], n=3) )
            sim = float(np.mean(sims)) if sims else 0.0
        nums_local = [pool[i]["num"] for i in neigh_idx[:min(5,len(neigh_idx))]]
        cnt = Counter([n for n in nums_local if n is not None])
        ans_cons = (max(cnt.values())/len(nums_local)) if cnt else 0.0
        dir_sim = np.mean([1.0 - edit_distance_norm(pool[i]["ops"], chosen_anchor_ops) for i in neigh_idx[:min(3, len(neigh_idx))]]).item() if neigh_idx else 0.0
        num_stable = (best_num is not None and last_best_num is not None and equal_num(best_num, last_best_num))
        last_best_num = best_num

        converged = ((sim >= 0.80 and ans_cons >= 0.70 and dir_sim >= 0.75 and num_stable) or (loops_used >= len(k_parts)))
        if converged: break

        # more budget? add, update anchors, consensus, CMQ (guard ≥3)
        remain_parts = k_parts[loops_used:]
        if remain_parts and remain_parts[0] > 0:
            hint = (" Keep operation order: " + " ".join(cons_ops[:4]) + ".") if cons_ops else ""
            refine = f"{prompt}\n{hint} Recompute carefully. On the last line write: Answer: <number>."
            textsX, lensX = generate_k(tok, mdl, [refine], remain_parts[0], max_new_tokens, temperature, top_k, top_p)
            add_candidates(textsX[0], lensX); immediate_score_last(len(textsX[0]))
            # re-derive consensus & anchors from pool
            cons_ops = build_consensus_opseq([c["ops"] for c in pool])
            dir_scores = [direction_score(c["ops"], cons_ops) for c in pool]
            order = np.argsort(-np.array(dir_scores))
            anchors = [int(order[0])]
            for idx2 in order[1:]:
                if edit_distance_norm(pool[int(idx2)]["ops"], pool[anchors[0]]["ops"]) >= 0.4:
                    anchors.append(int(idx2)); break
            if not dual_anchor: anchors = anchors[:1]
            # update CMQ
            recent_ops = [c["ops"] for c in pool[-min(5, remain_parts[0]):]]
            if len(recent_ops) >= 3:
                cmq_now = cm_quality(cm_first, recent_ops)
                cmq_smoothed = cmq_now if cmq_smoothed is None else (0.7*cmq_smoothed + 0.3*cmq_now)

    # ---------- Accuracy flags ----------
    # "True if final answer equals OR text contains the gold number anywhere"
    def _ok_text_or_contains(txt, num):
        return equal_num(get_final_answer(txt), gold) or text_contains_gold(txt, gold)

    ok_adapt  = (equal_num(best_num, gold) or text_contains_gold(best_text, gold))

    # Baselines (ensure cached scores computed)
    missing_texts = [c["text"] for c in pool if c["lp"] is None or c["closure"] is None]
    if missing_texts:
        cl_all, lp_all = closure_lp_batch(tok, mdl, prompt, missing_texts, batch_size=lp_batch)
        for t,c_,l_ in zip(missing_texts, cl_all, lp_all):
            for c in pool:
                if c["text"] == t:
                    c["closure"] = c_; c["lp"] = l_

    nums_all = [c["num"] for c in pool]
    texts_all = [c["text"] for c in pool]

    # SC (tie by best LP)
    freq = Counter([n for n in nums_all if n is not None])
    pick_sc = None
    if len(freq) > 0:
        bc = max(freq.values()); tied = [n for n,c in freq.items() if c==bc]
        best_lp = -1e9; best_num_lp = None
        for c in pool:
            if c["num"] in tied and c["lp"] is not None and c["lp"] > best_lp:
                best_lp, best_num_lp = c["lp"], c["num"]
        pick_sc = tied[0] if len(tied)==1 else best_num_lp
    # Best-LP
    idx_lp = int(np.argmax([c["lp"] for c in pool]))
    pick_lp = pool[idx_lp]["num"]; text_lp = pool[idx_lp]["text"]
    # Closure
    idx_cl = int(np.argmax([c["closure"] for c in pool]))
    pick_closure = pool[idx_cl]["num"]; text_cl = pool[idx_cl]["text"]
    # First
    pick_first = pool[0]["num"]; text_first = pool[0]["text"]

    ok_sc      = (equal_num(pick_sc, gold)      or (pick_sc is not None and any(abs(pick_sc-float(g))<=1e-3 for g in [gold])) or text_contains_gold(" ".join(texts_all), gold))
    ok_lp      = (equal_num(pick_lp, gold)      or text_contains_gold(text_lp, gold))
    ok_closure = (equal_num(pick_closure, gold) or text_contains_gold(text_cl, gold))
    ok_first   = (equal_num(pick_first, gold)   or text_contains_gold(text_first, gold))

    return {
        "ok_adapt": int(ok_adapt), "ok_sc": int(ok_sc), "ok_lp": int(ok_lp), "ok_closure": int(ok_closure), "ok_first": int(ok_first),
        "pick_adapt": best_num, "pick_sc": pick_sc, "pick_lp": pick_lp, "pick_closure": pick_closure, "pick_first": pick_first,
        "pool_texts": texts_all, "pool_nums": nums_all,
        "token_budget_gen": int(total_gen_tokens)
    }

# ---------------- Bootstrap CI ----------------
def bootstrap_ci(flags: List[int], B: int=1000) -> Tuple[float,float,float]:
    n = len(flags); arr = np.array(flags, dtype=np.int32); mean = arr.mean()
    if B <= 0 or n <= 1: return (mean, mean, mean)
    samples = [arr[np.random.randint(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(samples, [2.5, 97.5]).tolist()
    return (mean, lo, hi)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--k", type=int, default=32)  # Augmenté pour générer plus de candidats
    ap.add_argument("--max_new_tokens", type=int, default=1024)  # Permet des réponses plus longues
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)  # Ajusté pour un meilleur échantillonnage
    ap.add_argument("--top_p", type=float, default=0.9)  # Ajusté pour un meilleur échantillonnage
    ap.add_argument("--limit", type=int, default=0, help="0 = full GSM8K test (≈1319).")  # Pas de limite
    ap.add_argument("--max_loops", type=int, default=5)
    ap.add_argument("--cmq_thresh", type=float, default=0.8)  # Seuil de qualité contextuelle ajusté
    ap.add_argument("--dual_anchor", action="store_true")
    ap.add_argument("--ablate_coherence", action="store_true")
    ap.add_argument("--ablate_closure", action="store_true")
    ap.add_argument("--lp_batch", type=int, default=16)  # Augmenté pour traiter plus de réponses par lot
    ap.add_argument("--out_csv", type=str, default="gsm8k_adaptive_intent_results_full.csv")  # Fichier de sortie
    ap.add_argument("--bootstrap", type=int, default=1000, help="e.g., 1000 for 95% CI")  # Bootstrap activé
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; switching to CPU.")
        args.device = "cpu"; args.dtype = "fp32"

    print(f"Loading model: {args.model} on {args.device} ({args.dtype})")
    tok, mdl = load_model(args.model, args.device, args.dtype)
    cfg = getattr(mdl, "config", None)
    commit = getattr(cfg, "_commit_hash", None) if cfg else None
    gpu_info = ""
    if args.device == "cuda":
        try:
            gpu_info = f" | GPU: {torch.cuda.get_device_name()} VRAM≈{round(torch.cuda.get_device_properties(0).total_memory/1e9,1)}GB"
        except Exception:
            pass
    print(f"torch={torch.__version__} transformers={__import__('transformers').__version__} commit={commit} seed=0{gpu_info}")

    print("Loading GSM8K test...")
    data_all = load_gsm8k("test")
    if args.limit and args.limit > 0: data_all = data_all[:args.limit]
    N = len(data_all)
    print(f"Eval items: {N} | k={args.k} | loops≤{args.max_loops} | cmq_thresh={args.cmq_thresh} | dual_anchor={args.dual_anchor}")

    t0 = time.time()
    acc_flags = {"first":[], "closure":[], "sc":[], "lp":[], "adapt":[]}
    rows = []
    total_tokens = 0

    pbar = tqdm(enumerate(data_all), total=N, desc="Items", dynamic_ncols=True)
    for idx, (prompt, gold, question) in pbar:
        res = adaptive_pick_for_item(
            tok, mdl, prompt, gold, question,
            k=args.k, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            max_loops=args.max_loops, cmq_thresh=args.cmq_thresh,
            dual_anchor=args.dual_anchor, ablate_coh=args.ablate_coherence, ablate_cl=args.ablate_closure,
            lp_batch=args.lp_batch
        )
        acc_flags["first"].append(int(res["ok_first"]))
        acc_flags["closure"].append(int(res["ok_closure"]))
        acc_flags["sc"].append(int(res["ok_sc"]))
        acc_flags["lp"].append(int(res["ok_lp"]))
        acc_flags["adapt"].append(int(res["ok_adapt"]))
        total_tokens += res.get("token_budget_gen", 0)

        done = idx+1
        pbar.set_postfix({
            "done%": f"{100.0*done/N:.1f}",
            "acc_adapt": fmt_float(np.mean(acc_flags["adapt"])),
            "acc_sc": fmt_float(np.mean(acc_flags["sc"])),
            "acc_lp": fmt_float(np.mean(acc_flags["lp"]))
        })

        rows.append([
            prompt, str(gold),
            json.dumps(res["pool_texts"]),
            json.dumps(res["pool_nums"]),
            int(res["ok_first"]), int(res["ok_closure"]), int(res["ok_sc"]), int(res["ok_lp"]), int(res["ok_adapt"]),
            res["pick_first"], res["pick_closure"], res["pick_sc"], res["pick_lp"], res["pick_adapt"]
        ])

    elapsed = time.time() - t0
    print("\n=== GSM8K Results (same total k generations) ===")
    for key, label in [("first","First"),("closure","Closure"),("sc","Self-Consistency"),("lp","Best-LP"),("adapt","ADAPTIVE")]:
        if args.bootstrap>0:
            mean, lo, hi = bootstrap_ci(acc_flags[key], args.bootstrap)
            print(f"Acc ({label:>16}): {mean:.3f}  [95% CI {lo:.3f},{hi:.3f}]")
        else:
            print(f"Acc ({label:>16}): {np.mean(acc_flags[key]):.3f}")
    print(f"Gen tokens≈     : {total_tokens}")
    print(f"Sec/Item        : {elapsed/max(1,N):.3f}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt","gold","candidates","candidate_numbers",
            "ok_first","ok_closure","ok_sc","ok_bestlp","ok_adaptive",
            "pick_first","pick_closure","pick_sc","pick_bestlp","pick_adaptive"
        ])
        w.writerows(rows)
    print(f"\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
