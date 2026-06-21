# compare_matrix_token_vs_vanilla_attention_implicit_shape_v1.py
# Goal: make SHAPE implicit (no shape embeddings, no shape-id head, no contrastive separation).
# The only "shape signal" is architectural:
# - fixed per-token masks (which elements exist)
# - hard compat gating in attention (matrix<->matrix only if SAME mask class)
#
# Compare:
# 1) Matrix-token transformer (MatLin + 2D masked tokens + hard compat)
# 2) Vanilla attention transformer (vector tokens + MHA) with the SAME hard compat gating
#
# Also prints balanced accuracy to avoid being fooled by label imbalance.

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 64
SHAPES = [(4, 16), (8, 8), (2, 32), (16, 4)]
assert all(r * c == D for r, c in SHAPES)

RMAX = max(r for r, _ in SHAPES)
CMAX = max(c for _, c in SHAPES)

MAT_IDS = list(range(len(SHAPES)))

TOK_PLUS   = len(SHAPES)
TOK_MATMUL = len(SHAPES) + 1
TOK_EQ     = len(SHAPES) + 2
TOK_TRUE   = len(SHAPES) + 3
TOK_FALSE  = len(SHAPES) + 4
TOK_PAD    = len(SHAPES) + 5

VOCAB = len(SHAPES) + 6
SEQ = 9

STEPS = 300
BATCH = 138
LR = 2e-3
REPORT_EVERY = 30
EVAL_BATCHES = 3

def shape_id(tid: int) -> int:
    # "shape class" is a property of the token type, but we do NOT embed it or supervise it.
    return tid if tid in MAT_IDS else -1

def shape_rc(sid: int):
    return SHAPES[sid]

# -----------------------------
# Dataset
# -----------------------------
def valid_add(a_sid, b_sid):
    return a_sid == b_sid

def valid_matmul(a_sid, b_sid):
    ra, ca = shape_rc(a_sid)
    rb, cb = shape_rc(b_sid)
    return ca == rb

def sample_example():
    a = random.choice(MAT_IDS)
    b = random.choice(MAT_IDS)
    op = TOK_PLUS if random.random() < 0.5 else TOK_MATMUL
    ok = valid_add(a, b) if op == TOK_PLUS else valid_matmul(a, b)
    lab = TOK_TRUE if ok else TOK_FALSE
    seq = [a, op, b, TOK_EQ, lab]
    seq = [TOK_PAD] * (SEQ - len(seq)) + seq
    return seq

def sample_batch(batch=BATCH):
    x = torch.empty(batch, SEQ, dtype=torch.long)
    y = torch.empty(batch, SEQ, dtype=torch.long)
    for i in range(batch):
        seq = sample_example()
        x[i] = torch.tensor(seq)
        y[i, :-1] = x[i, 1:]
        y[i, -1] = TOK_PAD
    return x.to(DEVICE), y.to(DEVICE)

# -----------------------------
# Metrics: raw + balanced accuracy
# -----------------------------
@torch.no_grad()
def tf_metrics(model, n_batches=10, batch=BATCH):
    model.eval()
    eq_pos = SEQ - 2
    lab_pos = SEQ - 1

    tp = tn = fp = fn = 0
    for _ in range(n_batches):
        x, _ = sample_batch(batch)
        logits = model(x)  # [B,T,V]
        p = logits[:, eq_pos, :]
        pred01 = torch.argmax(p[:, [TOK_TRUE, TOK_FALSE]], dim=-1)
        pred = torch.where(
            pred01 == 0,
            torch.tensor(TOK_TRUE, device=x.device),
            torch.tensor(TOK_FALSE, device=x.device),
        )
        gold = x[:, lab_pos]

        is_true = gold == TOK_TRUE
        is_false = gold == TOK_FALSE

        tp += (pred[is_true] == TOK_TRUE).sum().item()
        fn += (pred[is_true] == TOK_FALSE).sum().item()
        tn += (pred[is_false] == TOK_FALSE).sum().item()
        fp += (pred[is_false] == TOK_TRUE).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    acc_true = tp / max(1, tp + fn)
    acc_false = tn / max(1, tn + fp)
    bal_acc = 0.5 * (acc_true + acc_false)
    true_rate = (tp + fn) / max(1, tp + tn + fp + fn)
    return acc, bal_acc, acc_true, acc_false, true_rate

# -----------------------------
# Matrix-token model (implicit shape)
# -----------------------------
def masked_mean_var(x, m, eps=1e-5):
    denom = m.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mu = (x * m).sum(dim=(-2, -1), keepdim=True) / denom
    var = ((x - mu) ** 2 * m).sum(dim=(-2, -1), keepdim=True) / denom
    return mu, var + eps

class MaskedMatLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(RMAX, CMAX))
        self.b = nn.Parameter(torch.zeros(RMAX, CMAX))

    def forward(self, x, m):
        mu, var = masked_mean_var(x, m, self.eps)
        y = (x - mu) / torch.sqrt(var)
        return (y * self.g + self.b) * m

class MatLin(nn.Module):
    def __init__(self):
        super().__init__()
        self.WL = nn.Parameter(torch.randn(RMAX, RMAX) / math.sqrt(RMAX))
        self.WR = nn.Parameter(torch.randn(CMAX, CMAX) / math.sqrt(CMAX))
        self.B  = nn.Parameter(torch.zeros(RMAX, CMAX))

    def forward(self, x):
        x = torch.einsum("ij,btjk->btik", self.WL, x)
        x = torch.einsum("btik,kl->btil", x, self.WR)
        return x + self.B

class HardCompatSelfAttnMat(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = MatLin()
        self.k = MatLin()
        self.v = MatLin()
        self.o = MatLin()

    def forward(self, x, m, compat, valid_count):
        q = self.q(x) * m
        k = self.k(x) * m
        v = self.v(x) * m

        scale = valid_count.clamp_min(1.0).sqrt().unsqueeze(-1)  # [B,T,1]
        scores = torch.einsum("btrc,bsrc->bts", q, k) / (scale * scale.transpose(1, 2))

        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
        scores = scores.masked_fill(~compat, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        y = torch.einsum("bts,bsrc->btrc", attn, v)
        return self.o(y) * m

class VarShapeMatrixLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)

        self.ln1 = MaskedMatLayerNorm()
        self.attn = HardCompatSelfAttnMat()
        self.ln2 = MaskedMatLayerNorm()
        self.ff1 = MatLin()
        self.ff2 = MatLin()

        self.out = nn.Linear(D, VOCAB)

        # Precompute per-token masks + shape classes (NOT learned; no embeddings; no losses)
        masks = torch.zeros(VOCAB, RMAX, CMAX)
        sids  = torch.full((VOCAB,), -1, dtype=torch.long)
        for tid in range(VOCAB):
            sid = shape_id(tid)
            if sid >= 0:
                r, c = shape_rc(sid)
                masks[tid, :r, :c] = 1.0
                sids[tid] = sid
            else:
                masks[tid, :8, :8] = 1.0  # specials fixed view
        self.register_buffer("TOK_MASK", masks)
        self.register_buffer("TOK_SID",  sids)

    def embed(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        flat = self.tok(idx) + self.pos(pos)[None, :, :]  # [B,T,D]
        sid = self.TOK_SID[idx]  # [B,T]
        m = self.TOK_MASK[idx]   # [B,T,R,C]

        # Scatter flat -> (r,c) view (mask defines what exists)
        x = torch.zeros(B, T, RMAX, CMAX, device=idx.device)
        for tid in range(VOCAB):
            sel = (idx == tid)
            if not sel.any():
                continue
            sid0 = shape_id(tid)
            r, c = shape_rc(sid0) if sid0 >= 0 else (8, 8)
            x_sel = x[sel]
            x_sel[:, :r, :c] = flat[sel].view(-1, r, c)
            x[sel] = x_sel

        valid_count = m.sum(dim=(-2, -1))  # [B,T]

        # Hard compat (implicit shape): matrix<->matrix only if same sid; specials always allowed
        sid_q = sid.unsqueeze(2)
        sid_k = sid.unsqueeze(1)
        both_mat = (sid_q >= 0) & (sid_k >= 0)
        compat = (~both_mat) | (sid_q == sid_k)  # [B,T,T]
        return x, m, compat, valid_count

    def forward(self, idx):
        x, m, compat, valid_count = self.embed(idx)
        x = x + self.attn(self.ln1(x, m), m, compat, valid_count)
        x = x + (self.ff2(F.gelu(self.ff1(self.ln2(x, m)))) * m)

        # Readout flatten back to D per token view
        B, T = idx.shape
        out_flat = torch.zeros(B, T, D, device=idx.device)
        for tid in range(VOCAB):
            sel = (idx == tid)
            if not sel.any():
                continue
            sid0 = shape_id(tid)
            r, c = shape_rc(sid0) if sid0 >= 0 else (8, 8)
            out_flat[sel] = x[sel][:, :r, :c].contiguous().view(-1, D)

        return self.out(out_flat)  # [B,T,V]

# -----------------------------
# Vanilla attention baseline (implicit shape)
# -----------------------------
class HardCompatSelfAttnVec(nn.Module):
    def __init__(self, d_model=D, n_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h, compat):
        # h: [B,T,D], compat: [B,T,T] True=allowed
        T = h.size(1)
        causal = torch.triu(torch.ones(T, T, device=h.device), 1).bool()  # [T,T]
        allow = compat & (~causal.unsqueeze(0))  # [B,T,T]
        attn_mask = ~allow  # True=block in MHA
        B, T, _ = h.shape
        H = self.mha.num_heads

        attn_mask = attn_mask.unsqueeze(1)          # [B,1,T,T]
        attn_mask = attn_mask.expand(B, H, T, T)    # [B,H,T,T]
        attn_mask = attn_mask.reshape(B * H, T, T)  # [B*H,T,T]

        a, _ = self.mha(h, h, h, attn_mask=attn_mask)

        return a

class VanillaCompatLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)

        self.ln1 = nn.LayerNorm(D)
        self.attn = HardCompatSelfAttnVec(D, n_heads=4)
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

        self.out = nn.Linear(D, VOCAB)

        # Fixed token->shape class table (NOT embedded, NOT supervised)
        sids = torch.full((VOCAB,), -1, dtype=torch.long)
        for tid in range(VOCAB):
            sid = shape_id(tid)
            if sid >= 0:
                sids[tid] = sid
        self.register_buffer("TOK_SID", sids)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        h = self.emb(idx) + self.pos(pos)[None, :, :]  # [B,T,D]

        sid = self.TOK_SID[idx]  # [B,T]

        # SAME hard compat (implicit): matrix<->matrix only if same shape class
        sid_q = sid.unsqueeze(2)
        sid_k = sid.unsqueeze(1)
        both_mat = (sid_q >= 0) & (sid_k >= 0)
        compat = (~both_mat) | (sid_q == sid_k)  # [B,T,T]

        h = h + self.attn(self.ln1(h), compat)
        h = h + self.ff(self.ln2(h))

        return self.out(h)  # [B,T,V]

# -----------------------------
# Train loop (shared)
# -----------------------------
def train_one(model, name):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ema = None

    for step in range(1, STEPS + 1):
        x, y = sample_batch(BATCH)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=TOK_PAD)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.item()
        ema = l if ema is None else (0.95 * ema + 0.05 * l)

        if step == 1 or step % REPORT_EVERY == 0:
            acc, bal, atr, afr, tr = tf_metrics(model, n_batches=EVAL_BATCHES, batch=BATCH)
            print(
                f"{name} [step {step:4d}/{STEPS}] "
                f"loss={l:.4f} ema={ema:.4f} "
                f"acc={acc:.3f} bal={bal:.3f} "
                f"accT={atr:.3f} accF={afr:.3f} true_rate={tr:.3f}"
            )

def main():
    print("\nTASK: predict validity of matrix ops from token shapes (implicit shape)")
    print(f"Shapes={SHAPES}  D={D}  Padded={RMAX}x{CMAX}  SEQ={SEQ}  Vocab={VOCAB}")
    print("Rule: '+' valid iff same shape; '@' valid iff inner dims match. Output label TRUE/FALSE.")
    print("Hard compat: matrix<->matrix attention only if SAME shape class (specials always compatible).")
    print("No shape embeddings, no shape heads, no contrastive losses.")
    print(f"Device: {DEVICE}\n")

    mat = VarShapeMatrixLM()
    van = VanillaCompatLM()

    print("=== Train: Matrix-token transformer (implicit) ===")
    train_one(mat, "MAT")

    print("\n=== Train: Vanilla attention baseline (implicit) ===")
    train_one(van, "VAN")

    acc_m, bal_m, atr_m, afr_m, tr_m = tf_metrics(mat, n_batches=40, batch=BATCH)
    acc_v, bal_v, atr_v, afr_v, tr_v = tf_metrics(van, n_batches=40, batch=BATCH)

    print("\nFINAL METRICS (label predicted after '='):")
    print(f"MAT: acc={acc_m:.3f} bal={bal_m:.3f} accT={atr_m:.3f} accF={afr_m:.3f} true_rate={tr_m:.3f}")
    print(f"VAN: acc={acc_v:.3f} bal={bal_v:.3f} accT={atr_v:.3f} accF={afr_v:.3f} true_rate={tr_v:.3f}")

if __name__ == "__main__":
    main()

