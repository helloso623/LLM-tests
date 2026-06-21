# cnn_vs_models_harder_mixed_validity_low_compute_v3_FAST_FIXED.py

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 64

# A is your original shapes
SHAPES_A = [(4, 16), (8, 8), (2, 32), (16, 4)]
assert all(r * c == D for r, c in SHAPES_A)

# B is chosen so that r(B) covers every c(A): {16,8,32,4}
SHAPES_B = [(16, 4), (8, 8), (32, 2), (4, 16)]
assert all(r * c == D for r, c in SHAPES_B)

NSHA = len(SHAPES_A)
NSHB = len(SHAPES_B)

# Padding must cover BOTH sets
RMAX = max(max(r for r, _ in SHAPES_A), max(r for r, _ in SHAPES_B))  # 32
CMAX = max(max(c for _, c in SHAPES_A), max(c for _, c in SHAPES_B))  # 32

# Low compute
BATCH = 64            # must be divisible by 4
STEPS = 160
LR = 1e-3
REPORT_EVERY = 20

# Fixed eval sets
EVAL_SAMPLES = 1024
FINAL_EVAL_SAMPLES = 4096
INFER_BS = 512

# Task difficulty
PAT_AMP = 0.60
PAT_JITTER = 0.40
NOISE_STD = 0.55
DISTR_P = 0.15
LABEL_FLIP_P = 0.02
DECOY_IN_NEG_P = 0.20
DROPOUT = 0.10

AMP_ENABLED = True

# -----------------------------
# RNG streams (train/eval isolated)
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class RNGStreams:
    train_py: random.Random
    eval_py: random.Random
    train_t: torch.Generator
    eval_t: torch.Generator

def make_streams(seed: int) -> RNGStreams:
    train_py = random.Random(seed + 101)
    eval_py  = random.Random(seed + 202)
    dev = DEVICE if DEVICE in ("cuda", "cpu") else "cpu"
    train_t = torch.Generator(device=dev).manual_seed(seed + 303)
    eval_t  = torch.Generator(device=dev).manual_seed(seed + 404)
    return RNGStreams(train_py=train_py, eval_py=eval_py, train_t=train_t, eval_t=eval_t)

# -----------------------------
# Shapes / masks / validity
# -----------------------------
def shape_rc_A(sid: int):
    return SHAPES_A[sid]

def shape_rc_B(sid: int):
    return SHAPES_B[sid]

def valid_matmul(a_sid: int, b_sid: int) -> bool:
    return SHAPES_A[a_sid][1] == SHAPES_B[b_sid][0]

def make_masks_from_sids(sids: torch.Tensor, which: str, device):
    # sids: [N]
    ms = torch.zeros(sids.size(0), RMAX, CMAX, device=device)
    for i in range(sids.size(0)):
        if which == "A":
            r, c = shape_rc_A(int(sids[i]))
        else:
            r, c = shape_rc_B(int(sids[i]))
        ms[i, :r, :c] = 1.0
    return ms

# Precompute compatible/incompatible maps (A -> list of B)
COMPAT_B = {a: [b for b in range(NSHB) if valid_matmul(a, b)] for a in range(NSHA)}
INCOMP_B = {a: [b for b in range(NSHB) if not valid_matmul(a, b)] for a in range(NSHA)}
assert all(len(COMPAT_B[a]) > 0 for a in range(NSHA))
assert all(len(INCOMP_B[a]) > 0 for a in range(NSHA))

# -----------------------------
# Pattern + distractors
# -----------------------------
def insert_pattern_inplace(A: torch.Tensor, a_sids: torch.Tensor, py_rng: random.Random, t_rng: torch.Generator, amp: float):
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=A.device)
    for i in range(A.size(0)):
        r, c = shape_rc_A(int(a_sids[i]))
        if r < 2 or c < 2:
            continue
        ii = py_rng.randrange(0, r - 1)
        jj = py_rng.randrange(0, c - 1)
        jitter = torch.randn(2, 2, generator=t_rng, device=A.device) * PAT_JITTER
        A[i, ii:ii+2, jj:jj+2] += (base + jitter) * amp
    return A

def add_distractors_inplace(X: torch.Tensor, sids: torch.Tensor, which: str, py_rng: random.Random):
    for i in range(X.size(0)):
        r, c = shape_rc_A(int(sids[i])) if which == "A" else shape_rc_B(int(sids[i]))
        if r == 0 or c == 0:
            continue
        if py_rng.random() < DISTR_P:
            ii = py_rng.randrange(0, r)
            jj = py_rng.randrange(0, c)
            X[i, ii, jj] += (1.6 + py_rng.random() * 1.8) * (1 if py_rng.random() < 0.5 else -1)
    return X

# -----------------------------
# Fast batch construction (no retries)
# -----------------------------
def sample_pairs_for_modes(n: int, mode: str, py_rng: random.Random):
    # returns cpu tensors a_sids (0..NSHA-1), b_sids (0..NSHB-1)
    a = [py_rng.randrange(NSHA) for _ in range(n)]
    b = []
    if mode in ("pos", "neg_pat"):   # shape-valid
        for aa in a:
            b.append(py_rng.choice(COMPAT_B[aa]))
    else:                            # shape-invalid
        for aa in a:
            b.append(py_rng.choice(INCOMP_B[aa]))
    return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)

def make_batch(batch: int, device: str, py_rng: random.Random, t_rng: torch.Generator):
    assert batch % 4 == 0
    n_pos = batch // 2
    n_np  = batch // 4
    n_ns  = batch // 4

    a_pos, b_pos = sample_pairs_for_modes(n_pos, "pos", py_rng)
    a_np,  b_np  = sample_pairs_for_modes(n_np,  "neg_pat", py_rng)
    a_ns,  b_ns  = sample_pairs_for_modes(n_ns,  "neg_shape", py_rng)

    a_sids = torch.cat([a_pos, a_np, a_ns], dim=0).to(device)
    b_sids = torch.cat([b_pos, b_np, b_ns], dim=0).to(device)

    MA = make_masks_from_sids(a_sids, "A", device)
    MB = make_masks_from_sids(b_sids, "B", device)

    A = torch.randn(batch, RMAX, CMAX, generator=t_rng, device=device) * NOISE_STD * MA
    B = torch.randn(batch, RMAX, CMAX, generator=t_rng, device=device) * NOISE_STD * MB

    # pattern in POS + NEG_SHAPE
    A[:n_pos] = insert_pattern_inplace(A[:n_pos], a_sids[:n_pos], py_rng, t_rng, amp=PAT_AMP)
    A[n_pos+n_np:] = insert_pattern_inplace(A[n_pos+n_np:], a_sids[n_pos+n_np:], py_rng, t_rng, amp=PAT_AMP)

    # optional decoy in NEG_PAT only
    for i in range(n_pos, n_pos + n_np):
        if py_rng.random() < DECOY_IN_NEG_P:
            _ = insert_pattern_inplace(A[i:i+1], a_sids[i:i+1], py_rng, t_rng, amp=PAT_AMP * 0.35)

    A = add_distractors_inplace(A, a_sids, "A", py_rng)
    B = add_distractors_inplace(B, b_sids, "B", py_rng)

    Y = torch.zeros(batch, device=device, dtype=torch.long)
    Y[:n_pos] = 1

    if LABEL_FLIP_P > 0:
        flip = torch.rand(batch, generator=t_rng, device=device) < LABEL_FLIP_P
        Y = torch.where(flip, 1 - Y, Y)

    perm = torch.randperm(batch, generator=t_rng, device=device)
    return A[perm], MA[perm], B[perm], MB[perm], Y[perm]

@torch.no_grad()
def make_fixed_eval_set(n_samples: int, device: str, py_rng: random.Random, t_rng: torch.Generator):
    return make_batch(n_samples, device, py_rng, t_rng)

# -----------------------------
# Models
# -----------------------------
class MaskedCNN(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * channels, 2),
        )

    def encode(self, X, M):
        h = torch.stack([X, M], dim=1)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        denom = M.sum(dim=(1, 2)).clamp_min(1.0)
        rep = (h * M[:, None]).sum(dim=(2, 3)) / denom[:, None]
        return self.drop(rep)

    def forward(self, A, MA, B, MB):
        ra = self.encode(A, MA)
        rb = self.encode(B, MB)
        return self.head(torch.cat([ra, rb], dim=-1))

class SmallTransformerAB(nn.Module):
    def __init__(self, d_model=32, n_heads=2, n_layers=1):
        super().__init__()
        self.d = d_model
        self.in_proj = nn.Linear(2, d_model)
        self.sep = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(2 * RMAX * CMAX + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            activation="gelu",
            batch_first=True,
            dropout=DROPOUT,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * d_model, 2),
        )

    def _tokens(self, X, M):
        Bsz = X.size(0)
        xv = X.reshape(Bsz, -1, 1)
        mv = M.reshape(Bsz, -1, 1)
        t = torch.cat([xv, mv], dim=-1)
        h = self.in_proj(t)
        pad = (M.reshape(Bsz, -1) < 0.5)
        return h, pad

    def forward(self, A, MA, B, MB):
        hA, padA = self._tokens(A, MA)
        hB, padB = self._tokens(B, MB)

        sep = self.sep.expand(A.size(0), 1, self.d)
        h = torch.cat([hA, sep, hB], dim=1)
        h = h + self.pos[None, :h.size(1)]

        pad_sep = torch.zeros(A.size(0), 1, device=A.device, dtype=torch.bool)
        key_padding = torch.cat([padA, pad_sep, padB], dim=1)

        out = self.enc(h, src_key_padding_mask=key_padding)

        N = RMAX * CMAX
        outA = out[:, :N]
        outB = out[:, N+1:N+1+N]

        wA = MA.reshape(A.size(0), -1)
        wB = MB.reshape(B.size(0), -1)
        repA = (outA * wA[:, :, None]).sum(dim=1) / wA.sum(dim=1).clamp_min(1.0)[:, None]
        repB = (outB * wB[:, :, None]).sum(dim=1) / wB.sum(dim=1).clamp_min(1.0)[:, None]

        return self.head(torch.cat([repA, repB], dim=-1))

class MatLin(nn.Module):
    def __init__(self):
        super().__init__()
        self.WL = nn.Parameter(torch.randn(RMAX, RMAX) / math.sqrt(RMAX))
        self.WR = nn.Parameter(torch.randn(CMAX, CMAX) / math.sqrt(CMAX))
        self.B  = nn.Parameter(torch.zeros(RMAX, CMAX))

    def forward(self, x):
        x = torch.einsum("ij,bjk->bik", self.WL, x)
        x = torch.einsum("bik,kl->bil", x, self.WR)
        return x + self.B

class MatrixTokenAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.in_proj = nn.Conv2d(4, 1, kernel_size=1)
        self.m1 = MatLin()
        self.m2 = MatLin()
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(2 * D, 2 * D),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * D, 2),
        )

    def encode(self, X, M):
        h = torch.stack([X, M], dim=1)
        h = F.gelu(self.local(h))
        h = self.in_proj(h).squeeze(1)
        h = F.gelu(self.m1(h))
        h = F.gelu(self.m2(h))
        h = h * M
        rep = h.reshape(h.size(0), -1).view(h.size(0), D, -1).mean(dim=-1)
        return self.drop(rep)

    def forward(self, A, MA, B, MB):
        ra = self.encode(A, MA)
        rb = self.encode(B, MB)
        return self.head(torch.cat([ra, rb], dim=-1))

class VanillaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2),
        )

    def _feats(self, X, M):
        denom = M.sum(dim=(1, 2)).clamp_min(1.0)
        mu = (X * M).sum(dim=(1, 2)) / denom
        var = ((X - mu[:, None, None]) ** 2 * M).sum(dim=(1, 2)) / denom
        area = denom / (RMAX * CMAX)
        rows = (M.sum(dim=2) > 0).sum(dim=1).float()
        cols = (M.sum(dim=1) > 0).sum(dim=1).float().clamp_min(1.0)
        aspect = rows / cols
        return torch.stack([mu, var, area, aspect], dim=-1)

    def forward(self, A, MA, B, MB):
        fa = self._feats(A, MA)
        fb = self._feats(B, MB)
        return self.net(torch.cat([fa, fb], dim=-1))

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def eval_metrics(model, A, MA, B, MB, Y, batch=INFER_BS):
    model.eval()
    preds = []
    for i in range(0, A.size(0), batch):
        logits = model(A[i:i+batch], MA[i:i+batch], B[i:i+batch], MB[i:i+batch])
        preds.append(logits.argmax(dim=-1))
    pred = torch.cat(preds, dim=0)

    acc = (pred == Y).float().mean().item()
    y1 = (Y == 1)
    y0 = (Y == 0)
    acc1 = (pred[y1] == 1).float().mean().item() if y1.any() else 0.0
    acc0 = (pred[y0] == 0).float().mean().item() if y0.any() else 0.0
    bal = 0.5 * (acc1 + acc0)
    return acc, bal, acc1, acc0

# -----------------------------
# Train (best checkpoint by eval balanced acc)
# -----------------------------
def train_model(model, name, streams: RNGStreams, Aev, MAev, Bev, MBev, Yev):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    ema = None
    best_bal = -1.0
    best_state = None

    for step in range(1, STEPS + 1):
        A, MA, B, MB, Y = make_batch(BATCH, DEVICE, streams.train_py, streams.train_t)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            logits = model(A, MA, B, MB)
            loss = F.cross_entropy(logits, Y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        l = float(loss.item())
        ema = l if ema is None else (0.95 * ema + 0.05 * l)

        if step == 1 or step % REPORT_EVERY == 0:
            acc, bal, acc1, acc0 = eval_metrics(model, Aev, MAev, Bev, MBev, Yev, batch=INFER_BS)
            if bal > best_bal:
                best_bal = bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{name} [step {step:4d}/{STEPS}] "
                f"loss={l:.4f} ema={ema:.4f} "
                f"acc={acc:.3f} bal={bal:.3f} accT={acc1:.3f} accF={acc0:.3f} best_bal={best_bal:.3f}"
            )

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model

def main():
    seed_all(SEED)
    streams = make_streams(SEED)

    print("\nTASK: y = pattern(A) AND shape_valid(A@B), with A and B provided (values+masks)")
    print(f"A Shapes={SHAPES_A}")
    print(f"B Shapes={SHAPES_B}")
    print(f"Padded={RMAX}x{CMAX}  Device={DEVICE}")
    print(f"Train: steps={STEPS} batch={BATCH}  EvalSet={EVAL_SAMPLES}  FinalEval={FINAL_EVAL_SAMPLES}")
    print("Mixture: 50% POS, 25% NEG_PAT (shape ok), 25% NEG_SHAPE (pattern ok)\n")

    Aev, MAev, Bev, MBev, Yev = make_fixed_eval_set(EVAL_SAMPLES, DEVICE, streams.eval_py, streams.eval_t)
    Afin, MAfin, Bfin, MBfin, Yfin = make_fixed_eval_set(FINAL_EVAL_SAMPLES, DEVICE, streams.eval_py, streams.eval_t)

    models = {
        "CNN": MaskedCNN(channels=12),
        "TRF": SmallTransformerAB(d_model=32, n_heads=2, n_layers=1),
        "MAT": MatrixTokenAB(),
        "MLP": VanillaMLP(),
    }

    trained = {}
    for name, model in models.items():
        print(f"=== Train {name} ===")
        trained[name] = train_model(model, name, streams, Aev, MAev, Bev, MBev, Yev)
        print()

    print("=== FINAL EVAL (fixed, large) ===")
    for name, model in trained.items():
        acc, bal, acc1, acc0 = eval_metrics(model, Afin, MAfin, Bfin, MBfin, Yfin, batch=INFER_BS)
        print(f"{name}: acc={acc:.3f} bal={bal:.3f} accT={acc1:.3f} accF={acc0:.3f}")

if __name__ == "__main__":
    main()
