# cnn_vs_models_harder_mixed_validity_low_compute_v3_FAST_RECODE.py
#
# Recode with your settings + FIX: torch.Generator has no .cuda().
# Create generators directly on the target device and pass them into torch ops.

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config (your run settings)
# -----------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shapes from your run
SHAPES = [(4, 8), (8, 4), (2, 16), (16, 2)]
NSH = len(SHAPES)

# Padded sizes from your run
RMAX = 16
CMAX = 16

# Each shape area must be constant for some model heads (we use D below)
AREA = SHAPES[0][0] * SHAPES[0][1]
assert all(r * c == AREA for r, c in SHAPES), "All SHAPES must have same area"
D = 32  # must divide RMAX*CMAX (=256). 32 works: 256/32 = 8
assert (RMAX * CMAX) % D == 0

# Low compute (your run)
BATCH = 128
STEPS = 300
LR = 1e-3
REPORT_EVERY = 20

# Fixed eval sets (your run)
EVAL_SAMPLES = 4096
FINAL_EVAL_SAMPLES = 8192
INFER_BS = 512

# Difficulty (your run)
PAT_AMP = 0.80
PAT_JITTER = 0.45
NOISE_STD = 0.75
DISTR_P = 0.22
LABEL_FLIP_P = 0.05

# Optional knobs
DECOY_IN_NEG_P = 0.20  # only used in NEG_PAT
DROPOUT = 0.10
AMP_ENABLED = True

# -----------------------------
# RNG streams (train/eval isolated)
# -----------------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_generator(seed: int, device: str) -> torch.Generator:
    # Correct way: generator is created on the device. No .cuda() exists.
    return torch.Generator(device=device).manual_seed(seed)

@dataclass
class RNGStreams:
    train_py: random.Random
    eval_py: random.Random
    train_t: torch.Generator
    eval_t: torch.Generator

def make_streams(seed: int) -> RNGStreams:
    train_py = random.Random(seed + 101)
    eval_py = random.Random(seed + 202)
    train_t = make_generator(seed + 303, DEVICE)
    eval_t = make_generator(seed + 404, DEVICE)
    return RNGStreams(train_py=train_py, eval_py=eval_py, train_t=train_t, eval_t=eval_t)

# -----------------------------
# Shapes / masks / validity
# -----------------------------
def shape_rc(sid: int):
    return SHAPES[sid]

def valid_matmul(a_sid: int, b_sid: int) -> bool:
    # A is (ra, ca), B is (rb, cb) => ca must equal rb
    return SHAPES[a_sid][1] == SHAPES[b_sid][0]

def make_masks(sids: torch.Tensor, device: str):
    # sids: [N] on device
    ms = torch.zeros(sids.size(0), RMAX, CMAX, device=device)
    for i in range(sids.size(0)):
        r, c = shape_rc(int(sids[i]))
        ms[i, :r, :c] = 1.0
    return ms

# Precompute compatible/incompatible maps for speed
COMPAT_B = {a: [b for b in range(NSH) if valid_matmul(a, b)] for a in range(NSH)}
INCOMP_B = {a: [b for b in range(NSH) if not valid_matmul(a, b)] for a in range(NSH)}
assert all(len(COMPAT_B[a]) > 0 for a in range(NSH))
assert all(len(INCOMP_B[a]) > 0 for a in range(NSH))

# -----------------------------
# Pattern + distractors
# -----------------------------
def insert_pattern_inplace(A: torch.Tensor, a_sids: torch.Tensor, py_rng: random.Random, t_rng: torch.Generator, amp: float):
    """
    A: [N,RMAX,CMAX], insert 2x2 pattern within each sample's valid region.
    """
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=A.device)
    for i in range(A.size(0)):
        r, c = shape_rc(int(a_sids[i]))
        if r < 2 or c < 2:
            continue
        ii = py_rng.randrange(0, r - 1)
        jj = py_rng.randrange(0, c - 1)
        jitter = torch.randn(2, 2, generator=t_rng, device=A.device) * PAT_JITTER
        A[i, ii:ii+2, jj:jj+2] += (base + jitter) * amp
    return A

def add_distractors_inplace(X: torch.Tensor, sids: torch.Tensor, py_rng: random.Random):
    for i in range(X.size(0)):
        r, c = shape_rc(int(sids[i]))
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
    """
    mode:
      - "pos" / "neg_pat": need shape-valid pairs
      - "neg_shape": need shape-invalid pairs
    """
    a = [py_rng.randrange(NSH) for _ in range(n)]
    b = []
    if mode in ("pos", "neg_pat"):
        for aa in a:
            b.append(py_rng.choice(COMPAT_B[aa]))
    else:
        for aa in a:
            b.append(py_rng.choice(INCOMP_B[aa]))
    return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)

def make_batch(batch: int, device: str, py_rng: random.Random, t_rng: torch.Generator):
    """
    Mixture:
      50% pos:       pattern=1, shape_valid=1 => y=1
      25% neg_pat:   pattern=0, shape_valid=1 => y=0 (optional weak decoy)
      25% neg_shape: pattern=1, shape_valid=0 => y=0
    """
    assert batch % 4 == 0
    n_pos = batch // 2
    n_np = batch // 4
    n_ns = batch // 4

    a_pos, b_pos = sample_pairs_for_modes(n_pos, "pos", py_rng)
    a_np, b_np = sample_pairs_for_modes(n_np, "neg_pat", py_rng)
    a_ns, b_ns = sample_pairs_for_modes(n_ns, "neg_shape", py_rng)

    a_sids = torch.cat([a_pos, a_np, a_ns], dim=0).to(device)
    b_sids = torch.cat([b_pos, b_np, b_ns], dim=0).to(device)

    MA = make_masks(a_sids, device)
    MB = make_masks(b_sids, device)

    # base noise, masked to valid region
    A = torch.randn(batch, RMAX, CMAX, generator=t_rng, device=device) * NOISE_STD * MA
    B = torch.randn(batch, RMAX, CMAX, generator=t_rng, device=device) * NOISE_STD * MB

    # pattern in POS + NEG_SH
    A[:n_pos] = insert_pattern_inplace(A[:n_pos], a_sids[:n_pos], py_rng, t_rng, amp=PAT_AMP)
    A[n_pos + n_np:] = insert_pattern_inplace(A[n_pos + n_np:], a_sids[n_pos + n_np:], py_rng, t_rng, amp=PAT_AMP)

    # optional decoy in NEG_PAT only
    if DECOY_IN_NEG_P > 0:
        for i in range(n_pos, n_pos + n_np):
            if py_rng.random() < DECOY_IN_NEG_P:
                _ = insert_pattern_inplace(A[i:i+1], a_sids[i:i+1], py_rng, t_rng, amp=PAT_AMP * 0.35)

    # distractors
    A = add_distractors_inplace(A, a_sids, py_rng)
    B = add_distractors_inplace(B, b_sids, py_rng)

    # labels by construction
    Y = torch.zeros(batch, device=device, dtype=torch.long)
    Y[:n_pos] = 1

    # label noise
    if LABEL_FLIP_P > 0:
        flip = torch.rand(batch, generator=t_rng, device=device) < LABEL_FLIP_P
        Y = torch.where(flip, 1 - Y, Y)

    # shuffle within batch
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
        outB = out[:, N + 1:N + 1 + N]

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
        self.B = nn.Parameter(torch.zeros(RMAX, CMAX))

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
# Train (best checkpoint by eval balanced accuracy)
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

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("\nTASK: y = pattern(A) AND shape_valid(A@B), with A and B provided (values+masks)")
    print(f"Shapes={SHAPES}  Padded={RMAX}x{CMAX}  Device={DEVICE}")
    print(f"Train: steps={STEPS} batch={BATCH}  EvalSet={EVAL_SAMPLES}  FinalEval={FINAL_EVAL_SAMPLES}")
    print(f"Hardness: amp={PAT_AMP} noise={NOISE_STD} jitter={PAT_JITTER} distr_p={DISTR_P} flip_p={LABEL_FLIP_P} decoy_neg_p={DECOY_IN_NEG_P}")
    print("Mixture: 50% POS, 25% NEG_PAT (shape ok), 25% NEG_SHAPE (pattern ok)\n")

    print("Generating fixed eval sets (fast, no retries)...")
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
