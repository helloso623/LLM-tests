# cnn_vs_models_harder_mixed_validity_low_compute_v1.py
# Same spirit as your bench (CNN should beat attention on locality),
# but lower compute cost and less chance of ceiling.
#
# Changes for speed:
# - Smaller batch/steps
# - Smaller Transformer (d_model, heads, layers)
# - Smaller CNN channels
# - Fewer eval batches
#
# Changes to avoid perfect accuracy:
# - Harder pattern (lower amp, more noise/jitter/distractors)
# - Dropout in all models
# - Slightly higher label flip
# - Add a "decoy" pattern sometimes in negatives (confuses simple heuristics)

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

RMAX = max(r for r, _ in SHAPES)  # 16
CMAX = max(c for _, c in SHAPES)  # 32
NSH = len(SHAPES)

# Lower compute
BATCH = 64
STEPS = 220
LR = 2e-3
REPORT_EVERY = 20
EVAL_BATCHES = 4  # eval speed

# Harder / non-ceiling
PAT_AMP = 0.65
PAT_JITTER = 0.55
NOISE_STD = 0.85
DISTR_P = 0.28
LABEL_FLIP_P = 0.07
DECOY_NEG_P = 0.25  # add a weak decoy pattern in some negatives
DROPOUT = 0.10

# -----------------------------
# Shapes / validity
# -----------------------------
def sample_shape():
    sid = random.randrange(NSH)
    r, c = SHAPES[sid]
    return sid, r, c

def valid_matmul(a_sid, b_sid):
    ra, ca = SHAPES[a_sid]
    rb, cb = SHAPES[b_sid]
    return ca == rb

def make_mask(r, c, device):
    m = torch.zeros(RMAX, CMAX, device=device)
    m[:r, :c] = 1.0
    return m

# -----------------------------
# Local pattern
# -----------------------------
def insert_subtle_pattern(x, r, c, amp=PAT_AMP):
    if r < 2 or c < 2:
        return x, False
    i = random.randrange(0, r - 1)
    j = random.randrange(0, c - 1)
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=x.device)
    jitter = torch.randn(2, 2, device=x.device) * PAT_JITTER
    patch = (base + jitter) * amp
    x[i:i+2, j:j+2] += patch
    return x, True

def add_distractors(x, r, c):
    if r == 0 or c == 0:
        return x
    if random.random() < DISTR_P:
        i = random.randrange(0, r)
        j = random.randrange(0, c)
        x[i, j] += (2.0 + random.random() * 2.5) * (1 if random.random() < 0.5 else -1)
    return x

def has_pattern(x, m):
    # bookkeeping only
    xx = x * m
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=x.device)
    for i in range(RMAX - 1):
        for j in range(CMAX - 1):
            if m[i, j] == 0 or m[i, j+1] == 0 or m[i+1, j] == 0 or m[i+1, j+1] == 0:
                continue
            patch = xx[i:i+2, j:j+2]
            score = (patch * base).sum().abs().item()
            if score > (PAT_AMP * 1.55):
                return True
    return False

# -----------------------------
# Data sampling (balanced final label)
# -----------------------------
def sample_one(desired_label: int, device):
    a_sid, r, c = sample_shape()
    b_sid, rb, cb = sample_shape()
    ok_shape = valid_matmul(a_sid, b_sid)

    x = torch.randn(RMAX, CMAX, device=device) * NOISE_STD
    m = make_mask(r, c, device)
    x = x * m

    if desired_label == 1:
        # force compatibility if needed
        if not ok_shape:
            candidates = [sid for sid, (rr, cc) in enumerate(SHAPES) if rr == c]
            if candidates:
                b_sid = random.choice(candidates)
                ok_shape = True
        x, ok_pat = insert_subtle_pattern(x, r, c, amp=PAT_AMP)
        if not ok_pat:
            desired_label = 0

    if desired_label == 0:
        # break one condition
        break_pat = (random.random() < 0.5)
        if break_pat:
            ok_pat = False
        else:
            ok_pat = True
            if ok_shape:
                bad = [sid for sid in range(NSH) if not valid_matmul(a_sid, sid)]
                if bad:
                    b_sid = random.choice(bad)
                    ok_shape = False
            if ok_shape:
                ok_pat = False

        if ok_pat:
            x, inserted = insert_subtle_pattern(x, r, c, amp=PAT_AMP)
            ok_pat = inserted

        # decoy: some negatives contain a weaker pattern to confuse purely-local solutions
        if random.random() < DECOY_NEG_P:
            x, _ = insert_subtle_pattern(x, r, c, amp=PAT_AMP * 0.35)

    x = add_distractors(x, r, c)

    y = 1 if (ok_shape and has_pattern(x, m)) else 0
    if random.random() < LABEL_FLIP_P:
        y = 1 - y
    return x, m, y

def sample_batch_balanced(batch=BATCH, device=DEVICE):
    half = batch // 2
    xs, ms, ys = [], [], []
    while len(ys) < batch:
        want = 1 if len(ys) < half else 0
        x, m, y = sample_one(want, device)
        if y == want:
            xs.append(x); ms.append(m); ys.append(y)

    idx = list(range(batch))
    random.shuffle(idx)
    X = torch.stack([xs[i] for i in idx], dim=0)
    M = torch.stack([ms[i] for i in idx], dim=0)
    Y = torch.tensor([ys[i] for i in idx], device=device).long()
    return X, M, Y

# -----------------------------
# Models
# -----------------------------
class MaskedCNN(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(channels, 2))

    def forward(self, x, m):
        h = torch.stack([x, m], dim=1)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        pooled = (h * m[:, None, :, :]).sum(dim=(2, 3)) / denom[:, None]
        pooled = self.drop(pooled)
        return self.head(pooled)

class VanillaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 48),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(48, 2),
        )

    def forward(self, x, m):
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        mu = (x * m).sum(dim=(1, 2)) / denom
        var = ((x - mu[:, None, None]) ** 2 * m).sum(dim=(1, 2)) / denom
        # add two simple mask-derived features (lets it partially reason about shape without being told)
        area = denom / (RMAX * CMAX)
        aspect = (m.sum(dim=2) > 0).sum(dim=1).float() / ((m.sum(dim=1) > 0).sum(dim=1).float().clamp_min(1.0))
        feats = torch.stack([mu, var, area, aspect], dim=-1)
        return self.net(feats)

class SmallTransformer(nn.Module):
    def __init__(self, d_model=40, n_heads=4, n_layers=1):
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        self.pos = nn.Parameter(torch.randn(RMAX * CMAX, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=3 * d_model,
            activation="gelu",
            batch_first=True,
            dropout=DROPOUT,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(d_model, 2))

    def forward(self, x, m):
        B = x.size(0)
        xv = x.reshape(B, -1, 1)
        mv = m.reshape(B, -1, 1)
        t = torch.cat([xv, mv], dim=-1)
        h = self.in_proj(t) + self.pos[None, :, :]
        key_padding_mask = (m.reshape(B, -1) < 0.5)
        h = self.enc(h, src_key_padding_mask=key_padding_mask)
        w = m.reshape(B, -1)
        denom = w.sum(dim=1).clamp_min(1.0)
        pooled = (h * w[:, :, None]).sum(dim=1) / denom[:, None]
        return self.head(pooled)

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

class MatrixTokenClassifier(nn.Module):
    # closer to CNN-like: add a local conv before global MatLin mixing
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.in_proj = nn.Conv2d(4, 1, kernel_size=1)
        self.m1 = MatLin()
        self.m2 = MatLin()
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(D, 96), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(96, 2))

    def forward(self, x, m):
        h = torch.stack([x, m], dim=1)     # [B,2,R,C]
        h = F.gelu(self.local(h))          # [B,4,R,C] local bias like CNN
        h = self.in_proj(h).squeeze(1)     # [B,R,C]
        h = F.gelu(self.m1(h))
        h = F.gelu(self.m2(h))
        h = h * m
        flat = h.reshape(h.size(0), -1).view(h.size(0), D, -1).mean(dim=-1)  # [B,D]
        flat = self.drop(flat)
        return self.head(flat)

# -----------------------------
# Train / eval
# -----------------------------
@torch.no_grad()
def accuracy(model, n_batches=10, batch=BATCH, device=DEVICE):
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_batches):
        X, M, Y = sample_batch_balanced(batch=batch, device=device)
        logits = model(X, M)
        pred = logits.argmax(dim=-1)
        correct += (pred == Y).sum().item()
        total += Y.numel()
    return correct / max(1, total)

def train_model(model, name, steps=STEPS):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ema = None

    for step in range(1, steps + 1):
        X, M, Y = sample_batch_balanced(batch=BATCH, device=DEVICE)
        logits = model(X, M)
        loss = F.cross_entropy(logits, Y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.item()
        ema = l if ema is None else (0.95 * ema + 0.05 * l)

        if step == 1 or step % REPORT_EVERY == 0:
            acc = accuracy(model, n_batches=EVAL_BATCHES, batch=BATCH)
            print(f"{name} [step {step:4d}/{steps}] loss={l:.4f} ema={ema:.4f} acc={acc:.3f}")

def main():
    print("\nTASK: subtle local pattern AND implicit shape-validity (low compute, non-ceiling)")
    print(f"Shapes={SHAPES}  Padded={RMAX}x{CMAX}  Balanced labels  Device={DEVICE}")
    print(f"Hardness: amp={PAT_AMP} noise={NOISE_STD} jitter={PAT_JITTER} distr_p={DISTR_P} flip_p={LABEL_FLIP_P} decoy_p={DECOY_NEG_P}\n")

    cnn = MaskedCNN(channels=16)
    trf = SmallTransformer(d_model=40, n_heads=4, n_layers=1)
    mat = MatrixTokenClassifier()
    mlp = VanillaMLP()

    print("=== Train CNN ===")
    train_model(cnn, "CNN")

    print("\n=== Train Vanilla Transformer ===")
    train_model(trf, "TRF")

    print("\n=== Train Matrix-token model ===")
    train_model(mat, "MAT")

    print("\n=== Train Vanilla MLP baseline ===")
    train_model(mlp, "MLP")

    acc_c = accuracy(cnn, n_batches=30)
    acc_t = accuracy(trf, n_batches=30)
    acc_m = accuracy(mat, n_batches=30)
    acc_p = accuracy(mlp, n_batches=30)

    print("\nFINAL ACCURACY:")
    print(f"CNN : {acc_c:.3f}")
    print(f"TRF : {acc_t:.3f}")
    print(f"MAT : {acc_m:.3f}")
    print(f"MLP : {acc_p:.3f}")

if __name__ == "__main__":
    main()
