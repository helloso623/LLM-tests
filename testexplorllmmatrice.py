# exploration_shape_reward_bench_v1.py
# Harder, meaningful comparison with:
# - variable shapes for TWO matrices A and B (both provided, with masks)
# - label = (local pattern in A) AND (A @ B is shape-valid)
# - extra rewards (same for ALL models):
#     (1) shape separation: reps cluster by shape_id and different shapes spread out (supervised contrastive)
#     (2) fit reward: compatible (A,B) pairs should have higher similarity than incompatible pairs (margin)
# - exploration-favoring sampler: oversamples rare shape-pairs (diversity) + keeps eps-uniform exploration
# - noise + distractors + label flip => no perfect accuracy

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config (adjusted for faster training)
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 32  # Reduced from 64
SHAPES = [(4, 8), (8, 4), (2, 16), (16, 2)]  # Smaller shapes
assert all(r * c == D for r, c in SHAPES)

NSH = len(SHAPES)
RMAX = max(r for r, _ in SHAPES)   # 16 -> 8
CMAX = max(c for _, c in SHAPES)   # 32 -> 16

BATCH = 64  # Reduced from 128
STEPS = 100  # Reduced from 300
LR = 1e-3  # Lower learning rate for stability with fewer steps
REPORT_EVERY = 20  # Adjusted for fewer steps
EVAL_BATCHES = 5  # Reduced from 10

# Task hardness knobs (unchanged for meaningful comparison)
PAT_AMP = 0.8
PAT_JITTER = 0.45
NOISE_STD = 0.75
DISTR_P = 0.22
LABEL_FLIP_P = 0.05
DROPOUT = 0.10

# Reward weights (unchanged for fairness)
LAM_SEP = 0.08
LAM_FIT = 0.10

# Contrastive / fit params (unchanged)
SEP_TEMP = 0.2
FIT_MARGIN = 0.25

# Exploration sampler params (unchanged)
EPS_UNIFORM = 0.15
RARE_POWER = 0.75

# -----------------------------
# Shape + validity
# -----------------------------
def shape_rc(sid: int):
    return SHAPES[sid]

def valid_matmul(a_sid: int, b_sid: int) -> bool:
    ra, ca = shape_rc(a_sid)
    rb, cb = shape_rc(b_sid)
    return ca == rb

def make_mask(r, c, device):
    m = torch.zeros(RMAX, CMAX, device=device)
    m[:r, :c] = 1.0
    return m

# -----------------------------
# Exploration sampler (pair diversity)
# -----------------------------
class PairSampler:
    """
    Chooses (a_sid,b_sid) with:
    - epsilon uniform exploration
    - otherwise: probability proportional to (1 / (count+1))^RARE_POWER
    This favors exploring rare shape-pairs (diversity) without using model feedback.
    """
    def __init__(self, n_shapes: int):
        self.n = n_shapes
        self.count = torch.zeros(n_shapes, n_shapes, dtype=torch.long)

    def sample_pair(self):
        if random.random() < EPS_UNIFORM:
            a = random.randrange(self.n)
            b = random.randrange(self.n)
            self.count[a, b] += 1
            return a, b

        # rare-weighted sampling
        w = (1.0 / (self.count.float() + 1.0)).pow(RARE_POWER)
        w = w / w.sum()
        idx = torch.multinomial(w.view(-1), num_samples=1).item()
        a = idx // self.n
        b = idx % self.n
        self.count[a, b] += 1
        return int(a), int(b)

# -----------------------------
# Pattern + distractors
# -----------------------------
def insert_subtle_pattern(x, r, c):
    if r < 2 or c < 2:
        return x, False
    i = random.randrange(0, r - 1)
    j = random.randrange(0, c - 1)
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=x.device)
    jitter = torch.randn(2, 2, device=x.device) * PAT_JITTER
    patch = (base + jitter) * PAT_AMP
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

def has_pattern_heuristic(x, m):
    # dataset bookkeeping only (not seen by model)
    xx = x * m
    base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=x.device)
    for i in range(RMAX - 1):
        for j in range(CMAX - 1):
            if m[i, j] == 0 or m[i, j+1] == 0 or m[i+1, j] == 0 or m[i+1, j+1] == 0:
                continue
            patch = xx[i:i+2, j:j+2]
            score = (patch * base).sum().abs().item()
            if score > (PAT_AMP * 2.0):
                return True
    return False

# -----------------------------
# Data (balanced final label)
# -----------------------------
def sample_one(desired_label: int, sampler: PairSampler, device):
    """
    Provide BOTH matrices A and B, each with its own mask.
    Label = (pattern present in A) AND (A@B shape-valid).
    """
    a_sid, b_sid = sampler.sample_pair()
    ra, ca = shape_rc(a_sid)
    rb, cb = shape_rc(b_sid)

    ma = make_mask(ra, ca, device)
    mb = make_mask(rb, cb, device)

    # matrices (masked noise)
    A = (torch.randn(RMAX, CMAX, device=device) * NOISE_STD) * ma
    B = (torch.randn(RMAX, CMAX, device=device) * NOISE_STD) * mb

    ok_shape = valid_matmul(a_sid, b_sid)

    # control whether pattern is present in A
    if desired_label == 1:
        if not ok_shape:
            # try once to force compatible b
            candidates = [sid for sid, (rr, cc) in enumerate(SHAPES) if rr == ca]
            if candidates:
                b_sid = random.choice(candidates)
                rb, cb = shape_rc(b_sid)
                mb = make_mask(rb, cb, device)
                B = (torch.randn(RMAX, CMAX, device=device) * NOISE_STD) * mb
                ok_shape = True

        A, ok_pat = insert_subtle_pattern(A, ra, ca)
        if not ok_pat:
            desired_label = 0  # fallback if too small

    if desired_label == 0:
        # break at least one condition
        break_pat = (random.random() < 0.5)
        if break_pat:
            ok_pat = False
        else:
            ok_pat = True
            if ok_shape:
                bad = [sid for sid in range(NSH) if not valid_matmul(a_sid, sid)]
                if bad:
                    b_sid = random.choice(bad)
                    rb, cb = shape_rc(b_sid)
                    mb = make_mask(rb, cb, device)
                    B = (torch.randn(RMAX, CMAX, device=device) * NOISE_STD) * mb
                    ok_shape = False
            if ok_shape:
                ok_pat = False

        if ok_pat:
            A, inserted = insert_subtle_pattern(A, ra, ca)
            ok_pat = inserted

    # distractors
    A = add_distractors(A, ra, ca)
    B = add_distractors(B, rb, cb)

    # final label computed by heuristic presence (keeps generation honest under noise)
    pat_present = has_pattern_heuristic(A, ma)
    y = 1 if (pat_present and ok_shape) else 0

    # small label noise to avoid ceiling
    if random.random() < LABEL_FLIP_P:
        y = 1 - y

    return A, ma, B, mb, y, a_sid, b_sid, int(ok_shape)

def sample_batch_balanced(batch=BATCH, device=DEVICE):
    sampler = sample_batch_balanced.sampler  # type: ignore[attr-defined]
    xsA, msA, xsB, msB, ys, a_sids, b_sids, ok_s = [], [], [], [], [], [], [], []
    half = batch // 2

    while len(ys) < batch:
        want = 1 if len(ys) < half else 0
        A, ma, B, mb, y, a_sid, b_sid, ok_shape = sample_one(want, sampler, device)
        if y == want:
            xsA.append(A); msA.append(ma)
            xsB.append(B); msB.append(mb)
            ys.append(y)
            a_sids.append(a_sid); b_sids.append(b_sid); ok_s.append(ok_shape)

    idx = list(range(batch))
    random.shuffle(idx)

    A = torch.stack([xsA[i] for i in idx], dim=0)
    MA = torch.stack([msA[i] for i in idx], dim=0)
    B = torch.stack([xsB[i] for i in idx], dim=0)
    MB = torch.stack([msB[i] for i in idx], dim=0)
    Y = torch.tensor([ys[i] for i in idx], device=device).long()
    AS = torch.tensor([a_sids[i] for i in idx], device=device).long()
    BS = torch.tensor([b_sids[i] for i in idx], device=device).long()
    OK = torch.tensor([ok_s[i] for i in idx], device=device).long()
    return A, MA, B, MB, Y, AS, BS, OK

# attach sampler (model-agnostic exploration)
sample_batch_balanced.sampler = PairSampler(NSH)  # type: ignore[attr-defined]

# -----------------------------
# Rewards (same for all models)
# -----------------------------
def supervised_contrastive_loss(z, labels, temperature=0.2):
    # z: [N,D], labels: [N]
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=-1)
    sim = (z @ z.t()) / temperature
    N = z.size(0)
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    labels = labels.view(-1, 1)
    pos = (labels == labels.t()) & (~eye)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    num = (exp_sim * pos.float()).sum(dim=1, keepdim=True)

    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return z.new_tensor(0.0)

    loss = -torch.log((num / denom).clamp_min(1e-12))
    return loss[has_pos].mean()

def fit_margin_loss(repA, repB, ok_shape, margin=0.25):
    """
    Encourage: sim(A,B) higher when ok_shape=1 than when ok_shape=0.
    Uses a simple margin ranking loss by pairing random positives and negatives within batch.
    """
    repA = F.normalize(repA, dim=-1)
    repB = F.normalize(repB, dim=-1)
    sim = (repA * repB).sum(dim=-1)  # [B]

    pos = (ok_shape == 1).nonzero(as_tuple=False).view(-1)
    neg = (ok_shape == 0).nonzero(as_tuple=False).view(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return sim.new_tensor(0.0)

    k = min(pos.numel(), neg.numel(), 64)
    pos = pos[torch.randperm(pos.numel(), device=sim.device)[:k]]
    neg = neg[torch.randperm(neg.numel(), device=sim.device)[:k]]
    # want sim_pos >= sim_neg + margin
    return F.relu(margin - (sim[pos] - sim[neg])).mean()

# -----------------------------
# Models: all return (logits, repA, repB)
# -----------------------------
class MaskedCNN(nn.Module):
    def __init__(self, channels=16):  # Reduced from 32
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(channels, 2))

    def encode_one(self, x, m):
        h = torch.stack([x, m], dim=1)  # [B,2,R,C]
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        rep = (h * m[:, None, :, :]).sum(dim=(2, 3)) / denom[:, None]  # [B,Ch]
        return self.dropout(rep)

    def forward(self, A, MA, B, MB):
        repA = self.encode_one(A, MA)
        repB = self.encode_one(B, MB)
        # simple fusion for main task
        fused = repA + repB
        return self.head(fused), repA, repB

class VanillaTransformer(nn.Module):
    """
    Flatten cells; encode A tokens then B tokens with a learned separator.
    """
    def __init__(self, d_model=32, n_heads=2, n_layers=1):  # Reduced dimensions and layers
        super().__init__()
        self.d = d_model
        self.in_proj = nn.Linear(2, d_model)
        self.sep = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(2 * RMAX * CMAX + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,  # Reduced feedforward size
            activation="gelu", batch_first=True, dropout=DROPOUT
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(d_model, 2))

    def encode_one(self, x, m):
        Bsz = x.size(0)
        xv = x.reshape(Bsz, -1, 1)
        mv = m.reshape(Bsz, -1, 1)
        t = torch.cat([xv, mv], dim=-1)          # [B,N,2]
        h = self.in_proj(t)                      # [B,N,D]
        key_padding = (m.reshape(Bsz, -1) < 0.5) # [B,N]
        return h, key_padding

    def forward(self, A, MA, B, MB):
        hA, padA = self.encode_one(A, MA)
        hB, padB = self.encode_one(B, MB)

        sep = self.sep.expand(A.size(0), 1, self.d)
        h = torch.cat([hA, sep, hB], dim=1)  # [B, 2N+1, D]
        h = h + self.pos[None, :h.size(1), :]

        pad_sep = torch.zeros(A.size(0), 1, device=A.device, dtype=torch.bool)
        key_padding = torch.cat([padA, pad_sep, padB], dim=1)

        out = self.enc(h, src_key_padding_mask=key_padding)

        # pooled repA and repB as masked means over their segments
        N = RMAX * CMAX
        outA = out[:, :N, :]
        outB = out[:, N+1:N+1+N, :]

        wA = MA.reshape(A.size(0), -1)
        wB = MB.reshape(B.size(0), -1)
        repA = (outA * wA[:, :, None]).sum(dim=1) / wA.sum(dim=1).clamp_min(1.0)[:, None]
        repB = (outB * wB[:, :, None]).sum(dim=1) / wB.sum(dim=1).clamp_min(1.0)[:, None]

        fused = repA + repB
        return self.head(fused), repA, repB

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

class MatrixTokenModel(nn.Module):
    """
    A compact "matrix-token style" baseline:
    - keeps 2D grid
    - uses MatLin mixing (row/col structured)
    - extracts reps from masked mean of final 2D state
    """
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Conv2d(2, 1, kernel_size=1)  # value+mask -> 1 channel grid
        self.m1 = MatLin()
        self.m2 = MatLin()
        self.m3 = MatLin()
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(128, 2))

    def encode_one(self, x, m):
        h = torch.stack([x, m], dim=1)     # [B,2,R,C]
        h = self.in_proj(h).squeeze(1)     # [B,R,C]
        h = F.gelu(self.m1(h))
        h = F.gelu(self.m2(h))
        h = F.gelu(self.m3(h))
        h = h * m                          # keep invalid region inert
        # compress (RMAX*CMAX -> D) deterministically (no learned shape embedding)
        flat = h.reshape(h.size(0), -1).view(h.size(0), D, -1).mean(dim=-1)  # [B,D]
        return self.drop(flat)

    def forward(self, A, MA, B, MB):
        repA = self.encode_one(A, MA)
        repB = self.encode_one(B, MB)
        fused = repA + repB
        return self.head(fused), repA, repB

# -----------------------------
# Train / eval
# -----------------------------
@torch.no_grad()
def accuracy(model, n_batches=5, batch=BATCH, device=DEVICE):  # Reduced n_batches
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_batches):
        A, MA, B, MB, Y, AS, BS, OK = sample_batch_balanced(batch=batch, device=device)
        logits, _, _ = model(A, MA, B, MB)
        pred = logits.argmax(dim=-1)
        correct += (pred == Y).sum().item()
        total += Y.numel()
    return correct / max(1, total)

def train_model(model, name):
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ema = None

    for step in range(1, STEPS + 1):
        A, MA, B, MB, Y, AS, BS, OK = sample_batch_balanced(batch=BATCH, device=DEVICE)
        logits, repA, repB = model(A, MA, B, MB)

        # main task loss
        main = F.cross_entropy(logits, Y)

        # shape separation reward (applied to both A and B reps)
        sepA = supervised_contrastive_loss(repA, AS, temperature=SEP_TEMP)
        sepB = supervised_contrastive_loss(repB, BS, temperature=SEP_TEMP)
        sep = 0.5 * (sepA + sepB)

        # fit reward (compatible pairs closer)
        fit = fit_margin_loss(repA, repB, OK, margin=FIT_MARGIN)

        loss = main + LAM_SEP * sep + LAM_FIT * fit

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.item()
        ema = l if ema is None else (0.95 * ema + 0.05 * l)

        if step == 1 or step % REPORT_EVERY == 0:
            acc = accuracy(model, n_batches=EVAL_BATCHES, batch=BATCH)
            print(
                f"{name} [step {step:4d}/{STEPS}] "
                f"loss={l:.4f} ema={ema:.4f} acc={acc:.3f} "
                f"(main={main.item():.4f}, sep={sep.item():.4f}, fit={fit.item():.4f})"
            )

def main():
    print("\nTASK: (pattern in A) AND (A@B shape-valid), with exploration + shape-difference rewards")
    print(f"Shapes={SHAPES}  Padded={RMAX}x{CMAX}  Device={DEVICE}")
    print(f"Hardness: amp={PAT_AMP} noise={NOISE_STD} distr_p={DISTR_P} flip_p={LABEL_FLIP_P} dropout={DROPOUT}")
    print(f"Rewards: LAM_SEP={LAM_SEP} (shape spread), LAM_FIT={LAM_FIT} (fit), sampler eps={EPS_UNIFORM}\n")

    cnn = MaskedCNN(channels=32)
    trf = VanillaTransformer(d_model=64, n_heads=4, n_layers=2)
    mat = MatrixTokenModel()

    print("=== Train CNN ===")
    train_model(cnn, "CNN")

    print("\n=== Train Vanilla Transformer ===")
    train_model(trf, "TRF")

    print("\n=== Train Matrix-token style model ===")
    train_model(mat, "MAT")

    acc_c = accuracy(cnn, n_batches=40)
    acc_t = accuracy(trf, n_batches=40)
    acc_m = accuracy(mat, n_batches=40)

    print("\nFINAL ACCURACY:")
    print(f"CNN : {acc_c:.3f}")
    print(f"TRF : {acc_t:.3f}")
    print(f"MAT : {acc_m:.3f}")

if __name__ == "__main__":
    main()
