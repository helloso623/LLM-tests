"""
Goal-Biased Attention Experiment
================================
Tests whether injecting goal relevance into attention scores
lets a transformer hit the same accuracy with fewer heads.

This version adds:
- stronger lambda sweep
- optional focus on current-node outgoing edges first
- plots for 4-head goal-biased performance
- per-goal 4-head lambda curves
- ASCII-only output for Windows
"""

import json
import math
import os
import random
import time
import heapq
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

ATTR_NAMES = ["time", "cost", "risk", "effort"]

GOAL_WEIGHTS = {
    "FASTEST":             {"time": 1.0, "cost": 0.0, "risk": 0.0, "effort": 0.0},
    "CHEAPEST":            {"time": 0.0, "cost": 1.0, "risk": 0.0, "effort": 0.0},
    "SAFEST":              {"time": 0.0, "cost": 0.0, "risk": 1.0, "effort": 0.0},
    "LOWEST_EFFORT":       {"time": 0.0, "cost": 0.0, "risk": 0.0, "effort": 1.0},
    "BALANCED_TIME_COST":  {"time": 0.5, "cost": 0.5, "risk": 0.0, "effort": 0.0},
    "BALANCED_TIME_RISK":  {"time": 0.5, "cost": 0.0, "risk": 0.5, "effort": 0.0},
}

GOAL_LIST = list(GOAL_WEIGHTS.keys())
GOAL_TO_IDX = {g: i for i, g in enumerate(GOAL_LIST)}
IDX_TO_GOAL = {i: g for g, i in GOAL_TO_IDX.items()}


@dataclass
class WorldConfig:
    num_nodes: int = 12
    edge_density: float = 0.20
    attr_range: Tuple[int, int] = (1, 10)
    seed: int = 42


@dataclass
class ExperimentConfig:
    # Data
    num_train_graphs: int = 200
    num_val_graphs: int = 50
    samples_per_graph: int = 40

    # Model
    d_model: int = 64
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_seq_len: int = 256

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 25
    patience: int = 5

    # Sweep
    head_counts: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    model_types: List[str] = field(default_factory=lambda: ["baseline", "goal_token", "goal_biased"])

    # Target
    target_accuracy: float = 0.85

    # Goals
    goals: List[str] = field(default_factory=lambda: GOAL_LIST.copy())

    # Tokenization / experiment behavior
    prioritize_current_edges: bool = True
    current_edges_first_only: bool = False   # set True if you want a much easier task
    plot_focus_heads: int = 4

    # Runtime
    device: str = "cpu"
    results_dir: str = "results_goal_attention"
    seed: int = 123


# --------------------------------------------------
# UTILS
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# WORLD
# --------------------------------------------------

def generate_graph(num_nodes: int, edge_density: float, attr_range: Tuple[int, int], rng: random.Random) -> dict:
    lo, hi = attr_range
    edges = {}

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < edge_density:
                edges[(i, j)] = {a: rng.randint(lo, hi) for a in ATTR_NAMES}

    order = list(range(num_nodes))
    rng.shuffle(order)
    for k in range(len(order) - 1):
        u, v = order[k], order[k + 1]
        if (u, v) not in edges:
            edges[(u, v)] = {a: rng.randint(lo, hi) for a in ATTR_NAMES}
        if (v, u) not in edges:
            edges[(v, u)] = {a: rng.randint(lo, hi) for a in ATTR_NAMES}

    return {"num_nodes": num_nodes, "edges": edges}


def edge_cost(attrs: Dict[str, int], goal: str) -> float:
    weights = GOAL_WEIGHTS[goal]
    return sum(weights[a] * attrs[a] for a in ATTR_NAMES)


def shortest_distances_to_dest(graph: dict, dest: int, goal: str) -> List[float]:
    n = graph["num_nodes"]
    rev_adj = defaultdict(list)

    for (u, v), attrs in graph["edges"].items():
        rev_adj[v].append((u, edge_cost(attrs, goal)))

    dist = [float("inf")] * n
    dist[dest] = 0.0
    heap = [(0.0, dest)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for prev_node, w in rev_adj[node]:
            nd = d + w
            if nd < dist[prev_node]:
                dist[prev_node] = nd
                heapq.heappush(heap, (nd, prev_node))

    return dist


def optimal_next_actions(graph: dict, current: int, dest: int, goal: str) -> List[int]:
    dist = shortest_distances_to_dest(graph, dest, goal)
    if math.isinf(dist[current]):
        return []

    best_val = float("inf")
    candidates = []

    for (u, v), attrs in graph["edges"].items():
        if u != current:
            continue
        val = edge_cost(attrs, goal) + dist[v]
        if val < best_val - 1e-9:
            best_val = val
            candidates = [v]
        elif abs(val - best_val) <= 1e-9:
            candidates.append(v)

    return sorted(set(candidates))


# --------------------------------------------------
# VOCAB / TOKENIZATION
# --------------------------------------------------

class Vocabulary:
    def __init__(self, num_nodes: int, attr_range: Tuple[int, int]):
        self.num_nodes = num_nodes
        _, hi = attr_range
        self.max_attr_value = hi

        self.PAD = 0
        self.CLS = 1
        self.CURRENT = 2
        self.DEST = 3
        self.GOAL = 4
        self.EDGE = 5

        self.node_offset = 6
        self.goal_offset = self.node_offset + num_nodes
        self.val_offset = self.goal_offset + len(GOAL_LIST)

        self.vocab_size = self.val_offset + self.max_attr_value + 1

    def node_tok(self, n: int) -> int:
        return self.node_offset + n

    def goal_tok(self, goal: str) -> int:
        return self.goal_offset + GOAL_TO_IDX[goal]

    def val_tok(self, value: int) -> int:
        if value < 0 or value > self.max_attr_value:
            raise ValueError(f"attribute value {value} out of supported range")
        return self.val_offset + value


def ordered_edges_for_sample(
    graph: dict,
    current: int,
    prioritize_current_edges: bool,
    current_edges_first_only: bool,
):
    items = sorted(graph["edges"].items())

    if current_edges_first_only:
        return [item for item in items if item[0][0] == current]

    if not prioritize_current_edges:
        return items

    current_items = [item for item in items if item[0][0] == current]
    other_items = [item for item in items if item[0][0] != current]
    return current_items + other_items


def tokenize_sample(
    graph: dict,
    current: int,
    dest: int,
    goal: str,
    vocab: Vocabulary,
    max_len: int,
    include_goal_token: bool,
    prioritize_current_edges: bool,
    current_edges_first_only: bool,
) -> List[int]:
    toks = [vocab.CLS, vocab.CURRENT, vocab.node_tok(current), vocab.DEST, vocab.node_tok(dest)]
    if include_goal_token:
        toks.extend([vocab.GOAL, vocab.goal_tok(goal)])

    edge_items = ordered_edges_for_sample(
        graph=graph,
        current=current,
        prioritize_current_edges=prioritize_current_edges,
        current_edges_first_only=current_edges_first_only,
    )

    for (u, v), attrs in edge_items:
        edge_piece = [
            vocab.EDGE,
            vocab.node_tok(u),
            vocab.node_tok(v),
            vocab.val_tok(attrs["time"]),
            vocab.val_tok(attrs["cost"]),
            vocab.val_tok(attrs["risk"]),
            vocab.val_tok(attrs["effort"]),
        ]
        if len(toks) + len(edge_piece) > max_len:
            break
        toks.extend(edge_piece)

    if len(toks) < max_len:
        toks.extend([vocab.PAD] * (max_len - len(toks)))

    return toks[:max_len]


class RouteDataset(Dataset):
    def __init__(self, graphs: List[dict], vocab: Vocabulary, config: ExperimentConfig, include_goal_token: bool):
        self.samples = []
        rng = random.Random(config.seed + (1 if include_goal_token else 0) + len(graphs))

        for graph in graphs:
            n = graph["num_nodes"]
            made = 0
            tries = 0

            while made < config.samples_per_graph and tries < config.samples_per_graph * 20:
                tries += 1
                start = rng.randrange(n)
                dest = rng.randrange(n)
                if start == dest:
                    continue

                goal = rng.choice(config.goals)
                targets = optimal_next_actions(graph, start, dest, goal)
                if not targets:
                    continue

                tokens = tokenize_sample(
                    graph=graph,
                    current=start,
                    dest=dest,
                    goal=goal,
                    vocab=vocab,
                    max_len=config.max_seq_len,
                    include_goal_token=include_goal_token,
                    prioritize_current_edges=config.prioritize_current_edges,
                    current_edges_first_only=config.current_edges_first_only,
                )

                target_mask = torch.zeros(n, dtype=torch.float32)
                for t in targets:
                    target_mask[t] = 1.0

                self.samples.append({
                    "tokens": torch.tensor(tokens, dtype=torch.long),
                    "goal_idx": GOAL_TO_IDX[goal],
                    "target_mask": target_mask,
                })
                made += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["tokens"], s["goal_idx"], s["target_mask"]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class GoalRelevanceScorer(nn.Module):
    def __init__(self, d_model: int, num_goals: int):
        super().__init__()
        self.goal_embed = nn.Embedding(num_goals, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor, goal_idx: torch.Tensor) -> torch.Tensor:
        b, l, d = h.shape
        g = self.goal_embed(goal_idx).unsqueeze(1).expand(-1, l, -1)
        return self.proj(torch.cat([h, g], dim=-1)).squeeze(-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, goal_biased: bool, num_goals: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.goal_biased = goal_biased

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        if goal_biased:
            self.goal_scorer = GoalRelevanceScorer(d_model, num_goals)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        goal_idx: Optional[torch.Tensor],
        lam: float,
    ) -> torch.Tensor:
        b, l, d = x.shape

        q = self.wq(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.goal_biased and goal_idx is not None and lam != 0.0:
            rel = self.goal_scorer(x, goal_idx).unsqueeze(1).unsqueeze(2)
            attn = F.softmax(scores, dim=-1)

            rel = self.goal_scorer(x, goal_idx)
            rel = torch.sigmoid(rel)  # [0,1]

            rel = rel.unsqueeze(1).unsqueeze(2)

            # gate attention
            attn = attn * (1 + lam * rel)

            # re-normalize
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, d)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, goal_biased: bool, num_goals: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, goal_biased, num_goals)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask, goal_idx, lam):
        x = x + self.attn(self.ln1(x), key_padding_mask, goal_idx, lam)
        x = x + self.ff(self.ln2(x))
        return x


class ActionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_nodes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        model_type: str,
        num_goals: int,
    ):
        super().__init__()
        self.model_type = model_type

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        goal_biased = (model_type == "goal_biased")
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, goal_biased, num_goals)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_nodes)

    def forward(self, tokens: torch.Tensor, goal_idx: Optional[torch.Tensor], lam: float) -> torch.Tensor:
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device).unsqueeze(0)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        x = self.drop(x)

        key_padding_mask = (tokens != 0).float()

        for blk in self.blocks:
            x = blk(x, key_padding_mask, goal_idx, lam)

        x = self.ln(x)
        cls = x[:, 0, :]
        return self.head(cls)


# --------------------------------------------------
# TRAIN / EVAL
# --------------------------------------------------

def masked_multi_target_loss(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    counts = target_mask.sum(dim=-1).clamp(min=1.0)
    return -((log_probs * target_mask).sum(dim=-1) / counts).mean()


def compute_accuracy(logits: torch.Tensor, target_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    hits = target_mask[torch.arange(target_mask.size(0), device=target_mask.device), preds] > 0
    return hits.float().mean().item()


def train_one_epoch(model, loader, optimizer, device, lam: float):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for tokens, goal_idx, target_mask in loader:
        tokens = tokens.to(device)
        goal_idx = goal_idx.to(device)
        target_mask = target_mask.to(device)

        logits = model(tokens, goal_idx, lam)
        loss = masked_multi_target_loss(logits, target_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = tokens.size(0)
        total_loss += loss.item() * bs
        total_acc += compute_accuracy(logits, target_mask) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device, lam: float):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    per_goal_hits = defaultdict(int)
    per_goal_total = defaultdict(int)

    for tokens, goal_idx, target_mask in loader:
        tokens = tokens.to(device)
        goal_idx = goal_idx.to(device)
        target_mask = target_mask.to(device)

        logits = model(tokens, goal_idx, lam)
        loss = masked_multi_target_loss(logits, target_mask)

        preds = logits.argmax(dim=-1)
        hits = target_mask[torch.arange(target_mask.size(0), device=target_mask.device), preds] > 0

        bs = tokens.size(0)
        total_loss += loss.item() * bs
        total_acc += hits.float().mean().item() * bs
        total_n += bs

        for i in range(bs):
            g = int(goal_idx[i].item())
            per_goal_hits[g] += int(hits[i].item())
            per_goal_total[g] += 1

    per_goal = {}
    for g_idx, tot in sorted(per_goal_total.items()):
        per_goal[IDX_TO_GOAL[g_idx]] = per_goal_hits[g_idx] / tot

    return total_loss / total_n, total_acc / total_n, per_goal


def train_model(model, train_loader, val_loader, config: ExperimentConfig, lam: float):
    device = torch.device(config.device)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    best = -1.0
    best_per_goal = {}
    history = []
    patience_count = 0

    for epoch in range(config.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, lam)
        va_loss, va_acc, per_goal = evaluate(model, val_loader, device, lam)

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })

        if va_acc > best:
            best = va_acc
            best_per_goal = per_goal
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= config.patience:
            break

    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best, best_per_goal, history


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------

def save_focus_plots(results: List[dict], out_dir: str, focus_heads: int):
    os.makedirs(out_dir, exist_ok=True)

    # 1) All model types at chosen head count
    subset = [r for r in results if r["n_heads"] == focus_heads]

    plt.figure(figsize=(8, 5))

    baseline = [r for r in subset if r["model_type"] == "baseline"]
    goal_token = [r for r in subset if r["model_type"] == "goal_token"]
    goal_biased = sorted(
        [r for r in subset if r["model_type"] == "goal_biased"],
        key=lambda x: x["lambda"]
    )

    if baseline:
        plt.axhline(baseline[0]["best_val_acc"], linestyle="--", linewidth=1, label=f"baseline ({focus_heads} heads)")
    if goal_token:
        plt.axhline(goal_token[0]["best_val_acc"], linestyle=":", linewidth=2, label=f"goal_token ({focus_heads} heads)")
    if goal_biased:
        plt.plot(
            [r["lambda"] for r in goal_biased],
            [r["best_val_acc"] for r in goal_biased],
            marker="o",
            label=f"goal_biased ({focus_heads} heads)"
        )

    plt.xlabel("lambda")
    plt.ylabel("best validation accuracy")
    plt.title(f"Performance vs lambda at {focus_heads} heads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"performance_vs_lambda_{focus_heads}heads.png"), dpi=150)
    plt.close()

    # 2) Per-goal curves for goal-biased at chosen head count
    goal_biased = sorted(
        [r for r in results if r["model_type"] == "goal_biased" and r["n_heads"] == focus_heads],
        key=lambda x: x["lambda"]
    )

    if goal_biased:
        plt.figure(figsize=(9, 5))
        for goal in GOAL_LIST:
            xs = [r["lambda"] for r in goal_biased]
            ys = [r["per_goal_acc"].get(goal, 0.0) for r in goal_biased]
            plt.plot(xs, ys, marker="o", label=goal)

        plt.xlabel("lambda")
        plt.ylabel("per-goal accuracy")
        plt.title(f"Per-goal accuracy vs lambda ({focus_heads} heads, goal_biased)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"per_goal_vs_lambda_{focus_heads}heads.png"), dpi=150)
        plt.close()

    # 3) Per-goal bars comparing baseline / goal_token / best goal_biased at chosen head count
    best_biased = None
    if goal_biased:
        best_biased = max(goal_biased, key=lambda x: x["best_val_acc"])

    baseline_one = baseline[0] if baseline else None
    goal_token_one = goal_token[0] if goal_token else None

    if baseline_one and goal_token_one and best_biased:
        x = list(range(len(GOAL_LIST)))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(
            [i - width for i in x],
            [baseline_one["per_goal_acc"].get(g, 0.0) for g in GOAL_LIST],
            width=width,
            label="baseline"
        )
        plt.bar(
            x,
            [goal_token_one["per_goal_acc"].get(g, 0.0) for g in GOAL_LIST],
            width=width,
            label="goal_token"
        )
        plt.bar(
            [i + width for i in x],
            [best_biased["per_goal_acc"].get(g, 0.0) for g in GOAL_LIST],
            width=width,
            label=f"best goal_biased (lambda={best_biased['lambda']})"
        )
        plt.xticks(x, GOAL_LIST, rotation=30, ha="right")
        plt.ylabel("accuracy")
        plt.title(f"Per-goal comparison at {focus_heads} heads")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"per_goal_compare_{focus_heads}heads.png"), dpi=150)
        plt.close()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def print_summary(results: List[dict], config: ExperimentConfig):
    print("\n" + "=" * 60)
    print(f"SUMMARY - target accuracy {config.target_accuracy:.0%}")
    print("=" * 60)

    for model_type in config.model_types:
        print(f"\n{model_type.upper()}")
        type_results = [r for r in results if r["model_type"] == model_type]

        if model_type == "goal_biased":
            for lam in config.lambdas:
                lam_results = [r for r in type_results if abs(r["lambda"] - lam) < 1e-9]
                hits = [r for r in lam_results if r["hits_target"]]
                if hits:
                    min_heads = min(r["n_heads"] for r in hits)
                    best = max(lam_results, key=lambda x: x["best_val_acc"])
                    print(f"  lambda={lam:.2f}: min heads = {min_heads}, best acc = {best['best_val_acc']:.4f}")
                else:
                    best = max(lam_results, key=lambda x: x["best_val_acc"]) if lam_results else None
                    acc_str = f"{best['best_val_acc']:.4f}" if best else "N/A"
                    print(f"  lambda={lam:.2f}: never hits target (best = {acc_str})")
        else:
            hits = [r for r in type_results if r["hits_target"]]
            if hits:
                min_heads = min(r["n_heads"] for r in hits)
                best = max(type_results, key=lambda x: x["best_val_acc"])
                print(f"  min heads = {min_heads}, best acc = {best['best_val_acc']:.4f}")
            else:
                best = max(type_results, key=lambda x: x["best_val_acc"]) if type_results else None
                acc_str = f"{best['best_val_acc']:.4f}" if best else "N/A"
                print(f"  never hits target (best = {acc_str})")

    print("\n" + "-" * 60)
    print("KEY QUESTION: does goal-biased attention reduce min heads?")
    print("-" * 60)

    baseline_hits = [r for r in results if r["model_type"] == "baseline" and r["hits_target"]]
    baseline_min = min((r["n_heads"] for r in baseline_hits), default=None)

    biased_hits = [r for r in results if r["model_type"] == "goal_biased" and r["hits_target"]]
    biased_min = min((r["n_heads"] for r in biased_hits), default=None)

    if baseline_min is not None and biased_min is not None:
        if biased_min < baseline_min:
            print(f"YES - baseline needs {baseline_min} heads, goal-biased needs {biased_min}")
        elif biased_min == baseline_min:
            print(f"INCONCLUSIVE - both need {baseline_min} heads")
        else:
            print(f"NO - baseline needs {baseline_min}, goal-biased needs {biased_min}")
    elif baseline_min is not None and biased_min is None:
        print(f"NO - baseline hits target at {baseline_min} heads, goal-biased never hits target")
    elif baseline_min is None and biased_min is not None:
        print(f"YES (strongly) - baseline never hits target, goal-biased does at {biased_min} heads")
    else:
        print("INCONCLUSIVE - neither model hits the target")


# --------------------------------------------------
# RUNNER
# --------------------------------------------------

def run_experiment(config: ExperimentConfig):
    set_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)

    print("=" * 60)
    print("GOAL-BIASED ATTENTION EXPERIMENT")
    print("=" * 60)

    wc = WorldConfig(seed=config.seed)
    rng = random.Random(wc.seed)

    print("\n[1/4] Generating graphs...")
    train_graphs = [generate_graph(wc.num_nodes, wc.edge_density, wc.attr_range, rng) for _ in range(config.num_train_graphs)]
    val_graphs = [generate_graph(wc.num_nodes, wc.edge_density, wc.attr_range, rng) for _ in range(config.num_val_graphs)]

    vocab = Vocabulary(wc.num_nodes, wc.attr_range)

    print("[2/4] Building datasets...")
    train_no_goal = RouteDataset(train_graphs, vocab, config, include_goal_token=False)
    val_no_goal = RouteDataset(val_graphs, vocab, config, include_goal_token=False)
    train_goal = RouteDataset(train_graphs, vocab, config, include_goal_token=True)
    val_goal = RouteDataset(val_graphs, vocab, config, include_goal_token=True)

    print(f"  train samples (no goal token): {len(train_no_goal)}")
    print(f"  val samples   (no goal token): {len(val_no_goal)}")
    print(f"  train samples (with goal):     {len(train_goal)}")
    print(f"  val samples   (with goal):     {len(val_goal)}")

    loaders = {
        False: (
            DataLoader(train_no_goal, batch_size=config.batch_size, shuffle=True),
            DataLoader(val_no_goal, batch_size=config.batch_size, shuffle=False),
        ),
        True: (
            DataLoader(train_goal, batch_size=config.batch_size, shuffle=True),
            DataLoader(val_goal, batch_size=config.batch_size, shuffle=False),
        ),
    }

    total_runs = 0
    for m in config.model_types:
        for _ in config.head_counts:
            if m in ["baseline", "goal_token"]:
                total_runs += 1
            else:
                total_runs += len(config.lambdas)

    print("[3/4] Running experiment grid...")
    results = []
    run_idx = 0

    for model_type in config.model_types:
        for n_heads in config.head_counts:
            if model_type == "baseline":
                lambda_values = [0.0]
                include_goal = False
            elif model_type == "goal_token":
                lambda_values = [0.0]
                include_goal = True
            else:
                lambda_values = config.lambdas
                include_goal = True

            train_loader, val_loader = loaders[include_goal]

            d_model = config.d_model
            if d_model % n_heads != 0:
                d_model = n_heads * max(1, d_model // n_heads)

            for lam in lambda_values:
                run_idx += 1
                print(f"\nRun {run_idx}/{total_runs} | {model_type} | heads={n_heads} | lambda={lam}")

                model = ActionTransformer(
                    vocab_size=vocab.vocab_size,
                    num_nodes=wc.num_nodes,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=config.n_layers,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    max_seq_len=config.max_seq_len,
                    model_type=model_type,
                    num_goals=len(GOAL_LIST),
                )

                params = sum(p.numel() for p in model.parameters())
                t0 = time.time()
                best_acc, per_goal, history = train_model(model, train_loader, val_loader, config, lam)
                elapsed = time.time() - t0

                result = {
                    "model_type": model_type,
                    "n_heads": n_heads,
                    "lambda": lam,
                    "d_model": d_model,
                    "params": params,
                    "best_val_acc": best_acc,
                    "per_goal_acc": per_goal,
                    "epochs_trained": len(history),
                    "time_seconds": elapsed,
                    "hits_target": best_acc >= config.target_accuracy,
                }
                results.append(result)

                goal_str = " | ".join(
                    f"{g[:4]}={per_goal.get(g, 0.0):.2f}" for g in GOAL_LIST
                )
                print(f"  acc={best_acc:.4f} | target_hit={result['hits_target']} | time={elapsed:.1f}s | params={params}")
                print(f"  per_goal: {goal_str}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_json = os.path.join(config.results_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "config": asdict(config),
            "world_config": asdict(wc),
            "results": results,
        }, f, indent=2)

    print("[4/4] Saving plots...")
    save_focus_plots(results, config.results_dir, config.plot_focus_heads)

    print(f"\nSaved results to: {out_json}")
    print(f"Saved plots to: {config.results_dir}")
    print_summary(results, config)
    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    cfg = ExperimentConfig()

    if torch.cuda.is_available():
        cfg.device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        cfg.device = "mps"
        print("Using Apple MPS")
    else:
        cfg.device = "cpu"
        print("Using CPU")

    run_experiment(cfg)