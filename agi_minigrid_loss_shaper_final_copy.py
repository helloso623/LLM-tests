# agi_minigrid_loss_shaper_final_copy.py
# Faithful end-to-end runnable: PPO + adaptive loss-shaper with:
# - 3D CNN + single-head attention analyzer on stacked one-hot frames
# - automatic variable creation (actions + latent projections) with validation + decay
# - propositional rule miner (NOT/AND/OR/XOR/IMPLIES) with coverage + novelty + decay + disproval
# - explicit "bad component" tracking (anti-goals) and cancellation attempts via MC probes
# - transformer-like softmax memory recall over embeddings
# - compute governor: think ratio limit + RAM cap + adaptive throttles
# - rigorous MC acceptance: revert unless beats baseline by margin
#
# Requires: gymnasium, minigrid, stable_baselines3, torch, numpy (psutil optional)

from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import OneHotPartialObsWrapper, FlatObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

try:
    import psutil
except Exception:
    psutil = None


# ----------------------------
# Repro
# ----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Constants
# ----------------------------
W_DIM = 5  # local reward weights dimension


# ----------------------------
# Success predictor (P(success | episode start obs)) + embedding key
# ----------------------------
class SuccessPredictor(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, emb_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.emb = nn.Linear(hid, emb_dim)
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        e = torch.tanh(self.emb(h))
        p = torch.sigmoid(self.out(e)).squeeze(-1)
        return p, e


# ----------------------------
# Softmax "transformer-like" memory recall
# ----------------------------
@dataclass
class MemoryItem:
    key: torch.Tensor      # [D] normalized
    payload: torch.Tensor  # e.g., w vector [W_DIM]
    strength: float
    last_used_step: int


class SoftmaxMemory:
    def __init__(self, max_items: int = 256, decay: float = 0.93):
        self.max_items = int(max_items)
        self.decay = float(decay)
        self.items: List[MemoryItem] = []

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2) + 1e-8)

    def add(self, key: torch.Tensor, payload: torch.Tensor, *, strength: float, step: int):
        key_n = self._norm(key.detach().cpu())
        payload_cpu = payload.detach().cpu()

        # merge if very close and payload close
        best_i, best_sim = None, -1.0
        for i, it in enumerate(self.items):
            sim = float(torch.dot(key_n, it.key))
            if sim > best_sim:
                best_sim, best_i = sim, i

        if best_i is not None and best_sim > 0.985 and torch.allclose(self.items[best_i].payload, payload_cpu, atol=5e-3):
            self.items[best_i].strength += float(strength)
            self.items[best_i].last_used_step = int(step)
        else:
            self.items.append(MemoryItem(key=key_n, payload=payload_cpu, strength=float(strength), last_used_step=int(step)))

        if len(self.items) > self.max_items:
            self._evict()

    def _evict(self):
        # keep strong + recently used
        def score(it: MemoryItem) -> float:
            return float(it.strength) + 0.0005 * float(it.last_used_step)
        self.items.sort(key=score, reverse=True)
        self.items = self.items[: self.max_items]

    def recall_softmax(
        self,
        query: torch.Tensor,
        *,
        topk: int = 16,
        temperature: float = 0.35,
        step: int = 0,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[float, torch.Tensor]]]:
        """
        Transformer-like retrieval:
        - computes attention logits = dot(q, k) * strength / temperature
        - returns weighted sum payload, plus topk scored list
        """
        if not self.items:
            return None, []

        q = self._norm(query.detach().cpu())
        scored: List[Tuple[float, int]] = []
        for i, it in enumerate(self.items):
            sim = float(torch.dot(q, it.key))
            scored.append((sim * float(it.strength), i))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: int(topk)]

        logits = []
        payloads = []
        out_list: List[Tuple[float, torch.Tensor]] = []
        for s, i in scored:
            it = self.items[i]
            # "related not necessarily identical": strength modulates, and softmax mixes
            logit = float(s) / max(1e-6, float(temperature))
            logits.append(logit)
            payloads.append(it.payload)
            out_list.append((float(s), it.payload))
            it.last_used_step = int(step)

        logits_t = torch.tensor(logits, dtype=torch.float32)
        att = torch.softmax(logits_t, dim=0)  # [K]
        payload_stack = torch.stack(payloads, dim=0).float()  # [K, W_DIM]
        mixed = (att[:, None] * payload_stack).sum(dim=0)  # [W_DIM]
        return mixed, out_list

    def decay_all(self, pressure: float = 1.0):
        d = self.decay ** float(pressure)
        for it in self.items:
            it.strength *= d
        self.items = [it for it in self.items if it.strength > 0.05]


# ----------------------------
# Raw one-hot obs passthrough into info + robust episode-start obs capture
# ----------------------------
class InfoCaptureWrapper(gym.Wrapper):
    """
    - Stores one-hot image in info["obs_onehot"] every step
    - Stores flattened episode-start obs in info["episode_start_obs"] at reset and episode end
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["obs_onehot"] = self._extract_onehot(obs)  # Only capture onehot here
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["obs_onehot"] = self._extract_onehot(obs)  # Simplified logic
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _extract_onehot(obs) -> Optional[np.ndarray]:
        if isinstance(obs, dict) and "image" in obs:
            x = obs["image"]
            x = np.array(x, dtype=np.float32)
            if x.ndim == 3:
                return x
        return None


# ----------------------------
# 3D CNN + single-head attention analyzer
# ----------------------------
class SingleHeadAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.scale = 1.0 / math.sqrt(max(1, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        att = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,T,T]
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, V)  # [B,T,D]
        return out


class Analyzer3DCNN(nn.Module):
    """
    frames: [B,T,H,W,C]
    extras: [B,E]
    output y: [B,2] -> (predicted dGlobal, predicted risk)
    latent: [B,D]
    """
    def __init__(self, c_in: int, t: int, extras_dim: int, hid: int = 128, att_d: int = 128):
        super().__init__()
        self.t = int(t)

        self.conv = nn.Sequential(
            nn.Conv3d(c_in, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((self.t, 1, 1))

        self.proj = nn.Linear(96, att_d)
        self.att = SingleHeadAttention(att_d)

        self.extras = nn.Sequential(
            nn.Linear(extras_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, hid),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(att_d + hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 2),
        )

    def forward(self, frames: torch.Tensor, extras: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = frames.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,T,H,W]
        z = self.conv(x)
        z = self.pool(z)  # [B,96,T,1,1]
        z = z.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()  # [B,T,96]
        z = self.proj(z)  # [B,T,D]
        z = self.att(z)
        z = z.mean(dim=1)  # [B,D]

        e = self.extras(extras)
        y = self.head(torch.cat([z, e], dim=-1))
        return y, z


# ----------------------------
# Variables (validated + decay)
# ----------------------------
@dataclass
class AutoVariable:
    name: str
    proj: np.ndarray  # [D]
    thr: float
    alpha: float = 1.0
    beta: float = 1.0
    support: int = 0
    fired_total: int = 0
    total_seen: int = 0
    last_used_step: int = 0

    def trust(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def coverage(self) -> float:
        return float(self.fired_total / max(1, self.total_seen))

    def eval(self, latent: np.ndarray) -> bool:
        return bool(float(np.dot(latent, self.proj)) >= float(self.thr))

    def decay_step(self, base: float, pressure: float):
        b = float(base) ** float(pressure)
        self.alpha = 1.0 + (self.alpha - 1.0) * b
        self.beta = 1.0 + (self.beta - 1.0) * b


class VariableFactory:
    def __init__(self, latent_dim: int, max_vars: int, seed: int):
        self.latent_dim = int(latent_dim)
        self.max_vars = int(max_vars)
        self.rng = np.random.RandomState(seed + 4242)
        self.vars: List[AutoVariable] = []
        self._ctr = 0

    def maybe_create_from_latent(self, latent: np.ndarray, *, create_rate: float, step: int):
        if len(self.vars) >= self.max_vars:
            return
        if self.rng.rand() > float(create_rate):
            return
        proj = self.rng.randn(self.latent_dim).astype(np.float32)
        proj /= (np.linalg.norm(proj) + 1e-8)
        thr = float(self.rng.randn() * 0.25)
        name = f"LATVAR_{self._ctr}"
        self._ctr += 1
        self.vars.append(AutoVariable(name=name, proj=proj, thr=thr, last_used_step=int(step)))

    def update(self, latent: np.ndarray, success: bool, *, disprove_k: float, decay_base: float, pressure: float, step: int):
        for v in self.vars:
            v.decay_step(decay_base, pressure)

        for v in self.vars:
            v.total_seen += 1
            fired = v.eval(latent)
            if fired:
                v.fired_total += 1
                v.support += 1
                if success:
                    v.alpha += 1.0
                else:
                    v.beta += float(disprove_k)
                v.last_used_step = int(step)

        if len(self.vars) > self.max_vars:
            self._evict()

    def _evict(self):
        # keep trusted + balanced coverage + recently used
        def score(v: AutoVariable) -> float:
            cov = v.coverage()
            cov_pen = 0.0
            if cov < 0.10 or cov > 0.90:
                cov_pen = 2.0
            rec = 0.0005 * float(v.last_used_step)
            return v.trust() - cov_pen + rec - 0.0003 * float(v.support)
        self.vars.sort(key=score, reverse=True)
        self.vars = self.vars[: self.max_vars]

    def predicates(self, latent: np.ndarray, *, min_trust: float, min_support: int, cov_lo: float, cov_hi: float) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for v in self.vars:
            if v.support < int(min_support):
                continue
            if v.trust() < float(min_trust):
                continue
            cov = v.coverage()
            if cov < float(cov_lo) or cov > float(cov_hi):
                continue
            out[v.name] = bool(v.eval(latent))
        return out


# ----------------------------
# Rule engine with coverage + novelty + decay + disproval + directions
# ----------------------------
Literal = Tuple[str, bool]


def lit_eval(P: Dict[str, bool], lit: Literal) -> bool:
    name, pos = lit
    v = bool(P.get(name, False))
    return v if pos else (not v)


def rule_signature(expr: str) -> int:
    return (hash(expr) & 0x7FFFFFFF)


@dataclass
class RuleHypothesis:
    expr: str
    fn: Callable[[Dict[str, bool]], bool]
    predicts_success: bool
    alpha: float = 1.0
    beta: float = 1.0
    support: int = 0
    fired_total: int = 0
    total_seen: int = 0
    sig: int = 0
    last_used_step: int = 0

    succ_sum: Optional[np.ndarray] = None
    fail_sum: Optional[np.ndarray] = None
    succ_n: int = 0
    fail_n: int = 0

    def trust(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def coverage(self) -> float:
        return float(self.fired_total / max(1, self.total_seen))

    def decay_step(self, base: float, pressure: float):
        b = float(base) ** float(pressure)
        self.alpha = 1.0 + (self.alpha - 1.0) * b
        self.beta = 1.0 + (self.beta - 1.0) * b

    def update(self, fired: bool, success: bool, feat_ep: np.ndarray, *, disprove_k: float, step: int):
        self.total_seen += 1
        if not fired:
            return

        self.fired_total += 1
        self.support += 1
        correct = (bool(success) == bool(self.predicts_success))
        if correct:
            self.alpha += 1.0
        else:
            self.beta += float(disprove_k)

        if self.succ_sum is None:
            self.succ_sum = np.zeros_like(feat_ep, dtype=np.float64)
            self.fail_sum = np.zeros_like(feat_ep, dtype=np.float64)

        if success:
            self.succ_sum += feat_ep
            self.succ_n += 1
        else:
            self.fail_sum += feat_ep
            self.fail_n += 1

        self.last_used_step = int(step)

    def feat_direction(self) -> np.ndarray:
        if self.succ_sum is None or self.fail_sum is None:
            return np.zeros((W_DIM,), dtype=np.float32)

        succ_ok = (self.succ_n >= 3)
        fail_ok = (self.fail_n >= 3)

        succ_mean = self.succ_sum / max(1, self.succ_n)
        fail_mean = self.fail_sum / max(1, self.fail_n)

        if succ_ok and fail_ok:
            d = succ_mean - fail_mean
        elif fail_ok and self.succ_n == 0:
            d = -fail_mean
        elif succ_ok and self.fail_n == 0:
            d = succ_mean
        else:
            return np.zeros((W_DIM,), dtype=np.float32)

        return d.astype(np.float32)


class RuleEngine:
    def __init__(self, w_dim: int):
        self.w_dim = int(w_dim)
        self.rules: List[RuleHypothesis] = []
        self.predicates_known: List[str] = []
        self._recent_selected: Deque[int] = deque(maxlen=256)

    def ensure_predicates(self, names: List[str]):
        # incremental: if new predicates appear, add candidate rules involving them
        for n in names:
            if n not in self.predicates_known:
                self._add_predicate(n)

    def _add_predicate(self, name: str):
        existing = list(self.predicates_known)
        self.predicates_known.append(name)

        # single literal rules for new predicate
        for pos in (True, False):
            lit = (name, pos)
            expr_l = (name if pos else f"NOT {name}")
            fn_l = (lambda P, lit=lit: lit_eval(P, lit))
            for predicts_success in (True, False):
                expr = f"{expr_l} IMPLIES {'SUCCESS' if predicts_success else 'FAIL'}"
                self.rules.append(RuleHypothesis(expr=expr, fn=fn_l, predicts_success=predicts_success, sig=rule_signature(expr)))

        # pair rules between new predicate and all existing ones
        ops = ["AND", "OR", "XOR"]
        for other in existing:
            a, b = other, name
            for posa in (True, False):
                for posb in (True, False):
                    l1 = (a, posa)
                    l2 = (b, posb)
                    s1 = (a if posa else f"NOT {a}")
                    s2 = (b if posb else f"NOT {b}")

                    def fn_and(P, l1=l1, l2=l2): return lit_eval(P, l1) and lit_eval(P, l2)
                    def fn_or(P, l1=l1, l2=l2): return lit_eval(P, l1) or lit_eval(P, l2)
                    def fn_xor(P, l1=l1, l2=l2): return (lit_eval(P, l1) != lit_eval(P, l2))

                    fns = [fn_and, fn_or, fn_xor]
                    for op, fn in zip(ops, fns):
                        base = f"({s1} {op} {s2})"
                        for predicts_success in (True, False):
                            expr = f"{base} IMPLIES {'SUCCESS' if predicts_success else 'FAIL'}"
                            self.rules.append(RuleHypothesis(expr=expr, fn=fn, predicts_success=predicts_success, sig=rule_signature(expr)))

    def decay_all(self, base: float, pressure: float):
        for r in self.rules:
            r.decay_step(base, pressure)

    def update_episode(self, preds0: Dict[str, bool], success: bool, feat_ep: np.ndarray, *, disprove_k: float, step: int):
        for r in self.rules:
            try:
                fired = bool(r.fn(preds0))
            except Exception:
                fired = False
            r.update(fired, bool(success), feat_ep, disprove_k=disprove_k, step=step)

    def propose_delta_w(
        self,
        preds0: Dict[str, bool],
        *,
        min_support: int,
        min_trust: float,
        cov_lo: float,
        cov_hi: float,
        topk: int,
        novelty_lambda: float,
        eta: float,
        clip_norm: float,
    ) -> Tuple[np.ndarray, List[Tuple[str, float, int, float]]]:
        fired: List[RuleHypothesis] = []
        for r in self.rules:
            if r.support < int(min_support):
                continue
            if r.trust() < float(min_trust):
                continue
            cov = r.coverage()
            if cov < float(cov_lo) or cov > float(cov_hi):
                continue
            try:
                if r.fn(preds0):
                    fired.append(r)
            except Exception:
                continue

        def score(rr: RuleHypothesis) -> float:
            pen = float(novelty_lambda) if rr.sig in self._recent_selected else 0.0
            return rr.trust() + 0.001 * float(rr.support) - pen

        fired.sort(key=score, reverse=True)
        fired = fired[: int(topk)]

        expl = [(r.expr, r.trust(), r.support, r.coverage()) for r in fired]
        if not fired:
            return np.zeros((self.w_dim,), dtype=np.float32), expl

        d = np.zeros((self.w_dim,), dtype=np.float32)
        wsum = 0.0
        for r in fired:
            fd = r.feat_direction()
            if np.all(fd == 0):
                continue
            w = float(r.trust())
            d += w * fd
            wsum += w

        if wsum <= 1e-6 or np.all(d == 0):
            return np.zeros((self.w_dim,), dtype=np.float32), expl

        d = d / (wsum + 1e-8)
        d = d / (float(np.linalg.norm(d)) + 1e-8)
        d = np.clip(d, -float(clip_norm), float(clip_norm)).astype(np.float32)

        for r in fired:
            self._recent_selected.append(r.sig)

        return (float(eta) * d).astype(np.float32), expl


# ----------------------------
# Loss shaping wrapper (features + predicates)
# ----------------------------
class LossShapingWrapper(gym.Wrapper):
    """
    Features f:
      f0 env reward
      f1 new cell visited
      f2 picked key
      f3 doors opened delta
      f4 step cost (=1)

    Predicates:
      HAS_KEY, ANY_DOOR_OPEN, NEAR_DOOR, NEAR_KEY, NEAR_GOAL, STUCK
    """
    def __init__(self, env: gym.Env, *, use_shaped_reward: bool, initial_w: np.ndarray):
        super().__init__(env)
        self.use_shaped_reward = bool(use_shaped_reward)
        self._w = np.array(initial_w, dtype=np.float32).copy()
        assert self._w.shape == (W_DIM,)

        self._visited = set()
        self._prev_carry_key = False
        self._door_cells: List[Tuple[int, int]] = []
        self._prev_open_doors = 0
        self._ep_feat_sum = np.zeros((W_DIM,), dtype=np.float32)

        self._last_pos = None
        self._stuck_ctr = 0
        self._start_preds: Dict[str, bool] = {}

        # timing for think ratio (env-side)
        self._t_last = None
        self._env_time_acc = 0.0

    def pop_env_time_acc(self) -> float:
        x = float(self._env_time_acc)
        self._env_time_acc = 0.0
        return x

    def set_w(self, new_w: np.ndarray):
        w = np.array(new_w, dtype=np.float32).copy()
        assert w.shape == (W_DIM,)
        self._w = w

    def _get_agent_pos(self):
        uw = self.unwrapped
        if hasattr(uw, "agent_pos"):
            p = uw.agent_pos
            return (int(p[0]), int(p[1]))
        return None

    def _is_carrying_key(self) -> bool:
        carry = getattr(self.unwrapped, "carrying", None)
        if carry is None:
            return False
        if hasattr(carry, "type"):
            return str(carry.type) == "key"
        return False

    def _cache_doors(self):
        self._door_cells = []
        uw = self.unwrapped
        try:
            grid = uw.grid
            w, h = grid.width, grid.height
            for y in range(h):
                for x in range(w):
                    obj = grid.get(x, y)
                    if obj is None:
                        continue
                    if hasattr(obj, "type") and str(obj.type) == "door":
                        self._door_cells.append((x, y))
        except Exception:
            self._door_cells = []

    def _count_open_doors_cached(self) -> int:
        uw = self.unwrapped
        try:
            grid = uw.grid
            c = 0
            for (x, y) in self._door_cells:
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if hasattr(obj, "is_open") and bool(obj.is_open):
                    c += 1
            return int(c)
        except Exception:
            return 0

    def _near_type(self, obj_type: str) -> bool:
        uw = self.unwrapped
        if not hasattr(uw, "grid"):
            return False
        pos = self._get_agent_pos()
        if pos is None:
            return False
        x0, y0 = pos
        grid = uw.grid
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = x0 + dx, y0 + dy
                try:
                    obj = grid.get(x, y)
                except Exception:
                    obj = None
                if obj is None:
                    continue
                if hasattr(obj, "type") and str(obj.type) == obj_type:
                    return True
        return False

    def _compute_predicates(self) -> Dict[str, bool]:
        now_carry_key = self._is_carrying_key()
        now_open_doors = self._count_open_doors_cached()
        near_door = self._near_type("door")
        near_key = self._near_type("key")
        near_goal = self._near_type("goal")

        pos = self._get_agent_pos()
        if pos is None:
            stuck = False
        else:
            if self._last_pos == pos:
                self._stuck_ctr += 1
            else:
                self._stuck_ctr = 0
            self._last_pos = pos
            stuck = (self._stuck_ctr >= 3)

        return {
            "HAS_KEY": bool(now_carry_key),
            "ANY_DOOR_OPEN": bool(now_open_doors > 0),
            "NEAR_DOOR": bool(near_door),
            "NEAR_KEY": bool(near_key),
            "NEAR_GOAL": bool(near_goal),
            "STUCK": bool(stuck),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = set()
        pos = self._get_agent_pos()
        if pos is not None:
            self._visited.add(pos)

        self._prev_carry_key = self._is_carrying_key()
        self._cache_doors()
        self._prev_open_doors = self._count_open_doors_cached()
        self._ep_feat_sum = np.zeros((W_DIM,), dtype=np.float32)

        self._last_pos = self._get_agent_pos()
        self._stuck_ctr = 0
        self._start_preds = self._compute_predicates()

        self._t_last = time.time()

        info = dict(info)
        info["features"] = np.zeros((W_DIM,), dtype=np.float32)
        info["features_episode"] = np.zeros((W_DIM,), dtype=np.float32)
        info["is_success"] = False
        info["predicates_episode_start"] = dict(self._start_preds)
        info["predicates"] = dict(self._start_preds)
        return obs, info

    def step(self, action):
        # env time acc
        t = time.time()
        if self._t_last is not None:
            self._env_time_acc += max(0.0, t - self._t_last)
        self._t_last = t

        prev_carry_key = self._prev_carry_key
        prev_open_doors = self._prev_open_doors

        obs, reward, terminated, truncated, info = self.env.step(action)

        pos = self._get_agent_pos()
        now_carry_key = self._is_carrying_key()
        now_open_doors = self._count_open_doors_cached()

        f = np.zeros((W_DIM,), dtype=np.float32)
        f[0] = float(reward)

        if pos is not None and pos not in self._visited:
            f[1] = 1.0
            self._visited.add(pos)

        if (not prev_carry_key) and now_carry_key:
            f[2] = 1.0

        opened_delta = max(0, now_open_doors - prev_open_doors)
        if opened_delta > 0:
            f[3] = float(opened_delta)

        f[4] = 1.0  # step cost
        self._ep_feat_sum += f

        shaped = float(np.dot(self._w, f))
        is_success = bool((terminated or truncated) and (reward > 0.0))

        info = dict(info)
        info["is_success"] = is_success
        info["features"] = f
        if terminated or truncated:
            info["features_episode"] = self._ep_feat_sum.copy()

        self._prev_carry_key = now_carry_key
        self._prev_open_doors = now_open_doors

        info["predicates"] = self._compute_predicates()
        if terminated or truncated:
            info["predicates_episode_start"] = dict(self._start_preds)

        if self.use_shaped_reward:
            return obs, shaped, terminated, truncated, info
        return obs, float(reward), terminated, truncated, info


# ----------------------------
# Env builders
# ----------------------------
def make_env(env_id: str, seed: int, *, use_shaped_reward: bool, initial_w: np.ndarray) -> gym.Env:
    env = gym.make(env_id)
    env = OneHotPartialObsWrapper(env)   # dict obs with one-hot image
    env = InfoCaptureWrapper(env)        # capture obs_onehot while still dict
    env = FlatObsWrapper(env)            # flatten for PPO
    env = LossShapingWrapper(env, use_shaped_reward=use_shaped_reward, initial_w=initial_w)
    env.reset(seed=seed)
    return env


def make_vec_env(env_id: str, seed: int, n_envs: int, *, use_shaped_reward: bool, initial_w: np.ndarray):
    def thunk(i: int):
        return lambda: make_env(env_id, seed + i, use_shaped_reward=use_shaped_reward, initial_w=initial_w)
    env = DummyVecEnv([thunk(i) for i in range(n_envs)])
    env = VecMonitor(env)
    return env


def make_eval_env(env_id: str, seed: int, *, use_shaped_reward: bool, initial_w: np.ndarray):
    return make_vec_env(env_id, seed, n_envs=1, use_shaped_reward=use_shaped_reward, initial_w=initial_w)


def eval_success_rate_from_env(model: PPO, vec_env, episodes: int, deterministic: bool = True) -> float:
    assert getattr(vec_env, "num_envs", 1) == 1
    successes = 0
    for _ in range(int(episodes)):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            if done and infos[0].get("is_success", False):
                successes += 1
    return successes / float(episodes)


# ----------------------------
# Feature stats (direction)
# ----------------------------
@dataclass
class FeatureStats:
    succ_sum: np.ndarray
    fail_sum: np.ndarray
    succ_n: int
    fail_n: int

    @staticmethod
    def init():
        return FeatureStats(
            succ_sum=np.zeros((W_DIM,), dtype=np.float64),
            fail_sum=np.zeros((W_DIM,), dtype=np.float64),
            succ_n=0,
            fail_n=0,
        )

    def update(self, feat_ep: np.ndarray, success: bool):
        if success:
            self.succ_sum += feat_ep
            self.succ_n += 1
        else:
            self.fail_sum += feat_ep
            self.fail_n += 1

    def direction(self) -> np.ndarray:
        if self.succ_n >= 5 and self.fail_n >= 5:
            s = self.succ_sum / max(1, self.succ_n)
            f = self.fail_sum / max(1, self.fail_n)
            return (s - f).astype(np.float32)
        if self.fail_n >= 5 and self.succ_n == 0:
            f = self.fail_sum / max(1, self.fail_n)
            return (-f).astype(np.float32)
        if self.succ_n >= 5 and self.fail_n == 0:
            s = self.succ_sum / max(1, self.succ_n)
            return (s).astype(np.float32)
        return np.zeros((W_DIM,), dtype=np.float32)


# ----------------------------
# Compute governor
# ----------------------------
@dataclass
class ComputeGovernor:
    mc_eval_eps: int
    mc_candidates: int
    trigger_percentile: float
    trigger_cooldown: int
    mem_decay: float
    explore_eps: float

    max_mc_seconds_per_trigger: float
    max_rss_mb: float

    think_ratio_limit: float
    throttle_episodes: int

    def adapt(self, elapsed_sec: float, rss_mb: Optional[float]):
        if elapsed_sec > self.max_mc_seconds_per_trigger:
            self.mc_eval_eps = max(1, int(self.mc_eval_eps * 0.6))
            self.mc_candidates = max(4, int(self.mc_candidates * 0.7))
            self.trigger_percentile = min(25.0, float(self.trigger_percentile) + 2.0)
            self.explore_eps = max(0.0005, float(self.explore_eps) * 0.85)

        if rss_mb is not None and rss_mb > self.max_rss_mb:
            self.mem_decay = max(0.80, float(self.mem_decay) - 0.03)
            self.mc_candidates = max(4, int(self.mc_candidates * 0.7))
            self.mc_eval_eps = max(1, int(self.mc_eval_eps * 0.7))


# ----------------------------
# Ours meta-callback (faithful)
# ----------------------------
class OursMetaCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        device: str,
        w_init: np.ndarray,
        gov: ComputeGovernor,
        eval_eps: int,
        warmup_episodes: int,
        min_pred_samples: int,
        log_every_episodes: int,
        eval_every_episodes: int,
        # analyzer
        obs_stack_t: int,
        analyzer_lr: float,
        analyzer_train_every: int,
        analyzer_window: int,
        max_auto_vars: int,
        auto_var_create_rate: float,
        # rules
        rule_min_support: int,
        rule_min_trust: float,
        rule_cov_lo: float,
        rule_cov_hi: float,
        rule_topk: int,
        rule_disprove_k: float,
        rule_decay_base: float,
        rule_novelty_lambda: float,
        # acceptance
        mc_accept_margin: float,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.env_id = env_id
        self.seed = int(seed)
        self.device = str(device)
        self.w = np.array(w_init, dtype=np.float32).copy()
        self.gov = gov

        self.eval_eps = int(eval_eps)
        self.warmup_episodes = int(warmup_episodes)
        self.min_pred_samples = int(min_pred_samples)
        self.log_every_episodes = int(log_every_episodes)
        self.eval_every_episodes = int(eval_every_episodes)

        # analyzer config
        self.obs_stack_t = int(obs_stack_t)
        self.analyzer_lr = float(analyzer_lr)
        self.analyzer_train_every = int(analyzer_train_every)
        self.analyzer_window = int(analyzer_window)
        self.max_auto_vars = int(max_auto_vars)
        self.auto_var_create_rate = float(auto_var_create_rate)

        # rule config
        self.rule_min_support = int(rule_min_support)
        self.rule_min_trust = float(rule_min_trust)
        self.rule_cov_lo = float(rule_cov_lo)
        self.rule_cov_hi = float(rule_cov_hi)
        self.rule_topk = int(rule_topk)
        self.rule_disprove_k = float(rule_disprove_k)
        self.rule_decay_base = float(rule_decay_base)
        self.rule_novelty_lambda = float(rule_novelty_lambda)

        self.mc_accept_margin = float(mc_accept_margin)

        # predictor
        self.pred: Optional[SuccessPredictor] = None
        self.pred_opt: Optional[optim.Optimizer] = None
        self.pred_x: List[np.ndarray] = []
        self.pred_y: List[float] = []

        # memory
        self.mem = SoftmaxMemory(max_items=256, decay=self.gov.mem_decay)

        # feature stats
        self.feature_stats = FeatureStats.init()

        # rule engine
        self.rule_engine = RuleEngine(w_dim=W_DIM)

        # analyzer and vars
        self.analyzer: Optional[Analyzer3DCNN] = None
        self.analyzer_opt: Optional[optim.Optimizer] = None
        self.var_factory: Optional[VariableFactory] = None

        self.buf_frames: Deque[np.ndarray] = deque(maxlen=8000)
        self.buf_extras: Deque[np.ndarray] = deque(maxlen=8000)
        self.buf_targets: Deque[np.ndarray] = deque(maxlen=8000)

        self._frame_stack: Deque[np.ndarray] = deque(maxlen=self.obs_stack_t)

        # action trace => predicates
        self._ep_actions: List[int] = []

        # runtime
        self.episode_count = 0
        self.recent_success: List[int] = []
        self.p0_hist: List[float] = []
        self.cooldown = 0
        self._throttle_left = 0

        self._meta_time_acc = 0.0
        self._env_time_acc = 0.0

        # MC/Eval env
        self.mc_env = None
        self.eval_env = None

        self.rng = np.random.RandomState(self.seed + 1337)
        self.mc_seed_ctr = 0
        self.mc_seed_pool = [self.seed + 70001, self.seed + 70003, self.seed + 70007, self.seed + 70009]

        # last episode start preds for trigger use
        self._last_episode_start_preds: Dict[str, bool] = {}

        # explicit bad-components (anti-goals): predicate literal -> beta trust
        self.bad_alpha: Dict[str, float] = {}
        self.bad_beta: Dict[str, float] = {}

    # -------- system stats --------
    def _rss_mb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            p = psutil.Process()
            return float(p.memory_info().rss / (1024 * 1024))
        except Exception:
            return None

    def _avail_mb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            vm = psutil.virtual_memory()
            return float(vm.available / (1024 * 1024))
        except Exception:
            return None

    def _memory_pressure(self) -> float:
        rss = self._rss_mb()
        avail = self._avail_mb()
        if rss is None or avail is None or avail <= 1:
            return 1.0 + max(0.0, (len(self.mem.items) - 128) / 128.0)
        limit = 0.5 * avail  # 50% of available
        if rss <= limit:
            return 1.0
        return 1.0 + min(5.0, (rss - limit) / max(1.0, limit))

    # -------- predictor --------
    def _ensure_pred(self, obs_dim: int):
        if self.pred is None:
            self.pred = SuccessPredictor(obs_dim, hid=128, emb_dim=64).to(self.device)
            self.pred_opt = optim.Adam(self.pred.parameters(), lr=3e-4)

    def _train_pred_step(self, steps: int = 1, batch_size: int = 64):
        if self.pred is None or self.pred_opt is None:
            return
        if len(self.pred_x) < 128:
            return
        for _ in range(int(steps)):
            idx = self.rng.choice(len(self.pred_x), size=min(batch_size, len(self.pred_x)), replace=False)
            xb = torch.tensor(np.array([self.pred_x[i] for i in idx]), dtype=torch.float32, device=self.device)
            yb = torch.tensor(np.array([self.pred_y[i] for i in idx]), dtype=torch.float32, device=self.device)
            p, _ = self.pred(xb)
            loss = nn.BCELoss()(p.clamp(1e-4, 1.0 - 1e-4), yb)
            self.pred_opt.zero_grad()
            loss.backward()
            self.pred_opt.step()

    def _predict_p0_and_key(self, obs0: np.ndarray) -> Tuple[float, torch.Tensor]:
        self._ensure_pred(int(obs0.shape[-1]))
        x = torch.tensor(obs0[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p, emb = self.pred(x)
        return float(p.item()), emb[0].detach().cpu()

    # -------- analyzer --------
    def _ensure_analyzer(self, onehot: np.ndarray):
        if self.analyzer is not None:
            return
        assert onehot is not None and onehot.ndim == 3, f"Need onehot [H,W,C], got {None if onehot is None else onehot.shape}"
        _, _, C = onehot.shape
        extras_dim = 32

        self.analyzer = Analyzer3DCNN(
            c_in=int(C),
            t=self.obs_stack_t,
            extras_dim=extras_dim,
            hid=128,
            att_d=128,
        ).to(self.device)
        self.analyzer_opt = optim.Adam(self.analyzer.parameters(), lr=self.analyzer_lr)

        self.var_factory = VariableFactory(latent_dim=128, max_vars=self.max_auto_vars, seed=self.seed)

    def _extras_vec(self, *, avgJ: float, p0: float, think_ratio: float) -> np.ndarray:
        x = np.zeros((32,), dtype=np.float32)
        x[0:5] = self.w.astype(np.float32)
        x[5] = float(avgJ)
        x[6] = float(p0)
        x[7] = float(self.gov.mc_eval_eps)
        x[8] = float(self.gov.mc_candidates)
        x[9] = float(self.gov.trigger_percentile)
        x[10] = float(self.gov.explore_eps)
        x[11] = float(think_ratio)
        x[12] = float(self._throttle_left > 0)
        return x

    def _analyzer_forward(self, frames_stack: np.ndarray, extras: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_analyzer(frames_stack[-1])
        frames_t = torch.tensor(frames_stack[None, ...], dtype=torch.float32, device=self.device)
        extras_t = torch.tensor(extras[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y, latent = self.analyzer(frames_t, extras_t)
        return y.detach().cpu().numpy()[0], latent.detach().cpu().numpy()[0]

    def _train_analyzer_step(self, batch: int = 32, steps: int = 1):
        if self.analyzer is None or self.analyzer_opt is None:
            return
        if len(self.buf_frames) < 256:
            return
        for _ in range(int(steps)):
            idx = self.rng.choice(len(self.buf_frames), size=min(int(batch), len(self.buf_frames)), replace=False)
            frames = np.stack([self.buf_frames[i] for i in idx], axis=0).astype(np.float32)
            extras = np.stack([self.buf_extras[i] for i in idx], axis=0).astype(np.float32)
            targets = np.stack([self.buf_targets[i] for i in idx], axis=0).astype(np.float32)

            frames_t = torch.tensor(frames, dtype=torch.float32, device=self.device)
            extras_t = torch.tensor(extras, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

            pred, _ = self.analyzer(frames_t, extras_t)
            loss = ((pred - targets_t) ** 2).mean()

            self.analyzer_opt.zero_grad()
            loss.backward()
            self.analyzer_opt.step()

    # -------- action predicates ("anything it touches") --------
    @staticmethod
    def _action_predicates(actions: List[int]) -> Dict[str, bool]:
        if not actions:
            return {"ACT_NONE": True}
        a = np.array(actions, dtype=np.int32)
        out: Dict[str, bool] = {}

        counts = np.bincount(a, minlength=8)
        total = max(1, int(counts.sum()))
        frac = counts / float(total)

        top = int(np.argmax(counts))
        out[f"ACT_TOP_{top}"] = True

        max_run = 1
        run = 1
        for i in range(1, len(a)):
            if a[i] == a[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        out["ACT_REPEAT_BURST"] = bool(max_run >= 4)

        uniq = int(np.sum(counts > 0))
        out["ACT_LOW_DIVERSITY"] = bool(uniq <= 2)

        for idx in range(min(len(frac), 8)):
            if frac[idx] > 0.45:
                out[f"ACT_DOMINANT_{idx}"] = True

        if len(a) >= 2:
            for i in range(min(8, len(a) - 1)):
                out[f"SEQ2_{a[i]}_{a[i+1]}"] = True

        return out

    # -------- explicit bad-components (anti-goals) --------
    def _bad_key(self, lit_name: str) -> Tuple[float, float]:
        a = float(self.bad_alpha.get(lit_name, 1.0))
        b = float(self.bad_beta.get(lit_name, 1.0))
        return a, b

    def _bad_trust(self, lit_name: str) -> float:
        a, b = self._bad_key(lit_name)
        return float(a / (a + b))

    def _bad_update(self, lit_name: str, *, bad_occurred: bool, disprove_k: float):
        a = float(self.bad_alpha.get(lit_name, 1.0))
        b = float(self.bad_beta.get(lit_name, 1.0))
        if bad_occurred:
            a += 1.0
        else:
            b += float(disprove_k)
        self.bad_alpha[lit_name] = a
        self.bad_beta[lit_name] = b

    def _bad_decay(self, base: float, pressure: float):
        b = float(base) ** float(pressure)
        for k in list(self.bad_alpha.keys()):
            self.bad_alpha[k] = 1.0 + (float(self.bad_alpha[k]) - 1.0) * b
            self.bad_beta[k] = 1.0 + (float(self.bad_beta[k]) - 1.0) * b
            if self._bad_trust(k) < 0.10 and (self.bad_alpha[k] + self.bad_beta[k]) < 6.0:
                self.bad_alpha.pop(k, None)
                self.bad_beta.pop(k, None)

    def _infer_bad_components_from_rules(self, preds0: Dict[str, bool], success: bool):
        """
        If a strong rule predicts SUCCESS and success happened:
          the complement of its antecedent is treated as a "bad component" (anti-goal),
          because when antecedent is false, success is less likely.
        If a strong rule predicts FAIL and fail happened:
          the antecedent itself is a "bad component".
        We only do this for literal rules of the form "X IMPLIES SUCCESS/FAIL" or "NOT X ...".
        """
        # scan a small number of top fired rules (cheap)
        top = []
        for r in self.rule_engine.rules:
            if r.support < max(10, self.rule_min_support // 2):
                continue
            if r.trust() < max(0.60, self.rule_min_trust):
                continue
            try:
                fired = bool(r.fn(preds0))
            except Exception:
                continue
            if fired:
                top.append(r)
        top.sort(key=lambda rr: (rr.trust(), rr.support), reverse=True)
        top = top[:6]

        for r in top:
            # Only parse the simplest forms for faithful behavior (no liberties)
            # Examples:
            #  "HAS_KEY IMPLIES SUCCESS"
            #  "NOT HAS_KEY IMPLIES FAIL"
            expr = r.expr
            if " IMPLIES " not in expr:
                continue
            antecedent, cons = expr.split(" IMPLIES ", 1)
            antecedent = antecedent.strip()
            cons = cons.strip()

            # define "bad literal" string key
            if cons == "SUCCESS" and bool(success):
                # complement is bad: if antecedent is "X" then "NOT X" bad; if "NOT X" then "X" bad
                if antecedent.startswith("NOT "):
                    bad_lit = antecedent.replace("NOT ", "", 1)
                else:
                    bad_lit = f"NOT {antecedent}"
                self._bad_update(bad_lit, bad_occurred=True, disprove_k=1.5)

            if cons == "FAIL" and (not bool(success)):
                # antecedent itself is bad (as written)
                bad_lit = antecedent
                self._bad_update(bad_lit, bad_occurred=True, disprove_k=1.5)

    # -------- MC env --------
    def _reseed_mc(self):
        s = self.mc_seed_pool[self.mc_seed_ctr % len(self.mc_seed_pool)]
        self.mc_seed_ctr += 1
        try:
            if self.mc_env is not None:
                self.mc_env.close()
        except Exception:
            pass
        self.mc_env = make_vec_env(self.env_id, s, n_envs=1, use_shaped_reward=True, initial_w=self.w)
        self.mc_env.env_method("set_w", self.w)

    def _mc_score(self, cand_w: np.ndarray) -> float:
        assert getattr(self.mc_env, "num_envs", 1) == 1
        self.mc_env.env_method("set_w", cand_w)
        successes = 0
        for _ in range(int(self.gov.mc_eval_eps)):
            obs = self.mc_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.mc_env.step(action)
                done = bool(dones[0])
                if done and infos[0].get("is_success", False):
                    successes += 1
        return successes / float(max(1, int(self.gov.mc_eval_eps)))

    # -------- think ratio / throttling --------
    def _consume_env_time(self) -> float:
        # sum env_time across vec envs (each returns a float)
        try:
            times = self.training_env.env_method("pop_env_time_acc")
            return float(np.sum(np.array(times, dtype=np.float32)))
        except Exception:
            return 0.0

    def _think_ratio(self) -> float:
        denom = float(self._meta_time_acc + self._env_time_acc)
        if denom <= 1e-8:
            return 0.0
        return float(self._meta_time_acc / denom)

    def _apply_throttle_if_needed(self):
        tr = self._think_ratio()
        if tr > float(self.gov.think_ratio_limit):
            self._throttle_left = max(self._throttle_left, int(self.gov.throttle_episodes))
            self.gov.mc_candidates = max(4, int(self.gov.mc_candidates * 0.6))
            self.gov.mc_eval_eps = max(1, int(self.gov.mc_eval_eps * 0.7))
            self.gov.trigger_percentile = min(25.0, float(self.gov.trigger_percentile) + 2.0)
            self.gov.explore_eps = max(0.0005, float(self.gov.explore_eps) * 0.9)

    # -------- candidate proposals (w) --------
    def _propose_candidates(
        self,
        current_w: np.ndarray,
        recall_mix_w: Optional[torch.Tensor],
        feat_dir: np.ndarray,
        rule_delta: np.ndarray,
        rng: np.random.RandomState,
    ) -> List[np.ndarray]:
        cands: List[np.ndarray] = [current_w.copy()]

        if recall_mix_w is not None:
            w = recall_mix_w.detach().cpu().numpy().astype(np.float32)
            cands.append(w.copy())
            cands.append((w + 0.10 * rng.randn(W_DIM)).astype(np.float32))

        # random local
        for _ in range(max(0, int(self.gov.mc_candidates) - 6)):
            cands.append((current_w + 0.20 * rng.randn(W_DIM)).astype(np.float32))

        # feature direction (normalized)
        if np.any(feat_dir != 0):
            d = feat_dir.astype(np.float32)
            d = d / (float(np.linalg.norm(d)) + 1e-8)
            cands.append((current_w + 0.35 * d).astype(np.float32))
            cands.append((current_w - 0.35 * d).astype(np.float32))

        # rule delta injection
        if np.any(rule_delta != 0):
            cands.insert(1, np.clip(current_w + rule_delta, -2.0, 2.0).astype(np.float32))
            cands.insert(2, np.clip(current_w - rule_delta, -2.0, 2.0).astype(np.float32))

        # occasional wide exploration
        if rng.rand() < float(self.gov.explore_eps):
            cands.append(rng.uniform(-2.0, 2.0, size=(W_DIM,)).astype(np.float32))

        # clip / constraints
        out: List[np.ndarray] = []
        for w in cands:
            w2 = np.clip(w, -2.0, 2.0).astype(np.float32)
            w2[0] = max(0.0, float(w2[0]))            # keep env reward weight non-negative
            w2[1] = float(np.clip(w2[1], -0.5, 1.5))  # exploration proxy bounded
            out.append(w2)

        uniq: List[np.ndarray] = []
        for w in out:
            if not any(np.allclose(w, u, atol=1e-3) for u in uniq):
                uniq.append(w)
        return uniq

    # -------- trigger logic --------
    def _push_p0_hist(self, p0: float):
        if len(self.p0_hist) < 200:
            self.p0_hist.append(p0)
        else:
            self.p0_hist.pop(0)
            self.p0_hist.append(p0)

    def _maybe_trigger(
        self,
        *,
        obs0_flat: np.ndarray,
        key: torch.Tensor,
        p0: float,
        avgJ: float,
        frames_stack: Optional[np.ndarray],
        preds0_full: Dict[str, bool],
        pressure: float,
    ):
        if self.episode_count < self.warmup_episodes:
            self._push_p0_hist(p0)
            return
        if len(self.pred_y) < self.min_pred_samples:
            self._push_p0_hist(p0)
            return
        if self._throttle_left > 0:
            self._throttle_left -= 1
            self._push_p0_hist(p0)
            return
        if self.cooldown > 0:
            self.cooldown -= 1
            self._push_p0_hist(p0)
            return
        if len(self.p0_hist) < 50:
            self._push_p0_hist(p0)
            return

        thr = float(np.percentile(self.p0_hist, float(self.gov.trigger_percentile)))
        if not (p0 < thr):
            self._push_p0_hist(p0)
            return

        meta_t0 = time.time()

        # reseed MC
        self._reseed_mc()

        # memory recall (softmax)
        recall_mix, _ = self.mem.recall_softmax(key, topk=16, temperature=0.35, step=self.episode_count)

        # feature stats direction
        feat_dir = self.feature_stats.direction()

        # analyzer forward (creates latent vars; provides "judge" signal)
        y_pred = np.zeros((2,), dtype=np.float32)
        latent = None
        extras = self._extras_vec(avgJ=avgJ, p0=p0, think_ratio=self._think_ratio())
        if frames_stack is not None:
            y_pred, latent = self._analyzer_forward(frames_stack, extras)
            if self.var_factory is not None and latent is not None:
                self.var_factory.maybe_create_from_latent(latent, create_rate=self.auto_var_create_rate, step=self.episode_count)
                auto_preds = self.var_factory.predicates(latent, min_trust=0.60, min_support=10, cov_lo=0.10, cov_hi=0.90)
                preds0_full = dict(preds0_full)
                preds0_full.update(auto_preds)

        # ensure rule engine knows predicates (base + actions + auto vars)
        self.rule_engine.ensure_predicates(list(preds0_full.keys()))

        # decay rules under pressure
        self.rule_engine.decay_all(base=self.rule_decay_base, pressure=pressure)
        self._bad_decay(base=0.995, pressure=pressure)

        # rule-guided delta (validated)
        rule_delta, rule_expl = self.rule_engine.propose_delta_w(
            preds0_full,
            min_support=self.rule_min_support,
            min_trust=self.rule_min_trust,
            cov_lo=self.rule_cov_lo,
            cov_hi=self.rule_cov_hi,
            topk=min(self.rule_topk, max(1, int(self.rule_topk * (0.6 if pressure > 1.5 else 1.0)))),
            novelty_lambda=self.rule_novelty_lambda,
            eta=0.35,
            clip_norm=0.65,
        )

        # anti-goal cancellation attempts:
        # pick most trusted bad literal and force MC probe by adding rule_delta if it references a predicate
        bad_sorted = sorted(self.bad_alpha.keys(), key=lambda k: self._bad_trust(k), reverse=True)
        bad_top = bad_sorted[:2]
        # (kept cheap): just log them and let rules/MC handle via predicate-conditioned deltas

        candidates = self._propose_candidates(self.w, recall_mix, feat_dir, rule_delta, self.rng)

        # baseline score (rigorous revert)
        base_score = self._mc_score(self.w.copy())

        # MC evaluate candidates
        t0 = time.time()
        best_w = self.w.copy()
        best = float(base_score)
        for cand in candidates[: int(self.gov.mc_candidates)]:
            score = self._mc_score(cand)
            if score > best:
                best = score
                best_w = cand.copy()
        elapsed = time.time() - t0

        # adapt governor
        rss = self._rss_mb()
        self.gov.adapt(elapsed, rss)
        self.mem.decay = self.gov.mem_decay

        # accept/revert: must beat baseline by margin
        accept = (best >= base_score + float(self.mc_accept_margin))
        if accept:
            self.w = best_w
            self.training_env.env_method("set_w", self.w)
            self.mem.add(key, torch.tensor(self.w, dtype=torch.float32), strength=1.0, step=self.episode_count)

        self.cooldown = int(self.gov.trigger_cooldown)
        self._meta_time_acc += float(time.time() - meta_t0)

        # logs
        self.logger.record("ours/triggered", 1.0)
        self.logger.record("ours/mc_baseJ", float(base_score))
        self.logger.record("ours/mc_bestJ", float(best))
        self.logger.record("ours/mc_accept", float(1.0 if accept else 0.0))
        self.logger.record("ours/mc_seconds", float(elapsed))
        self.logger.record("ours/think_ratio", float(self._think_ratio()))
        self.logger.record("ours/ana_dJ_pred", float(y_pred[0]))
        self.logger.record("ours/ana_risk_pred", float(y_pred[1]))
        if rss is not None:
            self.logger.record("system/rss_mb", float(rss))
        for i in range(W_DIM):
            self.logger.record(f"ours/w{i}", float(self.w[i]))

        if self.verbose:
            msg = (
                f"[TRIGGER] ep={self.episode_count} p0={p0:.3f} thr={thr:.3f} "
                f"baseJ={base_score:.3f} bestJ={best:.3f} accept={int(accept)} w={self.w}"
            )
            if rss is not None:
                msg += f" rssMB={rss:.0f}"
            msg += f" mc_eps={self.gov.mc_eval_eps} mc_cands={self.gov.mc_candidates}"
            if rule_expl:
                top = "; ".join([f"{e} (t={t:.2f},n={n},c={c:.2f})" for (e, t, n, c) in rule_expl[:3]])
                msg += f" | RULES: {top} | dW={rule_delta}"
            if bad_top:
                bt = ", ".join([f"{b}(p={self._bad_trust(b):.2f})" for b in bad_top])
                msg += f" | BAD: {bt}"
            msg += f" | ANA=[dJ={y_pred[0]:+.3f}, risk={y_pred[1]:+.3f}]"
            print(msg)

        self._apply_throttle_if_needed()

    # -------- SB3 hooks --------
    def _on_training_start(self) -> None:
        self.training_env.env_method("set_w", self.w)

        self.eval_env = make_eval_env(self.env_id, self.seed + 88888, use_shaped_reward=True, initial_w=self.w)
        self.eval_env.env_method("set_w", self.w)

        self._reseed_mc()

        # seed p0 history
        for _ in range(10):
            obs = self.eval_env.reset()
            obs0 = np.array(obs, dtype=np.float32).squeeze(0)
            p0, _ = self._predict_p0_and_key(obs0)
            self._push_p0_hist(p0)

    def _on_training_end(self) -> None:
        try:
            if self.mc_env is not None:
                self.mc_env.close()
        except Exception:
            pass
        try:
            if self.eval_env is not None:
                self.eval_env.close()
        except Exception:
            pass

    def _on_step(self) -> bool:
        # consume env time measured in wrapper
        self._env_time_acc += self._consume_env_time()

        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        actions = self.locals.get("actions", None)
        if dones is None or infos is None:
            return True

        # collect per-step frames and actions (faithful: store every "moment", bounded by stack/buffers)
        for i, d in enumerate(dones):
            if actions is not None:
                try:
                    self._ep_actions.append(int(actions[i]))
                except Exception:
                    pass

            onehot = infos[i].get("obs_onehot", None) if isinstance(infos[i], dict) else None
            if onehot is not None:
                onehot = np.array(onehot, dtype=np.float32)
                self._frame_stack.append(onehot)

            if not bool(d):
                continue

            meta_t0 = time.time()

            self.episode_count += 1

            succ = 1 if infos[i].get("is_success", False) else 0
            feats_ep = np.array(infos[i].get("features_episode", np.zeros((W_DIM,), dtype=np.float32)), dtype=np.float32)
            self.feature_stats.update(feats_ep, bool(succ))

            self.recent_success.append(succ)
            if len(self.recent_success) > 200:
                self.recent_success.pop(0)
            avgJ = float(np.mean(self.recent_success)) if self.recent_success else 0.0

            # predictor training from true episode-start obs (captured by wrapper, not SB3 internals)
            start_obs = infos[i].get("episode_start_obs", None)
            if start_obs is not None:
                start_obs = np.array(start_obs, dtype=np.float32).reshape(-1)
                self._ensure_pred(int(len(start_obs)))
                self.pred_x.append(start_obs)
                self.pred_y.append(float(succ))
                if len(self.pred_x) > 6000:
                    self.pred_x = self.pred_x[-5000:]
                    self.pred_y = self.pred_y[-5000:]
                self._train_pred_step(steps=1, batch_size=64)

            # episode-start predicates + action predicates
            preds0 = infos[i].get("predicates_episode_start", {})
            if not isinstance(preds0, dict):
                preds0 = {}
            act_preds = self._action_predicates(self._ep_actions)
            self._ep_actions = []

            preds0_full = dict(preds0)
            preds0_full.update(act_preds)
            self._last_episode_start_preds = dict(preds0_full)

            # ensure rule engine knows all predicates
            self.rule_engine.ensure_predicates(list(preds0_full.keys()))

            # pressure
            pressure = self._memory_pressure()

            # rule decay + episode update (validated, disproval faster)
            self.rule_engine.decay_all(base=self.rule_decay_base, pressure=pressure)
            self.rule_engine.update_episode(
                preds0_full,
                bool(succ),
                feats_ep,
                disprove_k=self.rule_disprove_k,
                step=self.episode_count,
            )

            # infer bad-components from high-trust rules (your "split good/bad" idea)
            self._infer_bad_components_from_rules(preds0_full, bool(succ))

            # build frames_stack (pad)
            frames_stack = None
            if len(self._frame_stack) > 0:
                while len(self._frame_stack) < self.obs_stack_t:
                    self._frame_stack.appendleft(self._frame_stack[0])
                # only keep last T
                frames_stack = np.stack(list(self._frame_stack)[-self.obs_stack_t:], axis=0).astype(np.float32)
                # analyzer init
                if self.analyzer is None:
                    self._ensure_analyzer(frames_stack[-1])

            # analyzer targets: exponential success preference + risk
            kexp = 3.0
            global_val = math.exp(kexp * float(avgJ)) if succ == 1 else -math.exp(kexp * float(1.0 - avgJ))
            risk_val = 1.0 - float(succ)

            # p0 for next episode start: use current reset obs if provided, else fallback to current start_obs
            # (we do not depend on SB3 new_obs locals)
            obs0_flat = infos[i].get("episode_start_obs", None)
            if obs0_flat is None:
                obs0_flat = start_obs if start_obs is not None else np.zeros((1,), dtype=np.float32)
            obs0_flat = np.array(obs0_flat, dtype=np.float32).reshape(-1)

            p0, key = self._predict_p0_and_key(obs0_flat)

            # analyzer train buffer and variable validation
            if frames_stack is not None:
                think_ratio = self._think_ratio()
                extras = self._extras_vec(avgJ=avgJ, p0=p0, think_ratio=think_ratio)

                y_pred, latent = self._analyzer_forward(frames_stack, extras)

                if self.var_factory is not None:
                    self.var_factory.update(
                        latent,
                        bool(succ),
                        disprove_k=2.0,
                        decay_base=0.995,
                        pressure=pressure,
                        step=self.episode_count,
                    )

                self.buf_frames.append(frames_stack)
                self.buf_extras.append(extras)
                self.buf_targets.append(np.array([global_val, risk_val], dtype=np.float32))

                if (self.episode_count % max(1, int(self.analyzer_train_every))) == 0:
                    steps = 2 if pressure <= 1.2 else 1
                    self._train_analyzer_step(batch=32, steps=steps)

            # memory decay
            self.mem.decay_all(pressure=pressure)
            self.mem.decay = max(0.80, float(self.mem.decay) ** float(pressure))

            # trigger (MC + recall + rule reasoning + cancellation pressure)
            self._maybe_trigger(
                obs0_flat=obs0_flat,
                key=key,
                p0=p0,
                avgJ=avgJ,
                frames_stack=frames_stack,
                preds0_full=preds0_full,
                pressure=pressure,
            )

            # shrink storage if above RAM goal (faithful: >50% available => reduce)
            if pressure > 1.5:
                # buffers
                while len(self.buf_frames) > 2000:
                    self.buf_frames.popleft()
                    self.buf_extras.popleft()
                    self.buf_targets.popleft()
                # memory
                if len(self.mem.items) > 128:
                    self.mem._evict()
                # vars
                if self.var_factory is not None and len(self.var_factory.vars) > max(64, self.max_auto_vars // 2):
                    self.var_factory.max_vars = max(64, self.max_auto_vars // 2)
                    self.var_factory._evict()

            # continuous logs
            self.logger.record("ours/J_recent", float(avgJ))
            self.logger.record("ours/p0", float(p0))
            self.logger.record("ours/memory_items", float(len(self.mem.items)))
            self.logger.record("ours/rules_count", float(len(self.rule_engine.rules)))
            self.logger.record("ours/auto_vars", float(len(self.var_factory.vars) if self.var_factory else 0))
            self.logger.record("ours/think_ratio", float(self._think_ratio()))
            for k in range(W_DIM):
                self.logger.record(f"ours/w{k}", float(self.w[k]))
            self.logger.record("ours/triggered", 0.0)

            # periodic console logs
            if self.verbose and (self.episode_count % max(1, int(self.log_every_episodes)) == 0):
                rss = self._rss_mb()
                msg = (
                    f"ep={self.episode_count:5d} recentJ={avgJ:.3f} "
                    f"mem={len(self.mem.items):3d} rules={len(self.rule_engine.rules):6d} "
                    f"autoVars={(len(self.var_factory.vars) if self.var_factory else 0):4d} "
                    f"w={self.w} predN={len(self.pred_y)} thinkR={self._think_ratio():.2f}"
                )
                if rss is not None:
                    msg += f" rssMB={rss:.0f}"
                if self._throttle_left > 0:
                    msg += f" THROTTLE({self._throttle_left})"
                print(msg)

            # periodic eval
            if self.verbose and (self.episode_count % max(1, int(self.eval_every_episodes)) == 0):
                self.eval_env.env_method("set_w", self.w)
                j_eval = eval_success_rate_from_env(self.model, self.eval_env, self.eval_eps, deterministic=True)
                self.logger.record("ours/J_eval", float(j_eval))
                self.logger.dump(step=self.num_timesteps)
                print(f"[eval] ep={self.episode_count:5d} OURS J={j_eval:.3f} w={self.w}")

            self._meta_time_acc += float(time.time() - meta_t0)

        return True


# ----------------------------
# PPO builder
# ----------------------------
def make_ppo(
    env,
    *,
    seed: int,
    policy_hid: int,
    lr: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    sb3_verbose: int,
    tb_logdir: Optional[str],
):
    policy_kwargs = dict(net_arch=[int(policy_hid), int(policy_hid)])
    return PPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        learning_rate=float(lr),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=4,
        gamma=float(gamma),
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=float(ent_coef),
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=int(sb3_verbose),
        device="cpu",
        tensorboard_log=tb_logdir,
    )


# ----------------------------
# IMPORTANT FIX: wrapper order so one-hot is available for analyzer
# ----------------------------
def make_env(env_id: str, seed: int, *, use_shaped_reward: bool, initial_w: np.ndarray) -> gym.Env:
    env = gym.make(env_id)
    env = OneHotPartialObsWrapper(env)   # dict obs with one-hot image
    env = InfoCaptureWrapper(env)        # capture obs_onehot while still dict
    env = FlatObsWrapper(env)            # flatten for PPO
    env = LossShapingWrapper(env, use_shaped_reward=use_shaped_reward, initial_w=initial_w)
    env.reset(seed=seed)
    return env


# ----------------------------
# Train / baseline compare
# ----------------------------
def train_standard(args) -> float:
    w0 = np.zeros((W_DIM,), dtype=np.float32)
    train_env = make_vec_env(args.env_id, args.seed, args.n_envs, use_shaped_reward=False, initial_w=w0)
    model = make_ppo(
        train_env,
        seed=args.seed,
        policy_hid=args.policy_hid,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        sb3_verbose=args.sb3_verbose,
        tb_logdir=args.tb_logdir,
    )
    model.learn(total_timesteps=int(args.total_timesteps), tb_log_name="standard")

    eval_env = make_eval_env(args.env_id, args.seed + 2222, use_shaped_reward=False, initial_w=w0)
    j = eval_success_rate_from_env(model, eval_env, int(args.eval_eps), deterministic=True)
    eval_env.close()
    train_env.close()
    return float(j)


def train_ours(args) -> Tuple[float, np.ndarray]:
    w_init = np.array([1.0, 0.0, 0.2, 0.2, -0.05], dtype=np.float32)

    train_env = make_vec_env(args.env_id, args.seed, args.n_envs, use_shaped_reward=True, initial_w=w_init)
    model = make_ppo(
        train_env,
        seed=args.seed,
        policy_hid=args.policy_hid,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        sb3_verbose=args.sb3_verbose,
        tb_logdir=args.tb_logdir,
    )

    gov = ComputeGovernor(
        mc_eval_eps=int(args.mc_eval_eps),
        mc_candidates=int(args.mc_candidates),
        trigger_percentile=float(args.trigger_percentile),
        trigger_cooldown=int(args.trigger_cooldown),
        mem_decay=float(args.memory_decay),
        explore_eps=float(args.explore_eps),
        max_mc_seconds_per_trigger=float(args.max_mc_seconds_per_trigger),
        max_rss_mb=float(args.max_rss_mb),
        think_ratio_limit=float(args.think_ratio_limit),
        throttle_episodes=int(args.throttle_episodes),
    )

    cb = OursMetaCallback(
        env_id=args.env_id,
        seed=args.seed,
        device="cpu",
        w_init=w_init,
        gov=gov,
        eval_eps=int(args.eval_eps),
        warmup_episodes=int(args.warmup_episodes),
        min_pred_samples=int(args.min_pred_samples),
        log_every_episodes=int(args.log_every_episodes),
        eval_every_episodes=int(args.eval_every_episodes),
        obs_stack_t=int(args.obs_stack_t),
        analyzer_lr=float(args.analyzer_lr),
        analyzer_train_every=int(args.analyzer_train_every),
        analyzer_window=int(args.analyzer_window),
        max_auto_vars=int(args.max_auto_vars),
        auto_var_create_rate=float(args.auto_var_create_rate),
        rule_min_support=int(args.rule_min_support),
        rule_min_trust=float(args.rule_min_trust),
        rule_cov_lo=float(args.rule_cov_lo),
        rule_cov_hi=float(args.rule_cov_hi),
        rule_topk=int(args.rule_topk),
        rule_disprove_k=float(args.rule_disprove_k),
        rule_decay_base=float(args.rule_decay_base),
        rule_novelty_lambda=float(args.rule_novelty_lambda),
        mc_accept_margin=float(args.mc_accept_margin),
        verbose=1,
    )

    model.learn(total_timesteps=int(args.total_timesteps), tb_log_name="ours", callback=[cb])

    eval_env = make_eval_env(args.env_id, args.seed + 3333, use_shaped_reward=True, initial_w=cb.w.copy())
    eval_env.env_method("set_w", cb.w.copy())
    j = eval_success_rate_from_env(model, eval_env, int(args.eval_eps), deterministic=True)
    eval_env.close()
    train_env.close()
    return float(j), cb.w.copy()


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "ours", "compare"], default="compare")
    p.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-8x8-v0")

    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--n_envs", type=int, default=4)

    # PPO
    p.add_argument("--policy_hid", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--sb3_verbose", type=int, default=0)

    # OURS trigger
    p.add_argument("--warmup_episodes", type=int, default=200)
    p.add_argument("--min_pred_samples", type=int, default=500)
    p.add_argument("--trigger_percentile", type=float, default=10.0)
    p.add_argument("--trigger_cooldown", type=int, default=10)

    # MC budgets
    p.add_argument("--mc_candidates", type=int, default=16)
    p.add_argument("--mc_eval_eps", type=int, default=6)
    p.add_argument("--mc_accept_margin", type=float, default=0.02)

    # memory/explore
    p.add_argument("--memory_decay", type=float, default=0.93)
    p.add_argument("--explore_eps", type=float, default=0.002)

    # compute governor
    p.add_argument("--max_mc_seconds_per_trigger", type=float, default=2.0)
    p.add_argument("--max_rss_mb", type=float, default=2500.0)
    p.add_argument("--think_ratio_limit", type=float, default=0.50)
    p.add_argument("--throttle_episodes", type=int, default=10)

    # analyzer
    p.add_argument("--obs_stack_t", type=int, default=8)
    p.add_argument("--analyzer_lr", type=float, default=3e-4)
    p.add_argument("--analyzer_train_every", type=int, default=5)
    p.add_argument("--analyzer_window", type=int, default=8000)
    p.add_argument("--max_auto_vars", type=int, default=256)
    p.add_argument("--auto_var_create_rate", type=float, default=0.02)

    # rules
    p.add_argument("--rule_min_support", type=int, default=25)
    p.add_argument("--rule_min_trust", type=float, default=0.68)
    p.add_argument("--rule_cov_lo", type=float, default=0.10)
    p.add_argument("--rule_cov_hi", type=float, default=0.90)
    p.add_argument("--rule_topk", type=int, default=8)
    p.add_argument("--rule_disprove_k", type=float, default=2.0)
    p.add_argument("--rule_decay_base", type=float, default=0.995)
    p.add_argument("--rule_novelty_lambda", type=float, default=0.35)

    # eval/logging
    p.add_argument("--eval_eps", type=int, default=50)
    p.add_argument("--log_every_episodes", type=int, default=50)
    p.add_argument("--eval_every_episodes", type=int, default=200)

    # tensorboard
    p.add_argument("--tb_logdir", type=str, default=None)

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_all_seeds(args.seed)

    # env check
    _ = gym.make(args.env_id)
    _.close()

    if args.mode == "standard":
        j = train_standard(args)
        print(f"FINAL STANDARD J={j:.3f}")
        return

    if args.mode == "ours":
        j, w = train_ours(args)
        print(f"FINAL OURS J={j:.3f} w={w}")
        return

    print("=== TRAIN STANDARD (SB3 PPO) ===")
    j_std = train_standard(args)
    print(f"[RESULT] STANDARD J={j_std:.3f}")

    print("\n=== TRAIN OURS (SB3 PPO + Faithful Loss-Shaper) ===")
    j_ours, w = train_ours(args)
    print(f"[RESULT] OURS J={j_ours:.3f} w={w}")

    print("\n=== SUMMARY ===")
    print(f"env_id={args.env_id} timesteps={args.total_timesteps} n_envs={args.n_envs}")
    print(f"STANDARD J={j_std:.3f}")
    print(f"OURS     J={j_ours:.3f} w={w}")


if __name__ == "__main__":
    main()


