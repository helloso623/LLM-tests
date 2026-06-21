# agi_minigrid_loss_shaper_final.py
# Faithful, end-to-end runnable, patched final version.

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    import psutil  # optional
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
# Memory (associative, decaying)
# ----------------------------
W_DIM = 5


@dataclass
class MemoryItem:
    key: torch.Tensor
    w: torch.Tensor
    strength: float


class HypothesisMemory:
    def __init__(self, max_items: int = 256, decay: float = 0.93):
        self.max_items = max_items
        self.decay = decay
        self.items: List[MemoryItem] = []

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2) + 1e-8)

    def add_or_strengthen(self, key: torch.Tensor, w: torch.Tensor, add_strength: float = 1.0):
        key_n = self._norm(key.detach().cpu())
        w_cpu = w.detach().cpu()

        best_i, best_sim = None, -1.0
        for i, it in enumerate(self.items):
            sim = float(torch.dot(key_n, it.key))
            if sim > best_sim:
                best_sim, best_i = sim, i

        if best_i is not None and best_sim > 0.985 and torch.allclose(self.items[best_i].w, w_cpu, atol=1e-3):
            self.items[best_i].strength += add_strength
        else:
            self.items.append(MemoryItem(key=key_n, w=w_cpu, strength=add_strength))
            if len(self.items) > self.max_items:
                self.items.sort(key=lambda it: it.strength, reverse=True)
                self.items = self.items[: self.max_items]

    def retrieve(self, key: torch.Tensor, topk: int = 5) -> List[Tuple[float, torch.Tensor]]:
        if not self.items:
            return []
        key_n = self._norm(key.detach().cpu())
        scored = []
        for it in self.items:
            sim = float(torch.dot(key_n, it.key))
            scored.append((sim * it.strength, it.w))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topk]

    def decay_all(self, pressure: float = 1.0):
        d = self.decay ** pressure
        for it in self.items:
            it.strength *= d
        self.items = [it for it in self.items if it.strength > 0.05]


# ----------------------------
# Environment wrapper: compute features + shaped reward (w·f)
# IMPORTANT: w is internal (not appended to obs). This keeps SB3 stable.
# ----------------------------
class LossShapingWrapper(gym.Wrapper):
    """
    Per-step features f:
      f0 = env reward (sparse success proxy)
      f1 = new cell visited (exploration proxy)
      f2 = picked key (progress proxy)
      f3 = doors opened delta (progress proxy)
      f4 = step cost (always 1)

    Local reward = w·f
    Also accumulates episode feature sum: info["features_episode"] at episode end.
    """

    def __init__(self, env: gym.Env, *, use_shaped_reward: bool, initial_w: np.ndarray):
        super().__init__(env)
        self.use_shaped_reward = use_shaped_reward
        self._w = np.array(initial_w, dtype=np.float32).copy()
        assert self._w.shape == (W_DIM,)

        self._visited = set()
        self._prev_carry_key = False

        self._door_cells: List[Tuple[int, int]] = []
        self._prev_open_doors = 0

        self._ep_feat_sum = np.zeros((W_DIM,), dtype=np.float32)

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

        info["features"] = np.zeros((W_DIM,), dtype=np.float32)
        info["features_episode"] = np.zeros((W_DIM,), dtype=np.float32)
        info["is_success"] = False
        return obs, info

    def step(self, action):
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

        info["is_success"] = is_success
        info["features"] = f
        if terminated or truncated:
            info["features_episode"] = self._ep_feat_sum.copy()

        self._prev_carry_key = now_carry_key
        self._prev_open_doors = now_open_doors

        if self.use_shaped_reward:
            return obs, shaped, terminated, truncated, info
        else:
            return obs, float(reward), terminated, truncated, info


def make_env(env_id: str, seed: int, *, use_shaped_reward: bool, initial_w: np.ndarray) -> gym.Env:
    env = gym.make(env_id)
    env = OneHotPartialObsWrapper(env)
    env = FlatObsWrapper(env)
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
    assert getattr(vec_env, "num_envs", 1) == 1, "eval_success_rate_from_env requires num_envs == 1"
    successes = 0
    for _ in range(episodes):
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
# Guided hypothesis proposal
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

    def update(self, features_episode: np.ndarray, success: bool):
        if success:
            self.succ_sum += features_episode
            self.succ_n += 1
        else:
            self.fail_sum += features_episode
            self.fail_n += 1

    def direction(self) -> np.ndarray:
        if self.succ_n < 5 or self.fail_n < 5:
            return np.zeros((W_DIM,), dtype=np.float32)
        succ_mean = self.succ_sum / max(1, self.succ_n)
        fail_mean = self.fail_sum / max(1, self.fail_n)
        return (succ_mean - fail_mean).astype(np.float32)


def propose_candidates(
    current_w: np.ndarray,
    retrieved: List[Tuple[float, torch.Tensor]],
    *,
    feat_dir: np.ndarray,
    n_rand: int,
    explore_eps: float,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    cands: List[np.ndarray] = [current_w.copy()]

    for _, wt in retrieved:
        w = wt.detach().cpu().numpy().astype(np.float32)
        cands.append(w.copy())
        cands.append((w + 0.10 * rng.randn(W_DIM)).astype(np.float32))

    for _ in range(n_rand):
        cands.append((current_w + 0.20 * rng.randn(W_DIM)).astype(np.float32))

    if np.any(feat_dir != 0):
        step = 0.35
        cands.append((current_w + step * feat_dir).astype(np.float32))
        cands.append((current_w - step * feat_dir).astype(np.float32))

    if rng.rand() < explore_eps:
        cands.append(rng.uniform(-2.0, 2.0, size=(W_DIM,)).astype(np.float32))

    out: List[np.ndarray] = []
    for w in cands:
        w2 = np.clip(w, -2.0, 2.0).astype(np.float32)
        w2[0] = max(0.0, float(w2[0]))
        w2[1] = float(np.clip(w2[1], -0.5, 1.5))
        out.append(w2)

    uniq: List[np.ndarray] = []
    for w in out:
        if not any(np.allclose(w, u, atol=1e-3) for u in uniq):
            uniq.append(w)
    return uniq


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

    def adapt(self, elapsed_sec: float, rss_mb: Optional[float]) -> None:
        if elapsed_sec > self.max_mc_seconds_per_trigger:
            self.mc_eval_eps = max(1, int(self.mc_eval_eps * 0.6))
            self.mc_candidates = max(4, int(self.mc_candidates * 0.7))
            self.trigger_percentile = min(25.0, self.trigger_percentile + 2.0)
            self.explore_eps = max(0.0005, self.explore_eps * 0.85)

        if rss_mb is not None and rss_mb > self.max_rss_mb:
            self.mem_decay = max(0.80, self.mem_decay - 0.03)
            self.mc_candidates = max(4, int(self.mc_candidates * 0.7))
            self.mc_eval_eps = max(1, int(self.mc_eval_eps * 0.7))


# ----------------------------
# Episode-start observation callback (for predictor training)
# ----------------------------
class EpisodeStartObsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.last_reset_obs: Dict[int, np.ndarray] = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        new_obs = self.locals.get("new_obs", None)
        if dones is None or infos is None or new_obs is None:
            return True

        for i, d in enumerate(dones):
            if bool(d):
                if i in self.last_reset_obs:
                    infos[i]["episode_start_obs"] = self.last_reset_obs[i]
                self.last_reset_obs[i] = np.array(new_obs[i], dtype=np.float32)
            else:
                if i not in self.last_reset_obs:
                    self.last_reset_obs[i] = np.array(new_obs[i], dtype=np.float32)
        return True


# ----------------------------
# OURS meta-callback
# ----------------------------
class OursMetaCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_id: str,
        device: str,
        seed: int,
        w_init: np.ndarray,
        gov: ComputeGovernor,
        eval_eps: int,
        log_every_episodes: int,
        eval_every_episodes: int,
        warmup_episodes: int,
        min_pred_samples: int,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.env_id = env_id
        self.device = device
        self.seed = seed

        self.w = np.array(w_init, dtype=np.float32).copy()
        self.gov = gov
        self.eval_eps = eval_eps
        self.log_every_episodes = log_every_episodes
        self.eval_every_episodes = eval_every_episodes
        self.warmup_episodes = warmup_episodes
        self.min_pred_samples = min_pred_samples

        self.mem = HypothesisMemory(max_items=256, decay=self.gov.mem_decay)
        self.feature_stats = FeatureStats.init()

        self.episode_count = 0
        self.recent_success: List[int] = []
        self.p0_hist: List[float] = []
        self.cooldown = 0

        self.pred: Optional[SuccessPredictor] = None
        self.pred_opt: Optional[optim.Optimizer] = None
        self.pred_x: List[np.ndarray] = []
        self.pred_y: List[float] = []

        self.mc_env = None
        self.eval_env = None

        self.rng = np.random.RandomState(seed + 1337)
        self.mc_seed_ctr = 0
        self.mc_seed_pool = [seed + 70001, seed + 70003, seed + 70007, seed + 70009]

    def _rss_mb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            p = psutil.Process()
            return p.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def _ensure_pred(self, obs_dim: int):
        if self.pred is None:
            self.pred = SuccessPredictor(obs_dim, hid=128, emb_dim=64).to(self.device)
            self.pred_opt = optim.Adam(self.pred.parameters(), lr=3e-4)

    def _train_pred_step(self, steps: int = 1, batch_size: int = 64):
        if self.pred is None or self.pred_opt is None:
            return
        if len(self.pred_x) < 128:
            return
        for _ in range(steps):
            idx = self.rng.choice(len(self.pred_x), size=min(batch_size, len(self.pred_x)), replace=False)
            xb = torch.tensor(np.array([self.pred_x[i] for i in idx]), dtype=torch.float32, device=self.device)
            yb = torch.tensor(np.array([self.pred_y[i] for i in idx]), dtype=torch.float32, device=self.device)
            p, _ = self.pred(xb)
            loss = nn.BCELoss()(p.clamp(1e-4, 1 - 1e-4), yb)
            self.pred_opt.zero_grad()
            loss.backward()
            self.pred_opt.step()

    def _predict_p0_and_key(self, obs0: np.ndarray) -> Tuple[float, torch.Tensor]:
        self._ensure_pred(obs0.shape[-1])
        x = torch.tensor(obs0[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p, emb = self.pred(x)
        return float(p.item()), emb[0].detach().cpu()

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
        for _ in range(self.gov.mc_eval_eps):
            obs = self.mc_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.mc_env.step(action)
                done = bool(dones[0])
                if done and infos[0].get("is_success", False):
                    successes += 1
        return successes / float(self.gov.mc_eval_eps)

    def _push_p0_hist(self, p0: float):
        if len(self.p0_hist) < 200:
            self.p0_hist.append(p0)
        else:
            self.p0_hist.pop(0)
            self.p0_hist.append(p0)

    def _maybe_trigger(self, obs0: np.ndarray, key: torch.Tensor, p0: float):
        if self.episode_count < self.warmup_episodes:
            return

        # Don’t allow triggers until predictor has enough labeled episodes
        if len(self.pred_y) < self.min_pred_samples:
            self._push_p0_hist(p0)
            return

        if self.cooldown > 0:
            self.cooldown -= 1
            self._push_p0_hist(p0)
            return

        # bootstrap history
        if len(self.p0_hist) < 50:
            self._push_p0_hist(p0)
            return

        thr = float(np.percentile(self.p0_hist, self.gov.trigger_percentile))
        triggered = p0 < thr

        if not triggered:
            self._push_p0_hist(p0)
            return

        # triggered path: rebuild mc_env to reseed robustly
        self._reseed_mc()

        retrieved = self.mem.retrieve(key, topk=5)
        feat_dir = self.feature_stats.direction()

        candidates = propose_candidates(
            self.w,
            retrieved,
            feat_dir=feat_dir,
            n_rand=max(4, self.gov.mc_candidates - len(retrieved) * 2),
            explore_eps=self.gov.explore_eps,
            rng=self.rng,
        )

        t0 = time.time()
        best_w = self.w.copy()
        best = -1.0
        for cand in candidates[: self.gov.mc_candidates]:
            score = self._mc_score(cand)
            if score > best:
                best = score
                best_w = cand.copy()
        elapsed = time.time() - t0

        rss = self._rss_mb()
        self.gov.adapt(elapsed, rss)
        self.mem.decay = self.gov.mem_decay

        self.w = best_w
        self.training_env.env_method("set_w", self.w)
        self.mem.add_or_strengthen(key, torch.tensor(self.w, dtype=torch.float32), add_strength=1.0)
        self.cooldown = self.gov.trigger_cooldown

        # tensorboard logs
        self.logger.record("ours/triggered", 1.0)
        self.logger.record("ours/mc_bestJ", float(best))
        self.logger.record("ours/mc_seconds", float(elapsed))
        self.logger.record("ours/mc_eval_eps", float(self.gov.mc_eval_eps))
        self.logger.record("ours/mc_candidates", float(self.gov.mc_candidates))
        self.logger.record("ours/trigger_percentile", float(self.gov.trigger_percentile))
        if rss is not None:
            self.logger.record("system/rss_mb", float(rss))
        for i in range(W_DIM):
            self.logger.record(f"ours/w{i}", float(self.w[i]))

        if self.verbose:
            msg = f"[TRIGGER] ep={self.episode_count} p0={p0:.3f} thr={thr:.3f} mcJ={best:.3f} w={self.w}"
            if rss is not None:
                msg += f" rssMB={rss:.0f}"
            msg += f" mc_eps={self.gov.mc_eval_eps} mc_cands={self.gov.mc_candidates}"
            print(msg)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_w", self.w)

        # create eval env (1 env)
        self.eval_env = make_eval_env(self.env_id, self.seed + 88888, use_shaped_reward=True, initial_w=self.w)
        self.eval_env.env_method("set_w", self.w)

        # initial MC env
        self._reseed_mc()

        # seed history a bit
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
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        new_obs = self.locals.get("new_obs", None)
        if dones is None or infos is None or new_obs is None:
            return True

        for i, d in enumerate(dones):
            if not bool(d):
                continue

            self.episode_count += 1

            succ = 1 if infos[i].get("is_success", False) else 0
            feats_ep = np.array(infos[i].get("features_episode", np.zeros((W_DIM,), dtype=np.float32)), dtype=np.float32)
            self.feature_stats.update(feats_ep, bool(succ))

            self.recent_success.append(succ)
            if len(self.recent_success) > 200:
                self.recent_success.pop(0)
            avgJ = float(np.mean(self.recent_success)) if self.recent_success else 0.0

            start_obs = infos[i].get("episode_start_obs", None)
            if start_obs is not None:
                self._ensure_pred(int(len(start_obs)))
                self.pred_x.append(np.array(start_obs, dtype=np.float32))
                self.pred_y.append(float(succ))
                if len(self.pred_x) > 6000:
                    self.pred_x = self.pred_x[-5000:]
                    self.pred_y = self.pred_y[-5000:]
                self._train_pred_step(steps=1, batch_size=64)

            # Trigger check uses next episode start observation (new_obs on done is reset obs)
            obs0 = np.array(new_obs[i], dtype=np.float32)
            p0, key = self._predict_p0_and_key(obs0)

            # tensorboard continuous logs
            self.logger.record("ours/J_recent", float(avgJ))
            self.logger.record("ours/p0", float(p0))
            self.logger.record("ours/triggered", 0.0)
            self.logger.record("ours/memory_items", float(len(self.mem.items)))
            for k in range(W_DIM):
                self.logger.record(f"ours/w{k}", float(self.w[k]))

            self._maybe_trigger(obs0, key, p0)

            # memory pressure decay
            pressure = 1.0 + max(0.0, (len(self.mem.items) - 128) / 128)
            self.mem.decay_all(pressure=pressure)

            if self.verbose and (self.episode_count % self.log_every_episodes == 0):
                rss = self._rss_mb()
                msg = f"ep={self.episode_count:5d} recentJ={avgJ:.3f} mem={len(self.mem.items):3d} w={self.w} predN={len(self.pred_y)}"
                if rss is not None:
                    msg += f" rssMB={rss:.0f}"
                print(msg)

            if self.verbose and (self.episode_count % self.eval_every_episodes == 0):
                self.eval_env.env_method("set_w", self.w)
                j_eval = eval_success_rate_from_env(self.model, self.eval_env, self.eval_eps, deterministic=True)
                self.logger.record("ours/J_eval", float(j_eval))
                self.logger.dump(step=self.num_timesteps)
                print(f"[eval] ep={self.episode_count:5d} OURS J={j_eval:.3f} w={self.w}")

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
    policy_kwargs = dict(net_arch=[policy_hid, policy_hid])
    return PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=sb3_verbose,
        device="cpu",
        tensorboard_log=tb_logdir,
    )


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
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="standard", callback=[EpisodeStartObsCallback()])

    eval_env = make_eval_env(args.env_id, args.seed + 2222, use_shaped_reward=False, initial_w=w0)
    j = eval_success_rate_from_env(model, eval_env, args.eval_eps, deterministic=True)
    eval_env.close()

    train_env.close()
    return j


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
        mc_eval_eps=args.mc_eval_eps,
        mc_candidates=args.mc_candidates,
        trigger_percentile=args.trigger_percentile,
        trigger_cooldown=args.trigger_cooldown,
        mem_decay=args.memory_decay,
        explore_eps=args.explore_eps,
        max_mc_seconds_per_trigger=args.max_mc_seconds_per_trigger,
        max_rss_mb=args.max_rss_mb,
    )

    cb = OursMetaCallback(
        env_id=args.env_id,
        device="cpu",
        seed=args.seed,
        w_init=w_init,
        gov=gov,
        eval_eps=args.eval_eps,
        log_every_episodes=args.log_every_episodes,
        eval_every_episodes=args.eval_every_episodes,
        warmup_episodes=args.warmup_episodes,
        min_pred_samples=args.min_pred_samples,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_timesteps, tb_log_name="ours", callback=[EpisodeStartObsCallback(), cb])

    eval_env = make_eval_env(args.env_id, args.seed + 3333, use_shaped_reward=True, initial_w=cb.w.copy())
    eval_env.env_method("set_w", cb.w.copy())
    j = eval_success_rate_from_env(model, eval_env, args.eval_eps, deterministic=True)
    eval_env.close()

    train_env.close()
    return j, cb.w.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "ours", "compare"], default="compare")
    p.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-8x8-v0")

    # long runs
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--n_envs", type=int, default=4)

    # PPO params
    p.add_argument("--policy_hid", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--sb3_verbose", type=int, default=0)

    # OURS params
    p.add_argument("--warmup_episodes", type=int, default=200)
    p.add_argument("--min_pred_samples", type=int, default=500)
    p.add_argument("--trigger_percentile", type=float, default=10.0)
    p.add_argument("--trigger_cooldown", type=int, default=10)

    p.add_argument("--mc_candidates", type=int, default=16)
    p.add_argument("--mc_eval_eps", type=int, default=6)
    p.add_argument("--memory_decay", type=float, default=0.93)
    p.add_argument("--explore_eps", type=float, default=0.002)

    # compute governor budgets
    p.add_argument("--max_mc_seconds_per_trigger", type=float, default=2.0)
    p.add_argument("--max_rss_mb", type=float, default=2500.0)

    # eval/logging
    p.add_argument("--eval_eps", type=int, default=50)
    p.add_argument("--log_every_episodes", type=int, default=50)
    p.add_argument("--eval_every_episodes", type=int, default=200)

    # tensorboard
    p.add_argument("--tb_logdir", type=str, default=None)

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_all_seeds(args.seed)

    # env existence check
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

    print("\n=== TRAIN OURS (SB3 PPO + Triggered Loss Shaping) ===")
    j_ours, w = train_ours(args)
    print(f"[RESULT] OURS J={j_ours:.3f} w={w}")

    print("\n=== SUMMARY ===")
    print(f"env_id={args.env_id} timesteps={args.total_timesteps} n_envs={args.n_envs}")
    print(f"STANDARD J={j_std:.3f}")
    print(f"OURS     J={j_ours:.3f} w={w}")


if __name__ == "__main__":
    main()
