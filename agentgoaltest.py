import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
Chat-ready prototype:
- loads an open-weights Qwen causal LM
- keeps a continuous latent state z
- applies autonomous drift with a vector field e(z)
- steers next-token generation with a Jacobian/gradient signal
- runs an interactive terminal chat loop

This is inference-only. No finetuning required.
The vector field modules are initialized randomly unless you load weights.
So the mechanism is real, but behavior will be weak until you train or hand-set
those modules. The base language quality still comes from Qwen.
"""


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    latent_dim: int = 256
    hidden_dim: int = 512
    steering_strength: float = 6.0
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 128
    drift_steps_per_token: int = 2
    drift_step_size: float = 0.15
    use_jacobian_guidance: bool = True


class MLP(nn.Module):
    def __init__(self, inp: int, hid: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DriftField(nn.Module):
    """Continuous vector field e(z)."""
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.field = MLP(latent_dim, hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.field(z)


class EventProjector(nn.Module):
    """Maps the latest token/event embedding into latent state movement."""
    def __init__(self, hidden_size: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = MLP(hidden_size, hidden_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class GoalScorer(nn.Module):
    """
    Differentiable scalar objective.
    This is NOT reward learning; it's a direct latent potential used at inference.
    Default form: learned scorer over [z, g].
    """
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = MLP(latent_dim * 2, hidden_dim, 1)

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, g], dim=-1)
        return self.scorer(x).squeeze(-1)


class JacobianFieldChat(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=cfg.dtype,
            trust_remote_code=True,
        ).to(cfg.device)
        self.lm.eval()

        hidden_size = self.lm.config.hidden_size
        self.state_proj = MLP(hidden_size, cfg.hidden_dim, cfg.latent_dim).to(cfg.device)
        self.goal_proj = MLP(hidden_size, cfg.hidden_dim, cfg.latent_dim).to(cfg.device)
        self.event_proj = EventProjector(hidden_size, cfg.latent_dim, cfg.hidden_dim).to(cfg.device)
        self.drift = DriftField(cfg.latent_dim, cfg.hidden_dim).to(cfg.device)
        self.goal_scorer = GoalScorer(cfg.latent_dim, cfg.hidden_dim).to(cfg.device)

    @torch.no_grad()
    def _encode_text_hidden(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(text, return_tensors="pt").to(self.cfg.device)
        out = self.lm(**toks, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]          # [1, T, H]
        pooled = hidden[:, -1, :]               # [1, H]
        return pooled, hidden

    def encode_state(self, text: str) -> torch.Tensor:
        pooled, _ = self._encode_text_hidden(text)
        return self.state_proj(pooled)

    def encode_goal(self, goal_text: str) -> torch.Tensor:
        pooled, _ = self._encode_text_hidden(goal_text)
        return self.goal_proj(pooled)

    def autonomous_drift(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the world vector field multiple small steps."""
        for _ in range(self.cfg.drift_steps_per_token):
            z = z + self.cfg.drift_step_size * self.drift(z)
        return z

    def apply_event(self, z: torch.Tensor, event_hidden: torch.Tensor) -> torch.Tensor:
        dz = self.event_proj(event_hidden)
        z = z + dz
        z = self.autonomous_drift(z)
        return z

    def potential(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.goal_scorer(z, g)

    def steered_next_logits(self, prompt: str, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)

        with torch.enable_grad():
            out = self.lm(**toks, output_hidden_states=True, use_cache=False)
            logits = out.logits[:, -1, :].float().requires_grad_(True)

            if not self.cfg.use_jacobian_guidance:
                return logits.detach()

            probs = F.softmax(logits / self.cfg.temperature, dim=-1)
            emb = self.lm.get_input_embeddings().weight.float()           # [V, H]
            expected_embed = probs @ emb                                  # [1, H]

            z_after = self.apply_event(z, expected_embed)
            value = self.potential(z_after, g).sum()
            grad = torch.autograd.grad(value, logits, retain_graph=False, create_graph=False)[0]
            steered = logits + self.cfg.steering_strength * grad
            return steered.detach()

    @torch.no_grad()
    def sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = torch.topk(logits, k=min(self.cfg.top_k, logits.shape[-1]), dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, topk_idx, topk_vals)
        probs = F.softmax(filtered / self.cfg.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def token_to_hidden(self, token_id: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(token_id).squeeze(1)

    def generate_reply(
        self,
        conversation_text: str,
        z: torch.Tensor,
        g: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, torch.Tensor]:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        text = conversation_text
        generated = []

        for _ in range(max_new_tokens):
            logits = self.steered_next_logits(text, z, g)
            next_token = self.sample_token(logits)
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated.append(token_text)
            text += token_text

            event_hidden = self.token_to_hidden(next_token)
            z = self.apply_event(z, event_hidden)

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if "
User:" in "".join(generated):
                break

        reply = "".join(generated)
        if "
User:" in reply:
            reply = reply.split("
User:", 1)[0]
        return reply.strip(), z


def build_prompt(history: str, user_message: str) -> str:
    if history.strip():
        return history + f"
User: {user_message}
Assistant:"
    return f"User: {user_message}
Assistant:"


def run_chat(goal_text: str, model_name: str = "Qwen/Qwen3-0.6B") -> None:
    cfg = Config(model_name=model_name)
    system = JacobianFieldChat(cfg)

    history = ""
    z = system.encode_state("Conversation start.")
    g = system.encode_goal(goal_text)

    print("Interactive Jacobian Field Chat")
    print("Type /reset to reset state, /quit to exit.
")
    print(f"Goal anchor: {goal_text}
")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() == "/quit":
            break
        if user_message.lower() == "/reset":
            history = ""
            z = system.encode_state("Conversation start.")
            g = system.encode_goal(goal_text)
            print("State reset.
")
            continue

        prompt = build_prompt(history, user_message)

        # Apply user message as an event to the latent world state.
        pooled, hidden = system._encode_text_hidden(f"User: {user_message}")
        user_event_hidden = hidden[:, -1, :]
        z = system.apply_event(z, user_event_hidden)

        reply, z = system.generate_reply(prompt, z, g)
        print(f"Assistant: {reply}
")

        history = prompt + " " + reply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen Jacobian world-field chat")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Track the real-world consequences of events and respond coherently.",
        help="Goal anchor text encoded into latent goal space",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_chat(goal_text=args.goal, model_name=args.model)


if __name__ == "__main__":
    main()
