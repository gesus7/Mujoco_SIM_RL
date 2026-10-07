"""Small, strict configuration shared by the launcher and training worker."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 0
    num_envs: int = 256
    num_eval_envs: int = 32
    num_timesteps: int = 20_000_000
    batch_size: int = 128
    num_minibatches: int = 8
    unroll_length: int = 10
    num_updates_per_batch: int = 8
    num_evals: int = 21
    max_wall_hours: float = 7.5
    memory_fraction: float = 0.8
    profile: str = "baseline"
    mass_range: tuple = (0.8, 1.2)
    friction_range: tuple = (0.8, 1.2)
    obs_noise_std: float = 0.01
    delay_steps: int = 1
    reward_mode: str = "upstream"
    max_grad_norm: float | None = None

    def validate(self):
        positive = ("num_envs", "num_eval_envs", "num_timesteps", "batch_size",
                    "num_minibatches", "unroll_length", "num_updates_per_batch", "num_evals")
        for key in positive:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.num_evals < 2:
            raise ValueError("num_evals >= 2 is required for periodic checkpoints")
        for key in ("seed", "delay_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.batch_size * self.num_minibatches % self.num_envs:
            raise ValueError("batch_size * num_minibatches must be divisible by num_envs")
        for key, low, high in (("max_wall_hours", 0, 8), ("memory_fraction", 0, 0.9)):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"{key} must be in ({low}, {high}]")
        if self.profile not in {"baseline", "physics", "sensor", "combined"}:
            raise ValueError("unknown profile")
        if self.reward_mode not in {"upstream", "progress", "progress_hold"}:
            raise ValueError("unknown reward_mode")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive or null")
        for key in ("mass_range", "friction_range"):
            value = getattr(self, key)
            if len(value) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value) or not 0 < value[0] <= value[1]:
                raise ValueError(f"invalid {key}")
        if not math.isfinite(self.obs_noise_std) or self.obs_noise_std < 0:
            raise ValueError("obs_noise_std must be finite and nonnegative")
        if not self.name or Path(self.name).name != self.name or any(c in self.name for c in "/\\:"):
            raise ValueError("name must be a simple nonempty filename")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()

    def to_dict(self):
        return asdict(self)
