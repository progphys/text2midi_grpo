from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config import train_config
from core.critics.metrics import CriticSpec
from core.critics.observer_client import ObserverCritic
from core.utils.runtime import resolve_device
from omegaconf import OmegaConf


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def latest_checkpoint_dir(experiment: str) -> Path:
    experiment_dir = PROJECT_ROOT / "outputs" / "checkpoints" / experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoint directory not found: {experiment_dir}")
    step_dirs = sorted(
        [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith("step_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No saved step directories found in {experiment_dir}")
    return step_dirs[-1]


def load_experiment_cfg(experiment: str):
    return train_config("text2midi", experiment)


def build_final_observer_spec() -> CriticSpec:
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "text2midi" / "reward" / "final_observer.yaml")
    critic_cfg = cfg.reward.observer_critic
    return CriticSpec(
        name="final_observer",
        client=ObserverCritic.from_config(PROJECT_ROOT, critic_cfg),
        center=float(critic_cfg.get("score_center", 0.0)),
        scale=float(critic_cfg.get("score_scale", 1.0)),
        normalize_scores=bool(critic_cfg.get("normalize_scores", False)),
    )


def default_output_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "ab_test_final_observer" / run_name


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_mean_ci(values: list[float], samples: int = 4000, seed: int = 123) -> dict[str, float]:
    import random

    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(draw) / len(draw))
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
    }


def get_device():
    return resolve_device()
