from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from core.critics.observer_client import ObserverCritic, ObserverItem, normalize_observer_score
from core.rewards import infer_metadata_from_score
from text2midi.prompting import parse_prompt_metadata


@dataclass
class CriticSpec:
    name: str
    client: ObserverCritic
    center: float
    scale: float
    normalize_scores: bool


def build_default_metric_critics(project_root: str | Path) -> list[CriticSpec]:
    project_root = Path(project_root)
    config_dir = project_root / "configs" / "text2midi" / "reward"
    sources = [
        ("observer_default", config_dir / "observer.yaml"),
        ("observer_fixed_pairs", config_dir / "observer_fixed_pairs.yaml"),
        ("final_observer", config_dir / "final_observer.yaml"),
        ("final_final", config_dir / "final_final.yaml"),
    ]

    specs: list[CriticSpec] = []
    for name, config_path in sources:
        cfg = OmegaConf.load(config_path)
        critic_cfg = cfg.reward.observer_critic
        specs.append(
            CriticSpec(
                name=name,
                client=ObserverCritic.from_config(project_root, critic_cfg),
                center=float(critic_cfg.get("score_center", -20.0)),
                scale=float(critic_cfg.get("score_scale", 5.0)),
                normalize_scores=bool(critic_cfg.get("normalize_scores", True)),
            )
        )
    return specs


def score_symbolic_scores_with_critics(
    project_root: str | Path,
    critic_specs: list[CriticSpec],
    scores: list,
    captions: list[str],
    group_size: int | None = None,
    tmp_prefix: str = "critic_metrics_",
) -> list[dict[str, float | str | None]]:
    results: list[dict[str, float | str | None]] = [{} for _ in scores]

    if not critic_specs:
        return results

    with tempfile.TemporaryDirectory(prefix=tmp_prefix, dir=project_root) as tmp_dir:
        tmp_root = Path(tmp_dir)
        items: list[ObserverItem] = []
        valid_indices: list[int] = []

        for idx, (score, caption) in enumerate(zip(scores, captions)):
            metadata = parse_prompt_metadata(caption)
            if score is None:
                for spec in critic_specs:
                    results[idx][spec.name] = 0.0
                    results[idx][f"{spec.name}_raw"] = 0.0
                    results[idx][f"{spec.name}_error"] = "decode_failed"
                continue
            if metadata is None:
                metadata = infer_metadata_from_score(score)
            if not metadata or not metadata.get("key") or not metadata.get("mode"):
                for spec in critic_specs:
                    results[idx][spec.name] = 0.0
                    results[idx][f"{spec.name}_raw"] = 0.0
                    results[idx][f"{spec.name}_error"] = "metadata_unavailable"
                continue

            midi_path = tmp_root / f"rollout_{idx:04d}.mid"
            try:
                score.dump_midi(str(midi_path))
            except Exception as exc:
                for spec in critic_specs:
                    results[idx][spec.name] = 0.0
                    results[idx][f"{spec.name}_raw"] = 0.0
                    results[idx][f"{spec.name}_error"] = f"midi_dump_failed:{exc}"
                continue

            items.append(
                ObserverItem(
                    id=f"rollout_{idx:04d}",
                    midi_path=str(midi_path),
                    key=str(metadata["key"]),
                    mode=str(metadata["mode"]),
                    bpm=float(metadata.get("bpm") or 120.0),
                    meter_numerator=int(metadata.get("meter_numerator") or 4),
                    meter_denominator=int(metadata.get("meter_denominator") or 4),
                )
            )
            valid_indices.append(idx)

        if not items:
            return results

        for spec in critic_specs:
            payload = spec.client.score_items(items)
            for local_idx, row in enumerate(payload.get("results", [])):
                global_idx = valid_indices[local_idx]
                raw_score = row.get("score")
                error = row.get("error")
                if raw_score is None:
                    results[global_idx][spec.name] = 0.0
                    results[global_idx][f"{spec.name}_raw"] = 0.0
                    results[global_idx][f"{spec.name}_error"] = error or "critic_score_missing"
                    continue

                raw_score = float(raw_score)
                if spec.normalize_scores:
                    results[global_idx][spec.name] = normalize_observer_score(
                        raw_score,
                        center=spec.center,
                        scale=spec.scale,
                    )
                else:
                    results[global_idx][spec.name] = raw_score
                results[global_idx][f"{spec.name}_raw"] = raw_score
                results[global_idx][f"{spec.name}_error"] = error
            if group_size and group_size > 1:
                _attach_group_ranks(results, spec.name, group_size)

    return results


def _attach_group_ranks(
    results: list[dict[str, float | str | None]],
    metric_name: str,
    group_size: int,
) -> None:
    for group_start in range(0, len(results), group_size):
        group = results[group_start : group_start + group_size]
        valid = [
            (idx, float(row[f"{metric_name}_raw"]))
            for idx, row in enumerate(group)
            if row.get(f"{metric_name}_raw") is not None and not row.get(f"{metric_name}_error")
        ]
        if not valid:
            for row in group:
                row[f"{metric_name}_rank"] = None
                row[f"{metric_name}_rank_score"] = 0.0
                row[f"{metric_name}_pairwise_win_rate"] = 0.0
            continue
        valid.sort(key=lambda pair: (-pair[1], pair[0]))
        n = len(valid)
        for rank, (local_idx, _raw) in enumerate(valid, start=1):
            row = group[local_idx]
            row[f"{metric_name}_rank"] = rank
            row[f"{metric_name}_rank_score"] = 1.0 if n == 1 else 1.0 - (rank - 1) / (n - 1)
            if n == 1:
                row[f"{metric_name}_pairwise_win_rate"] = 1.0
            else:
                wins = 0.0
                for other_local_idx, other_raw in valid:
                    if other_local_idx == local_idx:
                        continue
                    if _raw > other_raw:
                        wins += 1.0
                    elif _raw == other_raw:
                        wins += 0.5
                row[f"{metric_name}_pairwise_win_rate"] = wins / (n - 1)
        for idx, row in enumerate(group):
            if all(idx != local_idx for local_idx, _ in valid):
                row[f"{metric_name}_rank"] = None
                row[f"{metric_name}_rank_score"] = 0.0
                row[f"{metric_name}_pairwise_win_rate"] = 0.0
