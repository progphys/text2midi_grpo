from __future__ import annotations

import json
import random
import re
from pathlib import Path

from omegaconf import OmegaConf

from core.config import inference_config
from core.critics.observer_client import ObserverCritic, ObserverItem
from core.rewards import batch_rewards
from core.utils.runtime import resolve_device, seed_everything, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import parse_prompt_metadata


_KEY_MODE_RE = re.compile(r"\b(?:in|set in|written in)\s+([A-G](?:#|b)?)\s+(major|minor)\b", re.IGNORECASE)
_METER_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
_BPM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*BPM\b", re.IGNORECASE)


def _fallback_prompt_metadata(prompt: str) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    key_match = _KEY_MODE_RE.search(prompt)
    if key_match:
        key = key_match.group(1)
        result["key"] = key[0].upper() + key[1:]
        result["mode"] = key_match.group(2).lower()
    meter_match = _METER_RE.search(prompt)
    if meter_match:
        result["meter_numerator"] = int(meter_match.group(1))
        result["meter_denominator"] = int(meter_match.group(2))
    bpm_match = _BPM_RE.search(prompt)
    if bpm_match:
        result["bpm"] = float(bpm_match.group(1))
    return result


def _resolve_metadata(args) -> dict[str, int | float | str]:
    parsed = parse_prompt_metadata(args.caption) or _fallback_prompt_metadata(args.caption)
    metadata = {
        "key": args.key or parsed.get("key"),
        "mode": args.mode or parsed.get("mode"),
        "bpm": args.bpm if args.bpm is not None else parsed.get("bpm"),
        "meter_numerator": args.meter_numerator if args.meter_numerator is not None else parsed.get("meter_numerator"),
        "meter_denominator": args.meter_denominator if args.meter_denominator is not None else parsed.get("meter_denominator"),
    }
    missing = [name for name, value in metadata.items() if value is None]
    if missing:
        raise ValueError(
            "Prompt metadata could not be fully parsed. Provide missing fields explicitly: "
            + ", ".join(missing)
        )
    return metadata


def _attach_ranks(rows: list[dict[str, object]]) -> None:
    valid = [
        (idx, float(row["score"]))
        for idx, row in enumerate(rows)
        if row.get("score") is not None and not row.get("error")
    ]
    valid.sort(key=lambda pair: (-pair[1], pair[0]))
    n = len(valid)
    for rank, (local_idx, raw_score) in enumerate(valid, start=1):
        wins = 0.0
        for other_idx, other_score in valid:
            if other_idx == local_idx:
                continue
            if raw_score > other_score:
                wins += 1.0
            elif raw_score == other_score:
                wins += 0.5
        rows[local_idx]["rank"] = rank
        rows[local_idx]["rank_score"] = 1.0 if n <= 1 else 1.0 - (rank - 1) / (n - 1)
        rows[local_idx]["pairwise_win_rate"] = 1.0 if n <= 1 else wins / (n - 1)
    for idx, row in enumerate(rows):
        if not any(idx == valid_idx for valid_idx, _score in valid):
            row["rank"] = None
            row["rank_score"] = 0.0
            row["pairwise_win_rate"] = 0.0


def run_prompt_critic_probe(args) -> None:
    cfg = inference_config("text2midi")
    setup_logging()
    seed_everything(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device()

    num_samples = int(args.num_samples)
    max_len = int(args.max_len or cfg.inference.max_len)
    temperature = float(args.temperature or cfg.inference.temperature)
    metadata = _resolve_metadata(args)

    output_root = Path(args.output_dir) if args.output_dir else Path(cfg.paths.output_dir) / "prompt_critic_probe"
    output_root.mkdir(parents=True, exist_ok=True)

    adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)
    input_ids, attention_mask = adapter.tokenize_captions([args.caption])
    generated = adapter.generate_repeated(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=num_samples,
        max_len=max_len,
        temperature=temperature,
    )
    scores = adapter.decode_scores(generated)

    items: list[ObserverItem] = []
    reward_rows: list[dict[str, float]] = batch_rewards(scores, cfg.reward, captions=[args.caption] * num_samples)
    # We only use reward_rows to expose invalid/basic symbolic diagnostics from the existing reward stack.
    manifest_items: list[dict[str, object]] = []
    for idx, score in enumerate(scores):
        safe_caption = re.sub(r"[^\w\s-]", "", args.caption[:60]).strip().replace(" ", "_")
        midi_path = output_root / f"{idx:02d}_{safe_caption}.mid"
        if score is None:
            manifest_items.append(
                {
                    "index": idx,
                    "midi_path": None,
                    "error": "decode_failed",
                }
            )
            continue
        score.dump_midi(str(midi_path))
        item = ObserverItem(
            id=f"sample_{idx:02d}",
            midi_path=str(midi_path.resolve()),
            key=str(metadata["key"]),
            mode=str(metadata["mode"]),
            bpm=float(metadata["bpm"]),
            meter_numerator=int(metadata["meter_numerator"]),
            meter_denominator=int(metadata["meter_denominator"]),
        )
        items.append(item)
        manifest_items.append(
            {
                "index": idx,
                "midi_path": str(midi_path.resolve()),
                "error": None,
                "invalid": float(reward_rows[idx].get("invalid", 0.0) or 0.0),
            }
        )

    critic_cfg = OmegaConf.load(Path(cfg.project_root) / args.critic_config).reward.observer_critic
    critic = ObserverCritic.from_config(cfg.project_root, critic_cfg)
    payload = critic.score_items(items) if items else {"results": []}
    results = list(payload.get("results", []))
    scored_positions = [idx for idx, row in enumerate(manifest_items) if row.get("midi_path")]

    for result, manifest_idx in zip(results, scored_positions):
        row = manifest_items[manifest_idx]
        row["score"] = result.get("score")
        row["critic_error"] = result.get("error")
        row["key"] = metadata["key"]
        row["mode"] = metadata["mode"]
        row["bpm"] = metadata["bpm"]
        row["meter_numerator"] = metadata["meter_numerator"]
        row["meter_denominator"] = metadata["meter_denominator"]

    for idx, row in enumerate(manifest_items):
        if row.get("midi_path") and "score" not in row:
            row["score"] = None
            row["critic_error"] = "critic_result_missing"

    _attach_ranks(manifest_items)
    summary = {
        "device": str(device),
        "caption": args.caption,
        "metadata": metadata,
        "num_samples": num_samples,
        "max_len": max_len,
        "temperature": temperature,
        "critic_config": str((Path(cfg.project_root) / args.critic_config).resolve()),
        "items": manifest_items,
        "filtered": [row for row in sorted(manifest_items, key=lambda row: (row.get("score") is None, -(float(row.get("score") or -1e9)))) if row.get("score") is not None],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {device}")
    print(f"Output dir: {output_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    for row in summary["filtered"]:
        print(
            f"sample={row['index']:02d} score={float(row['score']):.4f} "
            f"rank={row['rank']} pairwise={float(row['pairwise_win_rate']):.3f} midi={row['midi_path']}"
        )
