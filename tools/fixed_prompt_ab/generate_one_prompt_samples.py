#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config import train_config
from core.utils.runtime import load_env, resolve_device, seed_everything, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import generate_prompt, parse_prompt_metadata
from tools.ab_test_final_observer.common import dump_json, load_json, resolve_path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate N samples for base and selected checkpoints on one fixed prompt."
    )
    parser.add_argument(
        "--selection-json",
        default="outputs/model_registry/fixed_prompt_best_checkpoints.json",
        help="JSON produced by select_best_checkpoints.py.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generation-chunk-size", type=int, default=30)
    parser.add_argument("--prompt-preset", default="debug_fixed_prompt_two_track")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument(
        "--only-model",
        action="append",
        default=[],
        help="Generate only selected label(s), e.g. base or rules_fixed. Can be passed multiple times.",
    )
    return parser.parse_args()


def safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", value).strip("_")


def default_output_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "fixed_prompt_ab" / run_name


def generate_for_model(
    *,
    label: str,
    experiment: str,
    checkpoint_dir: str | None,
    prompt: str,
    output_root: Path,
    num_samples: int,
    max_len: int,
    temperature: float,
    generation_chunk_size: int,
    device: torch.device,
) -> list[dict]:
    cfg = train_config("text2midi", experiment)
    cfg.grpo.generation_chunk_size = generation_chunk_size
    cfg.inference = {"generation_chunk_size": generation_chunk_size}

    with_lora = checkpoint_dir is not None
    adapter = Text2MidiAdapter(cfg, device=device, with_lora=with_lora, checkpoint_path=checkpoint_dir)
    input_ids, attention_mask = adapter.tokenize_captions([prompt])

    group = safe_label(label)
    prompt_dir = output_root / group / "prompt_000"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")

    rows = []
    metadata = parse_prompt_metadata(prompt)
    sample_index = 0
    while sample_index < num_samples:
        batch_size = min(generation_chunk_size, num_samples - sample_index)
        generated = adapter.generate_repeated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=batch_size,
            max_len=max_len,
            temperature=temperature,
        )
        scores = adapter.decode_scores(generated)

        for local_index, score in enumerate(scores):
            current_index = sample_index + local_index
            midi_path = prompt_dir / f"sample_{current_index:03d}.mid"
            decode_ok = score is not None
            error = None
            if score is not None:
                try:
                    score.dump_midi(str(midi_path))
                except Exception as exc:  # noqa: BLE001 - keep the batch manifest useful.
                    decode_ok = False
                    error = str(exc)

            rows.append(
                {
                    "prompt_index": 0,
                    "prompt": prompt,
                    "prompt_metadata": metadata,
                    "model_name": group,
                    "model_label": label,
                    "experiment": experiment,
                    "checkpoint_dir": checkpoint_dir,
                    "sample_index": current_index,
                    "midi_path": str(midi_path.resolve()) if decode_ok else None,
                    "decode_ok": bool(decode_ok),
                    "error": error,
                }
            )
        sample_index += batch_size

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    setup_logging()
    load_env(PROJECT_ROOT)
    seed_everything(args.seed)

    selection_path = resolve_path(args.selection_json)
    if selection_path is None or not selection_path.exists():
        raise FileNotFoundError(f"Selection JSON not found: {selection_path}")
    selection = load_json(selection_path)

    run_name = args.run_name or datetime.now().strftime("fixed_prompt_ab_%Y%m%d_%H%M%S")
    output_root = resolve_path(args.output_dir) or default_output_dir(run_name)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    prompt = generate_prompt(rng, preset=args.prompt_preset)
    (output_root / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")

    models = []
    selected_models = list(selection.get("models", []))
    if not selected_models:
        raise ValueError(f"No models found in selection JSON: {selection_path}")
    if not args.no_base:
        models.append(
            {
                "label": "base",
                "experiment": selected_models[0]["experiment"],
                "selected_checkpoint": None,
            }
        )
    models.extend(selected_models)
    if args.only_model:
        requested = {safe_label(label) for label in args.only_model}
        models = [model for model in models if safe_label(str(model["label"])) in requested]
        if not models:
            raise ValueError(f"No requested models found: {sorted(requested)}")

    device = resolve_device()
    manifest = []
    for model in models:
        label = str(model["label"])
        experiment = str(model["experiment"])
        checkpoint_dir = model.get("selected_checkpoint")
        log.info("Generating label=%s checkpoint=%s", label, checkpoint_dir or "base")
        manifest.extend(
            generate_for_model(
                label=label,
                experiment=experiment,
                checkpoint_dir=checkpoint_dir,
                prompt=prompt,
                output_root=output_root,
                num_samples=args.num_samples,
                max_len=args.max_len,
                temperature=args.temperature,
                generation_chunk_size=args.generation_chunk_size,
                device=device,
            )
        )

    generations_path = output_root / "generations.json"
    if generations_path.exists():
        existing_manifest = load_json(generations_path)
        if not isinstance(existing_manifest, list):
            raise ValueError(f"Existing generations manifest is not a list: {generations_path}")
        current_model_names = {safe_label(str(model["label"])) for model in models}
        existing_manifest = [
            row
            for row in existing_manifest
            if str(row.get("model_name", row.get("model_label", ""))) not in current_model_names
        ]
        manifest = existing_manifest + manifest

    run_config = {
        "selection_json": str(selection_path.resolve()),
        "output_root": str(output_root.resolve()),
        "prompt_preset": args.prompt_preset,
        "prompt": prompt,
        "num_samples": args.num_samples,
        "max_len": args.max_len,
        "temperature": args.temperature,
        "generation_chunk_size": args.generation_chunk_size,
        "seed": args.seed,
        "models": models,
        "device": str(device),
        "generated_model_labels": [str(model["label"]) for model in models],
        "manifest_rows": len(manifest),
    }
    dump_json(output_root / "run_config.json", run_config)
    dump_json(generations_path, manifest)
    print(str(generations_path.resolve()))


if __name__ == "__main__":
    main()
