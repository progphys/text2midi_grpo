#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2midi.prompting import parse_prompt_metadata

from tools.ab_test_final_observer.common import dump_json, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild generations.json from saved A/B sample folders.")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/ab_test_final_observer/<run_name>.")
    parser.add_argument("--output-json", default=None, help="Optional output path. Defaults to <run-dir>/generations.json.")
    return parser.parse_args()


def iter_prompt_dirs(model_root: Path):
    if not model_root.exists():
        return
    for prompt_dir in sorted(
        [path for path in model_root.iterdir() if path.is_dir() and path.name.startswith("prompt_")]
    ):
        yield prompt_dir


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    output_json = resolve_path(args.output_json) if args.output_json else run_dir / "generations.json"
    rows: list[dict] = []

    for model_name in ("base", "final"):
        model_root = run_dir / model_name
        for prompt_dir in iter_prompt_dirs(model_root):
            prompt_index = int(prompt_dir.name.split("_")[-1])
            prompt_path = prompt_dir / "prompt.txt"
            if not prompt_path.exists():
                continue
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            prompt_metadata = parse_prompt_metadata(prompt)

            sample_paths = sorted(prompt_dir.glob("sample_*.mid"))
            for sample_path in sample_paths:
                stem = sample_path.stem
                try:
                    sample_index = int(stem.split("_")[-1])
                except ValueError:
                    continue
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "prompt_metadata": prompt_metadata,
                        "model_name": model_name,
                        "sample_index": sample_index,
                        "midi_path": str(sample_path.resolve()),
                        "decode_ok": True,
                    }
                )

    rows.sort(key=lambda row: (row["prompt_index"], row["model_name"], row["sample_index"]))
    dump_json(output_json, rows)
    print(str(output_json.resolve()))


if __name__ == "__main__":
    main()
