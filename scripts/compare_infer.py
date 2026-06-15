#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paired base-vs-finetuned MIDI samples from synthetic prompts"
    )
    parser.add_argument("--experiment", type=str, help="Experiment name to auto-resolve the latest saved step directory")
    parser.add_argument("--checkpoint-path", type=str, help="Direct path to a fine-tuned LoRA checkpoint directory")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--skip-critics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.compare_infer_app import run_compare_inference

    run_compare_inference(args)
