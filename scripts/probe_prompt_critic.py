#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple base-model MIDI candidates for one prompt and score them with the observer critic."
    )
    parser.add_argument("--caption", required=True, type=str)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--critic-config",
        type=str,
        default="configs/text2midi/reward/observer_fixed_pairs.yaml",
        help="Path to reward config yaml that contains reward.observer_critic",
    )
    parser.add_argument("--key", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--meter-numerator", type=int)
    parser.add_argument("--meter-denominator", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.probe_prompt_critic_app import run_prompt_critic_probe

    run_prompt_critic_probe(args)
