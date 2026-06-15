#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a batch of base-model MIDI files for prompt probing.")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", type=str, default="melody_accompaniment_narrow")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.probe_base_app import run_base_probe

    run_base_probe(args)
