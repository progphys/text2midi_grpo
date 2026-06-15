#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIDI from a text caption")
    parser.add_argument("--caption", type=str)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--to-wav", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from text2midi.infer_app import run_inference

    run_inference(args)
