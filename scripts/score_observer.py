#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.critics.observer_client import ObserverCritic, ObserverItem


def parse_args():
    parser = argparse.ArgumentParser(description="Invoke the external observer critic through the project wrapper")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--checkpoint", default="supplementary/critic_assets/observer/best.pt")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--chord-weights-yaml", default="src/core/critics/observer/assets/learned_weights.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--project-midi", action="store_true")
    parser.add_argument("--projected-output-dir", default="outputs/observer_projected")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    observer_items = [ObserverItem(**item) for item in items]
    critic = ObserverCritic(
        project_root=PROJECT_ROOT,
        checkpoint_path=args.checkpoint,
        python_executable=args.python_executable,
        chord_weights_yaml=args.chord_weights_yaml,
        device=args.device,
        batch_size=args.batch_size,
        project_midi=args.project_midi,
        projected_output_dir=args.projected_output_dir,
    )
    scored = critic.score_items(observer_items)
    filtered = critic.filter_results(scored, min_score=args.min_score, top_k=args.top_k)
    print(json.dumps({"payload": scored, "filtered": filtered}, ensure_ascii=False, indent=2))
