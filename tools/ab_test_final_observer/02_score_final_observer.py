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

from core.critics.observer_client import ObserverItem, normalize_observer_score

from tools.ab_test_final_observer.common import (
    build_final_observer_spec,
    dump_json,
    load_json,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated MIDI with final_observer.")
    parser.add_argument("--generations-json", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generations_path = resolve_path(args.generations_json)
    output_json = resolve_path(args.output_json) if args.output_json else generations_path.parent / "scored_results.json"

    rows = load_json(generations_path)
    spec = build_final_observer_spec()

    items: list[ObserverItem] = []
    valid_indices: list[int] = []
    for idx, row in enumerate(rows):
        meta = row.get("prompt_metadata")
        if not row.get("decode_ok") or not row.get("midi_path") or not meta:
            continue
        items.append(
            ObserverItem(
                id=f"sample_{idx:05d}",
                midi_path=row["midi_path"],
                key=str(meta["key"]),
                mode=str(meta["mode"]),
                bpm=float(meta["bpm"]),
                meter_numerator=int(meta["meter_numerator"]),
                meter_denominator=int(meta["meter_denominator"]),
            )
        )
        valid_indices.append(idx)

    payload = spec.client.score_items(items) if items else {"results": []}

    for row in rows:
        row["final_observer"] = None
        row["final_observer_raw"] = None
        row["final_observer_error"] = "not_scored"

    for local_idx, result in enumerate(payload.get("results", [])):
        global_idx = valid_indices[local_idx]
        raw_score = result.get("score")
        error = result.get("error")
        rows[global_idx]["final_observer_raw"] = raw_score
        rows[global_idx]["final_observer_error"] = error
        if raw_score is not None:
            rows[global_idx]["final_observer"] = (
                normalize_observer_score(float(raw_score), center=spec.center, scale=spec.scale)
                if spec.normalize_scores
                else float(raw_score)
            )

    dump_json(output_json, rows)
    dump_json(output_json.parent / "final_observer_payload.json", payload)
    print(str(output_json.resolve()))


if __name__ == "__main__":
    main()
