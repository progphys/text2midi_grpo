from __future__ import annotations

import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

from core.rewards import (
    TONIC_TO_PC,
    estimate_key_mode,
    extract_score_bpm,
    tempo_bin_index,
)
from text2midi.prompting import parse_prompt_metadata


@dataclass(frozen=True)
class Text2MidiMetricRow:
    tb: float | None
    tbt: float | None
    ck: float | None
    ckd: float | None
    cr_proxy_zlib: float | None


def _normalize_key_name(key: str | None) -> str | None:
    if not key:
        return None
    key = str(key).strip()
    if not key:
        return None
    head = key[0].upper()
    accidental = key[1:].replace("♭", "b").replace("♯", "#")
    if accidental.lower() == "b":
        accidental = "b"
    return f"{head}{accidental}"


def _key_pc(key: str | None) -> int | None:
    normalized = _normalize_key_name(key)
    if not normalized:
        return None
    return TONIC_TO_PC.get(normalized.replace("b", "B").upper())


def _is_relative_duplicate(
    estimated_key: str | None,
    estimated_mode: str | None,
    target_key: str | None,
    target_mode: str | None,
) -> bool:
    estimated_pc = _key_pc(estimated_key)
    target_pc = _key_pc(target_key)
    if estimated_pc is None or target_pc is None:
        return False
    if estimated_mode == target_mode and estimated_pc == target_pc:
        return True
    if target_mode == "major" and estimated_mode == "minor":
        return estimated_pc == (target_pc + 9) % 12
    if target_mode == "minor" and estimated_mode == "major":
        return estimated_pc == (target_pc + 3) % 12
    return False


def _score_event_tokens(score) -> list[str]:
    tokens: list[str] = []
    try:
        tpq = max(int(getattr(score, "tpq", 480) or 480), 1)
        for track_idx, track in enumerate(getattr(score, "tracks", []) or []):
            for note in getattr(track, "notes", []) or []:
                onset = round(float(note.time) / tpq, 3)
                duration = round(float(note.duration) / tpq, 3)
                pitch = int(note.pitch)
                tokens.append(f"{track_idx}:{onset}:{duration}:{pitch}")
    except Exception:
        return []
    tokens.sort()
    return tokens


def compression_ratio_proxy(score) -> float | None:
    """A lightweight structural proxy, not the COSIATEC CR from the paper."""
    tokens = _score_event_tokens(score)
    if len(tokens) < 2:
        return None
    raw = " ".join(tokens).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    if not compressed:
        return None
    return float(len(raw) / max(len(compressed), 1))


def text2midi_metric_row(score, caption: str) -> Text2MidiMetricRow:
    metadata = parse_prompt_metadata(caption)
    if score is None or metadata is None:
        return Text2MidiMetricRow(None, None, None, None, None)

    target_bpm = float(metadata["bpm"])
    target_bin = tempo_bin_index(target_bpm)
    actual_bin = tempo_bin_index(extract_score_bpm(score))
    tb = None
    tbt = None
    if target_bin is not None and actual_bin is not None:
        distance = abs(target_bin - actual_bin)
        tb = 1.0 if distance == 0 else 0.0
        tbt = 1.0 if distance <= 1 else 0.0

    estimated_key, estimated_mode = estimate_key_mode(score)
    target_key = str(metadata["key"])
    target_mode = str(metadata["mode"])
    ck = None
    ckd = None
    if estimated_key and estimated_mode and target_key and target_mode:
        estimated_key_norm = _normalize_key_name(estimated_key)
        target_key_norm = _normalize_key_name(target_key)
        ck = 1.0 if estimated_key_norm == target_key_norm and estimated_mode == target_mode else 0.0
        ckd = 1.0 if _is_relative_duplicate(estimated_key, estimated_mode, target_key, target_mode) else 0.0

    return Text2MidiMetricRow(
        tb=tb,
        tbt=tbt,
        ck=ck,
        ckd=ckd,
        cr_proxy_zlib=compression_ratio_proxy(score),
    )


def _mean_defined(values: Iterable[float | None]) -> float:
    defined = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not defined:
        return 0.0
    return sum(defined) / len(defined)


def summarize_text2midi_metrics(scores: list, captions: list[str]) -> dict[str, float]:
    rows = [text2midi_metric_row(score, caption) for score, caption in zip(scores, captions)]
    return {
        "text2midi/tb_pct": 100.0 * _mean_defined(row.tb for row in rows),
        "text2midi/tbt_pct": 100.0 * _mean_defined(row.tbt for row in rows),
        "text2midi/ck_pct": 100.0 * _mean_defined(row.ck for row in rows),
        "text2midi/ckd_pct": 100.0 * _mean_defined(row.ckd for row in rows),
        "text2midi/cr_proxy_zlib": _mean_defined(row.cr_proxy_zlib for row in rows),
        "text2midi/metadata_coverage": _mean_defined(
            1.0 if parse_prompt_metadata(caption) is not None else 0.0 for caption in captions
        ),
    }
