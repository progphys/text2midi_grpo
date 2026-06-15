"""
Reward functions for GRPO fine-tuning.

Each function takes a symusic.Score object and returns a float in [0, 1].
P_invalid returns 0 (valid) or 1 (invalid — will be subtracted with weight 2.0).

Final reward: R = w_key * R_key + w_rhythm * R_rhythm - w_invalid * P_invalid
"""

from __future__ import annotations
import numpy as np
from typing import List

# 12 pitch classes: C C# D D# E F F# G G# A A# B
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]  # natural minor


def _all_scale_sets() -> list[set[int]]:
    """All 24 major/minor scales as sets of pitch classes."""
    scales = []
    for root in range(12):
        for intervals in (MAJOR_INTERVALS, MINOR_INTERVALS):
            scales.append({(root + i) % 12 for i in intervals})
    return scales


_SCALES = _all_scale_sets()


def penalty_invalid(
    score,
    min_notes: int = 8,
    min_duration_sec: float = 5.0,
    max_repeat_ratio: float = 0.8,
) -> float:
    """
    Returns 1.0 if the MIDI is considered invalid, else 0.0.

    Checks:
    - score is None or has no tracks
    - total note count < min_notes
    - total duration < min_duration_sec
    - most-common pitch dominates more than max_repeat_ratio
    """
    try:
        if score is None:
            return 1.0

        all_notes = []
        for track in score.tracks:
            all_notes.extend(track.notes)

        if len(all_notes) < min_notes:
            return 1.0

        # Duration: symusic uses ticks; ticks_per_quarter is in score.tpq
        tpq = score.tpq if hasattr(score, "tpq") else 480
        # Rough estimate: assume tempo ~120 BPM → 500000 us/beat → 0.5 sec/beat
        # More accurate: use score.tempos if available
        try:
            total_ticks = max(n.time + n.duration for n in all_notes)
            beats = total_ticks / tpq
            duration_sec = beats * 0.5  # at 120 BPM
        except Exception:
            return 1.0

        if duration_sec < min_duration_sec:
            return 1.0

        pitches = [n.pitch for n in all_notes]
        most_common_count = max(pitches.count(p) for p in set(pitches))
        if most_common_count / len(pitches) > max_repeat_ratio:
            return 1.0

        return 0.0

    except Exception:
        return 1.0


def reward_key(score) -> float:
    """
    Fraction of notes matching the best-fit major or minor scale.
    Returns value in [0, 1].
    """
    try:
        all_notes = []
        for track in score.tracks:
            all_notes.extend(track.notes)

        if not all_notes:
            return 0.0

        pitch_classes = [n.pitch % 12 for n in all_notes]
        best = max(
            sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
            for scale in _SCALES
        )
        return float(best)

    except Exception:
        return 0.0


def reward_rhythm(score, subdivisions: int = 16) -> float:
    """
    How well note onsets align to a 1/subdivisions grid.

    Computes mean normalized deviation from nearest grid point.
    Returns 1.0 (perfect alignment) down to 0.0 (maximum misalignment).
    """
    try:
        all_notes = []
        for track in score.tracks:
            all_notes.extend(track.notes)

        if not all_notes:
            return 0.0

        tpq = score.tpq if hasattr(score, "tpq") else 480
        # ticks per subdivision (e.g. 1/16 at 480 tpq = 120 ticks)
        grid = tpq * 4 // subdivisions

        if grid == 0:
            return 0.0

        deviations = []
        for note in all_notes:
            remainder = note.time % grid
            # Distance to nearest grid point (can round up or down)
            dev = min(remainder, grid - remainder)
            deviations.append(dev / grid)  # normalize to [0, 0.5]

        # 0.0 = perfect, 0.5 = max misalignment
        mean_dev = float(np.mean(deviations))
        # Convert to [0, 1] where 1 is best
        return 1.0 - 2.0 * mean_dev

    except Exception:
        return 0.0


def compute_reward(
    score,
    key_weight: float = 0.5,
    rhythm_weight: float = 0.5,
    invalid_penalty: float = 2.0,
    **invalid_kwargs,
) -> dict[str, float]:
    """
    Compute full reward dict for one generated MIDI (symusic.Score).

    Returns:
        {
            "total": float,
            "key": float,
            "rhythm": float,
            "invalid": float,   # 0 or 1
        }
    """
    p_inv = penalty_invalid(score, **invalid_kwargs)

    if p_inv > 0.0:
        return {"total": -invalid_penalty, "key": 0.0, "rhythm": 0.0, "invalid": 1.0}

    r_key = reward_key(score)
    r_rhy = reward_rhythm(score)
    total = key_weight * r_key + rhythm_weight * r_rhy

    return {"total": total, "key": r_key, "rhythm": r_rhy, "invalid": 0.0}


def batch_rewards(scores: list, cfg_reward) -> list[dict[str, float]]:
    """Compute rewards for a list of symusic.Score objects."""
    return [
        compute_reward(
            score,
            key_weight=cfg_reward.key_weight,
            rhythm_weight=cfg_reward.rhythm_weight,
            invalid_penalty=cfg_reward.invalid_penalty,
            min_notes=cfg_reward.min_notes,
            min_duration_sec=cfg_reward.min_duration_sec,
            max_repeat_ratio=cfg_reward.max_repeat_ratio,
        )
        for score in scores
    ]
