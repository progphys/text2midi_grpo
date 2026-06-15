"""
Model-agnostic reward functions operating on decoded symbolic scores.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import warnings

import numpy as np
from text2midi.prompting import parse_prompt_metadata

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from music21 import meter as music21_meter
        from music21.analysis import discrete as music21_discrete
except Exception:  # pragma: no cover - optional dependency fallback
    music21_meter = None
    music21_discrete = None

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
MODE_INTERVALS = {
    "major": MAJOR_INTERVALS,
    "minor": MINOR_INTERVALS,
}
if music21_discrete is not None:
    _KRUMHANSL = music21_discrete.KrumhanslSchmuckler()
    KRUMHANSL_MAJOR_PROFILE = np.array(_KRUMHANSL.getWeights("major"), dtype=np.float64)
    KRUMHANSL_MINOR_PROFILE = np.array(_KRUMHANSL.getWeights("minor"), dtype=np.float64)
else:
    _KRUMHANSL = None
    KRUMHANSL_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KRUMHANSL_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
METER_CANDIDATES = [(2, 4), (3, 4), (4, 4), (5, 4), (6, 8), (7, 8)]
TEMPO_BINS = [
    (60, 75),
    (76, 99),
    (100, 120),
    (121, 150),
    (151, 180),
]
TONIC_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


def _collect_notes(score) -> list:
    all_notes = []
    for track in score.tracks:
        all_notes.extend(track.notes)
    return all_notes


def _nonempty_tracks(score) -> list:
    return [track for track in getattr(score, "tracks", []) or [] if getattr(track, "notes", None)]


def _is_drum_track(track) -> bool:
    name = str(getattr(track, "name", "") or "").lower()
    if "drum" in name or "percussion" in name:
        return True
    if bool(getattr(track, "is_drum", False)):
        return True
    program = getattr(track, "program", None)
    return program is not None and int(program) < 0


def _score_tpq(score) -> int:
    return int(score.tpq if hasattr(score, "tpq") else 480)


def _score_duration_beats(score, notes: list | None = None) -> float:
    notes = notes if notes is not None else _collect_notes(score)
    if not notes:
        return 0.0
    return max(float(note.time + note.duration) for note in notes) / max(_score_tpq(score), 1)


def _triangular_score(value: float, low: float, target: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == target:
        return 1.0
    if value < target:
        return float((value - low) / max(target - low, 1e-8))
    return float((high - value) / max(high - target, 1e-8))


def _onset_groups(notes: list) -> dict[int, list]:
    groups: dict[int, list] = {}
    for note in notes:
        groups.setdefault(int(note.time), []).append(note)
    return groups


def _circular_distance(value: float, target: float, period: float) -> float:
    raw = abs(value - target)
    return min(raw, period - raw)


def _normalize_key_name(key: str | None) -> str | None:
    if not key:
        return None
    key = str(key).strip().upper()
    if len(key) >= 2 and key[1] == "B":
        key = key[0] + "b"
    return key.replace("b", "B")


def _scale_pitch_classes(root_pc: int, mode: str) -> set[int]:
    intervals = MODE_INTERVALS.get(mode, MAJOR_INTERVALS)
    return {(root_pc + interval) % 12 for interval in intervals}


def _duration_weighted_pitch_profile(score) -> np.ndarray | None:
    notes = _collect_notes(score)
    if not notes:
        return None
    profile = np.zeros(12, dtype=np.float64)
    for note in notes:
        duration = max(float(getattr(note, "duration", 0.0) or 0.0), 1.0)
        profile[int(note.pitch) % 12] += duration
    total = float(profile.sum())
    if total <= 0.0:
        return None
    return profile / total


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def _rotated_profile(mode: str, tonic_pc: int) -> np.ndarray:
    base = KRUMHANSL_MINOR_PROFILE if mode == "minor" else KRUMHANSL_MAJOR_PROFILE
    return np.roll(base, int(tonic_pc))


def _meter_measure_quarters(numerator: int, denominator: int) -> float:
    return float(numerator) * 4.0 / float(max(denominator, 1))


@lru_cache(maxsize=64)
def _cached_music21_time_signature(numerator: int, denominator: int):
    if music21_meter is None:
        return None
    try:
        return music21_meter.TimeSignature(f"{int(numerator)}/{int(denominator)}")
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _music21_accent_weight(numerator: int, denominator: int, position_quarters: float) -> float | None:
    time_signature = _cached_music21_time_signature(int(numerator), int(denominator))
    if time_signature is None:
        return None
    try:
        return float(time_signature.getAccentWeight(float(position_quarters), permitMeterModulus=True))
    except Exception:
        return None


def _fallback_accent_weight(numerator: int, denominator: int, position_quarters: float) -> float:
    measure = _meter_measure_quarters(numerator, denominator)
    beat = 4.0 / float(max(denominator, 1))
    position = position_quarters % max(measure, 1e-8)
    if _circular_distance(position, 0.0, measure) < 1e-6:
        return 1.0
    secondary = []
    if (numerator, denominator) == (4, 4):
        secondary = [2.0]
    elif (numerator, denominator) == (6, 8):
        secondary = [1.5]
    elif numerator in {5, 7}:
        secondary = [3.0 * beat if numerator == 5 else 4.0 * beat]
    if any(_circular_distance(position, item, measure) < 1e-6 for item in secondary):
        return 0.65
    if abs((position / beat) - round(position / beat)) < 1e-6:
        return 0.35
    return 0.0


def _onset_accent_weights(score) -> list[tuple[float, float]]:
    notes = _collect_notes(score)
    if not notes:
        return []
    tpq = max(_score_tpq(score), 1)
    grouped = _onset_groups(notes)
    weighted: list[tuple[float, float]] = []
    for onset_ticks, group in grouped.items():
        onset_quarters = float(onset_ticks) / float(tpq)
        weight = 0.0
        for note in group:
            duration_quarters = float(getattr(note, "duration", 0.0) or 0.0) / float(tpq)
            velocity = float(getattr(note, "velocity", 64.0) or 64.0) / 127.0
            bass_bonus = 0.35 if int(note.pitch) < 48 else 0.0
            long_note_bonus = min(max(duration_quarters, 0.0), 2.0) * 0.20
            weight += 1.0 + 0.35 * velocity + bass_bonus + long_note_bonus
        if len(group) >= 3:
            weight += 0.75
        weighted.append((onset_quarters, weight))
    return weighted


def _meter_template_fit(score, numerator: int, denominator: int) -> float:
    onsets = _onset_accent_weights(score)
    if not onsets:
        return 0.0
    measure = _meter_measure_quarters(numerator, denominator)
    if measure <= 0.0:
        return 0.0
    beat = 4.0 / float(max(denominator, 1))
    sigma = max(beat * 0.22, 0.08)
    total_weight = sum(weight for _, weight in onsets)
    if total_weight <= 0.0:
        return 0.0

    fit = 0.0
    for onset_quarters, weight in onsets:
        position = onset_quarters % measure
        best = 0.0
        steps = max(int(round(measure / max(sigma, 1e-8))) * 2, 16)
        for step in range(steps):
            template_pos = measure * step / steps
            template_weight = _music21_accent_weight(numerator, denominator, template_pos)
            if template_weight is None:
                template_weight = _fallback_accent_weight(numerator, denominator, template_pos)
            if template_weight <= 0.0:
                continue
            distance = _circular_distance(position, template_pos, measure)
            proximity = float(np.exp(-0.5 * (distance / sigma) ** 2))
            best = max(best, template_weight * proximity)
        fit += weight * best
    return float(max(0.0, min(1.0, fit / total_weight)))


def _all_scale_sets() -> list[set[int]]:
    scales = []
    for root in range(12):
        for intervals in (MAJOR_INTERVALS, MINOR_INTERVALS):
            scales.append({(root + i) % 12 for i in intervals})
    return scales


_SCALES = _all_scale_sets()


def estimate_key_mode(score) -> tuple[str | None, str | None]:
    try:
        all_notes = _collect_notes(score)
        if not all_notes:
            return None, None

        pitch_classes = [note.pitch % 12 for note in all_notes]
        best_score = -1.0
        best = (None, None)
        for tonic_name, tonic_pc in TONIC_TO_PC.items():
            if len(tonic_name) > 2 or "B#" in tonic_name or "E#" in tonic_name or "FB" in tonic_name or "CB" in tonic_name:
                continue
            for mode, intervals in MODE_INTERVALS.items():
                scale = {(tonic_pc + interval) % 12 for interval in intervals}
                coverage = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
                if coverage > best_score:
                    best_score = coverage
                    tonic = tonic_name.replace("B", "b") if len(tonic_name) > 1 else tonic_name
                    best = (tonic, mode)
        return best
    except Exception:
        return None, None


def extract_score_bpm(score) -> float | None:
    try:
        tempos = getattr(score, "tempos", None)
        if tempos:
            qpm = getattr(tempos[0], "qpm", None)
            if qpm:
                return float(qpm)
    except Exception:
        return None
    return None


def extract_score_time_signature(score) -> tuple[int | None, int | None]:
    try:
        time_signatures = getattr(score, "time_signatures", None)
        if time_signatures:
            numerator = getattr(time_signatures[0], "numerator", None)
            denominator = getattr(time_signatures[0], "denominator", None)
            if numerator and denominator:
                return int(numerator), int(denominator)
    except Exception:
        return None, None
    return None, None


def infer_metadata_from_score(score) -> dict[str, int | float | str | None]:
    key, mode = estimate_key_mode(score)
    bpm = extract_score_bpm(score)
    meter_numerator, meter_denominator = extract_score_time_signature(score)
    return {
        "key": key,
        "mode": mode,
        "bpm": bpm,
        "meter_numerator": meter_numerator,
        "meter_denominator": meter_denominator,
    }


def _relative_mode_key(target_key: str | None, target_mode: str | None) -> tuple[str | None, str | None]:
    tonic_pc = TONIC_TO_PC.get(_normalize_key_name(target_key)) if target_key else None
    if tonic_pc is None or target_mode not in {"major", "minor"}:
        return None, None
    if target_mode == "major":
        relative_pc = (tonic_pc + 9) % 12
        relative_mode = "minor"
    else:
        relative_pc = (tonic_pc + 3) % 12
        relative_mode = "major"
    pc_to_name = {value: key for key, value in TONIC_TO_PC.items() if len(key) <= 2 and "#" not in key and "B#" not in key and "E#" not in key and "FB" not in key and "CB" not in key}
    relative_key = pc_to_name.get(relative_pc)
    if relative_key:
        relative_key = relative_key.replace("B", "b") if len(relative_key) > 1 else relative_key
    return relative_key, relative_mode


def tempo_bin_index(bpm: float | None) -> int | None:
    if bpm is None or bpm <= 0:
        return None
    for idx, (low, high) in enumerate(TEMPO_BINS):
        if low <= bpm <= high:
            return idx
    if bpm < TEMPO_BINS[0][0]:
        return 0
    return len(TEMPO_BINS) - 1


def penalty_invalid(
    score,
    min_notes: int = 8,
    min_duration_sec: float = 5.0,
    max_repeat_ratio: float = 0.8,
) -> float:
    try:
        if score is None:
            return 1.0

        all_notes = _collect_notes(score)

        if len(all_notes) < min_notes:
            return 1.0

        tpq = score.tpq if hasattr(score, "tpq") else 480
        total_ticks = max(n.time + n.duration for n in all_notes)

        bpm = extract_score_bpm(score) or 120.0

        beats = total_ticks / tpq
        duration_sec = beats * (60.0 / bpm)
        if duration_sec < min_duration_sec:
            return 1.0

        pitches = [n.pitch for n in all_notes]
        most_common_count = max(pitches.count(p) for p in set(pitches))
        if most_common_count / len(pitches) > max_repeat_ratio:
            return 1.0

        return 0.0

    except Exception:
        return 1.0


def reward_key(score, target_key: str | None = None, target_mode: str | None = None) -> float:
    try:
        all_notes = _collect_notes(score)
        if not all_notes:
            return 0.0

        pitch_classes = [n.pitch % 12 for n in all_notes]
        tonic_pc = TONIC_TO_PC.get(_normalize_key_name(target_key)) if target_key else None
        if tonic_pc is not None and target_mode in MODE_INTERVALS:
            scale = _scale_pitch_classes(tonic_pc, target_mode)
            return float(sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes))

        best = max(sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes) for scale in _SCALES)
        return float(best)
    except Exception:
        return 0.0


def reward_key_profile(score, target_key: str | None = None, target_mode: str | None = None) -> float:
    try:
        tonic_pc = TONIC_TO_PC.get(_normalize_key_name(target_key)) if target_key else None
        if tonic_pc is None or target_mode not in MODE_INTERVALS:
            return 0.0
        observed = _duration_weighted_pitch_profile(score)
        if observed is None:
            return 0.0

        target_corr = _pearson_corr(observed, _rotated_profile(target_mode, tonic_pc))
        other_corrs = []
        for root in range(12):
            for mode in ("major", "minor"):
                if root == tonic_pc and mode == target_mode:
                    continue
                other_corrs.append(_pearson_corr(observed, _rotated_profile(mode, root)))
        best_other = max(other_corrs) if other_corrs else -1.0
        absolute = (target_corr + 1.0) * 0.5
        margin = max(0.0, min(1.0, (target_corr - best_other + 1.0) * 0.5))
        return float(max(0.0, min(1.0, 0.75 * absolute + 0.25 * margin)))
    except Exception:
        return 0.0


def reward_rhythm(score, subdivisions: int = 16) -> float:
    try:
        all_notes = _collect_notes(score)
        if not all_notes:
            return 0.0

        tpq = score.tpq if hasattr(score, "tpq") else 480
        grid = tpq * 4 // subdivisions
        if grid == 0:
            return 0.0

        deviations = []
        for note in all_notes:
            remainder = note.time % grid
            deviations.append(min(remainder, grid - remainder) / grid)

        return 1.0 - 2.0 * float(np.mean(deviations))
    except Exception:
        return 0.0


def reward_tempo(score, target_bpm: float | None = None) -> float:
    if not target_bpm or target_bpm <= 0:
        return 0.0
    actual_bpm = extract_score_bpm(score)
    if not actual_bpm or actual_bpm <= 0:
        return 0.0
    relative_error = abs(actual_bpm - float(target_bpm)) / max(float(target_bpm), 1.0)
    return float(max(0.0, 1.0 - min(relative_error, 1.0)))


def reward_meter(
    score,
    target_numerator: int | None = None,
    target_denominator: int | None = None,
) -> float:
    if not target_numerator or not target_denominator:
        return 0.0
    actual_numerator, actual_denominator = extract_score_time_signature(score)
    if not actual_numerator or not actual_denominator:
        return 0.0
    if actual_numerator == target_numerator and actual_denominator == target_denominator:
        return 1.0
    numerator_score = max(
        0.0,
        1.0 - abs(actual_numerator - target_numerator) / max(float(target_numerator), 1.0),
    )
    denominator_score = 1.0 if actual_denominator == target_denominator else 0.0
    return float(0.7 * numerator_score + 0.3 * denominator_score)


def reward_meter_template(
    score,
    target_numerator: int | None = None,
    target_denominator: int | None = None,
) -> float:
    try:
        if not target_numerator or not target_denominator:
            return 0.0
        target = (int(target_numerator), int(target_denominator))
        candidates = list(METER_CANDIDATES)
        if target not in candidates:
            candidates.append(target)
        fits = {candidate: _meter_template_fit(score, *candidate) for candidate in candidates}
        target_fit = fits.get(target, 0.0)
        best_fit = max(fits.values()) if fits else 0.0
        if best_fit <= 1e-8:
            return 0.0
        relative = target_fit / best_fit
        return float(max(0.0, min(1.0, 0.7 * relative + 0.3 * target_fit)))
    except Exception:
        return 0.0


def reward_key_exact(score, target_key: str | None = None, target_mode: str | None = None) -> float:
    estimated_key, estimated_mode = estimate_key_mode(score)
    if not estimated_key or not estimated_mode or not target_key or not target_mode:
        return 0.0
    return 1.0 if estimated_key == target_key and estimated_mode == target_mode else 0.0


def reward_key_relative(score, target_key: str | None = None, target_mode: str | None = None) -> float:
    estimated_key, estimated_mode = estimate_key_mode(score)
    if not estimated_key or not estimated_mode or not target_key or not target_mode:
        return 0.0
    if estimated_key == target_key and estimated_mode == target_mode:
        return 1.0
    relative_key, relative_mode = _relative_mode_key(target_key, target_mode)
    return 1.0 if estimated_key == relative_key and estimated_mode == relative_mode else 0.0


def reward_tempo_bin(score, target_bpm: float | None = None) -> float:
    target_bin = tempo_bin_index(target_bpm)
    actual_bin = tempo_bin_index(extract_score_bpm(score))
    if target_bin is None or actual_bin is None:
        return 0.0
    return 1.0 if target_bin == actual_bin else 0.0


def reward_tempo_bin_tolerant(score, target_bpm: float | None = None) -> float:
    target_bin = tempo_bin_index(target_bpm)
    actual_bin = tempo_bin_index(extract_score_bpm(score))
    if target_bin is None or actual_bin is None:
        return 0.0
    distance = abs(target_bin - actual_bin)
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def reward_duration_balance(score, target_bpm: float | None = None) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    duration_beats = _score_duration_beats(score, notes)
    # A practical generated MIDI should have enough material without becoming a
    # very long rollout artifact. The target peak is 16 bars in 4/4.
    return _triangular_score(duration_beats, low=8.0, target=64.0, high=160.0)


def reward_note_density(score) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    duration_beats = max(_score_duration_beats(score, notes), 1.0)
    notes_per_beat = len(notes) / duration_beats
    return _triangular_score(notes_per_beat, low=0.25, target=2.5, high=9.0)


def reward_moderate_note_density(
    score,
    low: float = 0.4,
    target: float = 3.0,
    high: float = 12.0,
) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    duration_beats = max(_score_duration_beats(score, notes), 1.0)
    notes_per_beat = len(notes) / duration_beats
    return _triangular_score(notes_per_beat, low=low, target=target, high=high)


def reward_pitch_range(score) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    pitches = [int(note.pitch) for note in notes]
    pitch_span = max(pitches) - min(pitches)
    return _triangular_score(float(pitch_span), low=4.0, target=28.0, high=72.0)


def reward_pitch_diversity(score) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    unique_pitch_classes = len({int(note.pitch) % 12 for note in notes})
    return _triangular_score(float(unique_pitch_classes), low=2.0, target=7.0, high=12.5)


def reward_polyphony_balance(score) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    groups = _onset_groups(notes)
    mean_notes_per_onset = float(np.mean([len(group) for group in groups.values()]))
    return _triangular_score(mean_notes_per_onset, low=0.9, target=2.0, high=6.0)


def reward_duration_variety(score) -> float:
    notes = _collect_notes(score)
    if not notes:
        return 0.0
    tpq = max(_score_tpq(score), 1)
    duration_bins = Counter(round(float(note.duration) / tpq, 2) for note in notes)
    if not duration_bins:
        return 0.0
    entropy = 0.0
    total = sum(duration_bins.values())
    for count in duration_bins.values():
        p = count / total
        entropy -= p * np.log2(max(p, 1e-8))
    max_entropy = np.log2(max(len(duration_bins), 2))
    return float(min(1.0, entropy / max_entropy))


def reward_stepwise_motion(score) -> float:
    notes = _collect_notes(score)
    if len(notes) < 2:
        return 0.0
    groups = _onset_groups(notes)
    melody_notes = [max(group, key=lambda note: (int(note.pitch), int(note.duration))) for _, group in sorted(groups.items())]
    if len(melody_notes) < 2:
        return 0.0
    intervals = [abs(int(curr.pitch) - int(prev.pitch)) for prev, curr in zip(melody_notes[:-1], melody_notes[1:])]
    stepwise_ratio = sum(1 for interval in intervals if interval <= 5) / len(intervals)
    leap_ratio = sum(1 for interval in intervals if interval >= 12) / len(intervals)
    return float(max(0.0, stepwise_ratio - 0.5 * leap_ratio))


def reward_repetition_balance(score) -> float:
    notes = _collect_notes(score)
    if len(notes) < 4:
        return 0.0
    pitch_classes = [int(note.pitch) % 12 for note in notes]
    most_common_ratio = max(Counter(pitch_classes).values()) / len(pitch_classes)
    # Too little repetition sounds random; too much is degenerative.
    return _triangular_score(most_common_ratio, low=0.05, target=0.22, high=0.65)


def reward_track_count(score, max_tracks: int = 4) -> float:
    try:
        track_count = len(_nonempty_tracks(score))
        if track_count <= 0:
            return 0.0
        if track_count <= max_tracks:
            return 1.0
        return float(max(0.0, 1.0 - (track_count - max_tracks) / max(float(max_tracks), 1.0)))
    except Exception:
        return 0.0


def reward_track_count_exact(score, target_tracks: int = 2) -> float:
    try:
        track_count = len(_nonempty_tracks(score))
        if track_count <= 0:
            return 0.0
        distance = abs(track_count - int(target_tracks))
        if distance == 0:
            return 1.0
        return float(max(0.0, 1.0 - distance / max(float(target_tracks), 1.0)))
    except Exception:
        return 0.0


def reward_no_drums(score) -> float:
    try:
        tracks = _nonempty_tracks(score)
        if not tracks:
            return 0.0
        for track in tracks:
            if _is_drum_track(track):
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def reward_drum_note_ratio(
    score,
    target_drum_note_ratio: float = 0.15,
    max_drum_note_ratio: float = 0.60,
) -> float:
    try:
        tracks = _nonempty_tracks(score)
        total_notes = sum(len(getattr(track, "notes", []) or []) for track in tracks)
        if total_notes <= 0:
            return 0.0
        drum_notes = sum(len(getattr(track, "notes", []) or []) for track in tracks if _is_drum_track(track))
        ratio = drum_notes / max(float(total_notes), 1.0)
        if ratio <= float(target_drum_note_ratio):
            return 1.0
        if ratio >= float(max_drum_note_ratio):
            return 0.0
        width = max(float(max_drum_note_ratio) - float(target_drum_note_ratio), 1e-8)
        return float(max(0.0, 1.0 - (ratio - float(target_drum_note_ratio)) / width))
    except Exception:
        return 0.0


def compute_reward(
    score,
    key_weight: float = 0.5,
    rhythm_weight: float = 0.5,
    key_conditioned_weight: float = 0.0,
    key_exact_weight: float = 0.0,
    key_relative_weight: float = 0.0,
    meter_weight: float = 0.0,
    tempo_weight: float = 0.0,
    tempo_bin_weight: float = 0.0,
    tempo_bin_tolerant_weight: float = 0.0,
    key_profile_weight: float = 0.0,
    meter_template_weight: float = 0.0,
    duration_balance_weight: float = 0.0,
    note_density_weight: float = 0.0,
    moderate_note_density_weight: float = 0.0,
    pitch_range_weight: float = 0.0,
    pitch_diversity_weight: float = 0.0,
    polyphony_balance_weight: float = 0.0,
    duration_variety_weight: float = 0.0,
    stepwise_motion_weight: float = 0.0,
    repetition_balance_weight: float = 0.0,
    track_count_weight: float = 0.0,
    track_count_exact_weight: float = 0.0,
    no_drums_weight: float = 0.0,
    drum_note_ratio_weight: float = 0.0,
    max_tracks: int = 4,
    target_tracks: int = 2,
    moderate_note_density_low: float = 0.4,
    moderate_note_density_target: float = 3.0,
    moderate_note_density_high: float = 12.0,
    target_drum_note_ratio: float = 0.15,
    max_drum_note_ratio: float = 0.60,
    invalid_penalty: float = 2.0,
    invalid_enabled: bool = True,
    rhythm_subdivisions: int = 16,
    target_key: str | None = None,
    target_mode: str | None = None,
    target_bpm: float | None = None,
    target_meter_numerator: int | None = None,
    target_meter_denominator: int | None = None,
    **invalid_kwargs,
) -> dict[str, float]:
    p_inv = penalty_invalid(score, **invalid_kwargs)
    if invalid_enabled and p_inv > 0.0:
        return {
            "total": -invalid_penalty,
            "key": 0.0,
            "key_conditioned": 0.0,
            "key_exact": 0.0,
            "key_relative": 0.0,
            "rhythm": 0.0,
            "meter": 0.0,
            "tempo": 0.0,
            "tempo_bin": 0.0,
            "tempo_bin_tolerant": 0.0,
            "key_profile": 0.0,
            "meter_template": 0.0,
            "duration_balance": 0.0,
            "note_density": 0.0,
            "moderate_note_density": 0.0,
            "pitch_range": 0.0,
            "pitch_diversity": 0.0,
            "polyphony_balance": 0.0,
            "duration_variety": 0.0,
            "stepwise_motion": 0.0,
            "repetition_balance": 0.0,
            "track_count": 0.0,
            "track_count_exact": 0.0,
            "no_drums": 0.0,
            "drum_note_ratio": 0.0,
            "invalid": 1.0,
        }

    r_key = reward_key(score)
    r_key_conditioned = reward_key(score, target_key=target_key, target_mode=target_mode)
    r_key_exact = reward_key_exact(score, target_key=target_key, target_mode=target_mode)
    r_key_relative = reward_key_relative(score, target_key=target_key, target_mode=target_mode)
    r_rhy = reward_rhythm(score, subdivisions=rhythm_subdivisions)
    r_meter = reward_meter(
        score,
        target_numerator=target_meter_numerator,
        target_denominator=target_meter_denominator,
    )
    r_tempo = reward_tempo(score, target_bpm=target_bpm)
    r_tempo_bin = reward_tempo_bin(score, target_bpm=target_bpm)
    r_tempo_bin_tolerant = reward_tempo_bin_tolerant(score, target_bpm=target_bpm)
    r_key_profile = reward_key_profile(score, target_key=target_key, target_mode=target_mode)
    r_meter_template = reward_meter_template(
        score,
        target_numerator=target_meter_numerator,
        target_denominator=target_meter_denominator,
    )
    r_duration_balance = reward_duration_balance(score, target_bpm=target_bpm)
    r_note_density = reward_note_density(score)
    r_moderate_note_density = reward_moderate_note_density(
        score,
        low=moderate_note_density_low,
        target=moderate_note_density_target,
        high=moderate_note_density_high,
    )
    r_pitch_range = reward_pitch_range(score)
    r_pitch_diversity = reward_pitch_diversity(score)
    r_polyphony_balance = reward_polyphony_balance(score)
    r_duration_variety = reward_duration_variety(score)
    r_stepwise_motion = reward_stepwise_motion(score)
    r_repetition_balance = reward_repetition_balance(score)
    r_track_count = reward_track_count(score, max_tracks=max_tracks)
    r_track_count_exact = reward_track_count_exact(score, target_tracks=target_tracks)
    r_no_drums = reward_no_drums(score)
    r_drum_note_ratio = reward_drum_note_ratio(
        score,
        target_drum_note_ratio=target_drum_note_ratio,
        max_drum_note_ratio=max_drum_note_ratio,
    )
    total = (
        key_weight * r_key
        + rhythm_weight * r_rhy
        + key_conditioned_weight * r_key_conditioned
        + key_exact_weight * r_key_exact
        + key_relative_weight * r_key_relative
        + meter_weight * r_meter
        + tempo_weight * r_tempo
        + tempo_bin_weight * r_tempo_bin
        + tempo_bin_tolerant_weight * r_tempo_bin_tolerant
        + key_profile_weight * r_key_profile
        + meter_template_weight * r_meter_template
        + duration_balance_weight * r_duration_balance
        + note_density_weight * r_note_density
        + moderate_note_density_weight * r_moderate_note_density
        + pitch_range_weight * r_pitch_range
        + pitch_diversity_weight * r_pitch_diversity
        + polyphony_balance_weight * r_polyphony_balance
        + duration_variety_weight * r_duration_variety
        + stepwise_motion_weight * r_stepwise_motion
        + repetition_balance_weight * r_repetition_balance
        + track_count_weight * r_track_count
        + track_count_exact_weight * r_track_count_exact
        + no_drums_weight * r_no_drums
        + drum_note_ratio_weight * r_drum_note_ratio
    )
    return {
        "total": total,
        "key": r_key,
        "key_conditioned": r_key_conditioned,
        "key_exact": r_key_exact,
        "key_relative": r_key_relative,
        "rhythm": r_rhy,
        "meter": r_meter,
        "tempo": r_tempo,
        "tempo_bin": r_tempo_bin,
        "tempo_bin_tolerant": r_tempo_bin_tolerant,
        "key_profile": r_key_profile,
        "meter_template": r_meter_template,
        "duration_balance": r_duration_balance,
        "note_density": r_note_density,
        "moderate_note_density": r_moderate_note_density,
        "pitch_range": r_pitch_range,
        "pitch_diversity": r_pitch_diversity,
        "polyphony_balance": r_polyphony_balance,
        "duration_variety": r_duration_variety,
        "stepwise_motion": r_stepwise_motion,
        "repetition_balance": r_repetition_balance,
        "track_count": r_track_count,
        "track_count_exact": r_track_count_exact,
        "no_drums": r_no_drums,
        "drum_note_ratio": r_drum_note_ratio,
        "invalid": float(p_inv),
    }


def batch_rewards(scores: list, cfg_reward, captions: list[str] | None = None) -> list[dict[str, float]]:
    rewards = []
    captions = captions or [None] * len(scores)
    for score, caption in zip(scores, captions):
        metadata = parse_prompt_metadata(caption) if caption else None
        rewards.append(
            compute_reward(
                score,
                key_weight=cfg_reward.key_weight,
                rhythm_weight=cfg_reward.rhythm_weight,
                key_conditioned_weight=float(cfg_reward.get("key_conditioned_weight", 0.0)),
                key_exact_weight=float(cfg_reward.get("key_exact_weight", 0.0)),
                key_relative_weight=float(cfg_reward.get("key_relative_weight", 0.0)),
                meter_weight=float(cfg_reward.get("meter_weight", 0.0)),
                tempo_weight=float(cfg_reward.get("tempo_weight", 0.0)),
                tempo_bin_weight=float(cfg_reward.get("tempo_bin_weight", 0.0)),
                tempo_bin_tolerant_weight=float(cfg_reward.get("tempo_bin_tolerant_weight", 0.0)),
                key_profile_weight=float(cfg_reward.get("key_profile_weight", 0.0)),
                meter_template_weight=float(cfg_reward.get("meter_template_weight", 0.0)),
                duration_balance_weight=float(cfg_reward.get("duration_balance_weight", 0.0)),
                note_density_weight=float(cfg_reward.get("note_density_weight", 0.0)),
                moderate_note_density_weight=float(cfg_reward.get("moderate_note_density_weight", 0.0)),
                pitch_range_weight=float(cfg_reward.get("pitch_range_weight", 0.0)),
                pitch_diversity_weight=float(cfg_reward.get("pitch_diversity_weight", 0.0)),
                polyphony_balance_weight=float(cfg_reward.get("polyphony_balance_weight", 0.0)),
                duration_variety_weight=float(cfg_reward.get("duration_variety_weight", 0.0)),
                stepwise_motion_weight=float(cfg_reward.get("stepwise_motion_weight", 0.0)),
                repetition_balance_weight=float(cfg_reward.get("repetition_balance_weight", 0.0)),
                track_count_weight=float(cfg_reward.get("track_count_weight", 0.0)),
                track_count_exact_weight=float(cfg_reward.get("track_count_exact_weight", 0.0)),
                no_drums_weight=float(cfg_reward.get("no_drums_weight", 0.0)),
                drum_note_ratio_weight=float(cfg_reward.get("drum_note_ratio_weight", 0.0)),
                max_tracks=int(cfg_reward.get("max_tracks", 4)),
                target_tracks=int(cfg_reward.get("target_tracks", 2)),
                moderate_note_density_low=float(cfg_reward.get("moderate_note_density_low", 0.4)),
                moderate_note_density_target=float(cfg_reward.get("moderate_note_density_target", 3.0)),
                moderate_note_density_high=float(cfg_reward.get("moderate_note_density_high", 12.0)),
                target_drum_note_ratio=float(cfg_reward.get("target_drum_note_ratio", 0.15)),
                max_drum_note_ratio=float(cfg_reward.get("max_drum_note_ratio", 0.60)),
                invalid_penalty=cfg_reward.invalid_penalty,
                invalid_enabled=bool(cfg_reward.get("invalid_enabled", True)),
                rhythm_subdivisions=cfg_reward.rhythm_subdivisions,
                target_key=metadata.get("key") if metadata else None,
                target_mode=metadata.get("mode") if metadata else None,
                target_bpm=metadata.get("bpm") if metadata else None,
                target_meter_numerator=metadata.get("meter_numerator") if metadata else None,
                target_meter_denominator=metadata.get("meter_denominator") if metadata else None,
                min_notes=cfg_reward.min_notes,
                min_duration_sec=cfg_reward.min_duration_sec,
                max_repeat_ratio=cfg_reward.max_repeat_ratio,
            )
        )
    return rewards
