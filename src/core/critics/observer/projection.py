from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi


DEFAULT_QUANTIZE_GRID_SEC = 0.125
DEFAULT_MIN_NOTE_DURATION_SEC = 0.05
MELODY_VELOCITY = 92
CHORDS_VELOCITY = 72


@dataclass
class TrackFeatures:
    index: int
    name: str
    note_count: int
    avg_pitch: float
    pitch_range: int
    onset_count: int
    mean_notes_per_onset: float
    polyphony_ratio: float
    chord_block_ratio: float
    stepwise_ratio: float
    melody_score: float


@dataclass
class ProjectionResult:
    input_path: str
    output_path: str
    strategy: str
    melody_source: str
    chords_sources: list[str]
    features: list[TrackFeatures]
    original_note_count: int = 0
    projected_note_count: int = 0
    melody_note_count: int = 0
    chords_note_count: int = 0


def _group_notes_by_onset(notes, precision: int = 4) -> dict[float, list[Any]]:
    grouped: dict[float, list[Any]] = {}
    for note in notes:
        onset = round(float(note.start), precision)
        grouped.setdefault(onset, []).append(note)
    return grouped


def _quantize_time(value: float, grid_sec: float) -> float:
    if grid_sec <= 0:
        return float(value)
    return round(float(value) / grid_sec) * grid_sec


def _stepwise_ratio(notes) -> float:
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    if len(ordered) < 2:
        return 0.0
    intervals = [abs(curr.pitch - prev.pitch) for prev, curr in zip(ordered[:-1], ordered[1:])]
    return sum(1 for interval in intervals if interval <= 5) / len(intervals)


def analyze_track(index: int, instrument: pretty_midi.Instrument) -> TrackFeatures:
    notes = list(instrument.notes)
    onset_groups = _group_notes_by_onset(notes)
    onset_sizes = [len(group) for group in onset_groups.values()] or [0]
    pitches = [note.pitch for note in notes] or [0]
    avg_pitch = mean(pitches)
    pitch_range = max(pitches) - min(pitches) if notes else 0
    mean_notes_per_onset = mean(onset_sizes)
    polyphony_ratio = sum(1 for size in onset_sizes if size > 1) / max(1, len(onset_sizes))
    chord_block_ratio = sum(1 for size in onset_sizes if size >= 3) / max(1, len(onset_sizes))
    stepwise_ratio = _stepwise_ratio(notes)

    melody_score = (
        0.30 * avg_pitch
        + 8.0 * stepwise_ratio
        + 6.0 * (1.0 - polyphony_ratio)
        + 4.0 * (1.0 - chord_block_ratio)
        + 2.0 * (1.0 / max(1.0, mean_notes_per_onset))
        + min(12.0, pitch_range) * 0.15
    )
    if len(notes) < 8:
        melody_score -= 18.0
    if avg_pitch < 48:
        melody_score -= 8.0

    return TrackFeatures(
        index=index,
        name=instrument.name or f"track_{index}",
        note_count=len(notes),
        avg_pitch=avg_pitch,
        pitch_range=pitch_range,
        onset_count=len(onset_groups),
        mean_notes_per_onset=mean_notes_per_onset,
        polyphony_ratio=polyphony_ratio,
        chord_block_ratio=chord_block_ratio,
        stepwise_ratio=stepwise_ratio,
        melody_score=melody_score,
    )


def _copy_note(
    note: pretty_midi.Note,
    velocity: int | None = None,
    quantize_grid_sec: float | None = None,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> pretty_midi.Note | None:
    start = float(note.start)
    end = float(note.end)
    if quantize_grid_sec:
        start = _quantize_time(start, quantize_grid_sec)
        end = _quantize_time(end, quantize_grid_sec)
    if end <= start:
        end = start + min_duration_sec
    duration = end - start
    if duration < min_duration_sec:
        end = start + min_duration_sec
    return pretty_midi.Note(
        velocity=int(velocity if velocity is not None else note.velocity),
        pitch=int(note.pitch),
        start=float(start),
        end=float(end),
    )


def _dedupe_notes(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
    seen: dict[tuple[int, float, float], pretty_midi.Note] = {}
    for note in notes:
        key = (int(note.pitch), round(float(note.start), 4), round(float(note.end), 4))
        current = seen.get(key)
        if current is None or note.velocity > current.velocity:
            seen[key] = note
    return sorted(seen.values(), key=lambda note: (note.start, note.pitch, note.end))


def _make_monophonic_topline(
    notes,
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> list[pretty_midi.Note]:
    grouped = _group_notes_by_onset(notes)
    topline: list[pretty_midi.Note] = []
    for onset in sorted(grouped):
        top_note = max(grouped[onset], key=lambda note: (note.pitch, note.end - note.start))
        copied = _copy_note(
            top_note,
            velocity=MELODY_VELOCITY,
            quantize_grid_sec=quantize_grid_sec,
            min_duration_sec=min_duration_sec,
        )
        if copied is not None:
            topline.append(copied)
    return _dedupe_notes(topline)


def _merge_notes(
    instruments: list[pretty_midi.Instrument],
    velocity: int | None = None,
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> list[pretty_midi.Note]:
    merged: list[pretty_midi.Note] = []
    for instrument in instruments:
        for note in instrument.notes:
            copied = _copy_note(
                note,
                velocity=velocity,
                quantize_grid_sec=quantize_grid_sec,
                min_duration_sec=min_duration_sec,
            )
            if copied is not None:
                merged.append(copied)
    return _dedupe_notes(merged)


def _project_by_tracks(
    pm: pretty_midi.PrettyMIDI,
    non_drum: list[pretty_midi.Instrument],
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument, ProjectionResult]:
    features = [analyze_track(index, instrument) for index, instrument in enumerate(non_drum)]
    ranked = sorted(features, key=lambda item: item.melody_score, reverse=True)
    melody_feature = ranked[0]
    melody_inst = non_drum[melody_feature.index]

    melody_notes = _make_monophonic_topline(
        melody_inst.notes,
        quantize_grid_sec=quantize_grid_sec,
        min_duration_sec=min_duration_sec,
    )
    accompaniment_sources = [inst for idx, inst in enumerate(non_drum) if idx != melody_feature.index]
    chords_notes = _merge_notes(
        accompaniment_sources,
        velocity=CHORDS_VELOCITY,
        quantize_grid_sec=quantize_grid_sec,
        min_duration_sec=min_duration_sec,
    )

    melody = pretty_midi.Instrument(program=melody_inst.program, is_drum=False, name="melody")
    melody.notes = melody_notes
    chords = pretty_midi.Instrument(program=0, is_drum=False, name="chords")
    chords.notes = chords_notes

    result = ProjectionResult(
        input_path="",
        output_path="",
        strategy="track_based",
        melody_source=melody_feature.name,
        chords_sources=[feature.name for feature in features if feature.index != melody_feature.index],
        features=features,
    )
    return melody, chords, result


def _project_by_events(
    pm: pretty_midi.PrettyMIDI,
    non_drum: list[pretty_midi.Instrument],
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> tuple[pretty_midi.Instrument, pretty_midi.Instrument, ProjectionResult]:
    all_notes = _merge_notes(
        non_drum,
        quantize_grid_sec=quantize_grid_sec,
        min_duration_sec=min_duration_sec,
    )
    grouped = _group_notes_by_onset(all_notes)
    melody = pretty_midi.Instrument(program=0, is_drum=False, name="melody")
    chords = pretty_midi.Instrument(program=0, is_drum=False, name="chords")

    for onset in sorted(grouped):
        group = sorted(grouped[onset], key=lambda note: (note.pitch, note.end - note.start), reverse=True)
        melody_note = _copy_note(
            group[0],
            velocity=MELODY_VELOCITY,
            quantize_grid_sec=quantize_grid_sec,
            min_duration_sec=min_duration_sec,
        )
        if melody_note is not None:
            melody.notes.append(melody_note)
        for note in group[1:]:
            chord_note = _copy_note(
                note,
                velocity=CHORDS_VELOCITY,
                quantize_grid_sec=quantize_grid_sec,
                min_duration_sec=min_duration_sec,
            )
            if chord_note is not None:
                chords.notes.append(chord_note)

    melody.notes = _dedupe_notes(melody.notes)
    chords.notes = _dedupe_notes(chords.notes)

    result = ProjectionResult(
        input_path="",
        output_path="",
        strategy="event_based_fallback",
        melody_source="top_voice_per_onset",
        chords_sources=["remaining_notes_per_onset"],
        features=[],
    )
    return melody, chords, result


def _borrow_chords_from_lower_voices(
    non_drum: list[pretty_midi.Instrument],
    melody_notes: list[pretty_midi.Note],
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> list[pretty_midi.Note]:
    melody_keys = {(round(note.start, 4), int(note.pitch)) for note in melody_notes}
    all_notes = _merge_notes(
        non_drum,
        velocity=CHORDS_VELOCITY,
        quantize_grid_sec=quantize_grid_sec,
        min_duration_sec=min_duration_sec,
    )
    borrowed = [
        note
        for note in all_notes
        if (round(note.start, 4), int(note.pitch)) not in melody_keys
    ]
    return _dedupe_notes(borrowed)


def project_to_observer_format(
    input_midi: str | Path,
    output_midi: str | Path,
    quantize_grid_sec: float = DEFAULT_QUANTIZE_GRID_SEC,
    min_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
) -> ProjectionResult:
    input_path = Path(input_midi)
    output_path = Path(output_midi)
    pm = pretty_midi.PrettyMIDI(str(input_path))
    non_drum = [instrument for instrument in pm.instruments if not instrument.is_drum and instrument.notes]
    original_note_count = sum(len(instrument.notes) for instrument in non_drum)

    if not non_drum:
        raise ValueError(f"No non-drum note tracks found in {input_path}")

    use_event_fallback = len(non_drum) == 1
    if not use_event_fallback:
        melody, chords, result = _project_by_tracks(
            pm,
            non_drum,
            quantize_grid_sec=quantize_grid_sec,
            min_duration_sec=min_duration_sec,
        )
        if not chords.notes or len(melody.notes) < 8:
            melody, chords, result = _project_by_events(
                pm,
                non_drum,
                quantize_grid_sec=quantize_grid_sec,
                min_duration_sec=min_duration_sec,
            )
    else:
        melody, chords, result = _project_by_events(
            pm,
            non_drum,
            quantize_grid_sec=quantize_grid_sec,
            min_duration_sec=min_duration_sec,
        )

    if not chords.notes:
        chords.notes = _borrow_chords_from_lower_voices(
            non_drum,
            melody.notes,
            quantize_grid_sec=quantize_grid_sec,
            min_duration_sec=min_duration_sec,
        )

    melody.notes = _dedupe_notes(melody.notes)
    chords.notes = _dedupe_notes(chords.notes)

    tempo_times, tempo_bpms = pm.get_tempo_changes()
    initial_tempo = float(tempo_bpms[0]) if len(tempo_bpms) else 120.0
    projected = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    projected.instruments = [melody, chords]

    for ts in pm.time_signature_changes:
        projected.time_signature_changes.append(
            pretty_midi.TimeSignature(int(ts.numerator), int(ts.denominator), float(ts.time))
        )
    for key_sig in getattr(pm, "key_signature_changes", []):
        projected.key_signature_changes.append(
            pretty_midi.KeySignature(int(key_sig.key_number), float(key_sig.time))
        )
    for lyric in getattr(pm, "lyrics", []):
        projected.lyrics.append(lyric)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected.write(str(output_path))

    result.input_path = str(input_path)
    result.output_path = str(output_path)
    result.original_note_count = original_note_count
    result.melody_note_count = len(melody.notes)
    result.chords_note_count = len(chords.notes)
    result.projected_note_count = len(melody.notes) + len(chords.notes)
    return result
