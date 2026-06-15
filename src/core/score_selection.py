from __future__ import annotations

from dataclasses import dataclass, asdict
from math import log1p
from statistics import mean


@dataclass
class TrackSelectionStats:
    raw_track_count: int
    candidate_track_count: int
    selected_track_count: int
    selected_track_indices: list[int]
    selected_track_names: list[str]
    track_decisions: list[dict]


def _track_notes(track) -> list:
    return list(getattr(track, "notes", []) or [])


def _is_drum_track(track) -> bool:
    name = str(getattr(track, "name", "") or "").lower()
    if "drum" in name or "percussion" in name:
        return True
    if bool(getattr(track, "is_drum", False)):
        return True
    program = getattr(track, "program", None)
    return program is not None and int(program) < 0


def _track_end(notes: list) -> float:
    if not notes:
        return 0.0
    return max(float(getattr(note, "time", 0.0) or 0.0) + float(getattr(note, "duration", 0.0) or 0.0) for note in notes)


def _track_rank_score(track, index: int) -> float:
    notes = _track_notes(track)
    if not notes:
        return -1e9
    pitches = [int(getattr(note, "pitch", 0) or 0) for note in notes]
    pitch_classes = {pitch % 12 for pitch in pitches}
    avg_pitch = mean(pitches)
    onset_count = len({int(getattr(note, "time", 0) or 0) for note in notes})
    mean_notes_per_onset = len(notes) / max(onset_count, 1)
    duration = _track_end(notes)

    score = 2.0 * log1p(len(notes))
    score += 0.45 * len(pitch_classes)
    score += 0.0008 * duration
    score += 0.6 if 40 <= avg_pitch <= 88 else -0.4
    score += 0.5 if mean_notes_per_onset <= 4.0 else -0.5
    score -= 0.01 * index
    return float(score)


def select_relevant_tracks(score, max_tracks: int = 4, drop_drums: bool = True):
    """Return a copied score containing the most useful non-drum tracks.

    The selection is intentionally conservative: it drops empty/drum tracks,
    ranks the remaining tracks by amount and variety of musical material, keeps
    up to max_tracks, and preserves their original order in the returned score.
    """

    if score is None:
        return None, TrackSelectionStats(0, 0, 0, [], [], [])

    tracks = list(getattr(score, "tracks", []) or [])
    candidates = []
    for index, track in enumerate(tracks):
        notes = _track_notes(track)
        if not notes:
            continue
        if drop_drums and _is_drum_track(track):
            continue
        candidates.append((index, track, _track_rank_score(track, index)))

    selected = sorted(candidates, key=lambda item: item[2], reverse=True)[: max(1, int(max_tracks))]
    selected = sorted(selected, key=lambda item: item[0])
    selected_indices = {int(index) for index, _, _ in selected}

    selected_score = score.copy()
    selected_score.tracks.clear()
    for _, track, _ in selected:
        selected_score.tracks.append(track.copy())
    try:
        selected_score.sort()
    except Exception:
        pass

    stats = TrackSelectionStats(
        raw_track_count=len([track for track in tracks if _track_notes(track)]),
        candidate_track_count=len(candidates),
        selected_track_count=len(selected),
        selected_track_indices=[int(index) for index, _, _ in selected],
        selected_track_names=[str(getattr(track, "name", "") or f"track_{index}") for index, track, _ in selected],
        track_decisions=[],
    )
    candidate_scores = {int(index): float(rank_score) for index, _, rank_score in candidates}
    for index, track in enumerate(tracks):
        notes = _track_notes(track)
        name = str(getattr(track, "name", "") or f"track_{index}")
        if not notes:
            decision = "reject"
            reason = "empty"
        elif drop_drums and _is_drum_track(track):
            decision = "reject"
            reason = "drum"
        elif index in selected_indices:
            decision = "accept"
            reason = "selected"
        else:
            decision = "reject"
            reason = "lower_rank"
        stats.track_decisions.append(
            {
                "track_index": int(index),
                "track_name": name,
                "decision": decision,
                "reason": reason,
                "note_count": int(len(notes)),
                "rank_score": candidate_scores.get(int(index)),
            }
        )
    return selected_score, stats


def select_scores_for_reward(scores: list, max_tracks: int = 4, drop_drums: bool = True):
    selected_scores = []
    stats = []
    for score in scores:
        selected, item_stats = select_relevant_tracks(score, max_tracks=max_tracks, drop_drums=drop_drums)
        selected_scores.append(selected)
        stats.append(asdict(item_stats))
    return selected_scores, stats
