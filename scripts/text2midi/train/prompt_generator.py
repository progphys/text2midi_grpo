"""
Synthetic prompt generator for GRPO training.

All musical parameters vary: key, mode, BPM, meter, instruments, style, mood, chord progression.
Two tracks are always present: melody + chords.
"""

from __future__ import annotations
import random
from dataclasses import dataclass

# ── Musical parameter pools ────────────────────────────────────────────────────

KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

MODES = ["major", "minor"]

# (bpm_min, bpm_max, label)  — label used in text naturally
BPM_RANGES = [
    (60,  75,  "slow"),
    (76,  99,  "moderate"),
    (100, 120, "moderately fast"),
    (121, 150, "fast"),
    (151, 180, "very fast"),
]

METERS = [
    (3, 4),
    (4, 4),
    (6, 8),
    (2, 4),
    (5, 4),
    (7, 8),
]

STYLES = [
    "pop", "jazz", "classical", "folk", "blues", "bossa nova",
    "electronic", "indie", "soul", "gospel", "ragtime", "new age",
    "film score", "ambient", "country", "r&b", "funk", "flamenco",
    "waltz", "tango",
]

MOODS = [
    "happy", "uplifting", "cheerful", "bright", "hopeful", "playful",
    "relaxed", "calm", "peaceful", "dreamy", "nostalgic", "warm",
    "energetic", "triumphant", "romantic", "tender", "melancholic",
    "bittersweet", "light-hearted", "whimsical", "contemplative",
    "mysterious", "dark", "longing", "joyful",
]

MELODY_INSTRUMENTS = [
    "piano", "acoustic guitar", "electric guitar", "flute", "violin",
    "cello", "trumpet", "clarinet", "oboe", "saxophone", "vibraphone",
    "marimba", "harmonica", "mandolin", "banjo", "ukulele",
    "recorder", "steel guitar", "electric piano", "theremin",
    "French horn", "bassoon", "piccolo",
]

CHORD_INSTRUMENTS = [
    "piano", "acoustic guitar", "electric guitar", "organ", "string ensemble",
    "choir", "accordion", "synthesizer", "harpsichord",
    "Rhodes piano", "jazz guitar", "nylon guitar", "banjo",
    "ukulele", "harp", "pad synthesizer",
]

TEXTURES = [
    "melodic and harmonic", "polyphonic", "homophonic", "lyrical",
    "flowing", "rhythmically driven", "delicate", "rich and full", "sparse",
]

DYNAMICS = [
    "soft and intimate", "moderately dynamic", "building gradually",
    "steady and consistent", "gently expressive", "lively and bright",
    "powerful and bold", "hushed and delicate",
]


def _chord_progressions(key: str, mode: str) -> list[tuple[str, str]]:
    """
    Return a pool of chord progressions as (roman_numerals, note_names) tuples.
    Note names are transposed to the actual key — simplified (no enharmonic correction).
    """
    # Chromatic scale for transposition
    chromatic = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    root_idx = chromatic.index(key) if key in chromatic else 0

    def shift(notes: list[str]) -> str:
        result = []
        for n in notes:
            base = n.rstrip("m7")
            suffix = n[len(base):]
            if base in chromatic:
                idx = (chromatic.index(base) + root_idx) % 12
                result.append(chromatic[idx] + suffix)
            else:
                result.append(n)
        return ", ".join(result)

    if mode == "major":
        return [
            ("I–IV–V–I",    shift(["C", "F", "G", "C"])),
            ("I–V–vi–IV",   shift(["C", "G", "Am", "F"])),
            ("I–vi–IV–V",   shift(["C", "Am", "F", "G"])),
            ("I–IV–I–V",    shift(["C", "F", "C", "G"])),
            ("ii–V–I",      shift(["Dm", "G", "C"])),
            ("I–IV–ii–V",   shift(["C", "F", "Dm", "G"])),
            ("I–iii–IV–V",  shift(["C", "Em", "F", "G"])),
            ("I–ii–V–I",    shift(["C", "Dm", "G", "C"])),
            ("I–V–IV–I",    shift(["C", "G", "F", "C"])),
            ("I–I–IV–V",    shift(["C", "C", "F", "G"])),
        ]
    else:  # minor
        return [
            ("i–iv–v–i",    shift(["Cm", "Fm", "Gm", "Cm"])),
            ("i–VI–III–VII",shift(["Cm", "Ab", "Eb", "Bb"])),
            ("i–iv–VII–III",shift(["Cm", "Fm", "Bb", "Eb"])),
            ("i–VII–VI–VII",shift(["Cm", "Bb", "Ab", "Bb"])),
            ("i–v–VI–III",  shift(["Cm", "Gm", "Ab", "Eb"])),
            ("i–iv–i–V",    shift(["Cm", "Fm", "Cm", "G"])),
            ("i–VI–VII–i",  shift(["Cm", "Ab", "Bb", "Cm"])),
            ("ii°–V–i",     shift(["Dm", "G", "Cm"])),
        ]


@dataclass
class PromptSpec:
    key: str
    mode: str
    bpm: int
    bpm_label: str
    meter_num: int
    meter_den: int
    style: str
    mood: str
    melody_instrument: str
    chord_instrument: str
    progression_name: str
    progression_notes: str
    texture: str
    dynamics: str


def _sample_spec(rng: random.Random) -> PromptSpec:
    key = rng.choice(KEYS)
    mode = rng.choice(MODES)

    bpm_min, bpm_max, bpm_label = rng.choice(BPM_RANGES)
    bpm = rng.randint(bpm_min, bpm_max)

    meter_num, meter_den = rng.choice(METERS)

    melody = rng.choice(MELODY_INSTRUMENTS)
    chord_pool = [c for c in CHORD_INSTRUMENTS if c != melody]
    chord = rng.choice(chord_pool)

    progressions = _chord_progressions(key, mode)
    prog_name, prog_notes = rng.choice(progressions)

    return PromptSpec(
        key=key,
        mode=mode,
        bpm=bpm,
        bpm_label=bpm_label,
        meter_num=meter_num,
        meter_den=meter_den,
        style=rng.choice(STYLES),
        mood=rng.choice(MOODS),
        melody_instrument=melody,
        chord_instrument=chord,
        progression_name=prog_name,
        progression_notes=prog_notes,
        texture=rng.choice(TEXTURES),
        dynamics=rng.choice(DYNAMICS),
    )


def _render(spec: PromptSpec, rng: random.Random) -> str:
    key_str    = f"{spec.key} {spec.mode}"
    meter_str  = f"{spec.meter_num}/{spec.meter_den} time signature"
    bpm_str    = f"{spec.bpm} BPM"
    tracks     = (
        f"{spec.melody_instrument} carrying the melody and "
        f"{spec.chord_instrument} providing harmonic accompaniment"
    )

    template = rng.randint(0, 2)

    if template == 0:
        return (
            f"A {spec.mood} {spec.style} piece in {key_str} at {bpm_str}, "
            f"with a {meter_str}. The composition is {spec.texture}, "
            f"featuring {tracks}. "
            f"The chord progression {spec.progression_name} "
            f"({spec.progression_notes}) gives the piece a {spec.dynamics} character."
        )
    elif template == 1:
        return (
            f"This {spec.style} track is written in {key_str} at {bpm_str} "
            f"in a {meter_str}. It features two main voices: {tracks}. "
            f"The overall mood is {spec.mood} and {spec.dynamics}, built on "
            f"the chord progression {spec.progression_notes} ({spec.progression_name})."
        )
    else:
        return (
            f"An instrumental {spec.style} composition in {key_str}, "
            f"{bpm_str}, {meter_str}. "
            f"The {spec.melody_instrument} leads the melody while the "
            f"{spec.chord_instrument} sustains the {spec.progression_name} "
            f"chord progression ({spec.progression_notes}). "
            f"The texture is {spec.texture} and the feel is {spec.mood}, {spec.dynamics}."
        )


def generate_prompt(rng: random.Random | None = None) -> str:
    if rng is None:
        rng = random.Random()
    spec = _sample_spec(rng)
    return _render(spec, rng)


def generate_batch(n: int, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    prompts: list[str] = []
    attempts = 0
    while len(prompts) < n and attempts < n * 10:
        p = generate_prompt(rng)
        if p not in seen:
            seen.add(p)
            prompts.append(p)
        attempts += 1
    return prompts


if __name__ == "__main__":
    for i, p in enumerate(generate_batch(10, seed=42), 1):
        print(f"[{i:02d}] {p}\n")
