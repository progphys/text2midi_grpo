from __future__ import annotations

import random
import re
from dataclasses import dataclass

KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MODES = ["major", "minor"]
BPM_RANGES = [
    (60, 75, "slow"),
    (76, 99, "moderate"),
    (100, 120, "moderately fast"),
    (121, 150, "fast"),
    (151, 180, "very fast"),
]
METERS = [(3, 4), (4, 4), (6, 8), (2, 4), (5, 4), (7, 8)]
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

NARROW_METERS = [(4, 4), (3, 4), (6, 8)]
NARROW_STYLES = [
    "pop", "folk", "indie", "film score", "ambient", "waltz", "new age",
]
NARROW_MOODS = [
    "warm", "tender", "nostalgic", "dreamy", "peaceful", "melancholic", "hopeful",
]
NARROW_MELODY_INSTRUMENTS = [
    "piano", "flute", "violin", "clarinet", "acoustic guitar",
]
NARROW_CHORD_INSTRUMENTS = [
    "piano", "acoustic guitar", "string ensemble", "organ", "Rhodes piano",
]
NARROW_TEXTURES = [
    "clear melody over steady accompaniment",
    "lyrical lead line with supportive chords",
    "singable melody with gentle harmonic backing",
]
NARROW_DYNAMICS = [
    "gently expressive",
    "steady and consistent",
    "soft and intimate",
]

FINAL_OBSERVER_STYLES = [
    "folk", "pop", "indie", "film score", "new age", "ambient", "classical",
]
FINAL_OBSERVER_MOODS = [
    "warm", "tender", "reflective", "peaceful", "nostalgic", "hopeful",
    "melancholic", "dreamy", "gentle", "lyrical",
]
FINAL_OBSERVER_MELODY_INSTRUMENTS = [
    "piano", "flute", "violin", "clarinet", "acoustic guitar", "oboe",
]
FINAL_OBSERVER_CHORD_INSTRUMENTS = [
    "piano", "acoustic guitar", "string ensemble", "Rhodes piano", "organ",
]
FINAL_OBSERVER_TEXTURES = [
    "The arrangement keeps the lead line clearly above the accompaniment.",
    "The accompaniment stays lower, sparse, and harmonically supportive.",
    "The texture avoids competing lead voices and keeps the melody easy to follow.",
    "The harmonic background uses steady chord tones under a singable melody.",
]

RULES3_STYLES = [
    "folk", "pop", "ambient", "film score", "new age", "classical",
]
RULES3_MOODS = [
    "warm", "tender", "peaceful", "nostalgic", "hopeful", "melancholic", "dreamy",
]
RULES3_MELODY_INSTRUMENTS = [
    "piano", "flute", "violin", "clarinet", "acoustic guitar",
]
RULES3_ACCOMPANIMENT_INSTRUMENTS = [
    "piano", "acoustic guitar", "string ensemble", "Rhodes piano",
]
RULES3_TEXTURES = [
    "with a sparse texture and a steady pulse",
    "with a light texture and clearly felt downbeats",
    "with simple harmonic support and an even rhythmic flow",
    "with a transparent texture and stable metric accents",
]

MODERATE_4TRACK_STYLES = [
    "folk", "pop", "ambient", "film score", "new age", "classical",
]
MODERATE_4TRACK_MOODS = [
    "warm", "peaceful", "hopeful", "nostalgic", "dreamy", "reflective",
]
MODERATE_4TRACK_INSTRUMENT_SETS = [
    ("piano", "acoustic guitar"),
    ("flute", "piano"),
    ("violin", "piano", "string ensemble"),
    ("clarinet", "Rhodes piano"),
    ("acoustic guitar", "string ensemble"),
    ("piano", "flute", "string ensemble"),
]
MODERATE_2TRACK_INSTRUMENT_PAIRS = [
    ("piano", "acoustic guitar"),
    ("flute", "piano"),
    ("violin", "piano"),
    ("clarinet", "Rhodes piano"),
    ("acoustic guitar", "string ensemble"),
    ("piano", "string ensemble"),
]

MELODY_FOCUS_PROMPT_BANK = [
    "A gentle and expressive piece in D minor at 96 BPM, with a 4/4 time signature. It features a clear and recognizable main melody in the upper register, while the remaining musical material stays supportive and secondary in the background.",
    "A melodic and emotional piece in G minor at 92 BPM, with a 4/4 time signature. It has a clearly defined leading melody, while the rest of the instruments provide soft harmonic support underneath.",
    "A warm and reflective piece in C major at 88 BPM, with a 4/4 time signature. It is centered around a clear upper-register melody, with the remaining musical material acting as gentle accompaniment below.",
    "A lyrical and memorable piece in A minor at 84 BPM, with a 3/4 time signature. It features one obvious main melody that stands out clearly above the accompaniment.",
    "A flowing and expressive piece in F major at 102 BPM, with a 6/8 time signature. It has a strong melody-forward character, where the highest line carries the main tune and the lower material stays soft and supportive.",
    "A calm and emotionally rich piece in Eb major at 90 BPM, with a 4/4 time signature. It features a clear singable melody in the foreground, with all other notes remaining secondary and supportive underneath.",
    "A gentle cinematic piece in B minor at 98 BPM, with a 4/4 time signature. It has a distinct leading melody that remains easy to follow, while the background material provides only harmonic support.",
    "A tender and melodic piece in D major at 86 BPM, with a 3/4 time signature. The top line forms a clear main melody and the rest of the musical texture stays underneath as accompaniment.",
    "A spacious and expressive piece in F minor at 94 BPM, with a 4/4 time signature. It features a prominent upper melody and light supportive material below it.",
    "A soft and lyrical piece in A major at 91 BPM, with a 4/4 time signature. It has one clearly recognizable melody in the upper layer, while the rest of the music remains subdued and accompaniment-like.",
    "A reflective and melodic piece in C minor at 100 BPM, with a 4/4 time signature. It is built around a single dominant main melody, with the remaining musical material staying lower and less attention-grabbing.",
    "An intimate and expressive piece in E minor at 89 BPM, with a 4/4 time signature. The highest musical line carries a clear melody and the rest of the arrangement stays in the background as gentle support.",
]

DEBUG_FIXED_PROMPT_TWO_TRACK = (
    "A moderate warm folk piece in C major at 96 BPM, with a 4/4 time signature. "
    "Use exactly two non-drum instrumental tracks: piano for the lead melody and acoustic guitar "
    "for steady accompaniment. Keep the texture clear, moderately dense, and stable, with no drums or percussion."
)

DEBUG_CURRICULUM_PROMPT_STAGE1 = (
    "A moderate warm folk piece in C major at 96 BPM, with a 4/4 time signature. "
    "Use exactly two non-drum instrumental tracks: piano for the lead melody and acoustic guitar "
    "for steady accompaniment. Keep the texture clear, moderately dense, and stable, with no drums or percussion."
)

DEBUG_CURRICULUM_PROMPT_STAGE2 = (
    "A moderate warm folk piece in G major at 96 BPM, with a 4/4 time signature. "
    "Use exactly two non-drum instrumental tracks: piano for the lead melody and acoustic guitar "
    "for steady accompaniment. Keep the texture clear, moderately dense, and stable, with no drums or percussion."
)

DEBUG_CURRICULUM_PROMPT_STAGE3 = (
    "A moderate warm folk piece in G major at 96 BPM, with a 3/4 time signature. "
    "Use exactly two non-drum instrumental tracks: piano for the lead melody and acoustic guitar "
    "for steady accompaniment. Keep the texture clear, moderately dense, and stable, with no drums or percussion."
)

DEBUG_CURRICULUM_PROMPT_STAGE4 = (
    "A moderate warm ambient piece in G major at 96 BPM, with a 3/4 time signature. "
    "Use exactly two non-drum instrumental tracks: piano for the lead melody and acoustic guitar "
    "for steady accompaniment. Keep the texture clear, moderately dense, and stable, with no drums or percussion."
)


def _chord_progressions(key: str, mode: str) -> list[tuple[str, str]]:
    chromatic = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    root_idx = chromatic.index(key) if key in chromatic else 0

    def shift(notes: list[str]) -> str:
        shifted = []
        for note in notes:
            base = note.rstrip("m7")
            suffix = note[len(base):]
            if base in chromatic:
                idx = (chromatic.index(base) + root_idx) % 12
                shifted.append(chromatic[idx] + suffix)
            else:
                shifted.append(note)
        return ", ".join(shifted)

    if mode == "major":
        return [
            ("I-IV-V-I", shift(["C", "F", "G", "C"])),
            ("I-V-vi-IV", shift(["C", "G", "Am", "F"])),
            ("I-vi-IV-V", shift(["C", "Am", "F", "G"])),
            ("ii-V-I", shift(["Dm", "G", "C"])),
        ]

    return [
        ("i-iv-v-i", shift(["Cm", "Fm", "Gm", "Cm"])),
        ("i-VI-III-VII", shift(["Cm", "Ab", "Eb", "Bb"])),
        ("i-v-VI-III", shift(["Cm", "Gm", "Ab", "Eb"])),
        ("ii°-V-i", shift(["Dm", "G", "Cm"])),
    ]


def _join_instruments(instruments: tuple[str, ...]) -> str:
    if len(instruments) == 1:
        return instruments[0]
    if len(instruments) == 2:
        return f"{instruments[0]} and {instruments[1]}"
    return ", ".join(instruments[:-1]) + f", and {instruments[-1]}"


@dataclass
class PromptSpec:
    key: str
    mode: str
    bpm: int
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
    bpm_min, bpm_max, _ = rng.choice(BPM_RANGES)
    melody = rng.choice(MELODY_INSTRUMENTS)
    chord = rng.choice([item for item in CHORD_INSTRUMENTS if item != melody])
    progression_name, progression_notes = rng.choice(_chord_progressions(key, mode))
    meter_num, meter_den = rng.choice(METERS)
    return PromptSpec(
        key=key,
        mode=mode,
        bpm=rng.randint(bpm_min, bpm_max),
        meter_num=meter_num,
        meter_den=meter_den,
        style=rng.choice(STYLES),
        mood=rng.choice(MOODS),
        melody_instrument=melody,
        chord_instrument=chord,
        progression_name=progression_name,
        progression_notes=progression_notes,
        texture=rng.choice(TEXTURES),
        dynamics=rng.choice(DYNAMICS),
    )


def _sample_narrow_spec(rng: random.Random) -> PromptSpec:
    key = rng.choice(KEYS)
    mode = rng.choice(MODES)
    bpm_min, bpm_max, _ = rng.choice(BPM_RANGES[:4])
    style = rng.choice(NARROW_STYLES)
    melody = rng.choice(NARROW_MELODY_INSTRUMENTS)
    chord = rng.choice([item for item in NARROW_CHORD_INSTRUMENTS if item != melody] or NARROW_CHORD_INSTRUMENTS)
    progression_name, progression_notes = rng.choice(_chord_progressions(key, mode))
    if style == "waltz":
        meter_num, meter_den = (3, 4)
    else:
        meter_num, meter_den = rng.choice(NARROW_METERS)
    return PromptSpec(
        key=key,
        mode=mode,
        bpm=rng.randint(bpm_min, bpm_max),
        meter_num=meter_num,
        meter_den=meter_den,
        style=style,
        mood=rng.choice(NARROW_MOODS),
        melody_instrument=melody,
        chord_instrument=chord,
        progression_name=progression_name,
        progression_notes=progression_notes,
        texture=rng.choice(NARROW_TEXTURES),
        dynamics=rng.choice(NARROW_DYNAMICS),
    )


def generate_prompt(rng: random.Random | None = None, preset: str = "broad") -> str:
    rng = rng or random.Random()
    if preset == "debug_fixed_prompt_two_track":
        return DEBUG_FIXED_PROMPT_TWO_TRACK
    if preset == "debug_curriculum_stage1":
        return DEBUG_CURRICULUM_PROMPT_STAGE1
    if preset == "debug_curriculum_stage2":
        return DEBUG_CURRICULUM_PROMPT_STAGE2
    if preset == "debug_curriculum_stage3":
        return DEBUG_CURRICULUM_PROMPT_STAGE3
    if preset == "debug_curriculum_stage4":
        return DEBUG_CURRICULUM_PROMPT_STAGE4
    if preset == "melody_focus_bank":
        return rng.choice(MELODY_FOCUS_PROMPT_BANK)
    if preset == "rules3_caption_narrow":
        key = rng.choice(KEYS)
        mode = rng.choice(MODES)
        bpm_min, bpm_max, tempo_word = rng.choice(BPM_RANGES[:4])
        bpm = rng.randint(bpm_min, bpm_max)
        meter_num, meter_den = rng.choice(NARROW_METERS)
        mood = rng.choice(RULES3_MOODS)
        style = rng.choice(RULES3_STYLES)
        melody = rng.choice(RULES3_MELODY_INSTRUMENTS)
        accompaniment = rng.choice(
            [item for item in RULES3_ACCOMPANIMENT_INSTRUMENTS if item != melody]
            or RULES3_ACCOMPANIMENT_INSTRUMENTS
        )
        texture = rng.choice(RULES3_TEXTURES)
        return (
            f"A {tempo_word} {mood} {style} piece in {key} {mode} at {bpm} BPM, "
            f"with a {meter_num}/{meter_den} time signature. "
            f"It features a clear {melody} melody over simple {accompaniment} accompaniment, "
            f"{texture}."
        )
    if preset == "moderate_4track_no_drums":
        key = rng.choice(KEYS)
        mode = rng.choice(MODES)
        bpm = rng.randint(84, 112)
        meter_num, meter_den = rng.choice(NARROW_METERS)
        mood = rng.choice(MODERATE_4TRACK_MOODS)
        style = rng.choice(MODERATE_4TRACK_STYLES)
        instruments = rng.choice(MODERATE_4TRACK_INSTRUMENT_SETS)
        instrument_text = _join_instruments(instruments)
        return (
            f"A moderate {mood} {style} piece in {key} {mode} at {bpm} BPM, "
            f"with a {meter_num}/{meter_den} time signature. "
            f"It uses a compact arrangement of no more than four instrumental parts, "
            f"featuring {instrument_text}, with no drums or percussion."
        )
    if preset == "moderate_3track_no_drums":
        key = rng.choice(KEYS)
        mode = rng.choice(MODES)
        bpm = rng.randint(84, 112)
        meter_num, meter_den = rng.choice(NARROW_METERS)
        mood = rng.choice(MODERATE_4TRACK_MOODS)
        style = rng.choice(MODERATE_4TRACK_STYLES)
        instruments = rng.choice(MODERATE_4TRACK_INSTRUMENT_SETS)
        instrument_text = _join_instruments(instruments[:3])
        return (
            f"A moderate {mood} {style} piece in {key} {mode} at {bpm} BPM, "
            f"with a {meter_num}/{meter_den} time signature. "
            f"It uses a compact arrangement of no more than three instrumental parts, "
            f"featuring {instrument_text}, with no drums or percussion."
        )
    if preset == "moderate_2track_no_drums":
        key = rng.choice(KEYS)
        mode = rng.choice(MODES)
        bpm = rng.randint(84, 112)
        meter_num, meter_den = rng.choice(NARROW_METERS)
        mood = rng.choice(MODERATE_4TRACK_MOODS)
        style = rng.choice(MODERATE_4TRACK_STYLES)
        lead, accompaniment = rng.choice(MODERATE_2TRACK_INSTRUMENT_PAIRS)
        return (
            f"A moderate {mood} {style} piece in {key} {mode} at {bpm} BPM, "
            f"with a {meter_num}/{meter_den} time signature. "
            f"Use exactly two non-drum instrumental tracks: {lead} for the lead melody "
            f"and {accompaniment} for steady accompaniment. "
            f"Keep the texture clear, moderately dense, and stable, with no drums or percussion."
        )
    if preset == "final_observer_aligned":
        key = rng.choice(KEYS)
        mode = rng.choice(MODES)
        bpm_min, bpm_max, _ = rng.choice(BPM_RANGES[:4])
        style = rng.choice(FINAL_OBSERVER_STYLES)
        mood = rng.choice(FINAL_OBSERVER_MOODS)
        melody = rng.choice(FINAL_OBSERVER_MELODY_INSTRUMENTS)
        chord = rng.choice([item for item in FINAL_OBSERVER_CHORD_INSTRUMENTS if item != melody] or FINAL_OBSERVER_CHORD_INSTRUMENTS)
        meter_num, meter_den = (3, 4) if style == "waltz" else rng.choice(NARROW_METERS)
        texture = rng.choice(FINAL_OBSERVER_TEXTURES)
        return (
            f"A {mood} {style} piece in {key} {mode} at {rng.randint(bpm_min, bpm_max)} BPM, "
            f"with a {meter_num}/{meter_den} time signature. "
            f"It features one clear main melody in the upper register played by {melody}, "
            f"while {chord} provides simple chordal accompaniment underneath. "
            f"{texture}"
        )
    if preset == "melody_accompaniment_narrow":
        spec = _sample_narrow_spec(rng)
        return (
            f"A {spec.mood} {spec.style} piece in {spec.key} {spec.mode} at {spec.bpm} BPM, "
            f"with a {spec.meter_num}/{spec.meter_den} time signature. "
            f"Use exactly two clearly separated musical roles: a single lead melody and a chordal accompaniment. "
            f"The {spec.melody_instrument} should play the lead melody while the {spec.chord_instrument} "
            f"supports the {spec.progression_name} progression ({spec.progression_notes}) with steady accompaniment patterns. "
            f"The texture is {spec.texture} and the dynamics feel {spec.dynamics}."
        )

    spec = _sample_spec(rng)
    return (
        f"A {spec.mood} {spec.style} piece in {spec.key} {spec.mode} at {spec.bpm} BPM, "
        f"with a {spec.meter_num}/{spec.meter_den} time signature. "
        f"The {spec.melody_instrument} leads the melody while the {spec.chord_instrument} "
        f"supports the {spec.progression_name} progression ({spec.progression_notes}). "
        f"The texture is {spec.texture} and the dynamics feel {spec.dynamics}."
    )


_PROMPT_META_RE = re.compile(
    r"\bin\s+(?P<key>[A-G](?:#|b)?)\s+(?P<mode>major|minor)\s+at\s+(?P<bpm>\d+)\s+BPM,\s+with\s+a\s+(?P<meter_num>\d+)/(?P<meter_den>\d+)\s+time signature",
    re.IGNORECASE,
)


def parse_prompt_metadata(prompt: str) -> dict[str, int | str] | None:
    match = _PROMPT_META_RE.search(prompt)
    if not match:
        return None
    values = match.groupdict()
    key = values["key"][0].upper() + values["key"][1:].replace("B", "b")
    return {
        "key": key,
        "mode": values["mode"].lower(),
        "bpm": int(values["bpm"]),
        "meter_numerator": int(values["meter_num"]),
        "meter_denominator": int(values["meter_den"]),
    }


def generate_batch(n: int, seed: int | None = None, preset: str = "broad") -> list[str]:
    rng = random.Random(seed)
    if preset == "melody_focus_bank":
        bank = list(MELODY_FOCUS_PROMPT_BANK)
        if n <= len(bank):
            rng.shuffle(bank)
            return bank[:n]
        prompts = []
        while len(prompts) < n:
            prompts.extend(bank)
        rng.shuffle(prompts)
        return prompts[:n]
    prompts = []
    seen = set()
    while len(prompts) < n:
        prompt = generate_prompt(rng, preset=preset)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts
