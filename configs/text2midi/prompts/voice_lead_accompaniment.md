# Voice Lead + Accompaniment Prompts

Ниже шаблоны промптов для base/finetuned inference, где мы максимально жестко просим:

- одну ведущую мелодическую линию;
- аккомпанемент как поддержку;
- минимум лишних слоев;
- отсутствие плотной аранжировки;
- ясное разделение ролей.

## Main Template

```text
Create a simple two-role musical piece in {KEY} {MODE} at {BPM} BPM with a {METER_NUM}/{METER_DEN} time signature.
Use exactly one clearly exposed lead voice for the main melody and one accompaniment layer for harmonic support.
The lead voice must carry a singable, foreground melodic line with longer connected phrases.
The accompaniment must stay in the background and play only supportive chords or broken-chord patterns.
Avoid drums, dense orchestration, counter-melodies, extra lead instruments, thick layered textures, and abrupt instrument changes.
Keep the arrangement sparse, transparent, and easy to separate into melody and accompaniment.
Style: {STYLE}. Mood: {MOOD}.
Suggested lead instrument: {LEAD_INSTRUMENT}.
Suggested accompaniment instrument: {ACCOMPANIMENT_INSTRUMENT}.
```

## Stronger Template

```text
Generate a sparse arrangement in {KEY} {MODE} at {BPM} BPM in {METER_NUM}/{METER_DEN}.
Important constraint: the music should behave like two musical roles only.
Role 1: one dominant lead voice that plays the full melody in the foreground.
Role 2: one accompaniment layer that supports the melody with chords, arpeggios, or sustained harmony.
Do not add drums, bass grooves as separate lead material, decorative counterpoint, or multiple competing melodic tracks.
The result should sound like melody plus accompaniment, not a full band arrangement.
Lead instrument: {LEAD_INSTRUMENT}.
Accompaniment instrument: {ACCOMPANIMENT_INSTRUMENT}.
Style: {STYLE}. Mood: {MOOD}.
```

## Observer-Oriented Template

```text
Compose a piece in {KEY} {MODE} at {BPM} BPM with a {METER_NUM}/{METER_DEN} meter.
Use one melody track and one chordal accompaniment track in clearly separated musical roles.
The melody should stay monophonic or near-monophonic and remain the perceptual foreground.
The accompaniment should provide harmonic support with repeated chord patterns, light arpeggiation, or sustained chords.
Avoid percussion-heavy writing, multi-layer ensemble textures, and overlapping competing melodies.
Keep the structure clean so the output can be interpreted as melody plus chords.
Style: {STYLE}. Mood: {MOOD}.
```

## Recommended Values

Для первых проверок лучше использовать:

- `METER`: `4/4`, `3/4`, `6/8`
- `BPM`: `72-128`
- `STYLE`: `pop`, `folk`, `ambient`, `film score`, `new age`, `waltz`
- `MOOD`: `warm`, `tender`, `nostalgic`, `dreamy`, `peaceful`
- `LEAD_INSTRUMENT`: `piano`, `violin`, `flute`, `clarinet`
- `ACCOMPANIMENT_INSTRUMENT`: `acoustic guitar`, `piano`, `organ`, `string ensemble`, `Rhodes piano`

## Ready-to-Use Examples

```text
Create a simple two-role musical piece in C major at 92 BPM with a 4/4 time signature.
Use exactly one clearly exposed lead voice for the main melody and one accompaniment layer for harmonic support.
The lead voice must carry a singable, foreground melodic line with longer connected phrases.
The accompaniment must stay in the background and play only supportive chords or broken-chord patterns.
Avoid drums, dense orchestration, counter-melodies, extra lead instruments, thick layered textures, and abrupt instrument changes.
Keep the arrangement sparse, transparent, and easy to separate into melody and accompaniment.
Style: folk. Mood: warm.
Suggested lead instrument: violin.
Suggested accompaniment instrument: acoustic guitar.
```

```text
Generate a sparse arrangement in A minor at 84 BPM in 3/4.
Important constraint: the music should behave like two musical roles only.
Role 1: one dominant lead voice that plays the full melody in the foreground.
Role 2: one accompaniment layer that supports the melody with chords, arpeggios, or sustained harmony.
Do not add drums, bass grooves as separate lead material, decorative counterpoint, or multiple competing melodic tracks.
The result should sound like melody plus accompaniment, not a full band arrangement.
Lead instrument: flute.
Accompaniment instrument: piano.
Style: waltz. Mood: nostalgic.
```

```text
Compose a piece in D major at 108 BPM with a 6/8 meter.
Use one melody track and one chordal accompaniment track in clearly separated musical roles.
The melody should stay monophonic or near-monophonic and remain the perceptual foreground.
The accompaniment should provide harmonic support with repeated chord patterns, light arpeggiation, or sustained chords.
Avoid percussion-heavy writing, multi-layer ensemble textures, and overlapping competing melodies.
Keep the structure clean so the output can be interpreted as melody plus chords.
Style: film score. Mood: hopeful.
```

## Practical Note

Даже такой prompt не гарантирует, что base Text2MIDI реально выдаст ровно две дорожки.
Но он сильнее подталкивает модель к:

- одной заметной мелодии;
- фоновому harmonic backing;
- меньшему числу конкурирующих инструментальных ролей.

Для GRPO лучше всего использовать этот стиль prompt вместе с:

- узким prompt distribution;
- projection в `melody/chords`;
- reward на structural separation;
- critic как дополнительный preference signal.
