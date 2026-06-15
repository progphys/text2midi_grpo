"""
Dataset for GRPO training.

Two modes:
  - real: loads captions from models/text2midi/captions/captions.json (JSONL)
  - synthetic: generates prompts via prompt_generator (C major, 120 BPM, 4/4, melody+chords)
"""

from __future__ import annotations
import random
from torch.utils.data import Dataset, IterableDataset


class CaptionDataset(Dataset):
    """Real captions from the MidiCaps JSONL dataset."""

    def __init__(self, captions_path: str, split: str = "train", seed: int = 42):
        import jsonlines

        with jsonlines.open(captions_path) as reader:
            all_items = list(reader)

        if split == "train":
            items = [x for x in all_items if not x.get("test_set", False)]
        elif split == "test":
            items = [x for x in all_items if x.get("test_set", False)]
        else:
            items = all_items

        rng = random.Random(seed)
        rng.shuffle(items)
        self.captions: list[str] = [x["caption"] for x in items if "caption" in x]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> str:
        return self.captions[idx]


class SyntheticDataset(IterableDataset):
    """
    Infinite stream of synthetic prompts.

    All prompts are fixed to: C major, 120 BPM, 4/4, melody + chords tracks.
    Everything else (style, mood, instruments, chord progression) is randomised.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def __iter__(self):
        from prompt_generator import generate_prompt
        rng = random.Random(self.seed)
        while True:
            yield generate_prompt(rng)


def make_dataloader(cfg, seed: int = 42):
    """
    Build a DataLoader based on cfg.prompt.num_prompts_per_step:
      - None  → real CaptionDataset (JSONL)
      - int   → SyntheticDataset (generated prompts)
    """
    from torch.utils.data import DataLoader
    import os

    batch_size = cfg.training.batch_size

    if cfg.prompt.num_prompts_per_step is None:
        captions_path = os.path.join(
            cfg.paths.model_repo, "captions/captions.json"
        )
        ds = CaptionDataset(captions_path, split="train", seed=seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=list,
        )
    else:
        ds = SyntheticDataset(seed=seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=list,
        )
