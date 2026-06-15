from __future__ import annotations

import os
import random

from torch.utils.data import Dataset, IterableDataset


class CaptionDataset(Dataset):
    def __init__(self, captions_path: str, split: str = "train", seed: int = 42):
        import jsonlines

        with jsonlines.open(captions_path) as reader:
            items = list(reader)

        if split == "train":
            items = [item for item in items if not item.get("test_set", False)]
        elif split == "test":
            items = [item for item in items if item.get("test_set", False)]

        rng = random.Random(seed)
        rng.shuffle(items)
        self.captions = [item["caption"] for item in items if "caption" in item]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> str:
        return self.captions[idx]


class SyntheticPromptDataset(IterableDataset):
    def __init__(
        self,
        seed: int = 42,
        preset: str = "broad",
        schedule_presets: list[str] | None = None,
    ):
        self.seed = seed
        self.preset = preset
        self.schedule_presets = list(schedule_presets or [])

    def __iter__(self):
        from text2midi.prompting import generate_prompt

        rng = random.Random(self.seed)
        if self.schedule_presets:
            step = 0
            while True:
                preset = self.schedule_presets[min(step, len(self.schedule_presets) - 1)]
                step += 1
                yield generate_prompt(rng, preset=preset)
        while True:
            yield generate_prompt(rng, preset=self.preset)


def make_dataloader(cfg, seed: int = 42):
    from torch.utils.data import DataLoader

    batch_size = cfg.training.batch_size
    if cfg.prompt.num_prompts_per_step is None:
        captions_path = os.path.join(cfg.paths.model_repo, "captions/captions.json")
        dataset = CaptionDataset(captions_path, split="train", seed=seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=list,
        )

    preset = str(cfg.prompt.get("preset", "broad"))
    schedule_presets = cfg.prompt.get("schedule_presets")
    return DataLoader(
        SyntheticPromptDataset(seed=seed, preset=preset, schedule_presets=schedule_presets),
        batch_size=batch_size,
        collate_fn=list,
    )
