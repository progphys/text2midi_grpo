# Text2MIDI GRPO Research

Research code for reinforcement fine-tuning of text-to-MIDI generation with `GRPO`, symbolic music rewards, and learned observer critics.

## What This Repository Contains

- `src/`: core training, evaluation, inference, and critic code
- `scripts/`: thin CLI entrypoints
- `configs/`: Hydra/OmegaConf configs for training, inference, evaluation, and rewards
- `tools/`: analysis and experiment utilities
- `models/`: local third-party model code and checkpoints used by the project
- `supplementary/`: paper, presentation, draft materials, templates, and archive artifacts

## Main Idea

The project fine-tunes a text-to-MIDI generator by:

1. sampling several MIDI candidates for the same prompt,
2. scoring them with symbolic music rules and learned critics,
3. converting group scores into GRPO advantages,
4. updating only the trainable adaptation layers.

The repository includes:

- rule-based reward components for musical structure,
- observer-based critics for ranking generated MIDI,
- training and evaluation pipelines,
- comparison scripts for base vs fine-tuned models.

## Quick Start

Install the main environment:

```bash
./run.sh init
```

Run inference:

```bash
./run.sh infer --caption "A calm piano solo in E minor"
```

Run training:

```bash
./run.sh train rules_plus_critic --synthetic --batch-size 4 --num-rollouts 4 --max-steps 100
```

Compare base and fine-tuned generations:

```bash
./run.sh compare --experiment rules_plus_critic --num-prompts 10
```

## WandB

Public Weights & Biases dashboard:

`https://wandb.ai/udalov0078-/text2midi-grpo`

## Notes For GitHub

- Local training outputs are not part of the repository.
- Local `wandb` runs, virtual environments, checkpoints, and heavy model weights should stay untracked.
- Research documents and presentation assets were moved to `supplementary/` to keep the codebase readable.

## Minimal Structure

```text
.
├── configs/
├── models/
├── scripts/
├── src/
├── supplementary/
├── tools/
├── requirements.txt
└── run.sh
```
