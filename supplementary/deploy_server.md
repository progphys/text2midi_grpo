# Text2MIDI GRPO Server Deployment

This bundle is meant for a clean server run. It contains our project code,
configs, reward/critic code, prompt files, critic checkpoints/assets, and `.env`
for W&B. It does not contain `.venv`, `.git`, `models/text2midi`, `outputs`, or
`wandb`. The init command recreates the environment and downloads the external
Text2MIDI model/weights.

## 1. Unpack Archive

```bash
tar -xzf text2midi_grpo_server_bundle.tar.gz
cd Text2midi
```

## 2. Initialize Environment

Classic server mode with a local `.venv`:

```bash
./run.sh init
```

This creates `.venv`, clones `amaai-lab/text2midi` into `models/text2midi`,
installs requirements, and downloads the base Text2MIDI weights/tokenizer.

Notebook/DataSphere mode without `.venv`:

```bash
./run.sh init-system
```

This installs requirements into the current/system `python3.10` environment and
uses the same model download logic. Use this mode if `python -m venv` fails with
`ensurepip is not available`.

W&B is already configured through `.env` in this bundle.

If you do not want W&B for a smoke test, add `--no-wandb` to the training
command.

## 3. Recommended FFN Finetune Run

If you want to train not with LoRA but with direct FFN finetuning, use:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --max-steps 1000 \
  --save-every 10
```

The experiment already sets:

- critic-only reward with `final_observer`
- `pairwise_win_rate`
- direct finetuning of FFN layers `linear1` and `linear2`
- `num_rollouts=18`
- `generation_chunk_size=18`
- `update_mini_batch=18`
- `rollout_max_len=1200`
- `lr=1e-5`
- warmup + cosine LR schedule
- `warmup_steps=13`
- `kl_ceiling=1.5`

If you prefer the older LoRA critic-only baseline, it is still available:

```bash
./run.sh train critic_rank_only_ffn_15rollouts \
  --synthetic \
  --max-steps 1000 \
  --save-every 10
```

## 4. Stronger Hardware Overrides

If the server has enough memory/throughput, increase prompt batch size:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --batch-size 6 \
  --max-steps 1000 \
  --save-every 10
```

If KL becomes unstable, lower the LR:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --set training.lr=7e-6 \
  --max-steps 1000 \
  --save-every 10
```

If you want a slightly stronger FFN update on a large GPU, try:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --set training.lr=2e-5 \
  --max-steps 1000 \
  --save-every 10
```

## 5. Resume Latest Checkpoint

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --resume-latest
```

## 6. Main W&B Graphs

For critic-only GRPO with `final_observer`, prioritize these graphs:

- `metric/final_observer_top1_raw`
- `metric/final_observer_top1_gap`
- `reward/total`
- `kl`
- `train/lr`
- `metrics/valid_rate`

`reward/total` can be flat because the critic reward is group-relative. The
most useful quality signal is usually whether `top1_raw` improves while `kl`
and `valid_rate` stay healthy.

## 7. Where Results Go

Checkpoints:

```bash
outputs/checkpoints/final_observer_only_ffn_finetune_18rollouts/
```

Generated rollout MIDI and projected observer MIDI:

```bash
outputs/
```

For a quick sanity test before the long run:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --max-steps 3 \
  --save-every 1 \
  --no-wandb
```
