# Full Text2MIDI Rules3 A100 Deploy

This archive contains the project wrapper code, configs, scripts, and the A100
track-selection GRPO experiment. It does not contain `.venv`, model weights,
checkpoints, or generated outputs.

## 1. Unpack

```bash
mkdir -p Text2midi
tar -xzf text2midi_rules3_full_deploy_a100_lr5e6.tar.gz -C Text2midi
cd Text2midi
```

If you unpack into an existing project root, use:

```bash
tar -xzf text2midi_rules3_full_deploy_a100_lr5e6.tar.gz
```

## 2. Install Text2MIDI, dependencies, and weights

```bash
./run.sh init
```

This creates `.venv`, clones `amaai-lab/text2midi` into `models/text2midi`,
installs dependencies, and downloads:

- `amaai-lab/text2midi` weights
- `vocab_remi.pkl`
- `google/flan-t5-base` tokenizer files

If you already have a preferred Python:

```bash
TEXT2MIDI_PYTHON=/path/to/python3.10 ./run.sh init
```

## 3. Run the experiment

```bash
WANDB_PROJECT=text2midi-rules \
WANDB_NAME=track_select_rules3_rejection_lr5e6_a100 \
./run.sh train track_select_rules3_attn_lora_5rollouts_a100 --synthetic
```

The experiment uses rejection sampling to accept generated candidates with
enough selected non-drum tracks, then applies only these three rewards:

- `tempo_bin_tolerant_weight: 1.0`
- `key_profile_weight: 1.0`
- `meter_template_weight: 1.0`

Learning rate is `5e-6`.

MIDI saving is disabled for this run:

```yaml
training:
  save_rollout_midis: false
```

Prompts and logs are still saved under:

```text
outputs/text2midi/track_select_rules3_attn_lora_5rollouts_a100/prompts/
outputs/logs/track_select_rules3_attn_lora_5rollouts_a100/train.log
```

