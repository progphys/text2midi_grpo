# Track Selection Rules3 A100 Deploy

Copy this archive into the Text2midi project root and unpack it there.

Run:

```bash
WANDB_PROJECT=text2midi-rules \
WANDB_NAME=track_select_rules3_rejection_lr5e6_a100 \
./run.sh train track_select_rules3_attn_lora_5rollouts_a100 --synthetic
```

This experiment uses rejection sampling to accept candidates with at least two selected non-drum tracks, then applies only these three rewards:

- `tempo_bin_tolerant_weight: 1.0`
- `key_profile_weight: 1.0`
- `meter_template_weight: 1.0`

Learning rate is `5e-6`.

MIDI saving is disabled:

```yaml
training:
  save_rollout_midis: false
```

Prompts and logs are still saved under `outputs/text2midi/.../prompts` and `outputs/logs/...`.
