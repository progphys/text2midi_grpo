# Обзор Экспериментов

В этом файле собраны experiment-конфиги из
`configs/text2midi/train/experiments/`.

Если в конкретном эксперименте параметр не переопределен, используются
значения по умолчанию из `configs/text2midi/train/base.yaml`:

- preset prompt'ов: `broad`
- число prompt'ов за шаг: `4`
- число rollout'ов в GRPO: `4`
- `generation_chunk_size`: `8`
- `update_mini_batch`: `4`
- максимальная длина rollout: `1200`
- температура: `1.0`
- `lr`: `5e-5`
- objective: `grpo`
- `batch_size`: `4`
- `max_steps`: `2000`
- `save_every`: `200`

## Обозначения

- `Семейство критика`
  - `observer_fixed_pairs`: `src/core/critics/observer_fixed_pairs/assets/last.pt`
  - `final_observer`: `src/core/critics/final_observer/assets/best.pt`
- `Адаптация`
  - `LoRA attn+ffn`: LoRA на `to_qkv`, `to_out`, `linear1`, `linear2`
  - `LoRA ffn`: LoRA на `linear1`, `linear2`
  - `LoRA attn`: LoRA на `to_qkv`, `to_out`
  - `Finetune ffn`: прямое дообучение выбранных модулей
- `Proj`
  - делается ли projection MIDI в `melody/chords` перед scoring критиком

## Сводная Таблица

| Эксперимент | Схема reward | Семейство критика | Сигнал критика | Proj | Адаптация | Rollouts | Gen chunk | Update MB | Max len | LR | Scheduler / KL |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| `critic_rank_only` | только критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn+ffn | 4 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `critic_rank_only_attn_15rollouts_lr7e5` | только критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn | 15 | 15 | 15 | 1200 | `7e-5` | cosine, warmup 30, min `2e-5`, KL ceiling `2.0` |
| `critic_rank_only_attn_ffn_15rollouts` | только критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn+ffn | 15 | 15 | 15 | 1200 | `5e-5` | cosine, warmup 30, min `2e-5`, KL ceiling `2.0` |
| `critic_rank_only_ffn_15rollouts` | только критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA ffn | 15 | 15 | 15 | 1200 | `5e-5` | cosine, warmup 30, min `2e-5`, KL ceiling `2.0` |
| `final_observer_only_attn_ffn_finetune_30rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | Finetune `to_qkv,to_out,linear1,linear2` | 30 | 30 | 30 | 1200 | `7e-6` | cosine, warmup 13, min `2e-6`, KL ceiling `1.5` |
| `final_observer_only_attn_finetune_30rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | Finetune `to_qkv,to_out` | 30 | 30 | 30 | 1200 | `5e-6` | cosine, warmup 13, min `1.5e-6`, KL ceiling `1.5` |
| `final_observer_only_attn_lora_18rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | LoRA attn | 18 | 18 | 18 | 1200 | `5e-6` | cosine, warmup 13, min `1.5e-6`, KL ceiling `1.5` |
| `final_observer_only_attn_lora_30rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | LoRA attn | 30 | 30 | 30 | 1200 | `5e-6` | cosine, warmup 13, min `1.5e-6`, KL ceiling `1.5` |
| `final_observer_only_ffn_18rollouts_stable` | только критик | `final_observer` | `pairwise_win_rate` | да | LoRA ffn | 18 | 18 | 18 | 1200 | `4e-5` | cosine, warmup 13, min `1.5e-5`, KL ceiling `1.5` |
| `final_observer_only_ffn_finetune_18rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | Finetune `linear1,linear2` | 18 | 18 | 18 | 1200 | `1e-5` | cosine, warmup 13, min `3e-6`, KL ceiling `1.5` |
| `final_observer_only_last2_attn_ffn_finetune_30rollouts` | только критик | `final_observer` | `pairwise_win_rate` | нет | Finetune последних decoder layers `16,17` на `to_qkv,to_out,linear1,linear2` | 30 | 30 | 30 | 1200 | `6e-6` | cosine, warmup 13, min `2e-6`, KL ceiling `1.5` |
| `key_meter_plus_critic` | `key_conditioned` + `meter` + критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn+ffn | 6 | 8 | 6 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `key_meter_plus_critic_ffn_only` | `key_conditioned` + `meter` + критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA ffn | 6 | 8 | 6 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `key_meter_plus_critic_ffn_only_lr1e4` | `key_conditioned` + `meter` + критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA ffn | 6 | 8 | 6 | 1200 | `1e-4` | без scheduler / без KL ceiling |
| `key_meter_plus_critic_lr1e4` | `key_conditioned` + `meter` + критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn+ffn | 6 | 8 | 6 | 1200 | `1e-4` | без scheduler / без KL ceiling |
| `key_plus_critic_lora` | `key_conditioned` + критик | `observer_fixed_pairs` | `pairwise_win_rate` | да | LoRA attn+ffn | 6 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `rules_plus_critic` | symbolic rules + критик | `observer_fixed_pairs` | унаследован default (`rank_score`) | да | LoRA attn+ffn | 4 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `rules_triplet` | только rules (`key_conditioned`,`meter`,`tempo_bin_tolerant`) | нет | нет | нет | LoRA attn+ffn | 4 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `rules_triplet_ffn_only` | только rules (`key_conditioned`,`meter`,`tempo_bin_tolerant`) | нет | нет | нет | LoRA ffn | 4 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |
| `symbolic_rules` | только symbolic rules | нет | нет | нет | LoRA attn+ffn | 4 | 8 | 4 | 1200 | `5e-5` | без scheduler / без KL ceiling |

## Веса Reward По Семействам

### Critic-only: `observer_fixed_pairs`

- `observer_weight: 1.0`
- все symbolic weights: `0.0`

Эксперименты:

- `critic_rank_only`
- `critic_rank_only_attn_15rollouts_lr7e5`
- `critic_rank_only_attn_ffn_15rollouts`
- `critic_rank_only_ffn_15rollouts`

### Critic-only: `final_observer`

- `observer_weight: 1.0`
- все symbolic weights: `0.0`

Эксперименты:

- `final_observer_only_ffn_18rollouts_stable`
- `final_observer_only_ffn_finetune_18rollouts`
- `final_observer_only_attn_lora_18rollouts`
- `final_observer_only_attn_lora_30rollouts`
- `final_observer_only_attn_finetune_30rollouts`
- `final_observer_only_attn_ffn_finetune_30rollouts`
- `final_observer_only_last2_attn_ffn_finetune_30rollouts`

### Смешанные: symbolic anchors + critic

`key_meter_plus_critic*`

- `key_conditioned_weight: 0.35`
- `meter_weight: 0.20`
- `observer_weight: 0.45`

`key_plus_critic_lora`

- `key_conditioned_weight: 0.45`
- `observer_weight: 0.55`

`rules_plus_critic`

- symbolic rules размазаны по многим термам
- `observer_weight: 0.25`

### Только rules

`rules_triplet*`

- `key_conditioned_weight: 0.45`
- `meter_weight: 0.30`
- `tempo_bin_tolerant_weight: 0.25`
- `observer_weight: 0.0`

`symbolic_rules`

- широкая смесь symbolic reward
- `observer_weight: 0.0`

## Комментарии

- Почти все эксперименты используют synthetic prompt'ы с preset по умолчанию `broad`, если при запуске не задано `--set prompt.preset=...`.
- Часть экспериментов делает projection перед scoring критиком:
  - старые `observer_fixed_pairs`-запуски обычно используют `project_midi: true`
  - многие новые direct-finetune run'ы с `final_observer` используют `project_midi: false`
- Для direct finetune экспериментов чекпоинты сохраняются как `trainable_state.pt`.
- Для LoRA-экспериментов чекпоинты сохраняются как adapter-файлы, например `adapter_model.safetensors`.
