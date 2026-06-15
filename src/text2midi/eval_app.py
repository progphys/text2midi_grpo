from __future__ import annotations

import logging
import os

from core.config import evaluation_config
from core.evaluation.evaluator import evaluate_prompts
from core.utils.runtime import load_env, resolve_device, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import generate_batch

log = logging.getLogger(__name__)


def run_evaluation(args) -> None:
    cfg = evaluation_config("text2midi", experiment=args.experiment)
    setup_logging()
    load_env(cfg.project_root)
    device = resolve_device()

    log.info("Device: %s", device)
    if args.baseline or not args.checkpoint:
        adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)
        run_name = "baseline"
    else:
        adapter = Text2MidiAdapter(
            cfg,
            device=device,
            with_lora=True,
            checkpoint_path=args.checkpoint,
        )
        step_tag = f"step_{args.step:05d}" if args.step else "final"
        run_name = f"{args.experiment}/{step_tag}"

    prompts = generate_batch(cfg.evaluation.num_prompts, seed=cfg.evaluation.seed)
    metrics = evaluate_prompts(
        adapter=adapter,
        prompts=prompts,
        generations_per_prompt=cfg.evaluation.generations_per_prompt,
        max_len=args.max_len or cfg.evaluation.max_len,
        temperature=args.temperature or cfg.evaluation.temperature,
        reward_cfg=cfg.reward,
    )

    use_wandb = cfg.logging.use_wandb and not args.no_wandb
    if use_wandb:
        import wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            wandb.login()
            wandb.init(
                project=cfg.logging.project,
                name=f"eval/{run_name}",
                job_type="eval",
                config={
                    "experiment": args.experiment or "baseline",
                    "checkpoint": args.checkpoint,
                    "step": args.step,
                },
            )
            wandb.log(metrics, step=args.step)
            wandb.finish()

    log.info("-" * 50)
    for key, value in metrics.items():
        log.info("%-24s %s", f"{key}:", f"{value:.4f}" if isinstance(value, float) else value)
    log.info("-" * 50)
