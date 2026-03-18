#!/usr/bin/env python3
"""
scripts/train.py
================
Entry point for LoRA / QLoRA supervised fine-tuning.

Quick start
-----------
Single GPU::

    python scripts/train.py \\
        --config configs/training_config.yaml \\
        --lora_config configs/lora_config.yaml

Multi-GPU with torchrun::

    torchrun --nproc_per_node=4 scripts/train.py \\
        --config configs/training_config.yaml \\
        --lora_config configs/lora_config.yaml

QLoRA (memory-efficient 4-bit training)::

    python scripts/train.py \\
        --config configs/training_config.yaml \\
        --lora_config configs/lora_config.yaml \\
        --use_qlora

The script honours environment variables set by ``torchrun`` /
``accelerate launch`` for distributed training.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Make sure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DataCollatorForSeq2Seq, InstructionDataset
from src.model.model_utils import (
    apply_lora,
    load_base_model,
    load_tokenizer,
    print_model_parameter_info,
)
from src.training.trainer import build_trainer, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model with LoRA / QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the main training configuration YAML file.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to the LoRA configuration YAML file.",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Enable QLoRA (4-bit quantisation).  Requires bitsandbytes.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="(Set automatically by torchrun – do not set manually.)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("Loading configurations from %s and %s", args.config, args.lora_config)
    cfg = load_config(args.config)
    lora_cfg = load_config(args.lora_config)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    lora_params = lora_cfg.get("lora", {})
    qlora_params = lora_cfg.get("qlora", {})

    model_name = model_cfg.get("name_or_path", "Qwen/Qwen2-7B-Instruct")
    use_qlora = args.use_qlora or qlora_params.get("load_in_4bit", False)

    # -----------------------------------------------------------------------
    # 1. Load tokenizer
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # -----------------------------------------------------------------------
    # 2. Load base model (with optional 4-bit quantisation)
    # -----------------------------------------------------------------------
    logger.info("Loading base model (%s)…", model_name)
    model = load_base_model(
        model_name,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_qlora=use_qlora,
        bnb_4bit_quant_type=qlora_params.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=qlora_params.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_use_double_quant=qlora_params.get("bnb_4bit_use_double_quant", True),
    )

    # -----------------------------------------------------------------------
    # 3. Apply LoRA adapter
    # -----------------------------------------------------------------------
    logger.info("Applying LoRA adapter (r=%d, alpha=%d)…",
                lora_params.get("r", 16), lora_params.get("lora_alpha", 32))
    model = apply_lora(
        model,
        r=lora_params.get("r", 16),
        lora_alpha=lora_params.get("lora_alpha", 32),
        lora_dropout=lora_params.get("lora_dropout", 0.05),
        target_modules=lora_params.get("target_modules", "all-linear"),
        bias=lora_params.get("bias", "none"),
        task_type=lora_params.get("task_type", "CAUSAL_LM"),
    )
    print_model_parameter_info(model)

    # -----------------------------------------------------------------------
    # 4. Load datasets
    # -----------------------------------------------------------------------
    data_format = data_cfg.get("data_format", "alpaca")
    max_length = data_cfg.get("max_length", 2048)

    train_file = data_cfg.get("train_file", "data/train.jsonl")
    eval_file = data_cfg.get("eval_file", None)

    logger.info("Loading training data from %s", train_file)
    train_dataset = InstructionDataset(
        train_file,
        tokenizer,
        max_length=max_length,
        data_format=data_format,
    )

    eval_dataset = None
    if eval_file and Path(eval_file).exists():
        logger.info("Loading evaluation data from %s", eval_file)
        eval_dataset = InstructionDataset(
            eval_file,
            tokenizer,
            max_length=max_length,
            data_format=data_format,
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_length)

    # -----------------------------------------------------------------------
    # 5. Build trainer and train
    # -----------------------------------------------------------------------
    # Pass deepspeed config path if provided
    ds_cfg = cfg.get("deepspeed", {})
    training_cfg["deepspeed"] = ds_cfg.get("config", None)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        training_config=training_cfg,
    )

    logger.info("Starting training…")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # -----------------------------------------------------------------------
    # 6. Save final checkpoint
    # -----------------------------------------------------------------------
    final_output_dir = os.path.join(training_cfg.get("output_dir", "outputs/model"), "final")
    logger.info("Saving final checkpoint to %s", final_output_dir)
    save_model(model, tokenizer, final_output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
