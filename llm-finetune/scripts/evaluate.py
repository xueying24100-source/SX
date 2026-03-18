#!/usr/bin/env python3
"""
scripts/evaluate.py
===================
Evaluate a fine-tuned model on an evaluation JSONL dataset and report
ROUGE and BLEU scores.

Usage
-----
::

    python scripts/evaluate.py \\
        --base_model Qwen/Qwen2-7B-Instruct \\
        --adapter_path outputs/qwen2-7b-lora/final \\
        --eval_file data/eval.jsonl \\
        --output_file evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.model.model_utils import load_lora_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LoRA model on a JSONL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="data/eval.jsonl")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results as JSON (optional).")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    args = parse_args()

    logger.info("Loading model and adapter from %s + %s", args.base_model, args.adapter_path)
    model, tokenizer = load_lora_for_inference(args.base_model, args.adapter_path)

    logger.info("Loading evaluation data from %s", args.eval_file)
    eval_samples = load_jsonl(args.eval_file)
    logger.info("  → %d evaluation samples", len(eval_samples))

    evaluator = ModelEvaluator(model, tokenizer)
    metrics = evaluator.evaluate_dataset(
        eval_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15} {v:.4f}")
    print("=" * 50 + "\n")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
