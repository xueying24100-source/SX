"""
trainer.py
==========
SFT (Supervised Fine-Tuning) trainer built on top of Hugging Face's
``transformers.Trainer``.

Why not use TRL's ``SFTTrainer`` directly?
------------------------------------------
TRL's ``SFTTrainer`` is convenient but hides many details.  For a project
that needs to demonstrate deep understanding in a technical interview, it is
more valuable to show that you understand the underlying ``Trainer`` API,
data collation, loss masking, and callback hooks.  We still leverage TRL's
``SFTTrainer`` optionally as a thin convenience wrapper.

Architecture
------------
``SFTTrainer`` (this file)
    ├── wraps ``transformers.Trainer``
    ├── adds custom ``TrainingCallback`` hooks for rich logging
    └── exposes ``train()`` / ``save()`` methods used by ``scripts/train.py``
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich progress callback
# ---------------------------------------------------------------------------

class LoggingCallback:
    """A minimal training callback that logs step metrics.

    Compatible with ``transformers.TrainerCallback`` interface.  Only the
    ``on_log`` hook is implemented; everything else is a no-op.
    """

    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs):
        if not logs:
            return
        step = state.global_step
        items = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in logs.items())
        logger.info("[step %6d]  %s", step, items)

    # no-op stubs for the full callback interface
    def on_train_begin(self, *args, **kwargs): pass
    def on_train_end(self, *args, **kwargs): pass
    def on_epoch_begin(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass
    def on_evaluate(self, *args, **kwargs): pass
    def on_save(self, *args, **kwargs): pass


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    training_config: Dict[str, Any],
):
    """Build and return a configured ``transformers.Trainer``.

    This function centralises all ``TrainingArguments`` construction and wires
    up datasets, data collator, and callbacks.

    Parameters
    ----------
    model:
        The model to train (base + LoRA adapter already applied).
    tokenizer:
        Tokenizer associated with ``model``.
    train_dataset:
        Training dataset (must implement ``__len__`` and ``__getitem__``).
    eval_dataset:
        Evaluation dataset (can be ``None`` to disable evaluation).
    data_collator:
        Collator that converts a list of dataset items into batched tensors.
    training_config:
        Dictionary of training hyper-parameters.  Keys correspond to
        ``TrainingArguments`` field names plus a few custom keys
        (``deepspeed_config``).

    Returns
    -------
    trainer:
        A ``transformers.Trainer`` ready to call ``.train()`` on.
    """
    try:
        from transformers import Trainer, TrainingArguments  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required. Install with: pip install transformers") from exc

    output_dir = training_config.get("output_dir", "outputs/model")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=training_config.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=eval_dataset is not None and training_config.get(
            "load_best_model_at_end", True
        ),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        report_to=training_config.get("report_to", "tensorboard"),
        deepspeed=training_config.get("deepspeed", None),
        label_names=["labels"],
    )

    try:
        from transformers import TrainerCallback  # noqa: PLC0415
        # Wrap our callback so it inherits from TrainerCallback properly
        class _WrappedCallback(TrainerCallback, LoggingCallback):
            pass
        callbacks = [_WrappedCallback()]
    except Exception:  # pragma: no cover
        callbacks = []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Trainer built. Output dir: %s", output_dir)
    return trainer


def save_model(model, tokenizer, output_dir: str) -> None:
    """Save the LoRA adapter (or full model) and tokenizer to ``output_dir``.

    When the model is a PEFT model only the adapter weights are saved,
    keeping the checkpoint size small.  The base model weights are not
    duplicated.

    Parameters
    ----------
    model:
        Trained model (PEFT or standard ``PreTrainedModel``).
    tokenizer:
        Associated tokenizer.
    output_dir:
        Directory where artifacts are written.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to %s", output_dir)
