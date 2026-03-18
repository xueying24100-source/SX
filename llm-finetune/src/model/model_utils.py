"""
model_utils.py
==============
Utilities for loading base models and applying LoRA / QLoRA adapters using the
Hugging Face `transformers` and `peft` libraries.

Key design decisions
--------------------
* Support both full-precision LoRA and 4-bit QLoRA out of the box.
* Accept a plain dict / OmegaConf DictConfig so callers are not forced to use
  any specific config framework.
* Keep tokenizer setup (padding, chat templates) in one place to avoid subtle
  mismatches between training and inference.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports – heavy ML libraries are imported only when actually needed so
# that unit tests that mock these modules can run without a GPU / full install.
# ---------------------------------------------------------------------------

def _import_torch():
    import torch  # noqa: PLC0415
    return torch


def _import_transformers():
    import transformers  # noqa: PLC0415
    return transformers


def _import_peft():
    import peft  # noqa: PLC0415
    return peft


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    """Load and configure a tokenizer.

    Sets ``pad_token`` to ``eos_token`` when no dedicated pad token is present
    (common for decoder-only models such as LLaMA / Qwen).  Also sets
    ``padding_side`` to ``"right"`` which is required for causal language
    models during training to avoid shifting labels incorrectly.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face Hub repo ID or local directory path.
    trust_remote_code:
        Whether to allow executing custom model code from the Hub (required
        for some models like Qwen / ChatGLM).

    Returns
    -------
    tokenizer:
        A fully configured ``PreTrainedTokenizerFast`` instance.
    """
    transformers = _import_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    # Ensure a pad token exists (decoder-only models often only have eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token: %s", tokenizer.eos_token)

    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name_or_path: str,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    use_qlora: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
    device_map: Union[str, dict, None] = "auto",
):
    """Load a causal language model with optional 4-bit quantisation (QLoRA).

    Parameters
    ----------
    model_name_or_path:
        Hugging Face Hub repo ID or local directory path.
    torch_dtype:
        Base dtype for model weights (``"bfloat16"`` / ``"float16"`` /
        ``"float32"``).  Ignored when ``use_qlora=True`` (bitsandbytes
        controls the dtype in that case).
    trust_remote_code:
        Required for models with custom modelling code (e.g., Qwen, ChatGLM).
    use_qlora:
        When ``True`` the model is loaded in 4-bit precision via
        ``bitsandbytes`` (QLoRA mode).
    bnb_4bit_quant_type:
        Quantisation type – ``"nf4"`` (NormalFloat4, recommended) or
        ``"fp4"``.
    bnb_4bit_compute_dtype:
        Dtype used for computation inside quantised layers.
    bnb_4bit_use_double_quant:
        When ``True`` quantises the quantisation constants themselves,
        saving an additional ~0.4 bits/parameter.
    device_map:
        How to distribute model layers across devices.  ``"auto"`` lets
        ``accelerate`` decide; pass ``None`` to load onto CPU only.

    Returns
    -------
    model:
        A loaded ``PreTrainedModel``, possibly with 4-bit quantisation applied.
    """
    torch = _import_torch()
    transformers = _import_transformers()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    quantization_config = None
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "bitsandbytes is required for QLoRA. "
                "Install it with: pip install bitsandbytes"
            ) from exc

        compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        logger.info("QLoRA mode: 4-bit quantisation enabled (quant_type=%s)", bnb_4bit_quant_type)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=dtype if quantization_config is None else None,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if use_qlora:
        # Prepare the model for k-bit training – casts layernorms to float32,
        # enables gradient computation on frozen quantised weights, etc.
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("Model prepared for k-bit training.")

    return model


def apply_lora(
    model,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Union[str, list] = "all-linear",
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """Wrap a base model with LoRA adapter layers using the ``peft`` library.

    After calling this function the model's original weights are frozen and
    only the LoRA matrices (A, B) are trainable, which drastically reduces the
    number of parameters that need to be updated.

    Parameters
    ----------
    model:
        Base ``PreTrainedModel`` (returned by :func:`load_base_model`).
    r:
        LoRA rank – the inner dimension of the low-rank decomposition
        (``ΔW = B·A`` where ``B ∈ R^{d×r}`` and ``A ∈ R^{r×k}``).
    lora_alpha:
        Scaling factor; the effective learning rate is scaled by
        ``lora_alpha / r``.
    lora_dropout:
        Dropout probability applied to LoRA layers.
    target_modules:
        Which modules to apply LoRA to.  Use ``"all-linear"`` to apply to
        every ``nn.Linear`` layer, or pass a list of module names.
    bias:
        Whether to train bias parameters: ``"none"`` / ``"all"`` /
        ``"lora_only"``.
    task_type:
        ``"CAUSAL_LM"`` for decoder-only models; ``"SEQ_2_SEQ_LM"`` for
        encoder-decoder models.

    Returns
    -------
    model:
        The model wrapped with a ``LoraModel`` – ready for fine-tuning.
    """
    peft = _import_peft()

    lora_config = peft.LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=getattr(peft.TaskType, task_type),
    )
    model = peft.get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    trainable_pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        "LoRA applied: trainable params = %d / %d (%.2f%%)",
        trainable,
        total,
        trainable_pct,
    )
    return model


def load_lora_for_inference(
    base_model_name_or_path: str,
    adapter_path: str,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    device_map: Union[str, dict, None] = "auto",
):
    """Load a base model and merge a saved LoRA adapter for inference.

    The adapter weights are merged into the base model weights and then
    unloaded, resulting in a standard ``PreTrainedModel`` with no PEFT
    overhead at inference time.

    Parameters
    ----------
    base_model_name_or_path:
        Hugging Face Hub repo ID or local path to the *base* model.
    adapter_path:
        Local directory containing the saved PEFT adapter (produced by
        ``model.save_pretrained(output_dir)``).
    torch_dtype:
        Weight dtype for the merged model.
    trust_remote_code:
        See :func:`load_base_model`.
    device_map:
        Device placement strategy.

    Returns
    -------
    model:
        The merged model ready for ``.generate()`` calls.
    tokenizer:
        Corresponding tokenizer.
    """
    peft = _import_peft()

    model = load_base_model(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    model = peft.PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    logger.info("LoRA adapter merged and unloaded. Model is ready for inference.")

    tokenizer = load_tokenizer(base_model_name_or_path, trust_remote_code=trust_remote_code)
    return model, tokenizer


def get_model_parameter_info(model) -> Tuple[int, int, float]:
    """Return (trainable_params, total_params, trainable_percentage).

    Useful for logging and sanity-checking how many parameters are being
    updated during a fine-tuning run.
    """
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def print_model_parameter_info(model) -> None:
    """Log a human-readable summary of trainable vs. total parameters."""
    trainable, total, pct = get_model_parameter_info(model)
    logger.info("=" * 50)
    logger.info("Model Parameter Summary")
    logger.info("  Trainable params : %15d", trainable)
    logger.info("  Total params     : %15d", total)
    logger.info("  Trainable %%      : %14.2f%%", pct)
    logger.info("=" * 50)
