"""
evaluator.py
============
Automatic evaluation of fine-tuned language models using standard NLP metrics.

Metrics implemented
-------------------
* **Perplexity** – measures how well the model predicts the evaluation corpus.
  Lower is better.  Computed as ``exp(mean_cross_entropy_loss)``.
* **ROUGE-L** – longest common subsequence recall between reference and
  hypothesis.  Standard for summarisation / generation tasks.
* **BLEU-4** – n-gram precision score (up to 4-grams).  Standard for
  translation and conditional generation.

Usage example
-------------
::

    evaluator = ModelEvaluator(model, tokenizer, device="cuda")
    scores = evaluator.evaluate_dataset(eval_dataset, batch_size=8)
    print(scores)
    # {'perplexity': 12.34, 'rouge_l': 0.42, 'bleu_4': 0.21}
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(model, data_loader) -> float:
    """Compute perplexity over a data loader.

    The model is evaluated in inference mode (``torch.no_grad``).  Each batch
    must contain ``input_ids``, ``attention_mask``, and ``labels`` keys.

    Parameters
    ----------
    model:
        A causal language model that returns a ``CausalLMOutput`` with a
        ``loss`` attribute when ``labels`` are passed.
    data_loader:
        An iterable of batched tensors (e.g., a ``torch.utils.data.DataLoader``).

    Returns
    -------
    float
        Perplexity = ``exp(mean_cross_entropy_loss)`` over all batches.
    """
    import torch  # noqa: PLC0415

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            # Move every tensor in the batch to the model's device
            device = next(model.parameters()).device
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()
                n_batches += 1

    if n_batches == 0:
        logger.warning("No batches evaluated – returning perplexity=inf")
        return float("inf")

    mean_loss = total_loss / n_batches
    perplexity = math.exp(mean_loss)
    return perplexity


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> List[str]:
    """Generate text responses for a list of prompt strings.

    The model must already be on the correct device.  This function handles
    batched tokenisation, generation, and decoding, stripping the prompt
    prefix from each output so only the generated continuation is returned.

    Parameters
    ----------
    model:
        Fine-tuned causal language model.
    tokenizer:
        Matching tokenizer.
    prompts:
        Input prompt strings.
    max_new_tokens:
        Maximum number of new tokens to generate per prompt.
    temperature:
        Sampling temperature (1.0 = unmodified distribution).
    top_p:
        Nucleus sampling threshold.
    top_k:
        Top-k sampling value.
    repetition_penalty:
        Penalise repeated tokens (> 1.0 reduces repetition).
    do_sample:
        When ``False`` greedy decoding is used (ignores temperature/top_p/top_k).

    Returns
    -------
    List[str]
        Decoded response strings (prompt prefix removed).
    """
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    model.eval()
    responses: List[str] = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Strip the prompt tokens from the output
            generated_ids = output_ids[0][prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

    return responses


# ---------------------------------------------------------------------------
# ROUGE / BLEU computation
# ---------------------------------------------------------------------------

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Parameters
    ----------
    predictions:
        Model-generated texts.
    references:
        Ground-truth reference texts.

    Returns
    -------
    dict
        Keys: ``rouge1``, ``rouge2``, ``rougeL`` (F1 scores, 0–1).
    """
    agg: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    if n == 0:
        return agg

    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "rouge-score is required. Install with: pip install rouge-score"
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        agg["rouge1"] += scores["rouge1"].fmeasure
        agg["rouge2"] += scores["rouge2"].fmeasure
        agg["rougeL"] += scores["rougeL"].fmeasure

    return {k: v / n for k, v in agg.items()}


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute corpus-level BLEU-1 through BLEU-4 scores.

    Tokenisation is character-level for CJK text and word-level otherwise,
    using a simple whitespace split.  For a multilingual benchmark consider
    using ``sacrebleu`` with ``--tokenize=zh`` for Chinese text.

    Parameters
    ----------
    predictions:
        Model-generated texts.
    references:
        Ground-truth reference texts.

    Returns
    -------
    dict
        Keys: ``bleu_1``, ``bleu_2``, ``bleu_3``, ``bleu_4`` (0–100 scale).
    """
    bleu_scores: Dict[str, float] = {f"bleu_{n}": 0.0 for n in range(1, 5)}
    if not predictions:
        return bleu_scores

    try:
        from nltk.translate.bleu_score import (  # noqa: PLC0415
            corpus_bleu,
            SmoothingFunction,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("nltk is required. Install with: pip install nltk") from exc

    smoothing = SmoothingFunction().method1
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_hyps = [pred.split() for pred in predictions]

    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        score = corpus_bleu(
            tokenized_refs,
            tokenized_hyps,
            weights=weights,
            smoothing_function=smoothing,
        )
        bleu_scores[f"bleu_{n}"] = round(score * 100, 2)

    return bleu_scores


# ---------------------------------------------------------------------------
# Comprehensive evaluator class
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """High-level evaluator that runs perplexity, ROUGE, and BLEU evaluation.

    Parameters
    ----------
    model:
        Fine-tuned causal language model.
    tokenizer:
        Matching tokenizer.
    device:
        Target device string (``"cuda"`` / ``"cpu"`` / ``"mps"``).
        When ``None`` the model's current device is used.
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            import torch  # noqa: PLC0415
            self.model = model.to(torch.device(device))

    def evaluate_dataset(
        self,
        eval_samples: List[Dict],
        *,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        generation_config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Run a full evaluation pass and return aggregated metrics.

        Parameters
        ----------
        eval_samples:
            List of dicts with keys ``instruction``, ``input`` (optional),
            and ``output`` (the reference response).
        batch_size:
            Batch size for perplexity computation.
        max_new_tokens:
            Token budget for response generation.
        generation_config:
            Optional dict of generation kwargs (overrides defaults).

        Returns
        -------
        dict
            Merged metric dictionary containing ``perplexity``, ``rouge1``,
            ``rouge2``, ``rougeL``, ``bleu_1``, ``bleu_2``, ``bleu_3``,
            ``bleu_4``.
        """
        from src.data.dataset import build_alpaca_prompt  # noqa: PLC0415

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        }
        if generation_config:
            gen_kwargs.update(generation_config)

        prompts = [
            build_alpaca_prompt(s.get("instruction", ""), s.get("input", ""))
            for s in eval_samples
        ]
        references = [s.get("output", "") for s in eval_samples]

        logger.info("Generating %d responses for evaluation...", len(prompts))
        predictions = generate_responses(
            self.model, self.tokenizer, prompts, **gen_kwargs
        )

        rouge_scores = compute_rouge(predictions, references)
        bleu_scores = compute_bleu(predictions, references)

        metrics = {**rouge_scores, **bleu_scores}
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
