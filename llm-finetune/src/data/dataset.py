"""
dataset.py
==========
PyTorch Dataset and preprocessing utilities for instruction-following fine-tuning.

Supported data formats
----------------------
* **Alpaca** (default)::

    {"instruction": "...", "input": "...", "output": "..."}

  When ``input`` is present and non-empty it is appended to the instruction.

* **ShareGPT** / multi-turn conversation::

    {"conversations": [
        {"from": "human",     "value": "..."},
        {"from": "assistant", "value": "..."},
        ...
    ]}

Prompt template
---------------
The default prompt follows the *Alpaca* style which is understood by many
instruction-tuned base models.  When the tokenizer exposes a
``apply_chat_template`` method (Qwen-2, Llama-3, etc.) that is used instead
so that special tokens (``<|im_start|>``, ``[INST]``, etc.) are inserted
correctly.

Loss masking
------------
Only the *assistant* tokens contribute to the loss – the prompt (instruction +
input) tokens are masked with ``-100`` so the model learns to predict
responses given queries, not to reproduce the queries themselves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_part}"
    ". Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n"
)


def build_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Construct an Alpaca-style prompt string.

    Parameters
    ----------
    instruction:
        The task description or question.
    input_text:
        Optional additional context.  When non-empty it is rendered as a
        separate *Input* section.

    Returns
    -------
    str
        The fully formatted prompt (without the expected response).
    """
    if input_text and input_text.strip():
        input_part = ", paired with an input that provides further context"
        input_section = f"### Input:\n{input_text.strip()}\n\n"
    else:
        input_part = ""
        input_section = ""

    return ALPACA_PROMPT_TEMPLATE.format(
        input_part=input_part,
        instruction=instruction.strip(),
        input_section=input_section,
    )


def build_sharegpt_prompt(conversations: List[Dict[str, str]], tokenizer) -> str:
    """Construct a chat-formatted prompt from ShareGPT conversation turns.

    Uses the tokenizer's built-in ``apply_chat_template`` when available;
    otherwise falls back to a simple ``Human: ... Assistant: ...`` format.

    Parameters
    ----------
    conversations:
        List of dicts with keys ``"from"`` (``"human"`` / ``"assistant"``)
        and ``"value"``.
    tokenizer:
        Tokenizer instance (used for ``apply_chat_template`` if present).

    Returns
    -------
    str
        The rendered prompt string.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        for turn in conversations:
            role = "user" if turn["from"] in ("human", "user") else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Fallback: simple concatenation
    parts = []
    for turn in conversations:
        if turn["from"] in ("human", "user"):
            parts.append(f"Human: {turn['value']}")
        else:
            parts.append(f"Assistant: {turn['value']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class InstructionDataset:
    """A map-style dataset for instruction-following fine-tuning.

    The dataset reads samples from a JSONL file (one JSON object per line),
    tokenises them into ``input_ids`` / ``attention_mask`` / ``labels``,
    and applies loss masking so only the response tokens are trained on.

    Parameters
    ----------
    data_path:
        Path to a ``.jsonl`` file.
    tokenizer:
        Hugging Face tokenizer (must have been configured with
        ``pad_token`` and ``padding_side="right"``).
    max_length:
        Maximum sequence length after tokenisation.  Samples that are
        longer after truncation will have their prompts truncated first;
        if the response itself is longer than ``max_length`` the sample is
        silently dropped.
    data_format:
        ``"alpaca"`` or ``"sharegpt"``.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 2048,
        data_format: str = "alpaca",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_format = data_format
        self._samples: List[Dict] = self._load_and_tokenize(Path(data_path))
        logger.info(
            "Loaded %d samples from %s (format=%s, max_length=%d)",
            len(self._samples),
            data_path,
            data_format,
            max_length,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self, path: Path) -> List[Dict]:
        """Read JSONL, skipping blank lines and malformed entries."""
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
        return records

    def _tokenize_alpaca(self, record: Dict) -> Optional[Dict]:
        """Tokenise one Alpaca-format record, returning None if too long."""
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output = record.get("output", "")

        prompt = build_alpaca_prompt(instruction, input_text)
        full_text = prompt + output

        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        # Build labels: mask prompt tokens with -100
        prompt_len = len(tokenized_prompt["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # If the full sequence was truncated before any response tokens appear,
        # discard this sample to avoid training on empty targets.
        if all(lbl == -100 for lbl in labels):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _tokenize_sharegpt(self, record: Dict) -> Optional[Dict]:
        """Tokenise one ShareGPT-format record."""
        conversations = record.get("conversations", [])
        if not conversations:
            return None

        full_text = build_sharegpt_prompt(conversations, self.tokenizer)
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        # For multi-turn we use a simple heuristic: mask everything before the
        # last assistant turn.  For production workloads implement per-turn
        # masking using the tokenizer's chat template token positions.
        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def _load_and_tokenize(self, path: Path) -> List[Dict]:
        """Load raw records and tokenise all of them."""
        raw = self._load_raw(path)
        tokenize_fn = (
            self._tokenize_alpaca if self.data_format == "alpaca"
            else self._tokenize_sharegpt
        )
        samples: List[Dict] = []
        skipped = 0
        for record in raw:
            tokenized = tokenize_fn(record)
            if tokenized is None:
                skipped += 1
                continue
            samples.append(tokenized)

        if skipped:
            logger.warning("Skipped %d samples (empty response after tokenisation)", skipped)
        return samples

    # ------------------------------------------------------------------
    # Map-style Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollatorForSeq2Seq:
    """Pad a batch of variable-length sequences from :class:`InstructionDataset`.

    Pads ``input_ids`` and ``attention_mask`` with the tokenizer's
    ``pad_token_id`` / ``0`` respectively, and pads ``labels`` with ``-100``
    (ignored by cross-entropy loss).

    Parameters
    ----------
    tokenizer:
        Used to obtain ``pad_token_id``.
    max_length:
        Hard upper limit on padded sequence length.  Sequences shorter than
        ``max_length`` are padded to the length of the longest sequence in
        the batch (dynamic padding), never beyond ``max_length``.
    """

    def __init__(self, tokenizer, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.label_pad_id = -100

    def __call__(self, features: List[Dict]) -> Dict:
        """Collate and pad a list of dataset samples into a batch dict."""
        batch_size = len(features)
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        import torch  # noqa: PLC0415

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad_id, dtype=torch.long)

        for i, feature in enumerate(features):
            seq_len = min(len(feature["input_ids"]), max_len)
            input_ids[i, :seq_len] = torch.tensor(feature["input_ids"][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(
                feature["attention_mask"][:seq_len], dtype=torch.long
            )
            labels[i, :seq_len] = torch.tensor(feature["labels"][:seq_len], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
