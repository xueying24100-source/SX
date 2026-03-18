"""
tests/test_dataset.py
=====================
Unit tests for the data processing pipeline in src/data/dataset.py.

All tests run without a real GPU or Hugging Face model by using a lightweight
mock tokenizer so the test suite is fast and CI-friendly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    DataCollatorForSeq2Seq,
    InstructionDataset,
    build_alpaca_prompt,
    build_sharegpt_prompt,
)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """A lightweight tokenizer stub that converts characters to integer ids."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1
    padding_side = "right"

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = False,
        return_tensors=None,
    ) -> Dict:
        # Simple char-level tokenisation: each char → its ordinal (clipped to 1000)
        ids = [min(ord(c), 1000) for c in text]
        if truncation:
            ids = ids[:max_length]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(min(i, 127)) for i in ids if i > 1)


# ---------------------------------------------------------------------------
# build_alpaca_prompt tests
# ---------------------------------------------------------------------------

class TestBuildAlpacaPrompt:
    def test_without_input(self):
        prompt = build_alpaca_prompt("Say hello")
        assert "Instruction" in prompt
        assert "Say hello" in prompt
        assert "Response" in prompt
        # No input section when input is empty
        assert "### Input" not in prompt

    def test_with_input(self):
        prompt = build_alpaca_prompt("Translate", "Hello world")
        assert "### Input" in prompt
        assert "Hello world" in prompt
        assert "paired with an input" in prompt

    def test_empty_instruction(self):
        prompt = build_alpaca_prompt("")
        # Should not crash; prompt should still contain structural markers
        assert "### Response" in prompt

    def test_whitespace_input_treated_as_empty(self):
        prompt = build_alpaca_prompt("Do something", "   ")
        # Whitespace-only input should be treated as absent
        assert "### Input" not in prompt


# ---------------------------------------------------------------------------
# build_sharegpt_prompt tests
# ---------------------------------------------------------------------------

class TestBuildSharegptPrompt:
    def test_fallback_format(self):
        """When tokenizer has no apply_chat_template, use the fallback format."""
        tokenizer = MockTokenizer()  # no apply_chat_template method
        conversations = [
            {"from": "human", "value": "What is LoRA?"},
            {"from": "assistant", "value": "LoRA is a fine-tuning method."},
        ]
        result = build_sharegpt_prompt(conversations, tokenizer)
        assert "Human:" in result
        assert "What is LoRA?" in result
        assert "Assistant:" in result
        assert "LoRA is a fine-tuning method." in result

    def test_with_chat_template(self):
        """When tokenizer has apply_chat_template, it should be called."""
        tokenizer = MockTokenizer()
        tokenizer.apply_chat_template = MagicMock(return_value="<chat>formatted</chat>")
        conversations = [
            {"from": "human", "value": "Hi"},
            {"from": "assistant", "value": "Hello!"},
        ]
        result = build_sharegpt_prompt(conversations, tokenizer)
        assert result == "<chat>formatted</chat>"
        tokenizer.apply_chat_template.assert_called_once()

    def test_user_role_alias(self):
        """Both 'user' and 'human' should map to the Human prefix."""
        tokenizer = MockTokenizer()
        conversations = [{"from": "user", "value": "question"}]
        result = build_sharegpt_prompt(conversations, tokenizer)
        assert "Human:" in result


# ---------------------------------------------------------------------------
# InstructionDataset tests
# ---------------------------------------------------------------------------

class TestInstructionDataset:
    def _write_jsonl(self, tmp_path: Path, records: List[Dict]) -> Path:
        path = tmp_path / "data.jsonl"
        with path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def test_basic_loading(self, tmp_path):
        records = [
            {"instruction": "Say hi", "input": "", "output": "Hi there!"},
            {"instruction": "What is 2+2?", "input": "", "output": "4"},
        ]
        path = self._write_jsonl(tmp_path, records)
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=256, data_format="alpaca")
        assert len(dataset) == 2

    def test_getitem_keys(self, tmp_path):
        records = [{"instruction": "Hello", "input": "", "output": "World"}]
        path = self._write_jsonl(tmp_path, records)
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=256)
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

    def test_loss_masking(self, tmp_path):
        """Prompt tokens should be masked with -100 in labels."""
        records = [{"instruction": "Say hello", "input": "", "output": "Hello World!"}]
        path = self._write_jsonl(tmp_path, records)
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=512)
        sample = dataset[0]
        labels = sample["labels"]
        # At least some tokens must be masked (the prompt)
        assert -100 in labels
        # At least some tokens must be unmasked (the response)
        assert any(lbl != -100 for lbl in labels)

    def test_max_length_respected(self, tmp_path):
        long_output = "A" * 2000
        records = [{"instruction": "Generate", "input": "", "output": long_output}]
        path = self._write_jsonl(tmp_path, records)
        tokenizer = MockTokenizer()
        max_length = 128
        dataset = InstructionDataset(path, tokenizer, max_length=max_length)
        for sample in dataset:
            assert len(sample["input_ids"]) <= max_length

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=256)
        assert len(dataset) == 0

    def test_malformed_json_skipped(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"instruction": "valid", "input": "", "output": "ok"}\n'
            "NOT VALID JSON\n"
            '{"instruction": "also valid", "input": "", "output": "yes"}\n'
        )
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=256)
        # Only 2 valid records should be loaded
        assert len(dataset) == 2

    def test_sharegpt_format(self, tmp_path):
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "What is AI?"},
                    {"from": "assistant", "value": "AI is artificial intelligence."},
                ]
            }
        ]
        path = self._write_jsonl(tmp_path, records)
        tokenizer = MockTokenizer()
        dataset = InstructionDataset(path, tokenizer, max_length=256, data_format="sharegpt")
        assert len(dataset) == 1
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample


# ---------------------------------------------------------------------------
# DataCollatorForSeq2Seq tests
# ---------------------------------------------------------------------------

class TestDataCollatorForSeq2Seq:
    def test_collates_to_same_length(self):
        import torch  # noqa: PLC0415

        tokenizer = MockTokenizer()
        collator = DataCollatorForSeq2Seq(tokenizer, max_length=64)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8], "attention_mask": [1, 1, 1, 1, 1],
             "labels": [-100, -100, 6, 7, 8]},
        ]
        batch = collator(features)

        # All tensors should have the same sequence length
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape == batch["labels"].shape

    def test_padding_values(self):
        import torch  # noqa: PLC0415

        tokenizer = MockTokenizer()
        collator = DataCollatorForSeq2Seq(tokenizer, max_length=10)
        features = [
            {"input_ids": [10, 20], "attention_mask": [1, 1], "labels": [10, 20]},
            {"input_ids": [30, 40, 50, 60], "attention_mask": [1, 1, 1, 1],
             "labels": [30, 40, 50, 60]},
        ]
        batch = collator(features)
        # Shorter sequence should be padded with pad_token_id (0) for input_ids
        assert batch["input_ids"][0, 2].item() == tokenizer.pad_token_id
        # Labels padding should use -100
        assert batch["labels"][0, 2].item() == -100

    def test_max_length_truncation(self):
        tokenizer = MockTokenizer()
        collator = DataCollatorForSeq2Seq(tokenizer, max_length=3)
        features = [
            {"input_ids": list(range(10)), "attention_mask": [1] * 10,
             "labels": list(range(10))},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape[1] <= 3
