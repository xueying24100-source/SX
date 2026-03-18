"""
tests/test_evaluator.py
=======================
Unit tests for src/evaluation/evaluator.py.

All tests mock out heavy dependencies (torch, rouge_score, nltk) so they run
without a GPU or model download.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import compute_bleu, compute_rouge


# ---------------------------------------------------------------------------
# compute_rouge
# ---------------------------------------------------------------------------

class TestComputeRouge:
    def test_identical_texts_high_score(self):
        """Identical prediction and reference should give maximum ROUGE."""
        try:
            from rouge_score import rouge_scorer  # noqa: F401
        except ImportError:
            pytest.skip("rouge-score not installed")

        text = "The quick brown fox jumps over the lazy dog."
        scores = compute_rouge([text], [text])
        assert scores["rouge1"] > 0.99
        assert scores["rougeL"] > 0.99

    def test_empty_lists_return_zeros(self):
        scores = compute_rouge([], [])
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_returns_expected_keys(self):
        try:
            from rouge_score import rouge_scorer  # noqa: F401
        except ImportError:
            pytest.skip("rouge-score not installed")

        scores = compute_rouge(["hello world"], ["hello world"])
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_different_texts_lower_score(self):
        try:
            from rouge_score import rouge_scorer  # noqa: F401
        except ImportError:
            pytest.skip("rouge-score not installed")

        scores = compute_rouge(["cat sat on mat"], ["the weather is sunny today"])
        assert scores["rouge1"] < 0.5


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------

class TestComputeBleu:
    def test_identical_texts_high_score(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        text = "the model generates high quality text"
        scores = compute_bleu([text], [text])
        assert scores["bleu_1"] > 80.0
        assert scores["bleu_4"] > 50.0

    def test_returns_expected_keys(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        scores = compute_bleu(["hello world"], ["hello world"])
        assert set(scores.keys()) == {"bleu_1", "bleu_2", "bleu_3", "bleu_4"}

    def test_scores_in_0_to_100_range(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        scores = compute_bleu(["foo bar baz"], ["foo bar baz"])
        for k, v in scores.items():
            assert 0.0 <= v <= 100.0, f"{k} = {v} out of range"

    def test_empty_lists_return_zeros(self):
        # No library needed — empty list short-circuits before import
        scores = compute_bleu([], [])
        assert all(v == 0.0 for v in scores.values())
