"""
tests/test_model_utils.py
=========================
Unit tests for src/model/model_utils.py.

All heavy ML imports (torch, transformers, peft) are mocked so the tests run
without a GPU or model download.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helper: build minimal mock objects
# ---------------------------------------------------------------------------

def make_mock_tokenizer(
    pad_token=None,
    eos_token="</s>",
    eos_token_id=1,
    padding_side="right",
):
    tok = MagicMock()
    tok.pad_token = pad_token
    tok.pad_token_id = None if pad_token is None else 0
    tok.eos_token = eos_token
    tok.eos_token_id = eos_token_id
    tok.padding_side = padding_side
    return tok


def make_mock_model(n_params=1_000_000, n_trainable=100_000):
    """Create a mock model with realistic parameter counts."""
    model = MagicMock()
    # Simulate parameters() returning a list of mock tensors
    frozen = MagicMock()
    frozen.numel.return_value = n_params - n_trainable
    frozen.requires_grad = False

    trainable = MagicMock()
    trainable.numel.return_value = n_trainable
    trainable.requires_grad = True

    model.parameters.return_value = [frozen, trainable]
    return model


# ---------------------------------------------------------------------------
# load_tokenizer
# ---------------------------------------------------------------------------

class TestLoadTokenizer:
    def _mock_transformers(self, mock_tok):
        """Return a mock transformers module with AutoTokenizer configured."""
        mock_tf = MagicMock()
        mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tok
        return mock_tf

    def test_sets_pad_token_when_missing(self):
        from src.model.model_utils import load_tokenizer

        mock_tok = make_mock_tokenizer(pad_token=None)
        with patch.dict("sys.modules", {"transformers": self._mock_transformers(mock_tok)}):
            result = load_tokenizer("any/model")

        assert result.pad_token == result.eos_token

    def test_does_not_overwrite_existing_pad_token(self):
        from src.model.model_utils import load_tokenizer

        mock_tok = make_mock_tokenizer(pad_token="<pad>")
        with patch.dict("sys.modules", {"transformers": self._mock_transformers(mock_tok)}):
            result = load_tokenizer("any/model")

        assert result.pad_token == "<pad>"

    def test_padding_side_set_to_right(self):
        from src.model.model_utils import load_tokenizer

        mock_tok = make_mock_tokenizer(pad_token="<pad>", padding_side="left")
        with patch.dict("sys.modules", {"transformers": self._mock_transformers(mock_tok)}):
            result = load_tokenizer("any/model")

        assert result.padding_side == "right"


# ---------------------------------------------------------------------------
# load_base_model
# ---------------------------------------------------------------------------

class TestLoadBaseModel:
    def _mock_transformers(self, mock_model):
        mock_tf = MagicMock()
        mock_tf.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_tf.BitsAndBytesConfig = MagicMock(return_value=MagicMock())
        return mock_tf

    def test_loads_causal_lm(self):
        from src.model.model_utils import load_base_model

        mock_model = MagicMock()
        mock_tf = self._mock_transformers(mock_model)
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": mock_torch}):
            result = load_base_model("any/model", torch_dtype="bfloat16")

        assert result is mock_model

    def test_qlora_creates_bnb_config(self):
        from src.model.model_utils import load_base_model

        mock_model = MagicMock()
        mock_tf = self._mock_transformers(mock_model)
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        mock_peft = MagicMock()
        mock_peft.prepare_model_for_kbit_training.return_value = mock_model

        with patch.dict("sys.modules", {
            "transformers": mock_tf,
            "torch": mock_torch,
            "peft": mock_peft,
        }):
            result = load_base_model("any/model", use_qlora=True)

        assert result is not None


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------

class TestApplyLora:
    def test_wraps_model_with_peft(self):
        from src.model.model_utils import apply_lora

        base_model = make_mock_model()
        peft_model = make_mock_model(n_trainable=16_000)  # LoRA makes few params trainable

        mock_lora_config = MagicMock()
        mock_peft = MagicMock()
        mock_peft.LoraConfig.return_value = mock_lora_config
        mock_peft.get_peft_model.return_value = peft_model
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": mock_peft}):
            # Reload the module to pick up the mock
            import importlib
            import src.model.model_utils as m
            importlib.reload(m)
            result = m.apply_lora(base_model, r=16, lora_alpha=32)

        assert result is peft_model

    def test_default_lora_rank(self):
        from src.model.model_utils import apply_lora

        base_model = make_mock_model()
        peft_model = make_mock_model()

        mock_peft = MagicMock()
        mock_peft.get_peft_model.return_value = peft_model
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"

        captured_config = {}

        def capture_config(**kwargs):
            captured_config.update(kwargs)
            return MagicMock()

        mock_peft.LoraConfig.side_effect = capture_config

        with patch.dict("sys.modules", {"peft": mock_peft}):
            import importlib, src.model.model_utils as m
            importlib.reload(m)
            m.apply_lora(base_model)  # use all defaults

        assert captured_config.get("r") == 16
        assert captured_config.get("lora_alpha") == 32


# ---------------------------------------------------------------------------
# get_model_parameter_info
# ---------------------------------------------------------------------------

class TestGetModelParameterInfo:
    def test_returns_correct_counts(self):
        from src.model.model_utils import get_model_parameter_info

        model = make_mock_model(n_params=1_000_000, n_trainable=100_000)
        trainable, total, pct = get_model_parameter_info(model)
        assert trainable == 100_000
        assert total == 1_000_000
        assert abs(pct - 10.0) < 1e-6

    def test_zero_params_model(self):
        from src.model.model_utils import get_model_parameter_info

        model = MagicMock()
        model.parameters.return_value = []
        trainable, total, pct = get_model_parameter_info(model)
        assert trainable == 0
        assert total == 0
        assert pct == 0.0
