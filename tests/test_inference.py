"""Tests for inference engine (mock GPU — runs on CPU in CI)."""

import pytest
from unittest.mock import MagicMock, patch
import torch


def test_generate_returns_required_keys():
    """generate() must return text, latency_ms, tokens_generated, tokens_per_second."""
    from src.engine.inference import generate

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.return_value = {"input_ids": mock_input_ids, "attention_mask": torch.ones(1, 3)}
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "This is a generated response."
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with patch("src.engine.inference.DEVICE", "cpu"):
        result = generate(mock_model, mock_tokenizer, "Test prompt", max_new_tokens=3)

    assert "text" in result
    assert "latency_ms" in result
    assert "tokens_generated" in result
    assert "tokens_per_second" in result
    assert result["latency_ms"] >= 0


def test_estimate_model_size():
    from src.engine.quantization import estimate_model_size
    import torch.nn as nn

    model = nn.Linear(1000, 1000)  # 1M params
    sizes = estimate_model_size(model)

    assert "params_billions" in sizes
    assert "fp16_gb" in sizes
    assert sizes["fp16_gb"] < sizes["fp32_gb"]
    assert sizes["int4_gb"] < sizes["int8_gb"]


def test_kv_cache_size_estimate():
    from src.engine.kv_cache import estimate_kv_cache_size_gb, PHI3_MINI_KV_CONFIG

    size = estimate_kv_cache_size_gb(
        **PHI3_MINI_KV_CONFIG,
        max_seq_len=2048,
        batch_size=1,
        dtype_bytes=2,
    )
    assert size > 0
    assert size < 10  # Should be well under 10 GB for these params
