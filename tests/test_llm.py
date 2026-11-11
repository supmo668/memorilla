"""Tests for ``inject_memory``: placeholder positions get memory tokens, and per-row count mismatches raise."""

import pytest
import torch

from memorilla.llm import inject_memory

MEMORY_TOKEN_ID = 7
NUM_MEMORIES = 2
HIDDEN = 3


def make_inputs(input_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build embeddings, memory tokens and ids for a batch of two rows.

    Args:
        input_ids: Token ids per row; ``MEMORY_TOKEN_ID`` marks a placeholder.

    Returns:
        The float32 embeddings ``[2, seq_len, HIDDEN]``, float64 memory tokens ``[2, NUM_MEMORIES, HIDDEN]`` and
        the ids ``[2, seq_len]``.
    """
    ids = torch.tensor(input_ids)
    batch, seq_len = ids.shape
    inputs_embeds = torch.arange(batch * seq_len * HIDDEN, dtype=torch.float32).view(batch, seq_len, HIDDEN)
    memory_tokens = -1.0 - torch.arange(batch * NUM_MEMORIES * HIDDEN, dtype=torch.float64)
    return inputs_embeds, memory_tokens.view(batch, NUM_MEMORIES, HIDDEN), ids


def test_inject_memory_places_tokens_in_order() -> None:
    m = MEMORY_TOKEN_ID
    inputs_embeds, memory_tokens, ids = make_inputs([[1, m, 2, m, 3], [m, 4, m, 5, 6]])

    out = inject_memory(inputs_embeds, memory_tokens, ids, MEMORY_TOKEN_ID)

    expected = inputs_embeds.clone()
    expected[0, 1] = memory_tokens[0, 0].float()
    expected[0, 3] = memory_tokens[0, 1].float()
    expected[1, 0] = memory_tokens[1, 0].float()
    expected[1, 2] = memory_tokens[1, 1].float()
    assert out.dtype == inputs_embeds.dtype
    assert out.shape == inputs_embeds.shape
    torch.testing.assert_close(out, expected)


def test_inject_memory_rejects_per_row_mismatch_with_right_total() -> None:
    m = MEMORY_TOKEN_ID
    inputs_embeds, memory_tokens, ids = make_inputs([[m, m, m, 1], [m, 2, 3, 4]])

    with pytest.raises(ValueError, match=r"Row 0 has 3 .* 2 memory tokens"):
        inject_memory(inputs_embeds, memory_tokens, ids, MEMORY_TOKEN_ID)


def test_inject_memory_rejects_extra_placeholders() -> None:
    m = MEMORY_TOKEN_ID
    inputs_embeds, memory_tokens, ids = make_inputs([[m, m, 1, 2], [m, m, m, 3]])

    with pytest.raises(ValueError, match=r"Row 1 has 3"):
        inject_memory(inputs_embeds, memory_tokens, ids, MEMORY_TOKEN_ID)
