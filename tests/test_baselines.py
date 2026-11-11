"""Tests for the text-only baselines: argument parsing, retrieval, output labels and the fit-rate summary."""

from pathlib import Path
import sys

import numpy as np
import pytest

from evaluate_baselines import CLOSED_BOOK, RAG, context_summary, method_label, parse_args, retrieve
from memorilla.embeddings import EmbeddingStore, build_embeddings, embedding_dir
from tests.helpers import FakeEncoder, make_rows, write_split


@pytest.mark.parametrize("top_k", ["0", "-1"])
def test_parse_args_rejects_non_positive_top_k(
    top_k: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retrieval depth below one is a usage error that names the flag."""
    monkeypatch.setattr(sys, "argv", ["evaluate_baselines.py", "--benchmark", "nq", "--method", RAG, "--top_k", top_k])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "argument --top_k" in stderr
    assert "must be a positive integer" in stderr


def test_context_summary_rejects_empty_splits() -> None:
    """Summarising no prompts fails with a message that says the split is empty."""
    with pytest.raises(ValueError, match="no rows"):
        context_summary([], max_model_len=8, max_input_tokens=4)


def test_retrieve_orders_by_similarity_and_caps_at_document_count(data_repo: Path, fake_encoder: FakeEncoder) -> None:
    """Retrieval returns each row's documents most similar first and never more than the row has."""
    rows = make_rows()
    dataset = write_split(data_repo, "toy", "dev", rows)
    build_embeddings(dataset, embedding_dir("toy", "dev", data_repo), fake_encoder)
    store = EmbeddingStore("toy", "dev", root=data_repo)

    retrieved = retrieve(dataset, store, top_k=3)

    for row, documents in zip(rows, retrieved, strict=True):
        question = fake_encoder.vector(row["question"].strip())
        vectors = np.stack([fake_encoder.vector(doc) for doc in row["documents"]])
        similarities = vectors @ question / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(question))
        expected = [row["documents"][index] for index in np.argsort(-similarities)]
        assert documents == expected[: min(3, len(row["documents"]))]
    assert [len(documents) for documents in retrieved] == [2, 3, 2]


def test_method_label() -> None:
    """Retrieval folders carry the depth; other methods use their own name."""
    assert method_label(RAG, 5) == "rag_top5"
    assert method_label(CLOSED_BOOK, 5) == "closed_book"
