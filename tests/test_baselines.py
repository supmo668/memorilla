"""Tests for the text-only baselines: argument parsing, retrieval, precomputed contexts, output labels, the fit-rate
summary and generation with a fake engine."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from datasets import Dataset
import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from evaluate_baselines import (
    CLOSED_BOOK,
    CONTEXT_FILE,
    GENERATION_RESERVE,
    METHODS,
    PRECOMPUTED,
    RAG,
    build_prompts,
    context_summary,
    context_window,
    generate_baseline,
    load_contexts,
    method_label,
    parse_args,
    prepare_prompts,
    retrieve,
)
from memorilla.benchmarks import ACCURACY, BENCHMARKS, CHOICE, BaselineProtocol, Benchmark
from memorilla.data import EVAL_COLUMNS, load_split
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


TOY = Benchmark(
    name="toy",
    config="toy",
    split="test",
    system_prompt="You are",
    max_length=128,
    max_new_tokens=4,
    scoring=CHOICE,
    metrics=(ACCURACY,),
    baseline=BaselineProtocol("You are a helpful", max_new_tokens=4, full_context_length=256, suffix=" answer briefly"),
)
TOY_QUESTIONS = [str(row["question"]).strip() for row in make_rows()[:2]]


@pytest.fixture
def toy(monkeypatch: pytest.MonkeyPatch) -> Benchmark:
    """Register the ``toy`` benchmark (the ``data_repo`` test split) for the duration of a test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The benchmark.
    """
    monkeypatch.setitem(BENCHMARKS, TOY.name, TOY)
    return TOY


def toy_split(data_repo: Path) -> Dataset:
    """Load the two-row ``toy`` test split of the ``data_repo`` fixture.

    Args:
        data_repo: The data repository.

    Returns:
        The split.
    """
    return load_split("toy", "test", data_repo, EVAL_COLUMNS)


def toy_records() -> list[dict[str, Any]]:
    """Return a valid contexts file for the ``toy`` test split.

    Returns:
        One record per row, in row order.
    """
    return [
        {"benchmark": "toy", "row": 0, "question": TOY_QUESTIONS[0], "contexts": ["paris is the capital", "france"]},
        {"benchmark": "toy", "row": 1, "question": TOY_QUESTIONS[1], "contexts": []},
    ]


def write_contexts(path: Path, records: list[Any]) -> Path:
    """Write records as JSON lines (strings are written as they are).

    Args:
        path: Output file.
        records: Records, or raw lines.

    Returns:
        The file.
    """
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records))
    return path


def precomputed_args(data_repo: Path, contexts: Path) -> argparse.Namespace:
    """Arguments of a ``--method precomputed`` run on the local data repository.

    Args:
        data_repo: The data repository.
        contexts: The contexts file.

    Returns:
        The arguments ``prepare_prompts`` reads.
    """
    return argparse.Namespace(
        method=PRECOMPUTED, contexts=str(contexts), top_k=5, data_repo=str(data_repo), data_dir=str(data_repo)
    )


def test_parse_args_accepts_precomputed_with_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--method precomputed`` takes the contexts file from ``--contexts``."""
    argv = ["evaluate_baselines.py", "--benchmark", "nq", "--method", PRECOMPUTED, "--contexts", "c.jsonl"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    assert (args.method, args.contexts) == (PRECOMPUTED, "c.jsonl")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--method", PRECOMPUTED], "--contexts is required with --method precomputed"),
        (["--method", RAG, "--contexts", "c.jsonl"], "--contexts only applies to --method precomputed"),
    ],
)
def test_parse_args_ties_contexts_to_precomputed(
    extra: list[str], message: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--contexts`` is required by, and only accepted with, ``--method precomputed``."""
    monkeypatch.setattr(sys, "argv", ["evaluate_baselines.py", "--benchmark", "nq", *extra])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_precomputed_label_and_window() -> None:
    """Precomputed contexts get their own folder and the retrieval context window."""
    assert PRECOMPUTED in METHODS
    assert method_label(PRECOMPUTED, 5) == "precomputed"
    benchmark = BENCHMARKS["triviaqa"]
    assert context_window(benchmark, PRECOMPUTED) == context_window(benchmark, RAG)


def test_load_contexts_returns_contexts_in_row_order(data_repo: Path, toy: Benchmark, tmp_path: Path) -> None:
    """Lines may come in any order; contexts come back per row, in the order given, empty lists kept."""
    path = write_contexts(tmp_path / "c.jsonl", toy_records()[::-1])
    assert load_contexts(path, toy, toy_split(data_repo)) == [["paris is the capital", "france"], []]


def test_load_contexts_ignores_other_registered_benchmarks_and_blank_lines(
    data_repo: Path, toy: Benchmark, tmp_path: Path
) -> None:
    """One file can serve ``--benchmark all``: lines of other registered benchmarks are skipped."""
    other = {"benchmark": "triviaqa", "row": 0, "question": "anything", "contexts": [1]}
    path = write_contexts(tmp_path / "c.jsonl", [other, toy_records()[0], "", toy_records()[1]])
    assert load_contexts(path, toy, toy_split(data_repo)) == [["paris is the capital", "france"], []]


def bad(field: str, value: Any) -> list[Any]:
    """Return the valid toy records with one field of the second record replaced.

    Args:
        field: Field name.
        value: Its new value.

    Returns:
        The records.
    """
    records = toy_records()
    records[1] = {**records[1], field: value}
    return records


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (toy_records()[:1], r"c\.jsonl: no line for row\(s\) \[1\] of benchmark 'toy'"),
        ([*toy_records(), toy_records()[0]], r"c\.jsonl:3: row 0 of benchmark 'toy' already given on line 1"),
        (bad("row", 2), r"c\.jsonl:2: row 2 is out of range: 'toy' has 2 rows"),
        (bad("row", -1), r"c\.jsonl:2: row -1 is out of range"),
        (bad("row", "1"), r"c\.jsonl:2: 'row' must be an integer"),
        (bad("row", True), r"c\.jsonl:2: 'row' must be an integer"),
        (bad("question", "who wrote paris"), r"c\.jsonl:2: question does not match row 1 of 'toy'"),
        (bad("contexts", ["hamlet", 3]), r"c\.jsonl:2: 'contexts' must be a list of strings"),
        (bad("contexts", "hamlet"), r"c\.jsonl:2: 'contexts' must be a list of strings"),
        (bad("benchmark", "no_such_benchmark"), r"c\.jsonl:2: unknown benchmark 'no_such_benchmark'"),
        ([toy_records()[0], "{not json"], r"c\.jsonl:2: not valid JSON"),
        ([toy_records()[0], "[1, 2]"], r"c\.jsonl:2: expected a JSON object"),
        (
            [toy_records()[0], {"benchmark": "toy", "row": 1}],
            r"c\.jsonl:2: missing key\(s\) \['contexts', 'question'\]",
        ),
    ],
)
def test_load_contexts_rejects_bad_files(
    records: list[Any], message: str, data_repo: Path, toy: Benchmark, tmp_path: Path
) -> None:
    """Every malformed file fails at load time with the file, the line and what is wrong."""
    path = write_contexts(tmp_path / "c.jsonl", records)
    with pytest.raises(ValueError, match=message):
        load_contexts(path, toy, toy_split(data_repo))


def test_load_contexts_accepts_unstripped_questions(data_repo: Path, toy: Benchmark, tmp_path: Path) -> None:
    """The question echo is compared after stripping, like the split's own question."""
    records = toy_records()
    records[0]["question"] = " what is the capital of france "
    path = write_contexts(tmp_path / "c.jsonl", records)
    assert load_contexts(path, toy, toy_split(data_repo))[0] == ["paris is the capital", "france"]


def test_build_prompts_keeps_question_and_context_order(
    data_repo: Path, toy: Benchmark, tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Loaded contexts go before the question in the order given; an empty list is a closed-book prompt."""
    records = toy_records()
    records[0]["contexts"] = ["france", "paris is the capital"]
    path = write_contexts(tmp_path / "c.jsonl", records)
    dataset = toy_split(data_repo)

    prompts, lengths = build_prompts(tokenizer, toy, dataset, load_contexts(path, toy, dataset), 1000)

    texts = [tokenizer.decode(ids) for ids in prompts]
    assert "user france paris is the capital what is the capital of france answer briefly" in texts[0]
    assert "user who wrote hamlet answer briefly" in texts[1]
    assert lengths == [len(ids) for ids in prompts]


def test_prepare_prompts_uses_precomputed_contexts(
    data_repo: Path, toy: Benchmark, tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    """``prepare_prompts`` puts the file's contexts where RAG would put retrieved documents."""
    path = write_contexts(tmp_path / "c.jsonl", toy_records())
    dataset = toy_split(data_repo)
    max_model_len, yarn_factor = context_window(toy, PRECOMPUTED)
    budget = max_model_len - toy.baseline.max_new_tokens - GENERATION_RESERVE
    expected, _ = build_prompts(tokenizer, toy, dataset, load_contexts(path, toy, dataset), budget)

    prompt_set = prepare_prompts(toy, precomputed_args(data_repo, path), tokenizer)

    assert prompt_set.prompts == expected
    assert (prompt_set.max_model_len, prompt_set.yarn_factor) == (max_model_len, yarn_factor)
    assert prompt_set.context["num_prompts"] == 2


def test_prepare_prompts_fails_on_a_stale_contexts_file(
    data_repo: Path, toy: Benchmark, tmp_path: Path, tokenizer: PreTrainedTokenizerFast
) -> None:
    """A file built for another revision of the split fails before any engine starts."""
    path = write_contexts(tmp_path / "c.jsonl", bad("question", "who wrote paris"))
    with pytest.raises(ValueError, match="question does not match row 1"):
        prepare_prompts(toy, precomputed_args(data_repo, path), tokenizer)


class FakeCompletion:
    """Mimics one vLLM generation result."""

    def __init__(self, text: str) -> None:
        """Wrap the generated text.

        Args:
            text: The text.
        """
        self.outputs = [SimpleNamespace(text=text)]


class FakeEngine:
    """Stand-in for a vLLM ``LLM``: answers each request with a fixed text and records the calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[list[dict[str, list[int]]], Any]] = []

    def generate(self, requests: list[dict[str, list[int]]], sampling_params: Any) -> list[FakeCompletion]:
        """Return one completion per request.

        Args:
            requests: Token-id requests.
            sampling_params: Sampling settings.

        Returns:
            The completions.
        """
        self.calls.append((requests, sampling_params))
        return [FakeCompletion(f" paris {index}<|im_end|>") for index in range(len(requests))]


def test_generate_baseline_writes_one_prediction_per_row(
    data_repo: Path,
    toy: Benchmark,
    tmp_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With precomputed contexts, generation sends every prompt once and returns a cleaned prediction per row."""
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **options: options))
    path = write_contexts(tmp_path / "c.jsonl", toy_records())
    prompt_set = prepare_prompts(toy, precomputed_args(data_repo, path), tokenizer)
    engine = FakeEngine()
    output_dir = tmp_path / "results" / method_label(PRECOMPUTED, 5)

    jobs = generate_baseline(engine, [prompt_set], output_dir)

    [(benchmark, dataset, predictions, job_dir)] = jobs
    assert benchmark is toy
    assert len(dataset) == 2
    assert predictions == ["paris 0", "paris 1"]
    assert job_dir == output_dir / "toy"
    [(requests, sampling)] = engine.calls
    assert requests == [{"prompt_token_ids": ids} for ids in prompt_set.prompts]
    assert sampling["truncate_prompt_tokens"] == prompt_set.context["max_input_tokens"]
    assert json.loads((job_dir / CONTEXT_FILE).read_text()) == prompt_set.context
