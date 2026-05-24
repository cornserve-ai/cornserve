"""Tests for OmniFlexTask routing and validation."""

from __future__ import annotations

import uuid
from collections import Counter

import pytest
from cornserve_tasklib.task.composite.omni import (
    OmniFlexGroupConfig,
    OmniFlexTask,
    ThinkerLLMConfig,
    _select_group,
    _select_thinker,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
N_SAMPLES = 5000
TOLERANCE = 0.03  # ±3%


def _make_request_ids(n: int) -> list[str]:
    """Generate n random hex request IDs (matching production uuid4.hex)."""
    return [uuid.uuid4().hex for _ in range(n)]


# ── Validation ────────────────────────────────────────────────────────


def test_empty_groups_rejected():
    with pytest.raises(ValueError, match="groups must contain at least one group"):
        OmniFlexTask(
            model_id=MODEL_ID,
            groups=[],
        )


def test_empty_thinkers_rejected():
    with pytest.raises(ValueError, match="thinkers must be non-empty"):
        OmniFlexTask(
            model_id=MODEL_ID,
            groups=[OmniFlexGroupConfig(thinkers=[])],
        )


def test_missing_batch_size_rejected():
    with pytest.raises(ValueError, match="missing from eric_max_batch_sizes"):
        OmniFlexTask(
            model_id=MODEL_ID,
            groups=[
                OmniFlexGroupConfig(
                    offloaded_modalities=["img"],
                    eric_max_batch_sizes={},  # missing "img"
                    thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)],
                )
            ],
        )


def test_invalid_group_index_rejected():
    with pytest.raises(ValueError, match="group index 5 out of range"):
        OmniFlexTask(
            model_id=MODEL_ID,
            groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
            type_routing_weights={1: {5: 1.0}},
        )


def test_negative_weight_rejected():
    with pytest.raises(ValueError, match="weight must be non-negative"):
        OmniFlexTask(
            model_id=MODEL_ID,
            groups=[
                OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)]),
                OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=1, max_num_seqs=64)]),
            ],
            type_routing_weights={1: {0: 1.0, 1: -0.5}},
        )


def test_single_group_no_type_routing_ok():
    """Single group with empty type_routing_weights is valid."""
    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
    )
    assert len(task.groups) == 1


# ── _select_group determinism ─────────────────────────────────────────


def test_select_group_deterministic():
    """Same request_id always selects the same group."""
    cdf = [(0, 0.3), (1, 0.7), (2, 1.0)]
    rid = uuid.uuid4().hex
    first = _select_group(rid, cdf)
    for _ in range(100):
        assert _select_group(rid, cdf) == first


def test_select_group_distribution():
    """Group selection follows weight distribution."""
    # Weights: group0=30%, group1=70%
    cdf = [(0, 0.3), (1, 1.0)]
    counts = Counter(_select_group(uuid.uuid4().hex, cdf) for _ in range(N_SAMPLES))
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.3) < TOLERANCE
    assert abs(counts[1] / total - 0.7) < TOLERANCE


def test_select_group_three_way():
    """Three-way split: 20/50/30."""
    cdf = [(0, 0.2), (1, 0.7), (2, 1.0)]
    counts = Counter(_select_group(uuid.uuid4().hex, cdf) for _ in range(N_SAMPLES))
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.2) < TOLERANCE
    assert abs(counts[1] / total - 0.5) < TOLERANCE
    assert abs(counts[2] / total - 0.3) < TOLERANCE


# ── _select_thinker determinism ───────────────────────────────────────


def test_select_thinker_deterministic():
    """Same request_id always selects same thinker."""
    cdf = [0.4, 1.0]
    rid = uuid.uuid4().hex
    first = _select_thinker(rid, cdf)
    for _ in range(100):
        assert _select_thinker(rid, cdf) == first


def test_select_thinker_distribution():
    """Thinker selection follows weight distribution."""
    # 40/60 split
    cdf = [0.4, 1.0]
    counts = Counter(_select_thinker(uuid.uuid4().hex, cdf) for _ in range(N_SAMPLES))
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.4) < TOLERANCE
    assert abs(counts[1] / total - 0.6) < TOLERANCE


def test_select_thinker_independent_of_group():
    """Thinker selection (second half of request_id) is independent of group
    selection (full request_id)."""
    cdf_group = [(0, 0.5), (1, 1.0)]
    cdf_thinker = [0.5, 1.0]
    rids = _make_request_ids(1000)
    group_results = [_select_group(rid, cdf_group) for rid in rids]
    thinker_results = [_select_thinker(rid, cdf_thinker) for rid in rids]
    # They should not be perfectly correlated
    assert group_results != thinker_results


# ── OmniFlexTask._get_group_idx ──────────────────────────────────────


def test_get_group_idx_single_group():
    """Single group always returns 0."""
    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
    )
    for _ in range(100):
        assert task._get_group_idx(uuid.uuid4().hex, 1) == 0


def test_get_group_idx_distribution():
    """Multi-group routing follows weights."""
    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[
            OmniFlexGroupConfig(
                offloaded_modalities=["img"],
                eric_max_batch_sizes={"img": 4},
                thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)],
            ),
            OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=1, max_num_seqs=64)]),
        ],
        type_routing_weights={1: {0: 0.6, 1: 0.4}},
    )
    counts = Counter(task._get_group_idx(uuid.uuid4().hex, 1) for _ in range(N_SAMPLES))
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.6) < TOLERANCE
    assert abs(counts[1] / total - 0.4) < TOLERANCE


# ── OmniFlexTask._get_thinker_idx ────────────────────────────────────


def test_get_thinker_idx_single():
    """Single thinker always returns 0."""
    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
    )
    for _ in range(100):
        assert task._get_thinker_idx(uuid.uuid4().hex, 0) == 0


def test_get_thinker_idx_distribution():
    """Multi-thinker routing follows weights within group."""
    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[
            OmniFlexGroupConfig(
                thinkers=[
                    ThinkerLLMConfig(tp_size=1, max_num_seqs=8, weight=0.3),
                    ThinkerLLMConfig(tp_size=2, max_num_seqs=16, weight=0.7),
                ]
            )
        ],
    )
    counts = Counter(task._get_thinker_idx(uuid.uuid4().hex, 0) for _ in range(N_SAMPLES))
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.3) < TOLERANCE
    assert abs(counts[1] / total - 0.7) < TOLERANCE


# ── Subtask discovery ─────────────────────────────────────────────────


def test_subtask_discovery_mono():
    """Mono task: 1 LLM text + 1 LLM emb + 1 TV = 3 unit tasks."""
    from cornserve.task.base import discover_unit_tasks

    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
    )
    unit_tasks = discover_unit_tasks([task])
    # LLMUnitTask + LLMEmbeddingUnitTask + OmniTalkerVocoderTask
    assert len(unit_tasks) == 3


def test_subtask_discovery_multi_group():
    """2 groups (1 thinker each) + TV = 5 unit tasks."""
    from cornserve.task.base import discover_unit_tasks

    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[
            OmniFlexGroupConfig(
                offloaded_modalities=["img"],
                eric_max_batch_sizes={"img": 4},
                thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)],
            ),
            OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=1, max_num_seqs=64)]),
        ],
        type_routing_weights={1: {0: 0.6, 1: 0.4}},
    )
    unit_tasks = discover_unit_tasks([task])
    # Group0: 1 EncoderTask + 1 LLMUnitTask + 1 LLMEmbeddingUnitTask = 3
    # Group1: 1 LLMUnitTask + 1 LLMEmbeddingUnitTask = 2
    # Shared: 1 OmniTalkerVocoderTask = 1
    # Total = 6
    assert len(unit_tasks) == 6


def test_subtask_discovery_vocoder_fission():
    """With vocoder_fission: TalkerEmbedding + AudioGenerator instead of TV."""
    from cornserve.task.base import discover_unit_tasks

    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[OmniFlexGroupConfig(thinkers=[ThinkerLLMConfig(tp_size=2, max_num_seqs=32)])],
        vocoder_fission=True,
    )
    unit_tasks = discover_unit_tasks([task])
    # LLMUnitTask + LLMEmbeddingUnitTask + OmniTalkerEmbeddingTask + AudioGeneratorTask
    assert len(unit_tasks) == 4


def test_subtask_discovery_multi_thinker():
    """2 thinkers in 1 group: 2 LLM pairs + TV = 5."""
    from cornserve.task.base import discover_unit_tasks

    task = OmniFlexTask(
        model_id=MODEL_ID,
        groups=[
            OmniFlexGroupConfig(
                thinkers=[
                    ThinkerLLMConfig(tp_size=1, max_num_seqs=8, weight=0.4),
                    ThinkerLLMConfig(tp_size=2, max_num_seqs=16, weight=0.6),
                ]
            )
        ],
    )
    unit_tasks = discover_unit_tasks([task])
    # 2 * (LLMUnitTask + LLMEmbeddingUnitTask) + OmniTalkerVocoderTask
    assert len(unit_tasks) == 5
