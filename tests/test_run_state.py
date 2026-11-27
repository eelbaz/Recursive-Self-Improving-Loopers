"""Given/When/Then tests for recoverable and attributable training runs."""

import json
from pathlib import Path

import pytest

from rse_looper.run_state import RunLedger, checkpoint_fingerprint, load_math_inputs


@pytest.fixture
def model_dir(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "test"}')
    (model / "model.safetensors").write_bytes(b"original checkpoint bytes")
    return model


def _gate(correct=6):
    return {
        "R3": {"correct": correct, "completed": 9, "n": 10},
        "R4": {"correct": correct, "completed": 9, "n": 10},
        "RK2": {"correct": correct, "completed": 9, "n": 10},
        "total_correct": correct * 3,
        "total_completed": 27,
        "max_rep": 0.0,
    }


def _ledger(tmp_path, model_dir, **kwargs):
    return RunLedger(
        tmp_path / "run", config={"max_gens": 4},
        data_fingerprints={"train": "train-sha", "selection": "gate-sha"},
        initial_checkpoint=checkpoint_fingerprint(model_dir), **kwargs,
    )


def test_given_committed_generation_when_resuming_then_lineage_and_gate_are_restored(tmp_path, model_dir):
    """Given an accepted checkpoint; when resumed; then its exact lineage survives."""
    successor = tmp_path / "successor"
    successor.mkdir()
    (successor / "model.safetensors").write_bytes(b"successor checkpoint bytes")
    delta = tmp_path / "delta.pt"
    delta.write_bytes(b"delta bytes")
    with _ledger(tmp_path, model_dir) as run:
        run.initialize_gate(_gate())
        run.begin_generation()
        run.record_candidate(1.0, _gate(7), accepted=True, reason="accuracy_improved")
        run.commit_generation(checkpoint=successor, selected_scale=1.0, delta_path=delta)

    with _ledger(tmp_path, model_dir, resume=True) as run:
        assert run.state["next_generation"] == 1
        assert run.state["accepted_generations"] == 1
        assert run.state["current_gate"] == _gate(7)
        assert run.state["current_checkpoint"]["path"] == str(successor.resolve())
        committed = [e for e in run.state["events"] if e["event"] == "generation_committed"][-1]
        assert committed["parent_checkpoint"]["sha256"] == checkpoint_fingerprint(model_dir)["sha256"]
        assert committed["output_checkpoint"]["sha256"] == checkpoint_fingerprint(successor)["sha256"]
        assert len(committed["delta_sha256"]) == 64


def test_given_interrupted_export_when_resuming_then_uncommitted_candidate_is_not_promoted(tmp_path, model_dir):
    """Given an interrupted accepted candidate; when resumed; then the parent remains current."""
    with pytest.raises(RuntimeError, match="export failed"):
        with _ledger(tmp_path, model_dir) as run:
            run.initialize_gate(_gate())
            first_attempt = run.begin_generation()
            run.record_candidate(1.0, _gate(7), accepted=True, reason="accuracy_improved")
            raise RuntimeError("export failed")
    # A stale derived JSONL must be repaired from the atomic authoritative manifest.
    (tmp_path / "run" / "events.jsonl").write_text("partial write")
    with _ledger(tmp_path, model_dir, resume=True) as run:
        assert run.state["next_generation"] == 0
        assert run.state["accepted_generations"] == 0
        assert run.state["current_gate"] == _gate()
        assert run.begin_generation() != first_attempt
        events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
        assert events == run.state["events"]


@pytest.mark.parametrize("mismatch", ["config", "data", "checkpoint"])
def test_given_existing_run_when_resume_inputs_change_then_resume_is_rejected(tmp_path, model_dir, mismatch):
    """Given durable state; when its inputs change; then resume refuses incompatible history."""
    with _ledger(tmp_path, model_dir) as run:
        run.initialize_gate(_gate())
    config = {"max_gens": 5 if mismatch == "config" else 4}
    data = {"train": "changed" if mismatch == "data" else "train-sha", "selection": "gate-sha"}
    if mismatch == "checkpoint":
        (model_dir / "model.safetensors").write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="mismatch|changed"):
        with RunLedger(tmp_path / "run", config=config, data_fingerprints=data,
                       initial_checkpoint=checkpoint_fingerprint(model_dir), resume=True):
            pass


def test_given_existing_run_when_resume_is_not_requested_then_history_is_not_overwritten(tmp_path, model_dir):
    """Given a previous run; when started fresh at its path; then its history is protected."""
    with _ledger(tmp_path, model_dir):
        pass
    before = (tmp_path / "run" / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="resume"):
        with _ledger(tmp_path, model_dir):
            pass
    assert (tmp_path / "run" / "run_manifest.json").read_bytes() == before


def test_given_rejected_candidate_when_committing_as_winner_then_lineage_is_not_advanced(tmp_path, model_dir):
    """Given a rejected candidate; when promoted accidentally; then the ledger blocks it."""
    with _ledger(tmp_path, model_dir) as run:
        run.initialize_gate(_gate())
        run.begin_generation()
        run.record_candidate(1.0, _gate(5), accepted=False, reason="accuracy_regression")
        with pytest.raises(ValueError, match="accepted"):
            run.commit_generation(checkpoint=model_dir, selected_scale=1.0)
        assert run.state["next_generation"] == 0


def test_given_explicit_disjoint_inputs_when_loading_then_exact_source_and_selected_ids_are_recorded(tmp_path):
    """Given local datasets; when loaded; then hashes and the precise selection subset are retained."""
    train, gate = tmp_path / "train.jsonl", tmp_path / "gate.jsonl"
    train.write_text(json.dumps({"id": "t1", "problem": "one plus one", "answer": "2"}) + "\n")
    gate.write_text("\n".join(json.dumps({"id": f"g{i}", "problem": f"gate problem {i}", "answer": str(i)}) for i in range(3)) + "\n")
    inputs = load_math_inputs(train, gate, gate_size=2)
    assert [r["id"] for r in inputs.gate_rows] == ["g0", "g1"]
    assert inputs.fingerprints["selection"]["selected_ids"] == ["g0", "g1"]
    assert len(inputs.fingerprints["train"]["sha256"]) == 64
    assert inputs.fingerprints["selection"]["row_count"] == 3


def test_given_overlap_outside_selected_gate_when_loading_then_full_source_overlap_is_rejected(tmp_path):
    """Given overlap beyond the gate slice; when validating; then future leakage is still rejected."""
    train, gate = tmp_path / "train.jsonl", tmp_path / "gate.jsonl"
    shared = {"id": "t1", "problem": "one plus one", "answer": "2"}
    train.write_text(json.dumps(shared) + "\n")
    gate.write_text(json.dumps({"id": "g1", "problem": "two plus two", "answer": "4"}) + "\n" + json.dumps(shared) + "\n")
    with pytest.raises(ValueError, match="overlap|disjoint"):
        load_math_inputs(train, gate, gate_size=1)


def test_given_regressing_gate_when_caller_claims_acceptance_then_ledger_recomputes_and_rejects(tmp_path, model_dir):
    """Given a falsely accepted regression; when recorded; then the rule cannot be bypassed."""
    with _ledger(tmp_path, model_dir) as run:
        run.initialize_gate(_gate())
        run.begin_generation()
        with pytest.raises(ValueError, match="decision|acceptance"):
            run.record_candidate(1.0, _gate(5), accepted=True, reason="accuracy_improved")
        assert not run.state["pending_generation"]["candidates"]


def test_given_different_implementation_when_resuming_then_recorded_source_identity_is_checked(tmp_path, model_dir, monkeypatch):
    """Given a run from different code; when resumed; then a source change cannot go unnoticed."""
    import rse_looper.run_state as state
    with _ledger(tmp_path, model_dir):
        pass
    monkeypatch.setattr(state, "execution_fingerprint", lambda: {"source_sha256": "changed", "runtime": {}})
    with pytest.raises(ValueError, match="implementation|runtime"):
        with _ledger(tmp_path, model_dir, resume=True):
            pass


def test_given_incomparable_gate_sizes_or_completion_totals_when_deciding_then_evidence_is_rejected():
    """Given altered denominators or invented completions; when deciding; then reject invalid comparisons."""
    from rse_looper.run_state import assess_gate_candidate
    previous = _gate()
    incomparable = _gate(7)
    incomparable["R3"]["n"] = 100
    with pytest.raises(ValueError, match="size|count|denominator"):
        assess_gate_candidate(previous, incomparable)
    invented = _gate()
    invented["total_completed"] = 1000
    with pytest.raises(ValueError, match="completed|completion"):
        assess_gate_candidate(previous, invented)


def test_given_model_loading_runtime_when_fingerprinted_then_accelerate_is_part_of_compatibility():
    """Given device-mapped model loading; when recording runtime; then its accelerate dependency is bound."""
    from rse_looper.run_state import execution_fingerprint
    assert "accelerate" in execution_fingerprint()["runtime"]["packages"]
