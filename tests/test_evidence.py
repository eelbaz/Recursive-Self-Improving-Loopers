"""Given/When/Then contracts for evidence, provenance, and historical replay."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from rse_looper import evidence


RESULTS = Path(__file__).resolve().parents[1] / "results"


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def dataset(tmp_path, role="test", prefix="q"):
    rows = [{"id": f"{prefix}{i}", "problem": f"{prefix} problem {i}", "answer": str(i)} for i in range(4)]
    path = write_rows(tmp_path / f"{prefix}.jsonl", rows)
    manifest = evidence.build_data_manifest(path, dataset="synthetic", revision="v1", split=role, role=role)
    return path, rows, manifest


def provenance(manifest, run_id):
    return {
        "schema_version": 1,
        "kind": "evaluation-provenance",
        "run_id": run_id,
        "dataset_sha256": manifest["sha256"],
        "checkpoint": {"model_id": "synthetic", "revision": "a" * 40, "sha256": "b" * 64},
        "code_revision": "c" * 40,
        "grader": {"name": "synthetic-checker", "revision": "v1"},
        "decoding": {"strategy": "greedy", "max_new_tokens": 64, "seed": 42, "loop_passes": 3},
        "selection": {"used_for_selection": False, "frozen_before_run": True},
    }


def gate(r3, r4, rk2, completed=12):
    return {
        "R3": {"correct": r3}, "R4": {"correct": r4}, "RK2": {"correct": rk2},
        "total_correct": r3 + r4 + rk2, "total_completed": completed, "max_rep": 0.0,
    }


def test_given_legacy_results_when_replayed_then_expose_known_acceptance_contradictions():
    # Given: original aggregates, left unchanged by this repair.
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in RESULTS.glob("*.json")}
    # When: replay declared parent-to-winner transitions.
    report = evidence.audit_legacy_results(RESULTS)
    # Then: five accepted entries include a 118 -> 114 regression and excessive mode drops.
    assert report["status"] == "contradictions_found"
    assert report["gate_replay"]["declared_acceptances"] == 5
    regressions = [f for f in report["findings"] if f["code"] == "accepted_total_regression"]
    assert any(f["generation"] == 3 and f["previous"] == 118 and f["candidate"] == 114 for f in regressions)
    assert any(f["code"] == "accepted_mode_drop" and f["generation"] == 1 for f in report["findings"])
    assert any(f["code"] == "missing_item_traces" for f in report["findings"])
    assert any(f["code"] == "missing_provenance" for f in report["findings"])
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in RESULTS.glob("*.json")}


def test_given_clean_synthetic_history_when_replayed_then_accept_only_valid_transitions():
    # Given: one strict gain and one correctly rejected regression.
    first = {"scale": 1.0, **gate(2, 2, 2)}
    worse = {"scale": 1.0, **gate(1, 1, 1)}
    history = {"gate_size_per_mode": 4, "init_gate": gate(1, 1, 1), "generations": [
        {"gen": 0, "candidates": [first], "accepted": True, "winner_scale": 1.0},
        {"gen": 1, "candidates": [worse], "accepted": False},
    ]}
    # When: replay the complete history.
    report = evidence.replay_gate_history(history)
    # Then: a clean trace passes; checks are not hard-coded to fail every input.
    assert report["status"] == "consistent"
    assert report["findings"] == []
    assert report["declared_acceptances"] == 1
    assert report["final_total_correct"] == 6


def test_given_hashed_dataset_when_validated_then_preserve_exact_identity(tmp_path):
    # Given: a frozen JSONL data source and a manifest.
    path, rows, manifest = dataset(tmp_path)
    # When: validate it without changing the file.
    verified = evidence.validate_data_manifest(path, manifest, expected_role="test")
    # Then: exact bytes, rows, IDs, and semantic problem identities are recorded.
    assert verified == rows
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["row_count"] == 4
    assert manifest["item_ids"] == ["q0", "q1", "q2", "q3"]
    assert len(manifest["problem_hashes"]) == 4


def test_given_modified_data_when_validated_then_hash_mismatch_is_fatal(tmp_path):
    # Given: a manifest committed before the input changes.
    path, rows, manifest = dataset(tmp_path)
    rows[0]["answer"] = "999"
    write_rows(path, rows)
    # When / Then: changed answers invalidate the exact-byte commitment.
    with pytest.raises(ValueError, match="sha256|hash"):
        evidence.validate_data_manifest(path, manifest)


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_problem", "missing_id", "bad_answer", "duplicate_json_key", "nonfinite_json"])
def test_given_invalid_dataset_when_manifested_then_reject(tmp_path, mutation):
    # Given: malformed identities, duplicate content, or ambiguous JSON.
    rows = [{"id": "a", "problem": "Question A", "answer": "1"}, {"id": "b", "problem": "Question B", "answer": "2"}]
    if mutation == "duplicate_id":
        rows[1]["id"] = "a"
    elif mutation == "duplicate_problem":
        rows[1]["problem"] = "  Question   A  "
    elif mutation == "missing_id":
        del rows[1]["id"]
    elif mutation == "bad_answer":
        rows[1]["answer"] = True
    path = write_rows(tmp_path / "bad.jsonl", rows)
    if mutation == "duplicate_json_key":
        path.write_text('{"id":"a","id":"b","problem":"Q","answer":"1"}\n')
    elif mutation == "nonfinite_json":
        path.write_text('{"id":"a","problem":"Q","answer":"1","x":NaN}\n')
    # When / Then: refuse to create a manifest that launders invalid data.
    with pytest.raises(ValueError):
        evidence.build_data_manifest(path, dataset="synthetic", revision="v1", split="test", role="test")


@pytest.mark.parametrize("same_id", [True, False])
def test_given_train_test_overlap_when_validating_splits_then_reject(tmp_path, same_id):
    # Given: leakage may preserve an ID or rename an identical problem.
    _, rows, train = dataset(tmp_path, role="train", prefix="train")
    leaked = {"id": rows[0]["id"] if same_id else "new-name", "problem": "New text" if same_id else rows[0]["problem"], "answer": "0"}
    path = write_rows(tmp_path / "leaked.jsonl", [leaked])
    test = evidence.build_data_manifest(path, dataset="synthetic", revision="v1", split="test", role="test")
    # When / Then: either ID overlap or canonical text overlap is disallowed.
    with pytest.raises(ValueError, match="overlap"):
        evidence.validate_split_manifests([train, test])


def test_given_disjoint_splits_when_validating_then_accept_without_claiming_external_dedup(tmp_path):
    # Given: separate local training and final-test problems.
    _, _, train = dataset(tmp_path, role="train", prefix="train")
    _, _, test = dataset(tmp_path, role="test", prefix="test")
    # When / Then: exact local disjointness is verifiable.
    assert evidence.validate_split_manifests([train, test]) is None


def test_given_synthetic_paired_runs_when_reported_then_include_reproducible_methods(tmp_path):
    # Given: a frozen test set, declared provenance, and complete boolean outcomes.
    _, rows, manifest = dataset(tmp_path)
    base = [{"id": row["id"], "correct": False} for row in rows]
    edit = [{"id": row["id"], "correct": i < 2} for i, row in enumerate(rows)]
    # When: compare the same samples with the same recorded protocol.
    result = evidence.compare_evaluation_runs(base, edit, provenance(manifest, "base"), provenance(manifest, "edit"), manifest, seed=7)
    # Then: exact probability and seeded interval are accompanied by explicit limits.
    assert result["comparison"]["p_value"] == 0.25
    assert result["comparison"]["n"] == 4
    assert result["status"] == "validated_metadata_and_pairing"
    assert result["limitations"]
    assert result["comparison"]["bootstrap_seed"] == 7


@pytest.mark.parametrize("fault", ["selection_reuse", "unfrozen", "dataset_hash", "missing_checkpoint", "decoder_mismatch", "missing_item"])
def test_given_unverifiable_or_unmatched_run_when_reported_then_fail_closed(tmp_path, fault):
    # Given: valid runs apart from one concrete provenance or coverage failure.
    _, rows, manifest = dataset(tmp_path)
    base = [{"id": row["id"], "correct": False} for row in rows]
    edit = [{"id": row["id"], "correct": True} for row in rows]
    bp, ep = provenance(manifest, "base"), provenance(manifest, "edit")
    if fault == "selection_reuse":
        ep["selection"]["used_for_selection"] = True
    elif fault == "unfrozen":
        ep["selection"]["frozen_before_run"] = False
    elif fault == "dataset_hash":
        ep["dataset_sha256"] = "d" * 64
    elif fault == "missing_checkpoint":
        del ep["checkpoint"]
    elif fault == "decoder_mismatch":
        ep["decoding"]["max_new_tokens"] = 128
    elif fault == "missing_item":
        edit.pop()
    # When / Then: no significance number is emitted for the invalid experiment.
    with pytest.raises(ValueError):
        evidence.compare_evaluation_runs(base, edit, bp, ep, manifest)


def test_given_legacy_cli_when_called_then_return_machine_readable_failure(capsys):
    # Given / When: an audit of known contradictory legacy records.
    code = evidence.cli(["legacy", "--results-dir", str(RESULTS)])
    report = json.loads(capsys.readouterr().out)
    # Then: CI can distinguish detected contradictions from successful validation.
    assert code == 1 and report["status"] == "contradictions_found"


def test_given_json_exponent_overflow_when_read_then_reject_nonfinite_number(tmp_path):
    # Given: valid JSON syntax whose exponent overflows a machine float.
    path = tmp_path / "overflow.jsonl"
    path.write_text('{"id":"a","correct":true,"elapsed":1e9999}\n')
    # When / Then: standard JSON syntax must not smuggle infinity into evidence.
    with pytest.raises(ValueError, match="finite"):
        evidence.read_jsonl(path)


def test_given_data_changes_during_validation_when_returned_then_use_the_verified_snapshot(tmp_path, monkeypatch):
    # Given: a frozen source, changed immediately after its verified bytes are read.
    path, rows, manifest = dataset(tmp_path)
    altered = copy.deepcopy(rows)
    altered[0]["answer"] = "changed after verification"
    original_read_bytes = Path.read_bytes
    def mutate_after_read(self):
        value = original_read_bytes(self)
        if self == path:
            write_rows(path, altered)
        return value
    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    # When: source validation reads the committed snapshot.
    returned = evidence.validate_data_manifest(path, manifest)
    # Then: callers receive the validated rows, never a second unverified read.
    assert returned == rows


@pytest.mark.parametrize("change_after_read", [False, True])
def test_given_compare_cli_when_artifacts_match_then_bind_report_to_input_bytes(tmp_path, capsys, monkeypatch, change_after_read):
    # Given: complete synthetic data, outcomes and declared provenance files.
    data_path, rows, manifest = dataset(tmp_path)
    files = {
        "manifest": manifest,
        "base-provenance": provenance(manifest, "base"),
        "edit-provenance": provenance(manifest, "edit"),
    }
    args = ["compare", "--data", str(data_path)]
    for name, value in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        args.extend([f"--{name}", str(path)])
    originals = {}
    for name, correct in (("base", False), ("edit", True)):
        path = write_rows(tmp_path / f"{name}.jsonl", [{"id": row["id"], "correct": correct} for row in rows])
        originals[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        args.extend([f"--{name}", str(path)])
    if change_after_read:
        read_bytes = Path.read_bytes
        def mutate_after_read(self):
            raw = read_bytes(self)
            if self in originals:
                self.write_bytes(raw + b"\n")
            return raw
        monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    # When: invoke the real lightweight CLI rather than a mocked report writer.
    assert evidence.cli(args) == 0
    report = json.loads(capsys.readouterr().out)
    # Then: its hashes refer to exactly the supplied evaluation files.
    assert report["comparison"]["p_value"] == 1 / 16
    assert report["base_jsonl_sha256"] == originals[tmp_path / "base.jsonl"]
    assert report["edit_jsonl_sha256"] == originals[tmp_path / "edit.jsonl"]
    assert report["provided_other_split_manifests"] == 0
