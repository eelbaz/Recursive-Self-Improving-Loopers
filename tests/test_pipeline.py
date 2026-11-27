"""Unit tests for the production autonomous pipeline CLI (rse_looper.pipeline)."""

import math
import pytest
import torch

from rse_looper.pipeline import PipelineConfig, main, merge_generation_deltas, parse_config


def test_parse_config_and_validation():
    cfg, dry_run = parse_config(["--stage", "fpsolve", "--gpus", "0,1", "--dry-run"])
    assert cfg.stage == "fpsolve"
    assert cfg.gpus == "0,1"
    assert cfg.resume is True
    assert dry_run is True

    with pytest.raises(ValueError):
        PipelineConfig(stage="invalid").validate()
    with pytest.raises(ValueError):
        PipelineConfig(fpsolve_max_rel=0.9).validate()
    with pytest.raises(ValueError):
        PipelineConfig(gspo_max_gens=0).validate()


def test_merge_generation_deltas_weighted_average_and_union_keys():
    d1 = {"a": torch.tensor([2.0, 4.0]), "b": torch.tensor([1.0])}
    d2 = {"a": torch.tensor([6.0, 8.0]), "c": torch.tensor([3.0])}
    m = merge_generation_deltas([d1, d2], weights=[1.0, 1.0], base_fingerprints=["0" * 64] * 2)
    assert torch.allclose(m["a"], torch.tensor([4.0, 6.0]))
    assert math.isclose(m["b"].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(m["c"].item(), 1.5, rel_tol=1e-6)

    with pytest.raises(ValueError):
        merge_generation_deltas([])
    with pytest.raises(ValueError):
        merge_generation_deltas([d1, d2], weights=[1.0])


def test_pipeline_dry_run_executes_preflight_cleanly():
    res = main(["--stage", "both", "--gpus", "0,1,2,3,4,5", "--dry-run"])
    assert res["dry_run"] is True
    assert res["stage"] == "both"
    assert res["gpus"] == [0, 1, 2, 3, 4, 5]
    assert "causal_conv1d_installed" in res["preflight"]


def test_stage_modules_expose_runnable_cli_parsers():
    from rse_looper.fpsolve import build_parser as fp_parser
    from rse_looper.gspo import main as gspo_main

    fp_args = fp_parser().parse_args(["--lam-keep", "4.0", "--max-rel", "0.25"])
    assert fp_args.lam_keep == 4.0 and fp_args.max_rel == 0.25

    gspo_res = gspo_main(["--gpus", "0,1", "--dry-run"])
    assert gspo_res["gpus"] == [0, 1]


def test_given_data_paths_when_running_pipeline_then_selection_inputs_reach_the_training_stage(tmp_path, monkeypatch):
    """Given explicit data; when invoking the pipeline; then the child receives the same paths."""
    import json
    import subprocess
    import rse_looper.pipeline as pipeline
    train, gate = tmp_path / "train.jsonl", tmp_path / "gate.jsonl"
    train.write_text(json.dumps({"id": "t", "problem": "one plus one", "answer": "2"}) + "\n")
    gate.write_text(json.dumps({"id": "g", "problem": "two plus two", "answer": "4"}) + "\n")
    commands = []
    def fake_training(command, **kwargs):
        commands.append(command)
        from pathlib import Path
        child_dir = Path(command[command.index("--work-dir") + 1])
        child_dir.mkdir(parents=True)
        (child_dir / "run_manifest.json").write_text(json.dumps({
            "status": "completed", "current_checkpoint": {"path": "accepted-model"},
            "current_gate": {}, "accepted_generations": 1,
        }))
    monkeypatch.setattr(subprocess, "run", fake_training)
    monkeypatch.setattr(pipeline, "preflight_gpu_check", lambda **kw: {"cuda_available": True})
    result = main(["--stage", "gspo", "--train-jsonl", str(train), "--gate-jsonl", str(gate),
                   "--gate-size", "1", "--work-dir", str(tmp_path / "run")])
    command = commands[0]
    assert command[command.index("--train-jsonl") + 1] == str(train)
    assert command[command.index("--gate-jsonl") + 1] == str(gate)
    assert command[command.index("--gate-size") + 1] == "1"
    assert result["stage"] == "gspo"


def test_given_successful_cli_when_invoked_then_json_is_printed_and_exit_status_is_zero(monkeypatch, capsys):
    """Given a successful pipeline; when used as a console script; then its dict is not an error exit."""
    import json
    import rse_looper.pipeline as pipeline
    monkeypatch.setattr(pipeline, "main", lambda argv=None: {"status": "completed"})
    assert pipeline.cli([]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}


def test_given_deltas_without_common_base_when_merging_then_ambiguous_lineage_is_rejected():
    """Given unknown or different delta origins; when merging; then unsafe averaging is blocked."""
    d = {"weight": torch.ones(2)}
    with pytest.raises(ValueError, match="base"):
        merge_generation_deltas([d, d])
    with pytest.raises(ValueError, match="base"):
        merge_generation_deltas([d, d], base_fingerprints=["0" * 64, "1" * 64])


@pytest.mark.parametrize("weights", [[-1.0, 2.0], [float("nan"), 1.0]])
def test_given_invalid_merge_weights_when_merging_then_nonconvex_or_nonfinite_results_are_rejected(weights):
    """Given invalid merge weights; when combining deltas; then corruption is rejected."""
    with pytest.raises(ValueError, match="weights"):
        merge_generation_deltas([{"w": torch.ones(2)}] * 2, weights=weights, base_fingerprints=["0" * 64] * 2)


def test_given_completed_stage_one_when_resuming_then_verified_output_is_reused(tmp_path, monkeypatch):
    """Given a verified first stage; when resuming stage two; then the initial checkpoint is not rewritten."""
    import subprocess
    from pathlib import Path
    import rse_looper.checkpoint as checkpoint
    import rse_looper.pipeline as pipeline
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base weights")
    output = tmp_path / "stage1"
    calls = []
    def fake_fpsolve(command, **kwargs):
        calls.append(command)
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"edited weights")
    monkeypatch.setattr(subprocess, "run", fake_fpsolve)
    monkeypatch.setattr("platform.platform", lambda: "test-platform")
    monkeypatch.setattr(checkpoint, "resolve_checkpoint_dir", lambda *args, **kwargs: base)
    cfg = PipelineConfig(stage="fpsolve", base_model=str(base), work_dir=str(tmp_path / "run"), gpus="2")
    assert Path(pipeline.run_fpsolve_stage(cfg, str(output))) == output
    assert Path(pipeline.run_fpsolve_stage(cfg, str(output))) == output
    assert len(calls) == 1
    assert calls[0][calls[0].index("--device") + 1] == "cuda:2"
    (base / "model.safetensors").write_bytes(b"changed base")
    with pytest.raises(ValueError, match="mismatch|changed"):
        pipeline.run_fpsolve_stage(cfg, str(output))


def test_given_cpu_only_runtime_when_executing_fpsolve_then_reject_before_model_resolution(tmp_path, monkeypatch):
    """Given no CUDA; when executing a CUDA stage; then fail before downloads or model work."""
    import rse_looper.checkpoint as checkpoint
    import rse_looper.pipeline as pipeline
    monkeypatch.setattr(pipeline, "enable_fast_math", lambda: False)
    monkeypatch.setattr(pipeline, "preflight_gpu_check", lambda **kwargs: {"cuda_available": False})
    monkeypatch.setattr(checkpoint, "resolve_checkpoint_dir", lambda *args, **kwargs: pytest.fail("Model resolution preceded CUDA validation"))
    with pytest.raises(RuntimeError, match="CUDA"):
        pipeline.main(["--stage", "fpsolve", "--work-dir", str(tmp_path / "run")])


def test_given_mutable_calibration_source_when_child_runs_then_consumed_snapshot_matches_recorded_hash(tmp_path, monkeypatch):
    """Given a source changed at launch; when calibrating; then the child consumes the captured bytes."""
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    import rse_looper.checkpoint as checkpoint
    import rse_looper.pipeline as pipeline
    base, output = tmp_path / "base", tmp_path / "edited"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    source = tmp_path / "prompts.jsonl"
    original = b'{"prompt":"original prompt"}\n'
    source.write_bytes(original)
    consumed, child_paths = [], []
    def fake_child(command, **kwargs):
        source.write_bytes(b'{"prompt":"changed after validation"}\n')
        child_path = Path(command[command.index("--prompts-jsonl") + 1])
        child_paths.append(child_path)
        consumed.append(child_path.read_bytes())
        output.mkdir()
        (output / "model.safetensors").write_bytes(b"edited")
    monkeypatch.setattr(subprocess, "run", fake_child)
    monkeypatch.setattr("platform.platform", lambda: "test-platform")
    monkeypatch.setattr(checkpoint, "resolve_checkpoint_dir", lambda *args, **kwargs: base)
    cfg = PipelineConfig(stage="fpsolve", base_model=str(base), prompts_jsonl=str(source),
                         work_dir=str(tmp_path / "run"), gpus="0")
    pipeline.run_fpsolve_stage(cfg, str(output))
    assert consumed == [original]
    assert child_paths[0] != source
    assert child_paths[0].parent == tmp_path / "run" / "fpsolve"
    manifest = json.loads((tmp_path / "run" / "fpsolve" / "stage1_manifest.json").read_text())
    assert manifest["request"]["calibration"]["sha256"] == hashlib.sha256(consumed[0]).hexdigest()
    assert manifest["request"]["calibration"]["selected_row_indices"] == [0]
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.run_fpsolve_stage(cfg, str(output))
    source.write_bytes(original)
    assert pipeline.run_fpsolve_stage(cfg, str(output)) == str(output)
    assert len(consumed) == 1


def test_given_conflicting_snapshot_path_when_preparing_calibration_then_existing_bytes_are_not_overwritten(tmp_path, monkeypatch):
    """Given unrelated bytes at the reserved snapshot name; when preparing; then preserve and reject them."""
    import hashlib
    import subprocess
    import rse_looper.checkpoint as checkpoint
    import rse_looper.pipeline as pipeline
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    source = tmp_path / "prompts.jsonl"
    original = b'{"prompt":"original prompt"}\n'
    source.write_bytes(original)
    work = tmp_path / "run" / "fpsolve"
    work.mkdir(parents=True)
    snapshot = work / f"calibration-{hashlib.sha256(original).hexdigest()}.jsonl"
    snapshot.write_bytes(b"unrelated bytes")
    monkeypatch.setattr(checkpoint, "resolve_checkpoint_dir", lambda *args, **kwargs: base)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Conflicting snapshot reached child"))
    monkeypatch.setattr("platform.platform", lambda: "test-platform")
    cfg = PipelineConfig(stage="fpsolve", base_model=str(base), prompts_jsonl=str(source), work_dir=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="snapshot"):
        pipeline.run_fpsolve_stage(cfg, str(tmp_path / "edited"))
    assert snapshot.read_bytes() == b"unrelated bytes"
