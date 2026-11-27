"""Run loop editing and labeled self-training with validated inputs and provenance.

Stage 1 uses a gradient-free local ridge solve. Stage 2 uses a fixed selection
gate and returns its last accepted checkpoint. Runtime and benchmark improvement
must be measured for each run; this command does not automatically merge models.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence
import torch

from rse_looper.preflight import enable_fast_math, preflight_gpu_check


@dataclass(frozen=True)
class PipelineConfig:
    stage: str = "both"  # "fpsolve" | "gspo" | "both"
    base_model: str = "thinkingmachines/Inkling-2-Mini"
    work_dir: str = "./runs/rse_looper"
    out_model: str = "./models/Inkblot-2-mini"
    gpus: str = "0,1,2,3,4,5"
    model_revision: str | None = None
    prompts_jsonl: str | None = None
    train_jsonl: str | None = None
    gate_jsonl: str | None = None
    gate_size: int = 64
    max_new_tokens: int = 1536
    gate_max_new_tokens: int = 1536
    # Stage 1 (FP-Solve, 0 backprop, 0 labels)
    fpsolve_lam_keep: float = 4.0
    fpsolve_ridge_frac: float = 0.06
    fpsolve_max_rel: float = 0.25
    fpsolve_num_prompts: int = 48
    # Stage 2 (Loop-Advantage GSPO + DAPO, 0 external teacher)
    gspo_max_gens: int = 10
    gspo_patience: int = 3
    gspo_problems_per_gen: int = 96
    gspo_steps_per_gen: int = 24
    gspo_lr: float = 2.0e-5
    require_causal_conv1d: bool = False
    resume: bool = True

    def validate(self) -> None:
        if self.stage not in ("fpsolve", "gspo", "both"):
            raise ValueError(f"Unsupported stage: {self.stage!r}")
        if self.fpsolve_lam_keep < 0.0 or not (0.0 < self.fpsolve_max_rel <= 0.5):
            raise ValueError("Invalid FP-Solve regularization or relative norm cap")
        if self.gspo_max_gens < 1 or self.gspo_patience < 1:
            raise ValueError("gspo_max_gens and gspo_patience must be >= 1")
        gpu_list = [int(x.strip()) for x in self.gpus.split(",") if x.strip()]
        if not gpu_list or len(set(gpu_list)) != len(gpu_list) or any(gpu < 0 for gpu in gpu_list):
            raise ValueError("Distinct nonnegative GPU ids must be provided")
        if any(value < 1 for value in (self.gate_size, self.max_new_tokens, self.gate_max_new_tokens,
                                        self.gspo_problems_per_gen, self.gspo_steps_per_gen)):
            raise ValueError("Dataset sizes, generation limits, and training steps must be positive")
        if not math.isfinite(self.gspo_lr) or self.gspo_lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.fpsolve_lam_keep) or not math.isfinite(self.fpsolve_ridge_frac) or self.fpsolve_ridge_frac <= 0:
            raise ValueError("FP-Solve regularization must be finite and ridge fraction positive")


def merge_generation_deltas(
    delta_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    *, base_fingerprints: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Average deltas declared relative to one exact base; callers must gate the result.

Sequential parent-relative deltas are not interchangeable with common-base
checkpoint deltas. This numerical utility is not used by the production pipeline.
"""
    if not delta_dicts:
        raise ValueError("delta_dicts must be non-empty")
    if (base_fingerprints is None or len(base_fingerprints) != len(delta_dicts)
            or len(set(base_fingerprints)) != 1
            or any(not isinstance(value, str) or len(value) != 64
                   or any(character not in "0123456789abcdef" for character in value) for value in base_fingerprints)):
        raise ValueError("Deltas require matching SHA-256 common-base fingerprints.")
    if weights is None:
        w_norm = [1.0 / len(delta_dicts)] * len(delta_dicts)
    else:
        if len(weights) != len(delta_dicts):
            raise ValueError("weights and delta_dicts must match in length")
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError("Merge weights must be finite and nonnegative")
        w_sum = sum(float(w) for w in weights)
        if w_sum <= 0.0:
            raise ValueError("weights must sum to a positive value")
        w_norm = [float(w) / w_sum for w in weights]

    keys = sorted({k for d in delta_dicts for k in d.keys()})
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc: torch.Tensor | None = None
        for d, w in zip(delta_dicts, w_norm):
            if k in d:
                tensor = d[k].detach().float().cpu()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite delta for {k}")
                if acc is not None and tensor.shape != acc.shape:
                    raise ValueError(f"Delta shape mismatch for {k}")
                term = tensor * w
                acc = term if acc is None else acc + term
        if acc is not None:
            merged[k] = acc
    return merged


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rse-looper",
        description="Run autonomous FP-Solve + Loop-Advantage GSPO self-improvement pipeline.",
    )
    ap.add_argument("--stage", choices=("fpsolve", "gspo", "both"), default="both")
    ap.add_argument("--base-model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--work-dir", default="./runs/rse_looper")
    ap.add_argument("--out-model", default="./models/Inkblot-2-mini")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--model-revision")
    ap.add_argument("--prompts-jsonl", help="Optional explicit calibration prompts for FP-Solve.")
    ap.add_argument("--train-jsonl", help="Labeled training data; required for GSPO.")
    ap.add_argument("--gate-jsonl", help="Disjoint selection data; required for GSPO.")
    ap.add_argument("--gate-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--gate-max-new-tokens", type=int, default=1536)
    ap.add_argument("--fpsolve-lam-keep", type=float, default=4.0)
    ap.add_argument("--fpsolve-ridge-frac", type=float, default=0.06)
    ap.add_argument("--fpsolve-max-rel", type=float, default=0.25)
    ap.add_argument("--gspo-max-gens", type=int, default=10)
    ap.add_argument("--gspo-patience", type=int, default=3)
    ap.add_argument("--gspo-problems-per-gen", type=int, default=96)
    ap.add_argument("--gspo-steps-per-gen", type=int, default=24)
    ap.add_argument("--gspo-lr", type=float, default=2.0e-5)
    ap.add_argument("--require-causal-conv1d", action="store_true")
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    ap.add_argument("--dry-run", action="store_true", help="Validate config and run preflight only.")
    return ap


def parse_config(argv: Sequence[str] | None = None) -> tuple[PipelineConfig, bool]:
    ns = build_parser().parse_args(argv)
    cfg = PipelineConfig(
        stage=ns.stage,
        base_model=ns.base_model,
        work_dir=ns.work_dir,
        out_model=ns.out_model,
        gpus=ns.gpus,
        model_revision=ns.model_revision,
        prompts_jsonl=ns.prompts_jsonl,
        train_jsonl=ns.train_jsonl,
        gate_jsonl=ns.gate_jsonl,
        gate_size=ns.gate_size,
        max_new_tokens=ns.max_new_tokens,
        gate_max_new_tokens=ns.gate_max_new_tokens,
        fpsolve_lam_keep=ns.fpsolve_lam_keep,
        fpsolve_ridge_frac=ns.fpsolve_ridge_frac,
        fpsolve_max_rel=ns.fpsolve_max_rel,
        gspo_max_gens=ns.gspo_max_gens,
        gspo_patience=ns.gspo_patience,
        gspo_problems_per_gen=ns.gspo_problems_per_gen,
        gspo_steps_per_gen=ns.gspo_steps_per_gen,
        gspo_lr=ns.gspo_lr,
        require_causal_conv1d=ns.require_causal_conv1d,
        resume=ns.resume,
    )
    cfg.validate()
    return cfg, bool(ns.dry_run)


def _store_calibration_snapshot(work, data: bytes, digest: str):
    """Create a read-only snapshot without replacing any existing path.

    The caller holds the stage lock. A temporary file and exclusive hard link
    make publication atomic, including if another process created the name.
    """
    import os
    from pathlib import Path
    import tempfile

    destination = Path(work) / f"calibration-{digest}.jsonl"
    if not destination.exists() and not destination.is_symlink():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".calibration-", dir=work, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            directory = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink()
    if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != data:
        raise ValueError("Calibration snapshot path contains different bytes or is not a regular snapshot file.")
    return destination


def run_fpsolve_stage(cfg: PipelineConfig, out_model: str) -> str:
    """Reuse stage one only when its inputs, implementation and output still match."""
    import fcntl
    import hashlib
    from pathlib import Path
    import subprocess
    import sys
    from rse_looper.checkpoint import resolve_checkpoint_dir
    from rse_looper.run_state import _atomic_text, checkpoint_fingerprint, execution_fingerprint

    work = Path(cfg.work_dir).resolve() / "fpsolve"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "stage1_manifest.json"
    output = Path(out_model).resolve()
    with (work / ".stage1.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is running FP-Solve in this work directory.") from exc

        calibration: dict = {"kind": "builtin_equations", "recipe_version": 1, "count": 48}
        calibration_bytes = None
        if cfg.prompts_jsonl:
            # Read once: validation, the manifest hash, and the child's input all
            # derive from this same byte sequence, even if the source later changes.
            calibration_bytes = Path(cfg.prompts_jsonl).read_bytes()
            prompts = [json.loads(line) for line in calibration_bytes.decode("utf-8").splitlines() if line.strip()]
            if not prompts or any(not isinstance(row, dict) or not isinstance(row.get("prompt"), str)
                                  or not row["prompt"].strip() for row in prompts):
                raise ValueError("Calibration JSONL requires nonempty prompt strings.")
            digest = hashlib.sha256(calibration_bytes).hexdigest()
            calibration = {"kind": "local_jsonl", "sha256": digest,
                           "snapshot_file": f"calibration-{digest}.jsonl",
                           "selected_row_indices": list(range(min(48, len(prompts))))}
        base = resolve_checkpoint_dir(cfg.base_model, revision=cfg.model_revision)
        request = {
            "base_checkpoint": checkpoint_fingerprint(base), "calibration": calibration,
            "execution_fingerprint": execution_fingerprint(), "out_model": str(output),
            "lam_keep": cfg.fpsolve_lam_keep, "ridge_frac": cfg.fpsolve_ridge_frac,
            "max_rel": cfg.fpsolve_max_rel, "device": f"cuda:{cfg.gpus.split(',')[0].strip()}",
        }
        if manifest_path.exists():
            if not cfg.resume:
                raise FileExistsError("FP-Solve stage already exists; use --resume or a new work directory.")
            previous = json.loads(manifest_path.read_text())
            if previous.get("schema_version") != 1 or previous.get("kind") != "fpsolve-stage":
                raise ValueError("Unsupported FP-Solve stage manifest.")
            if previous["request"] != request:
                raise ValueError("FP-Solve resume input, configuration or implementation mismatch.")
            if checkpoint_fingerprint(output) != previous["output_checkpoint"]:
                raise ValueError("FP-Solve output checkpoint changed; resume mismatch.")
            if calibration_bytes is not None:
                _store_calibration_snapshot(work, calibration_bytes, calibration["sha256"])
            return str(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("FP-Solve output exists without a matching stage manifest; use a new output path.")
        snapshot = None
        if calibration_bytes is not None:
            snapshot = _store_calibration_snapshot(work, calibration_bytes, calibration["sha256"])
        command = [
            sys.executable, "-m", "rse_looper.fpsolve", "--base-model", str(base),
            "--work-dir", str(work), "--out-model", str(output),
            "--lam-keep", str(cfg.fpsolve_lam_keep), "--ridge-frac", str(cfg.fpsolve_ridge_frac),
            "--max-rel", str(cfg.fpsolve_max_rel), "--device", request["device"],
        ]
        if snapshot is not None:
            command.extend(["--prompts-jsonl", str(snapshot)])
        subprocess.run(command, check=True)
        if snapshot is not None and (snapshot.is_symlink() or snapshot.read_bytes() != calibration_bytes):
            raise ValueError("Calibration snapshot changed during stage execution; output was not committed.")
        manifest = {"schema_version": 1, "kind": "fpsolve-stage", "request": request,
                    "output_checkpoint": checkpoint_fingerprint(output)}
        _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return str(output)


def main(argv: Sequence[str] | None = None) -> dict:
    cfg, dry_run = parse_config(argv)
    from rse_looper.run_state import load_math_inputs
    inputs = None
    if cfg.stage in ("gspo", "both") and (cfg.train_jsonl or cfg.gate_jsonl or not dry_run):
        inputs = load_math_inputs(cfg.train_jsonl, cfg.gate_jsonl, gate_size=cfg.gate_size)
    enable_fast_math()
    preflight = preflight_gpu_check(require_causal_conv1d=cfg.require_causal_conv1d)
    plan = {
        "stage": cfg.stage,
        "base_model": cfg.base_model,
        "out_model": cfg.out_model,
        "gpus": [int(x.strip()) for x in cfg.gpus.split(",") if x.strip()],
        "preflight": preflight,
        "dry_run": dry_run,
        "inputs_validated": inputs is not None,
        "output_policy": "last_accepted_checkpoint",
    }
    if dry_run:
        if cfg.stage in ("gspo", "both") and inputs is None:
            plan["required_inputs"] = ["--train-jsonl", "--gate-jsonl"]
        return plan
    if not preflight["cuda_available"]:
        raise RuntimeError("Pipeline execution requires CUDA GPUs; no model was resolved or trained.")

    import os
    import subprocess
    import sys

    os.makedirs(cfg.work_dir, exist_ok=True)
    cur_model = cfg.base_model

    if cfg.stage in ("fpsolve", "both"):
        fp_out = cfg.out_model if cfg.stage == "fpsolve" else os.path.join(cfg.work_dir, "stage1_fpsolve")
        cur_model = run_fpsolve_stage(cfg, fp_out)
        plan["final_model"] = cur_model

    if cfg.stage in ("gspo", "both"):
        cmd = [
            sys.executable, "-m", "rse_looper.gspo",
            "--init-model", cur_model,
            "--work-dir", os.path.join(cfg.work_dir, "gspo"),
            "--ckpt-prefix", cfg.out_model,
            "--gpus", cfg.gpus,
            "--train-jsonl", cfg.train_jsonl,
            "--gate-jsonl", cfg.gate_jsonl,
            "--gate-size", str(cfg.gate_size),
            "--max-new-tokens", str(cfg.max_new_tokens),
            "--gate-max-new-tokens", str(cfg.gate_max_new_tokens),
            "--max-gens", str(cfg.gspo_max_gens),
            "--patience", str(cfg.gspo_patience),
            "--problems-per-gen", str(cfg.gspo_problems_per_gen),
            "--steps-per-gen", str(cfg.gspo_steps_per_gen),
            "--lr", str(cfg.gspo_lr),
        ]
        if cfg.resume:
            cmd.append("--resume")
        if cfg.model_revision and cfg.stage == "gspo":
            cmd.extend(["--model-revision", cfg.model_revision])
        subprocess.run(cmd, check=True)
        from pathlib import Path
        manifest = Path(cfg.work_dir) / "gspo" / "run_manifest.json"
        if not manifest.is_file():
            raise RuntimeError("Training process returned without a durable run manifest.")
        result = json.loads(manifest.read_text())
        if result["status"] != "completed":
            raise RuntimeError("Training process returned before completing its run manifest.")
        plan.update(final_model=result["current_checkpoint"]["path"], final_gate=result["current_gate"],
                    accepted_gens=result["accepted_generations"], manifest=str(manifest))

    plan["status"] = "completed"
    return plan


def cli(argv: Sequence[str] | None = None) -> int:
    """Console entry point: print structured output and return an integer exit status."""
    import subprocess
    import sys
    try:
        result = main(argv)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"rse-looper: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
