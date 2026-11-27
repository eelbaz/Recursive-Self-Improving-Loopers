"""Validated local inputs and an atomic, recoverable training-run ledger.

The manifest is authoritative. ``events.jsonl`` is a readable projection rebuilt
from it on resume, so a crash between the two atomic writes cannot change lineage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def assess_gate_candidate(cur_gate: Mapping[str, Any], cand_gate: Mapping[str, Any],
                          max_single_regime_drop: int = 2) -> dict:
    """The one acceptance rule used by both training and the durable ledger."""
    regimes = ("R3", "R4", "RK2")
    if max_single_regime_drop < 0:
        raise ValueError("max_single_regime_drop must be nonnegative.")
    for gate in (cur_gate, cand_gate):
        if not math.isfinite(float(gate["max_rep"])) or not 0 <= gate["max_rep"] <= 100:
            raise ValueError("Gate repetition rate must be finite and between 0 and 100.")
        for regime in regimes:
            count = gate[regime]["correct"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Gate correctness counts must be nonnegative integers.")
            if "n" in gate[regime] and count > gate[regime]["n"]:
                raise ValueError("Gate correctness exceeds the number of evaluated problems.")
            if "n" in gate[regime] and (type(gate[regime]["n"]) is not int or gate[regime]["n"] < 1):
                raise ValueError("Gate size must be a positive integer.")
            if "completed" in gate[regime]:
                completed = gate[regime]["completed"]
                if type(completed) is not int or completed < count or ("n" in gate[regime] and completed > gate[regime]["n"]):
                    raise ValueError("Gate completed counts must cover correct answers and not exceed the gate size.")
        if sum(gate[regime]["correct"] for regime in regimes) != gate["total_correct"]:
            raise ValueError("Gate total_correct does not equal the regime counts.")
        if type(gate["total_completed"]) is not int or gate["total_completed"] < gate["total_correct"]:
            raise ValueError("Gate total_completed must be a valid completion count.")
        if all("completed" in gate[regime] for regime in regimes):
            if sum(gate[regime]["completed"] for regime in regimes) != gate["total_completed"]:
                raise ValueError("Gate total_completed does not equal regime completion counts.")
        if all("n" in gate[regime] for regime in regimes) and gate["total_completed"] > sum(gate[regime]["n"] for regime in regimes):
            raise ValueError("Gate total_completed exceeds the number of evaluations.")
    if any(cur_gate[regime].get("n") != cand_gate[regime].get("n") for regime in regimes):
        raise ValueError("Gate sizes/denominators must match between current and candidate checkpoints.")
    if cand_gate["max_rep"] > max(3.12, cur_gate["max_rep"] + 1.6):
        return {"accepted": False, "reason": "repetition_limit"}
    if any(cand_gate[regime]["correct"] < cur_gate[regime]["correct"] - max_single_regime_drop for regime in regimes):
        return {"accepted": False, "reason": "mode_regression"}
    if cand_gate["total_correct"] > cur_gate["total_correct"]:
        return {"accepted": True, "reason": "accuracy_improved"}
    if cand_gate["total_correct"] < cur_gate["total_correct"]:
        return {"accepted": False, "reason": "accuracy_regression"}
    if cand_gate["total_completed"] >= cur_gate["total_completed"] + 4:
        return {"accepted": True, "reason": "completion_improved"}
    return {"accepted": False, "reason": "no_gain"}


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checkpoint_fingerprint(path: str | Path) -> dict:
    """Identify exact local weights and configuration; never download a model."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    files = {}
    extensions = {".safetensors", ".json", ".txt", ".model", ".tiktoken", ".py", ".jinja"}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if any(part in {".cache", ".git", "__pycache__"} for part in relative.parts):
            continue
        if item.is_file() and item.suffix in extensions:
            files[relative.as_posix()] = file_sha256(item)
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"No safetensors weights found in checkpoint: {root}")
    return {"path": str(root), "sha256": hashlib.sha256(_canonical(files).encode()).hexdigest(), "files": files}


def execution_fingerprint() -> dict:
    """Record exact source and installed runtime identities used for this run."""
    import importlib.metadata
    import platform
    import sys

    package = Path(__file__).resolve().parent
    source_files = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    project = package.parent / "pyproject.toml"
    if project.is_file():
        source_files["pyproject.toml"] = file_sha256(project)
    versions = {}
    for name in ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "source_sha256": hashlib.sha256(_canonical(source_files).encode()).hexdigest(),
        "source_files": source_files,
        "runtime": {"python": sys.version, "platform": platform.platform(), "packages": versions},
    }


@dataclass(frozen=True)
class MathInputs:
    train_rows: list[dict]
    gate_rows: list[dict]
    fingerprints: dict


def load_math_inputs(train_path: str | Path, gate_path: str | Path, *, gate_size: int) -> MathInputs:
    """Validate complete disjoint sources before selecting the fixed gate subset."""
    from rse_looper.evidence import build_data_manifest, validate_data_manifest, validate_split_manifests

    if not train_path or not gate_path:
        raise ValueError("Explicit --train-jsonl and --gate-jsonl inputs are required.")
    if gate_size < 1:
        raise ValueError("gate_size must be positive.")
    manifests, all_rows = {}, {}
    for role, source in (("train", train_path), ("selection", gate_path)):
        source = Path(source)
        digest = file_sha256(source)
        manifest = build_data_manifest(
            source, dataset=f"local-jsonl:{source.name}", revision=f"sha256:{digest}", split=role, role=role,
        )
        if manifest["sha256"] != digest:
            raise ValueError("Input changed while constructing its dataset manifest.")
        all_rows[role] = validate_data_manifest(source, manifest, expected_role=role)
        manifests[role] = manifest
    validate_split_manifests(list(manifests.values()))
    if not all_rows["train"]:
        raise ValueError("Training data must not be empty.")
    if len(all_rows["selection"]) < gate_size:
        raise ValueError(f"Selection data has {len(all_rows['selection'])} rows; gate_size requires {gate_size}.")
    gate_rows = all_rows["selection"][:gate_size]
    manifests["selection"] = {**manifests["selection"], "selected_ids": [row["id"] for row in gate_rows]}
    return MathInputs(all_rows["train"], gate_rows, manifests)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class RunLedger:
    """Serialize run decisions and only promote successfully exported checkpoints."""

    def __init__(self, work_dir: str | Path, *, config: Mapping[str, Any],
                 data_fingerprints: Mapping[str, Any], initial_checkpoint: Mapping[str, Any],
                 resume: bool = False):
        self.work_dir = Path(work_dir).resolve()
        self.manifest_path = self.work_dir / "run_manifest.json"
        self.event_path = self.work_dir / "events.jsonl"
        self.config = json.loads(_canonical(config))
        self.data = json.loads(_canonical(data_fingerprints))
        self.initial = copy.deepcopy(dict(initial_checkpoint))
        self.execution = execution_fingerprint()
        self.resume = resume
        self.state: dict = {}
        self._lock = None

    def __enter__(self):
        import fcntl

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.work_dir / ".run.lock").open("a+")
        try:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is using run directory: {self.work_dir}") from exc
            if self.manifest_path.exists():
                if not self.resume:
                    raise FileExistsError(f"Run already exists at {self.work_dir}; use --resume with identical inputs.")
                self.state = json.loads(self.manifest_path.read_text())
                if self.state.get("schema_version") != SCHEMA_VERSION or self.state.get("kind") != "training-run":
                    raise ValueError("Unsupported run manifest schema; cannot resume.")
                if self.state["config"] != self.config:
                    raise ValueError("Resume configuration mismatch.")
                if self.state["data_fingerprints"] != self.data:
                    raise ValueError("Resume data fingerprint mismatch.")
                if self.state.get("execution_fingerprint") != self.execution:
                    raise ValueError("Resume implementation or runtime fingerprint mismatch.")
                if self.state["initial_checkpoint"]["sha256"] != self.initial["sha256"]:
                    raise ValueError("Initial checkpoint changed; resume fingerprint mismatch.")
                current = self.state["current_checkpoint"]
                if checkpoint_fingerprint(current["path"])["sha256"] != current["sha256"]:
                    raise ValueError("Current checkpoint changed; resume fingerprint mismatch.")
                self._write_events()
            else:
                self.state = {
                    "schema_version": SCHEMA_VERSION, "kind": "training-run", "status": "initialized",
                    "config": self.config, "data_fingerprints": self.data,
                    "execution_fingerprint": self.execution,
                    "initial_checkpoint": self.initial, "current_checkpoint": self.initial,
                    "current_gate": None, "next_generation": 0, "accepted_generations": 0,
                    "consecutive_rejections": 0, "attempt_count": 0, "pending_generation": None, "events": [],
                }
                self._event("run_initialized")
                self._save()
            return self
        except BaseException:
            self._lock.close()
            self._lock = None
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None and self.state:
                self.state["status"] = "interrupted"
                self._event("run_interrupted", error_type=exc_type.__name__, message=str(exc))
                self._save()
        finally:
            if self._lock is not None:
                self._lock.close()
                self._lock = None
        return False

    def _event(self, event: str, **details) -> None:
        record = {
            "schema_version": SCHEMA_VERSION, "event_id": len(self.state["events"]),
            "event": event, "at_utc": datetime.now(timezone.utc).isoformat(), **details,
        }
        self.state["events"].append(json.loads(_canonical(record)))

    def _write_events(self) -> None:
        _atomic_text(self.event_path, "".join(_canonical(event) + "\n" for event in self.state["events"]))

    def _save(self) -> None:
        _atomic_text(self.manifest_path, json.dumps(self.state, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self._write_events()

    def initialize_gate(self, gate: Mapping[str, Any]) -> None:
        if self.state["current_gate"] is not None:
            raise ValueError("Initial gate has already been committed.")
        assess_gate_candidate(gate, gate)
        self.state["current_gate"] = json.loads(_canonical(gate))
        self._event("initial_gate", checkpoint=self.state["current_checkpoint"], gate=gate)
        self._save()

    def begin_generation(self) -> Path:
        if self.state["current_gate"] is None:
            raise ValueError("An initial gate must be committed before generation starts.")
        self.state["attempt_count"] += 1
        generation, attempt = self.state["next_generation"], self.state["attempt_count"]
        directory = self.work_dir / f"generation_{generation:04d}" / f"attempt_{attempt:04d}"
        # A process can die after mkdir but before the manifest replacement.
        while directory.exists():
            self.state["attempt_count"] += 1
            attempt = self.state["attempt_count"]
            directory = self.work_dir / f"generation_{generation:04d}" / f"attempt_{attempt:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        self.state["status"] = "running"
        self.state["pending_generation"] = {"generation": generation, "attempt": attempt, "directory": str(directory), "candidates": []}
        self._event("generation_started", generation=generation, attempt=attempt,
                    parent_checkpoint=self.state["current_checkpoint"], previous_gate=self.state["current_gate"])
        self._save()
        return directory

    def record_training_batch(self, item_ids: list[str]) -> None:
        pending = self.state["pending_generation"]
        if pending is None:
            raise ValueError("No generation is active.")
        self._event("training_batch", generation=pending["generation"], attempt=pending["attempt"], item_ids=item_ids)
        self._save()

    def record_no_update(self, reason: str, **details) -> None:
        pending = self.state["pending_generation"]
        if pending is None or pending["candidates"]:
            raise ValueError("A no-update outcome requires an active generation without candidates.")
        self._event("generation_no_update", generation=pending["generation"], attempt=pending["attempt"],
                    reason=reason, **details)
        self._save()

    def record_artifact(self, kind: str, path: str | Path, **details) -> None:
        pending = self.state["pending_generation"]
        self._event("artifact_written", generation=pending["generation"] if pending else None,
                    attempt=pending["attempt"] if pending else None,
                    artifact_kind=kind, path=str(Path(path).resolve()), sha256=file_sha256(path), **details)
        self._save()

    def record_candidate(self, scale: float, gate: Mapping[str, Any], *, accepted: bool | None = None,
                         reason: str | None = None) -> None:
        pending = self.state["pending_generation"]
        if pending is None:
            raise ValueError("No generation is active.")
        if any(candidate["scale"] == scale for candidate in pending["candidates"]):
            raise ValueError("This candidate scale has already been recorded for the attempt.")
        decision = assess_gate_candidate(self.state["current_gate"], gate)
        if ((accepted is not None and accepted != decision["accepted"])
                or (reason is not None and reason != decision["reason"])):
            raise ValueError("Supplied acceptance decision does not match the validated gate rule.")
        candidate = json.loads(_canonical({"scale": scale, "gate": gate, **decision}))
        pending["candidates"].append(candidate)
        self._event("candidate_decision", generation=pending["generation"], attempt=pending["attempt"],
                    previous_gate=self.state["current_gate"], **candidate)
        self._save()

    def commit_generation(self, *, checkpoint: str | Path | None = None,
                          selected_scale: float | None = None, delta_path: str | Path | None = None) -> None:
        pending = self.state["pending_generation"]
        if pending is None:
            raise ValueError("No generation is active.")
        previous = copy.deepcopy(self.state["current_checkpoint"])
        selected = None
        if selected_scale is not None:
            selected = next((c for c in pending["candidates"] if c["scale"] == selected_scale), None)
            if selected is None or not selected["accepted"]:
                raise ValueError("Only an accepted recorded candidate can be committed.")
            if checkpoint is None or delta_path is None:
                raise ValueError("An accepted candidate requires its exported checkpoint and delta file.")
            output = checkpoint_fingerprint(checkpoint)
            if output["path"] == previous["path"]:
                raise ValueError("An accepted checkpoint must be exported to a new directory.")
            delta_hash = file_sha256(delta_path)
        else:
            if checkpoint is not None:
                raise ValueError("A rejected generation cannot promote a checkpoint.")
            if any(c["accepted"] for c in pending["candidates"]):
                raise ValueError("An accepted candidate exists; select it explicitly.")
            output, delta_hash = previous, file_sha256(delta_path) if delta_path else None

        if selected is not None:
            self.state["current_checkpoint"] = output
            self.state["current_gate"] = selected["gate"]
            self.state["accepted_generations"] += 1
            self.state["consecutive_rejections"] = 0
        else:
            self.state["consecutive_rejections"] += 1
        self._event("generation_committed", generation=pending["generation"], attempt=pending["attempt"],
                    accepted=selected is not None, selected_scale=selected_scale,
                    parent_checkpoint=previous, output_checkpoint=output, delta_sha256=delta_hash,
                    gate=self.state["current_gate"])
        self.state["next_generation"] += 1
        self.state["pending_generation"] = None
        self._save()

    def complete(self) -> None:
        if self.state["pending_generation"] is not None:
            raise ValueError("Cannot complete a run with an uncommitted generation.")
        if self.state["status"] != "completed":
            self.state["status"] = "completed"
            self._event("run_completed", final_checkpoint=self.state["current_checkpoint"], final_gate=self.state["current_gate"])
            self._save()
