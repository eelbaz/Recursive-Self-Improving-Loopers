"""Validate experimental evidence without loading a model or modifying results.

Manifest and legacy-audit commands use only the standard library. Paired
comparisons import the NumPy-based grader lazily. A valid schema establishes
artifact consistency, not the truth of declarations or independent replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Mapping, Sequence


ROLES = frozenset({"calibration", "train", "selection", "test"})
REGIMES = ("R3", "R4", "RK2")
LEGACY_FILES = (
    "fpsolve_results.json", "general_skills.json", "gspo_results.json",
    "integrity_audit.json", "l17_fault.json", "terminal_bench.json",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed.")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number {value!r} is not allowed.")
    return parsed


def _loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_invalid_constant, parse_float=_finite_float)


def load_json(path: str | Path) -> Any:
    return _loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    """Read every JSONL record, rejecting blank lines and non-object values."""
    return _parse_jsonl(Path(path).read_bytes())


def _parse_jsonl(raw: bytes) -> list[dict]:
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL row at line {line_number}.")
        try:
            row = _loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object.")
        rows.append(row)
    if not rows:
        raise ValueError("JSONL data must contain at least one record.")
    return rows


def _problem_hash(problem: str) -> str:
    # Local duplicate detection only: this does not detect paraphrases or
    # pretraining contamination. Preserve case and mathematical punctuation.
    normalized = " ".join(unicodedata.normalize("NFC", problem).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_data_manifest(
    path: str | Path, *, dataset: str, revision: str, split: str, role: str
) -> dict:
    """Commit to exact JSONL bytes, all IDs, and whitespace-normalized problems.

    Train/selection/test records require string id, problem, and answer fields.
    Calibration may omit answer. Provenance labels remain author declarations.
    """
    manifest, _ = _data_snapshot(Path(path).read_bytes(), dataset=dataset, revision=revision, split=split, role=role)
    return manifest


def _data_snapshot(raw: bytes, *, dataset: str, revision: str, split: str, role: str) -> tuple[dict, list[dict]]:
    for value, name in ((dataset, "dataset"), (revision, "revision"), (split, "split")):
        _nonempty_string(value, name)
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}.")
    rows = _parse_jsonl(raw)
    ids, problem_hashes = set(), set()
    for position, row in enumerate(rows):
        item_id = _nonempty_string(row.get("id"), f"row {position} id")
        problem = _nonempty_string(row.get("problem"), f"row {position} problem")
        if role != "calibration" or "answer" in row:
            _nonempty_string(row.get("answer"), f"row {position} answer")
        problem_hash = _problem_hash(problem)
        if item_id in ids:
            raise ValueError(f"Duplicate item id {item_id!r}.")
        if problem_hash in problem_hashes:
            raise ValueError(f"Duplicate normalized problem for item {item_id!r}.")
        ids.add(item_id)
        problem_hashes.add(problem_hash)
    item_ids, hashes = sorted(ids), sorted(problem_hashes)
    return {
        "schema_version": 1, "kind": "dataset-manifest", "dataset": dataset,
        "revision": revision, "split": split, "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows), "item_ids": item_ids, "problem_hashes": hashes,
        "ids_sha256": _digest(item_ids), "content_sha256": _digest(hashes),
    }, rows


def _validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be an object.")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("kind") != "dataset-manifest":
        raise ValueError("Expected a version 1 dataset-manifest.")
    for name in ("dataset", "revision", "split"):
        _nonempty_string(manifest.get(name), name)
    if manifest.get("role") not in ROLES:
        raise ValueError("Invalid manifest role.")
    count = _positive_int(manifest.get("row_count"), "row_count")
    for name in ("sha256", "ids_sha256", "content_sha256"):
        _sha256(manifest.get(name), name)
    for name in ("item_ids", "problem_hashes"):
        values = manifest.get(name)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"{name} must contain row_count entries.")
        for value in values:
            (_sha256 if name == "problem_hashes" else _nonempty_string)(value, name)
        if len(set(values)) != count or values != sorted(values):
            raise ValueError(f"{name} must be sorted and unique.")
    if _digest(manifest["item_ids"]) != manifest["ids_sha256"] or _digest(manifest["problem_hashes"]) != manifest["content_sha256"]:
        raise ValueError("Manifest identity list hash mismatch.")


def validate_data_manifest(
    path: str | Path, manifest: Mapping[str, Any], expected_role: str | None = None
) -> list[dict]:
    """Validate a stored commitment against the actual source data."""
    _validate_manifest_schema(manifest)
    if expected_role is not None and manifest["role"] != expected_role:
        raise ValueError(f"Manifest role must be {expected_role!r}.")
    actual, rows = _data_snapshot(Path(path).read_bytes(), **{key: manifest[key] for key in ("dataset", "revision", "split", "role")})
    for key in actual:
        if actual[key] != manifest[key]:
            raise ValueError(f"Manifest {key} hash/identity mismatch against source data.")
    return rows


def validate_split_manifests(manifests: Sequence[Mapping[str, Any]]) -> None:
    """Reject ID or canonical-problem overlap across any supplied data roles."""
    for index, manifest in enumerate(manifests):
        _validate_manifest_schema(manifest)
        for earlier in manifests[:index]:
            same_ids = set(manifest["item_ids"]) & set(earlier["item_ids"])
            same_problems = set(manifest["problem_hashes"]) & set(earlier["problem_hashes"])
            if same_ids or same_problems:
                raise ValueError(
                    f"Data overlap between {earlier['role']}/{earlier['split']} and "
                    f"{manifest['role']}/{manifest['split']}: {len(same_ids)} IDs, "
                    f"{len(same_problems)} normalized problems."
                )


def validate_evaluation_provenance(provenance: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Check a held-out declaration; its historical truth needs external evidence."""
    _validate_manifest_schema(manifest)
    if manifest["role"] != "test":
        raise ValueError("Final evaluation requires a test-role data manifest.")
    if not isinstance(provenance, Mapping) or type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1 or provenance.get("kind") != "evaluation-provenance":
        raise ValueError("Expected version 1 evaluation-provenance.")
    _nonempty_string(provenance.get("run_id"), "run_id")
    if provenance.get("dataset_sha256") != manifest["sha256"]:
        raise ValueError("Run dataset_sha256 differs from the frozen data manifest.")
    revision = provenance.get("code_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise ValueError("code_revision must be an immutable full Git commit hash.")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint identity is required.")
    for key in ("model_id", "revision"):
        _nonempty_string(checkpoint.get(key), f"checkpoint.{key}")
    _sha256(checkpoint.get("sha256"), "checkpoint.sha256")
    grader = provenance.get("grader")
    if not isinstance(grader, Mapping):
        raise ValueError("grader identity is required.")
    for key in ("name", "revision"):
        _nonempty_string(grader.get(key), f"grader.{key}")
    decoding = provenance.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ValueError("decoding configuration is required.")
    _nonempty_string(decoding.get("strategy"), "decoding.strategy")
    _positive_int(decoding.get("max_new_tokens"), "decoding.max_new_tokens")
    _positive_int(decoding.get("loop_passes"), "decoding.loop_passes")
    if type(decoding.get("seed")) is not int or decoding["seed"] < 0:
        raise ValueError("decoding.seed must be a non-negative integer.")
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping) or selection.get("used_for_selection") is not False or selection.get("frozen_before_run") is not True:
        raise ValueError("Final test must be declared frozen before the run and unused for selection.")


def compare_evaluation_runs(
    base_rows: Sequence[Mapping[str, Any]], edit_rows: Sequence[Mapping[str, Any]],
    base_provenance: Mapping[str, Any], edit_provenance: Mapping[str, Any],
    manifest: Mapping[str, Any], *, seed: int = 42, n_boot: int = 10000,
) -> dict:
    """Create a paired report only after full coverage and provenance validation."""
    from rse_looper.grader import compare_paired_runs, validate_binary_rows

    for provenance in (base_provenance, edit_provenance):
        validate_evaluation_provenance(provenance, manifest)
    if base_provenance["run_id"] == edit_provenance["run_id"]:
        raise ValueError("Base and edit run_id values must be distinct.")
    for key in ("code_revision", "grader"):
        if base_provenance[key] != edit_provenance[key]:
            raise ValueError(f"Paired evaluation protocols differ in {key}.")
    base_decode, edit_decode = dict(base_provenance["decoding"]), dict(edit_provenance["decoding"])
    base_passes, edit_passes = base_decode.pop("loop_passes"), edit_decode.pop("loop_passes")
    if base_decode != edit_decode:
        raise ValueError("Paired decoding configurations differ beyond loop_passes.")
    for name, rows in (("base", base_rows), ("edit", edit_rows)):
        indexed = validate_binary_rows(rows, name=name)
        if set(indexed) != set(manifest["item_ids"]):
            raise ValueError(f"{name} must cover every frozen dataset ID exactly once.")
    return {
        "schema_version": 1, "kind": "paired-evaluation-report",
        "status": "validated_metadata_and_pairing",
        "dataset_manifest": dict(manifest),
        "base_provenance": dict(base_provenance), "edit_provenance": dict(edit_provenance),
        "base_rows_sha256": _digest(sorted(base_rows, key=lambda row: row["id"])),
        "edit_rows_sha256": _digest(sorted(edit_rows, key=lambda row: row["id"])),
        "same_loop_depth": base_passes == edit_passes,
        "comparison": compare_paired_runs(base_rows, edit_rows, seed=seed, n_boot=n_boot),
        "limitations": [
            "Checks validate submitted artifacts and declarations; they do not independently verify model execution or checkpoint contents.",
            "A held-out declaration does not prove historical non-use. Publish dated commitments and all candidate-selection logs.",
            "P-values are one-sided and unadjusted. Predeclare comparisons and control repeated testing and model selection.",
            "The percentile bootstrap assumes exchangeable independent items; clustered tasks need an appropriate resampling unit.",
            "Equal loop depth does not establish equal measured inference cost. Record wall time, tokens, memory and hardware.",
        ],
    }


def _finding(code: str, severity: str, message: str, **details: Any) -> dict:
    return {"code": code, "severity": severity, "message": message, **details}


def _count_from_rate(rate: Any, n: int, name: str) -> int:
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 100:
        raise ValueError(f"Invalid percentage {name}.")
    count = round(rate * n / 100.0)
    if abs(rate - count * 100.0 / n) > 0.011:
        raise ValueError(f"{name} cannot be a two-decimal rounded count out of {n}.")
    return count


def _gate_counts(gate: Mapping[str, Any], n: int) -> dict:
    counts = {}
    for regime in REGIMES:
        value = gate.get(regime)
        count = value.get("correct") if isinstance(value, Mapping) else _count_from_rate(value, n, regime)
        if type(count) is not int or not 0 <= count <= n:
            raise ValueError(f"{regime}.correct must be an integer between 0 and {n}.")
        counts[regime] = count
    total = gate.get("total_correct")
    if type(total) is not int or total != sum(counts.values()):
        raise ValueError("Gate total_correct must equal the sum of all mode counts.")
    completed = gate.get("total_completed")
    if completed is None and "mean_comp" in gate:
        completed = _count_from_rate(gate["mean_comp"], n * len(REGIMES), "mean_comp")
    if completed is not None and (type(completed) is not int or not total <= completed <= n * len(REGIMES)):
        raise ValueError("Invalid gate total_completed.")
    max_rep = gate.get("max_rep")
    if type(max_rep) not in (int, float) or not math.isfinite(max_rep) or not 0 <= max_rep <= 100:
        raise ValueError("Invalid gate max_rep.")
    return {**counts, "total_correct": total, "total_completed": completed, "max_rep": max_rep}


def replay_gate_history(history: Mapping[str, Any], max_single_regime_drop: int = 2) -> dict:
    """Replay the published acceptance rule against the *declared* lineage.

    Rounded legacy rates are converted only when they identify an integer count
    at the declared sample size. This checks arithmetic, not missing raw trials.
    After an invalid declared acceptance the next comparison still uses the
    recorded parent, so the report never silently rewrites history.
    """
    n = _positive_int(history.get("gate_size_per_mode"), "gate_size_per_mode")
    current = _gate_counts(history["init_gate"], n)
    generations = history.get("generations")
    if not isinstance(generations, list):
        raise ValueError("generations must be a list.")
    findings, transitions, seen = [], [], set()
    accepted_count = 0
    for generation in generations:
        gen = generation.get("gen")
        if type(gen) is not int or gen < 0 or gen in seen:
            raise ValueError("Generation identifiers must be unique non-negative integers.")
        seen.add(gen)
        accepted = generation.get("accepted")
        if type(accepted) is not bool:
            raise ValueError("accepted must be a JSON boolean.")
        if not accepted:
            transitions.append({"generation": gen, "declared_accepted": False})
            continue
        accepted_count += 1
        candidates = generation.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Accepted generation requires candidates.")
        scale = generation.get("winner_scale")
        matches = [candidate for candidate in candidates if candidate.get("scale") == scale]
        if len(matches) != 1:
            raise ValueError("Accepted generation must name exactly one winning candidate.")
        candidate = _gate_counts(matches[0], n)
        previous_total, candidate_total = current["total_correct"], candidate["total_correct"]
        reasons = []
        if candidate_total < previous_total:
            reasons.append("total_regression")
            findings.append(_finding("accepted_total_regression", "contradiction", "Accepted winner decreased total correct outcomes.", generation=gen, previous=previous_total, candidate=candidate_total))
        for regime in REGIMES:
            drop = current[regime] - candidate[regime]
            if drop > max_single_regime_drop:
                reasons.append(f"{regime}_drop")
                findings.append(_finding("accepted_mode_drop", "contradiction", "Accepted winner exceeded the per-mode regression limit.", generation=gen, mode=regime, previous=current[regime], candidate=candidate[regime], allowed_drop=max_single_regime_drop))
        if candidate["max_rep"] > max(3.12, current["max_rep"] + 1.6):
            reasons.append("repetition")
            findings.append(_finding("accepted_repetition_violation", "contradiction", "Accepted winner exceeded the repetition limit.", generation=gen))
        if candidate_total == previous_total:
            if candidate["total_completed"] is None or current["total_completed"] is None:
                findings.append(_finding("tie_completion_unverifiable", "unverifiable", "Tied accuracy requires missing completion counts.", generation=gen))
                reasons.append("missing_completion_counts")
            elif candidate["total_completed"] < current["total_completed"] + 4:
                reasons.append("no_qualifying_gain")
                findings.append(_finding("accepted_without_qualifying_gain", "contradiction", "Accepted tied accuracy without four additional completions.", generation=gen))
        transitions.append({"generation": gen, "declared_accepted": True, "winner_scale": scale, "previous_total_correct": previous_total, "candidate_total_correct": candidate_total, "passes_published_rule": not reasons, "reasons": reasons})
        current = candidate
    return {
        "status": "contradictions_found" if any(f["severity"] == "contradiction" for f in findings) else ("incomplete" if findings else "consistent"),
        "declared_acceptances": accepted_count, "final_total_correct": current["total_correct"],
        "transitions": transitions, "findings": findings,
    }


def _check_aggregate_arithmetic(value: Any, path: str, findings: list[dict]) -> None:
    if isinstance(value, dict):
        if "correct" in value and "n" in value and ("acc" in value or "accuracy" in value):
            n, correct = value["n"], value["correct"]
            rate = value.get("acc", value.get("accuracy"))
            if type(rate) not in (int, float) or not math.isfinite(rate):
                raise ValueError(f"{path}: accuracy must be a finite number.")
            if type(n) is not int or n <= 0 or type(correct) is not int or not 0 <= correct <= n or abs(rate - 100.0 * correct / n) > 0.011:
                findings.append(_finding("aggregate_arithmetic_mismatch", "contradiction", "Recorded accuracy disagrees with its stated count.", path=path))
        for key, child in value.items():
            _check_aggregate_arithmetic(child, f"{path}/{key}", findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_aggregate_arithmetic(child, f"{path}/{index}", findings)


def audit_legacy_results(results_dir: str | Path) -> dict:
    """Read-only audit of the six historical summary artifacts, never a rerun."""
    directory = Path(results_dir)
    artifacts, hashes = {}, {}
    for name in LEGACY_FILES:
        path = directory / name
        raw = path.read_bytes()
        artifacts[name] = _loads(raw.decode("utf-8"))
        hashes[name] = hashlib.sha256(raw).hexdigest()
    replay = replay_gate_history(artifacts["gspo_results.json"])
    findings = list(replay["findings"])
    for name, data in artifacts.items():
        _check_aggregate_arithmetic(data, name, findings)
    findings.extend([
        _finding("missing_item_traces", "unverifiable", "Legacy aggregates contain no complete paired per-item outputs; paired confidence intervals and exact significance cannot be reconstructed."),
        _finding("missing_provenance", "unverifiable", "Legacy records do not bind scores to frozen data IDs/hashes, immutable code/model revisions, full evaluation configuration, or a predeclared selection protocol."),
        _finding("checkpoint_claims_unverified", "unverifiable", "The historical integrity JSON is a claim. Verifying unchanged expert/vision tensors requires comparing the actual checkpoints."),
        _finding("convergence_not_certified", "unverifiable", "Observed cosine similarities do not establish a contraction bound, fixed-point convergence, or semantic correctness."),
        _finding("timing_scope", "caveat", "Solve and optimizer-stage timing fields do not measure total calibration, rollout, gating, export and evaluation cost."),
    ])
    return {
        "schema_version": 1, "kind": "legacy-evidence-audit",
        "status": "contradictions_found" if any(f["severity"] == "contradiction" for f in findings) else "incomplete",
        "source_sha256": hashes, "gate_replay": replay, "findings": findings,
        "limitations": ["Historical summaries were read unchanged. No model was executed, no benchmark improvement reproduced, and no checkpoint bytes were verified."],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("legacy", help="Audit unchanged historical summary JSONs.")
    legacy.add_argument("--results-dir", default="results")
    manifest = commands.add_parser("manifest", help="Create an exact JSONL data commitment.")
    manifest.add_argument("data")
    for field in ("dataset", "revision", "split"):
        manifest.add_argument(f"--{field}", required=True)
    manifest.add_argument("--role", choices=sorted(ROLES), required=True)
    validate = commands.add_parser("validate-manifest", help="Verify a commitment against JSONL bytes.")
    validate.add_argument("data")
    validate.add_argument("manifest")
    compare = commands.add_parser("compare", help="Validate complete paired runs and declared test provenance.")
    for field in ("base", "edit", "base-provenance", "edit-provenance", "data", "manifest"):
        compare.add_argument(f"--{field}", required=True)
    compare.add_argument("--other-manifest", action="append", default=[], help="Training/calibration/selection manifest to check for exact overlap; repeatable.")
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--n-boot", type=int, default=10000)
    for command in (legacy, manifest, validate, compare):
        command.add_argument("--output", help="Write JSON to this path instead of stdout.")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "legacy":
            report = audit_legacy_results(args.results_dir)
            code = 1 if report["status"] == "contradictions_found" else 0
        elif args.command == "manifest":
            report = build_data_manifest(args.data, dataset=args.dataset, revision=args.revision, split=args.split, role=args.role)
            code = 0
        elif args.command == "validate-manifest":
            manifest = load_json(args.manifest)
            rows = validate_data_manifest(args.data, manifest)
            report = {"schema_version": 1, "kind": "manifest-validation", "status": "valid", "row_count": len(rows), "sha256": manifest["sha256"]}
            code = 0
        else:
            manifest = load_json(args.manifest)
            validate_data_manifest(args.data, manifest, expected_role="test")
            others = [load_json(path) for path in args.other_manifest]
            validate_split_manifests([*others, manifest])
            base_raw, edit_raw = Path(args.base).read_bytes(), Path(args.edit).read_bytes()
            report = compare_evaluation_runs(_parse_jsonl(base_raw), _parse_jsonl(edit_raw), load_json(args.base_provenance), load_json(args.edit_provenance), manifest, seed=args.seed, n_boot=args.n_boot)
            report["provided_other_split_manifests"] = len(others)
            report["base_jsonl_sha256"] = hashlib.sha256(base_raw).hexdigest()
            report["edit_jsonl_sha256"] = hashlib.sha256(edit_raw).hexdigest()
            code = 0
        text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"schema_version": 1, "status": "invalid_input", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
