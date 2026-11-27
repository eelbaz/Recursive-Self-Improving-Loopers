"""Package and historical-artifact hygiene; actual tensor checks live in test_checkpoint."""

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_given_public_package_when_modules_change_then_required_modules_still_compile():
    """Given a growing package, When inspecting sources, Then keep its public core and valid syntax."""
    pkg = REPO / "rse_looper"
    expected_modules = {
        "__init__.py",
        "checkpoint.py",
        "fpsolve.py",
        "grader.py",
        "gspo.py",
        "looping.py",
        "pipeline.py",
        "preflight.py",
    }
    actual_modules = {p.name for p in pkg.glob("*.py")}
    assert expected_modules <= actual_modules
    for fp in sorted(pkg.glob("*.py")):
        src = fp.read_text()
        ast.parse(src, filename=str(fp))
        assert "/workspace/" not in src, f"Hardcoded remote path in {fp.name}"


def test_result_artifacts_are_valid_and_finite():
    expected_results = {
        "l17_fault.json",
        "fpsolve_results.json",
        "gspo_results.json",
        "general_skills.json",
        "terminal_bench.json",
        "integrity_audit.json",
    }
    actual_results = {p.name for p in (REPO / "results").glob("*.json")}
    assert actual_results == expected_results
    for name in expected_results:
        txt = (REPO / "results" / name).read_text()
        assert "NaN" not in txt and "Infinity" not in txt
        data = json.loads(txt)
        assert isinstance(data, dict) and len(data) > 0


def test_historical_integrity_report_retains_its_original_claims():
    """Preserve the historical report; these assertions do not compare model weights."""
    audit = json.loads((REPO / "results" / "integrity_audit.json").read_text())
    assert audit["logical_total_params"] == 26_386_412_544
    assert audit["logical_text_params"] == 26_068_156_416
    assert audit["safetensors_keys"] == 1244
    assert audit["fpsolve"]["expert_keys_changed"] == 0
    assert audit["fpsolve"]["vision_keys_changed"] == 0
    assert audit["multi_gen_gspo"]["expert_keys_changed"] == 0
    assert audit["multi_gen_gspo"]["vision_keys_changed"] == 0


def test_no_credentials_committed_in_repository():
    token_prefixes = ("ghp_", "hf_", "sk_live_")
    for fp in REPO.rglob("*"):
        if fp.is_file() and not any(p in (".git", ".venv", ".pytest_cache", ".pytest-tmp", "runs", "models", "data") or p.startswith("pytest-of-") for p in fp.parts) and fp.suffix in (".py", ".tex", ".md", ".html", ".json", ".toml"):
            if fp.name == "test_integrity.py":
                continue
            txt = fp.read_text(errors="ignore")
            for prefix in token_prefixes:
                assert prefix not in txt, f"Potential credential prefix '{prefix}' found in {fp.relative_to(REPO)}"
