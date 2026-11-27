"""Unit tests for rse_looper.grader (parsing, math equivalence, repetition, paired stats)."""

import pytest

from rse_looper.grader import (
    answers_match,
    compare_paired_runs,
    detect_repetition,
    extract_boxed,
    grade_math_output,
    paired_bootstrap_ci,
    paired_permutation_test,
    split_inkling_output,
)


def test_split_inkling_output_handles_complete_and_truncated():
    raw_ok = "<|message_model|><|content_thinking|>3x = 15 so x = 5.<|content_text|>\\boxed{5}<|message_end|>"
    p = split_inkling_output(raw_ok)
    assert p["completed"] is True
    assert p["thinking_closed"] is True
    assert p["has_eos"] is True
    assert p["final_text"] == "\\boxed{5}"

    raw_trunc = "<|message_model|><|content_thinking|>Wait let me recheck..."
    pt = split_inkling_output(raw_trunc)
    assert pt["completed"] is False
    assert pt["thinking_closed"] is False


def test_extract_boxed_nested_braces_and_equivalence():
    assert extract_boxed("Thus the answer is \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert answers_match("\\dfrac{1}{2}", "0.5")
    assert answers_match("  \\left( 42 \\right)  ", "(42)")
    assert not answers_match("41", "42")


def test_repetition_detector_catches_loops_without_false_positives():
    loop_text = " ".join(["wait let me double check this step again carefully"] * 10)
    assert detect_repetition(loop_text) is True
    clean_text = "Step 1: factor x^2 - 5x + 6 = (x - 2)(x - 3). Therefore x = 2 or x = 3. Final answer: \\boxed{2, 3}."
    assert detect_repetition(clean_text) is False


def test_grade_math_output_end_to_end():
    raw = "<|content_thinking|>2^3 = 8<|content_text|>The answer is \\boxed{8}.<|message_end|>"
    g = grade_math_output(raw, "8")
    assert g["completed"] is True
    assert g["correct"] is True
    assert g["repetition"] is False
    assert g["pred"] == "8"


def test_paired_stats_reproducibility_and_significance():
    base = [0] * 40 + [1] * 24
    edit = [1] * 20 + [0] * 20 + [1] * 24  # +20 fixed, 0 broken
    p = paired_permutation_test(base, edit, n_perm=5000, seed=42)
    assert p < 0.001
    assert paired_permutation_test(base, base) == 1.0

    ci = paired_bootstrap_ci(base, edit, n_boot=5000, seed=42)
    assert ci["ci_lo"] <= ci["mean_diff"] <= ci["ci_hi"]
    assert ci["ci_lo"] > 0.0

    b_rows = [{"id": f"q{i}", "correct": bool(v)} for i, v in enumerate(base)]
    e_rows = [{"id": f"q{i}", "correct": bool(v)} for i, v in enumerate(edit)]
    cmp = compare_paired_runs(b_rows, e_rows)
    assert cmp["fixed"] == 20 and cmp["broken"] == 0 and cmp["p_value"] < 0.001
