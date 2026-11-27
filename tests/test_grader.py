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


@pytest.mark.parametrize("prediction,gold", [
    ("1e999", "9e999"),
    ("-1e999", "-9e999"),
    ("1e999", "1e999"),
    ("inf", "Infinity"),
    ("-inf", "-Infinity"),
    ("inf", "inf"),
    ("nan", "nan"),
    (float("inf"), float("inf")),
    ("1/0", "1/0"),
    (r"\frac{1}{0}", r"\frac{1}{0}"),
])
def test_given_invalid_numeric_answers_when_grading_then_never_reward_equivalence(prediction, gold):
    # Given: floating-point overflow, non-finite values, or a zero denominator.
    # When / Then: neither float comparison nor identical text creates a reward.
    assert answers_match(prediction, gold) is False


def test_given_overflowing_boxed_answer_when_grading_then_training_reward_is_false():
    # Given: two distinct values that both convert to positive float infinity.
    raw = r"<|content_text|>\boxed{1e999}<|message_end|>"
    # When: the public grader produces the reward used by training.
    graded = grade_math_output(raw, "9e999")
    # Then: completion is valid but the answer receives no correctness reward.
    assert graded["completed"] is True
    assert graded["correct"] is False


@pytest.mark.parametrize("prediction,gold", [
    ("100000000000000000001", "100000000000000000002"),
    ("100000000000000000001/1", "100000000000000000002"),
    (r"\frac{100000000000000000001}{3}", r"\frac{100000000000000000002}{3}"),
])
def test_given_distinct_exact_integers_or_fractions_when_grading_then_no_float_rounding_match(prediction, gold):
    # Given: exact rational answers whose float approximations collide or are close.
    # When / Then: their exact values, not relative float tolerance, determine equality.
    assert answers_match(prediction, gold) is False


@pytest.mark.parametrize("prediction,gold", [
    (r"\frac{1}{2}", "0.5"),
    ("2/6", "1/3"),
    (r"\frac{1.5}{0.3}", "5"),
    ("1.0000001", "1"),
    ("1e2", "100"),
    ("9" * 400, "9" * 400),
    (r"\infty", r"\infty"),
])
def test_given_valid_finite_or_exact_symbolic_answers_when_grading_then_preserve_supported_matches(prediction, gold):
    # Given: the existing numeric tolerance, simple fractions, and exact symbols.
    # When / Then: rejecting invalid floats does not extend or remove symbolic grading.
    assert answers_match(prediction, gold) is True


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


@pytest.mark.parametrize("bad_rows", [
    [{"id": "q1", "correct": True}, {"id": "q1", "correct": False}],
    [{"correct": True}],
    [{"id": "", "correct": True}],
    [{"id": 1, "correct": True}],
    [{"id": "q1"}],
    [{"id": "q1", "correct": "false"}],
    [{"id": "q1", "correct": 1}],
    [{"id": "q1", "correct": None}],
])
def test_given_invalid_pair_records_when_compared_then_reject_without_coercion(bad_rows):
    # Given: missing/duplicate identities or values that are not JSON booleans.
    valid = [{"id": "q1", "correct": False}]
    # When / Then: either run must fail closed, without bool() coercion or overwrite.
    for base, edit in ((bad_rows, valid), (valid, bad_rows)):
        with pytest.raises(ValueError):
            compare_paired_runs(base, edit)


def test_given_missing_items_when_compared_then_do_not_select_the_intersection():
    # Given: the edited run omitted a hard item from the original evaluation set.
    base = [{"id": "easy", "correct": False}, {"id": "hard", "correct": False}]
    edit = [{"id": "easy", "correct": True}]
    # When / Then: comparison reports the coverage error, not a 100% gain.
    with pytest.raises(ValueError, match="identical.*IDs|IDs.*identical"):
        compare_paired_runs(base, edit)


def test_given_two_fixes_when_compared_then_report_exact_paired_probability():
    # Given: two independent discordant pairs, both improvements.
    base = [{"id": f"q{i}", "correct": False} for i in range(2)]
    edit = [{"id": f"q{i}", "correct": True} for i in reversed(range(2))]
    # When: compare with different bootstrap seeds.
    first = compare_paired_runs(base, edit, seed=1)
    second = compare_paired_runs(base, edit, seed=9)
    # Then: exact one-sided McNemar/binomial probability is 1/4, never Monte Carlo.
    assert first["p_value"] == second["p_value"] == 0.25
    assert first["p_value_method"] == "exact_paired_binomial_greater"
    assert first["fixed"] == 2 and first["broken"] == 0


@pytest.mark.parametrize("function", [paired_bootstrap_ci, paired_permutation_test])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_given_nonfinite_measurements_when_testing_then_reject(function, bad):
    # Given: a failed measurement encoded as a non-finite number.
    # When / Then: it cannot silently become a confidence interval or p-value.
    with pytest.raises(ValueError, match="finite"):
        function([0.0, bad], [1.0, 1.0])


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"n_boot": True}, {"alpha": 0.0}, {"alpha": 1.0}])
def test_given_invalid_bootstrap_controls_when_testing_then_reject(kwargs):
    # Given / When / Then: invalid resampling controls cannot produce a report.
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.0], [1.0], **kwargs)


@pytest.mark.parametrize("n_perm", [0, -1, True])
def test_given_invalid_permutation_count_when_testing_then_reject(n_perm):
    # Given / When / Then: require a positive integer number of resamples.
    with pytest.raises(ValueError):
        paired_permutation_test([0.0], [1.0], n_perm=n_perm)
