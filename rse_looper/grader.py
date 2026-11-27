"""Deterministic math/code grading and paired statistical hypothesis testing."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence
import numpy as np


def split_inkling_output(raw_text: str) -> Dict[str, object]:
    """Parse Inkling-2-Mini special tags into thinking_text, final_text, and completion flags."""
    has_eos = "<|message_end|>" in raw_text or "<|endoftext|>" in raw_text
    cleaned = raw_text.replace("<|message_end|>", "").replace("<|endoftext|>", "")
    if "<|content_text|>" in cleaned:
        parts = cleaned.split("<|content_text|>", 1)
        thinking = parts[0].replace("<|content_thinking|>", "").replace("<|message_model|>", "").strip()
        final = parts[1].strip()
        thinking_closed = True
    else:
        thinking = cleaned.replace("<|content_thinking|>", "").replace("<|message_model|>", "").strip()
        final = ""
        thinking_closed = False
    return {
        "thinking_text": thinking,
        "final_text": final,
        "thinking_closed": thinking_closed,
        "has_eos": has_eos,
        "completed": bool(thinking_closed and len(final) > 0),
    }


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last balanced \\boxed{...} expression from text."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def normalize_math_answer(ans: Optional[str]) -> str:
    """Canonicalize LaTeX/numeric math strings for deterministic equivalence checks."""
    if ans is None:
        return ""
    s = str(ans).strip()
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "")
    s = s.rstrip(".")
    return s


def _numeric_answer(value: str) -> tuple[bool, Fraction | float | None]:
    """Return (recognized numeric syntax, valid value); None marks invalid numerics."""
    fraction = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", value)
    try:
        if fraction:
            return True, Fraction(fraction.group(1)) / Fraction(fraction.group(2))
        if re.fullmatch(r"[+-]?\d+", value) or re.fullmatch(r"-?\d+/\d+", value):
            return True, Fraction(value)
    except (ValueError, ZeroDivisionError):
        return True, None
    try:
        number = float(value)
    except ValueError:
        return False, None
    return True, number if math.isfinite(number) else None


def answers_match(pred: Optional[str], gold: str) -> bool:
    """Compare exact integers/fractions, finite floats, or identical normalized text.

    Decimal/scientific literals retain the historical 1e-6 numeric tolerance,
    but overflow, NaN, infinity, and zero denominators cannot earn a reward.
    Symbolic expressions receive only exact normalized-text matching.
    """
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if not p or not g:
        return False
    p_numeric, pn = _numeric_answer(p)
    g_numeric, gn = _numeric_answer(g)
    # Validate before exact-text matching, which otherwise accepts "nan" == "nan".
    if (p_numeric and pn is None) or (g_numeric and gn is None):
        return False
    if p == g:
        return True
    if not p_numeric or not g_numeric:
        return False
    if isinstance(pn, Fraction) and isinstance(gn, Fraction):
        return pn == gn
    try:
        pf, gf = float(pn), float(gn)
    except OverflowError:
        return False
    return math.isfinite(pf) and math.isfinite(gf) and math.isclose(pf, gf, rel_tol=1e-6, abs_tol=1e-6)


def detect_repetition(text: str, ngram: int = 8, min_repeats: int = 5) -> bool:
    """Detect degenerate verbatim n-gram repetition loops in generated text."""
    toks = text.split()
    if len(toks) < ngram * min_repeats:
        return False
    counts: Dict[tuple, int] = {}
    for i in range(len(toks) - ngram + 1):
        g = tuple(toks[i : i + ngram])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= min_repeats:
            return True
    return False


def grade_math_output(raw_text: str, gold_answer: str) -> Dict[str, object]:
    """Grade one raw Inkling-2-Mini math generation deterministically."""
    parsed = split_inkling_output(raw_text)
    pred = extract_boxed(str(parsed["final_text"]))
    if pred is None and parsed["completed"]:
        nums = re.findall(r"-?\d+(?:/\d+|\.\d+)?", str(parsed["final_text"]))
        pred = nums[-1] if nums else None
    correct = answers_match(pred, gold_answer)
    rep = detect_repetition(raw_text)
    return {
        "completed": bool(parsed["completed"]),
        "thinking_closed": bool(parsed["thinking_closed"]),
        "has_eos": bool(parsed["has_eos"]),
        "correct": bool(correct),
        "pred": pred,
        "repetition": bool(rep),
    }


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval for mean(b - a)."""
    diffs = _paired_differences(a, b)
    _positive_resamples(n_boot, "n_boot")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    rng = np.random.default_rng(seed)
    # Bound temporary memory while preserving a deterministic RNG stream.
    means = np.empty(n_boot, dtype=np.float64)
    batch = max(1, min(512, 1_000_000 // len(diffs)))
    for start in range(0, n_boot, batch):
        stop = min(n_boot, start + batch)
        idx = rng.integers(0, len(diffs), size=(stop - start, len(diffs)))
        means[start:stop] = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean_diff": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """One-sided paired sign-flip permutation p-value testing H1: mean(b) > mean(a)."""
    diffs = _paired_differences(a, b)
    _positive_resamples(n_perm, "n_perm")
    obs = float(diffs.mean())
    if obs <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceedances = 0
    batch = max(1, min(512, 1_000_000 // len(diffs)))
    for start in range(0, n_perm, batch):
        signs = rng.choice([-1.0, 1.0], size=(min(batch, n_perm - start), len(diffs)))
        perm_means = (signs * diffs[None, :]).mean(axis=1)
        exceedances += int(np.sum(perm_means >= obs - 1e-12))
    return float((exceedances + 1) / (n_perm + 1))


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    try:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Paired arrays must contain finite numeric measurements.") from exc
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or len(arr_a) == 0:
        raise ValueError("Paired arrays must be non-empty 1-D sequences of identical length.")
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        raise ValueError("Paired arrays must contain only finite measurements.")
    with np.errstate(over="ignore"):
        diffs = arr_b - arr_a
    if not np.isfinite(diffs).all():
        raise ValueError("Paired differences must be finite.")
    return diffs


def _positive_resamples(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_binary_rows(
    rows: Sequence[Mapping[str, object]], metric_key: str = "correct", name: str = "run"
) -> Dict[str, Mapping[str, object]]:
    """Validate an entire run before pairing; no dropped rows or truthiness coercion."""
    result: Dict[str, Mapping[str, object]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} row {position} must be an object.")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{name} row {position} requires a non-empty string id.")
        if item_id in result:
            raise ValueError(f"Duplicate item id {item_id!r} in {name}.")
        for key in {metric_key, "completed", "repetition"}:
            if key == metric_key or key in row:
                if type(row.get(key)) is not bool:
                    raise ValueError(f"{name} item {item_id!r}: {key} must be a JSON boolean.")
        result[item_id] = row
    if not result:
        raise ValueError(f"{name} must contain at least one item.")
    return result


def exact_paired_binomial_pvalue(fixed: int, broken: int) -> float:
    """Exact one-sided McNemar/binomial test, H1: edit improves binary outcomes.

    Conditional on discordant pairs, H0 assigns a fair coin to which run wins.
    This is an unadjusted per-comparison probability; selection/multiple testing
    and correlated benchmark items require separate experimental controls.
    """
    if any(type(value) is not int or value < 0 for value in (fixed, broken)):
        raise ValueError("Discordant pair counts must be non-negative integers.")
    discordant = fixed + broken
    if discordant == 0:
        return 1.0
    # Integer arithmetic avoids Monte Carlo error and premature rounding.
    numerator = sum(math.comb(discordant, k) for k in range(fixed, discordant + 1))
    return float(Fraction(numerator, 1 << discordant))


def compare_paired_runs(
    base_rows: Sequence[Mapping[str, object]],
    edit_rows: Sequence[Mapping[str, object]],
    metric_key: str = "correct",
    seed: int = 42,
    n_boot: int = 10000,
) -> Dict[str, object]:
    """Compare complete binary runs over identical item IDs, failing on bad input."""
    b_map = validate_binary_rows(base_rows, metric_key, "base")
    e_map = validate_binary_rows(edit_rows, metric_key, "edit")
    if b_map.keys() != e_map.keys():
        raise ValueError("Paired runs must have identical item IDs; missing or extra items are not dropped.")
    common = sorted(b_map)
    a = [float(b_map[i][metric_key]) for i in common]
    b = [float(e_map[i][metric_key]) for i in common]
    fixed = sum((not b_map[i][metric_key]) and e_map[i][metric_key] for i in common)
    broken = sum(b_map[i][metric_key] and (not e_map[i][metric_key]) for i in common)
    ci = paired_bootstrap_ci(a, b, seed=seed, n_boot=n_boot)
    p = exact_paired_binomial_pvalue(fixed, broken)
    return {
        "n": len(common),
        "base_rate": round(float(np.mean(a)) * 100.0, 2),
        "edit_rate": round(float(np.mean(b)) * 100.0, 2),
        "delta_pp": round(ci["mean_diff"] * 100.0, 2),
        "ci_95_pp": [round(ci["ci_lo"] * 100.0, 2), round(ci["ci_hi"] * 100.0, 2)],
        "fixed": fixed,
        "broken": broken,
        "p_value": p,
        "p_value_method": "exact_paired_binomial_greater",
        "ci_method": "paired_percentile_bootstrap_95",
        "bootstrap_seed": seed,
        "bootstrap_samples": n_boot,
        "multiplicity_adjusted": False,
    }
