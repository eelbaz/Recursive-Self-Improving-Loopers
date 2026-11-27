# Evidence, reproducibility, and controlled evaluation

The repository contains an experimental implementation and historical aggregate
results. The repairs make the implementation and future evidence more testable.
They do **not** reproduce the published model improvements, recover missing
historical outputs, or certify that the original training run followed the
published procedure.

## Root cause: five whys

1. **Why could an invalid comparison look like a gain?** The original paired
   comparator silently kept only shared IDs and converted arbitrary values with
   `bool()`. A missing difficult item, overwritten duplicate, or the string
   `"false"` could change the reported result.
2. **Why did these records reach the statistics?** There was no strict boundary
   requiring unique item IDs, complete coverage, and real JSON booleans.
3. **Why could a summary not be independently checked?** Aggregate rates were
   not bound to complete per-item traces, immutable checkpoint and code identities,
   or exact data manifests.
4. **Why did tests not expose the gap?** Tests checked summary values and utility
   behavior. They did not replay the recorded acceptance decisions or require the
   evidence needed to substantiate the experimental claims.
5. **Why was that sufficient to appear verified?** Artifact consistency was
   conflated with experimental reproduction. They require different evidence.

The new boundaries reject invalid evidence, preserve the historical record, and
describe which checks were actually performed. They cannot reconstruct evidence
that was never published.

## What the existing results establish

The six original files under `results/` remain unchanged. Audit them with:

```bash
python -m rse_looper.evidence legacy --results-dir results
# Or, after installation:
rse-looper-evidence legacy --results-dir results --output legacy-audit.json
```

The current legacy audit returns exit status **1**, intentionally. The JSON report
identifies contradictions rather than rewriting the input until the tests pass.
In `gspo_results.json`, the five declared accepted winners follow the total-correct
sequence `111 -> 112 -> 114 -> 118 -> 114 -> 116`. The fourth recorded acceptance
decreases the total from 118 to 114. Some accepted winners also violate the
published limit on a single mode's regression. Rounded percentages are converted
to counts only if they match an integer count at the stated sample size within
the artifact's two-decimal rounding precision.

Replay follows the **declared** accepted lineage even after an invalid decision.
This is necessary to audit what was recorded; silently rejecting that decision
would invent a different historical experiment. The independent runtime tests
cover how a newly executed run should actually accept, reject, and resume.

| Available check | What it establishes | What it does not establish |
| --- | --- | --- |
| Unit tests on synthetic tensors and records | Specified numerical, state, and validation behavior | Published benchmark gains or timings |
| Legacy aggregate replay | Arithmetic and acceptance-rule contradictions in the supplied summaries | The missing raw generations or the actual checkpoint lineage |
| Dataset manifest validation | Exact file identity, complete IDs, duplicate detection | Truth of source labels, absence of paraphrases, or pretraining contamination |
| Paired outcome comparison | A reproducible calculation on complete submitted outcomes | Correctness of a custom grader or authenticity of model execution |
| Actual checkpoint comparison | Properties of the checkpoint files provided to that comparison | The quality, safety, or origin of those weights |

The historical integrity JSON is an assertion, not a substitute for comparing
actual checkpoint tensors. The cosine summaries are observations, not a proof of
a contraction, a fixed point, semantic correctness, or AI alignment. Optimizer
or solve timings are stage timings, not the total cost of generating, selecting,
exporting and evaluating a model.

## Freeze the data before experimentation

Create separate JSONL files for calibration, training, selection, and final test.
Every row must have a unique nonempty string `id` and `problem`. Training,
selection, and test rows also require a nonempty string `answer`; calibration
may omit it. The answer is an external reference label even when the model
generates its own solution and no teacher model is called.

The FP-Solve command consumes the `prompt` field of its optional
`--prompts-jsonl` file. To use that same file with the evidence manifest tool,
include `id`, `problem`, and `prompt`, with `problem` equal to the actual prompt
text. For example:

```json
{"id":"calibration-0001","problem":"Explain how to solve 3x + 4 = 19.","prompt":"Explain how to solve 3x + 4 = 19."}
```

FP-Solve currently selects the first 48 supplied prompts and tokenizes each to
at most 256 tokens; record those limits with the source manifest. Without a
custom file it uses the 48 built-in equation prompts recorded in the code.
The pipeline automatically validates training/selection separation. Calibration
and final-test separation require the explicit manifest checks below; they are
not inferred from a file's name or from the stage-one source hash.

```json
{"id":"heldout-0001","problem":"What is 7 multiplied by 8?","answer":"56"}
{"id":"heldout-0002","problem":"What is the greatest common divisor of 18 and 24?","answer":"6"}
```

Create the commitment before generating candidate models:

```bash
rse-looper-evidence manifest heldout.jsonl \
  --dataset example-math --revision immutable-source-revision \
  --split final-test --role test --output heldout.manifest.json

rse-looper-evidence validate-manifest heldout.jsonl heldout.manifest.json
```

A version 1 `dataset-manifest` stores:

- `dataset`, `revision`, `split`, and `role` as explicit provenance declarations;
- `sha256` of the **exact JSONL bytes**, including answers and formatting;
- `row_count` and the complete sorted `item_ids` list;
- hashes of each problem after Unicode NFC and whitespace normalization;
- `ids_sha256` and `content_sha256` over canonical JSON encodings of those lists.

Changes to an answer, row, or even file formatting invalidate the byte hash.
Validation returns the same snapshot of rows whose bytes were checked, avoiding
a second unverified read. Blank lines, duplicate JSON keys, NaN/Infinity, overflowed
JSON numbers, duplicate IDs, and duplicate normalized problems are rejected.

The Python API is also usable by experiment runners:

```python
from rse_looper.evidence import (
    build_data_manifest,
    validate_data_manifest,
    validate_split_manifests,
)

train = build_data_manifest(
    "train.jsonl", dataset="example-math", revision="immutable-source-revision",
    split="train", role="train",
)
test = build_data_manifest(
    "heldout.jsonl", dataset="example-math", revision="immutable-source-revision",
    split="final-test", role="test",
)
validate_split_manifests([train, test])
rows = validate_data_manifest("heldout.jsonl", test, expected_role="test")
```

Cross-split validation rejects overlapping IDs **or** normalized problem hashes.
This detects exact local leakage, including a renamed duplicate. It cannot
detect semantically equivalent paraphrases, unseen source copies, or overlap
with base-model pretraining. Record the source sampling rule, full candidate
pool, revisions, and exclusions so another researcher can inspect those risks.

## Record complete per-item outcomes and provenance

Write one evaluation outcome per frozen test ID, including failures and
truncations. Never omit a failed item or run only the overlap between two sets.

```json
{"id":"heldout-0001","correct":false,"completed":true,"repetition":false}
{"id":"heldout-0002","correct":true,"completed":true,"repetition":false}
```

`correct` must be a JSON boolean. Optional `completed` and `repetition` fields
must also be booleans. Strings such as `"false"`, integers such as `0`/`1`, and
nulls are not silently coerced. Keep the raw model output, prompt, generation
configuration, errors, token counts, elapsed time, and verifier output alongside
these rows. The comparator checks outcome consistency; it does not rerun a
benchmark-specific grader from the boolean alone.

Create a separate `evaluation-provenance` document for each run. Replace the
placeholder hashes below with the actual values; placeholders are not valid
input:

```json
{
  "schema_version": 1,
  "kind": "evaluation-provenance",
  "run_id": "base-final-test-seed-42",
  "dataset_sha256": "<exact SHA-256 from the test manifest>",
  "checkpoint": {
    "model_id": "<model or local checkpoint identifier>",
    "revision": "<immutable source revision or recorded local revision>",
    "sha256": "<digest binding the actual checkpoint files used>"
  },
  "code_revision": "<full 40- or 64-character Git commit hash>",
  "grader": {"name": "<benchmark verifier>", "revision": "<pinned revision>"},
  "decoding": {
    "strategy": "greedy",
    "max_new_tokens": 2048,
    "seed": 42,
    "loop_passes": 3
  },
  "selection": {"used_for_selection": false, "frozen_before_run": true}
}
```

Include every other material decoding setting in `decoding`, such as sampling
temperature, stop tokens, the prompt/template revision and reasoning effort.
The paired reporter requires identical code, grader and decoding configuration,
with the sole allowed decoding difference being `loop_passes`; any such
difference is reported explicitly. For a sharded checkpoint, keep a manifest of
all file hashes and state how its fingerprint was formed. A free-form model name
or a mutable branch such as `main` is insufficient to identify the actual bytes.

`selection.used_for_selection=false` is a declaration, not a cryptographic proof
of non-use. Publish dated test commitments, source data manifests, all candidate
selection logs, and an immutable chosen checkpoint before final evaluation. A
selection gate reused to accept candidates is validation data; it cannot become
the untouched final test merely by calling it hidden.

## Compare the complete paired experiment

```bash
rse-looper-evidence compare \
  --base base.rows.jsonl --edit edited.rows.jsonl \
  --base-provenance base.provenance.json \
  --edit-provenance edited.provenance.json \
  --data heldout.jsonl --manifest heldout.manifest.json \
  --other-manifest train.manifest.json \
  --other-manifest selection.manifest.json \
  --seed 42 --n-boot 10000 --output paired-report.json
```

Repeat `--other-manifest` for every calibration, training, or selection source.
These external manifests must themselves be validated against their source
files; the comparison command checks their declared identities for overlap.
The report records how many other manifests were supplied. Omitting one is not
evidence that the corresponding split was independent.

The report includes the number of fixed and broken items, rate difference, a
seeded paired percentile-bootstrap 95% interval, and an **exact one-sided paired
binomial probability** (the conditional McNemar test for binary outcomes). The
probability tests the predeclared alternative that the edited model improves.
It is not rounded to five decimals and is not estimated by Monte Carlo. For
example, two fixed items and zero broken items give `p = 0.25`, regardless of
how many other items are unchanged. The confidence interval still depends on
the declared bootstrap seed and sample size.

These statistics assume independent/exchangeable items and are unadjusted for
multiple comparisons. A selection sweep, repeated peeking at the final test,
correlated tasks, or cherry-picked benchmark slice requires a different
experimental analysis. Predeclare the primary comparison, test population,
confidence procedure, and multiplicity correction. Do not compute paired
significance from aggregate totals when the discordant pairs are unavailable.

The status `validated_metadata_and_pairing` means exactly that. It does not mean
that independent reproduction succeeded or that a model-improvement claim is
established.

## Controls needed for a new model-improvement claim

1. Pin source code, dependencies, model/tokenizer revisions, actual checkpoint
   hashes, hardware, kernels, prompt templates, and all random seeds. Publish
   input manifests and a sampling rule before the run.
2. Test the default three-pass base and edited model on the same items with the
   same decoding budget. Separately label forced four-pass experiments. Inkling
   2 Mini's documented default is fixed at three passes; its developer released
   a separate variable-loop research model for varying depth.
3. Compare the unedited model, FP-Solve alone, GSPO alone, and their combination.
   Include a compute-matched nonrecursive training control to test whether
   sequential self-generated data contributes beyond additional optimization.
4. Keep calibration, training, acceptance/selection and final test disjoint.
   Save every candidate, accepted/rejected decision, parent checkpoint, rollback,
   restart, and final-model selection. Evaluate any weight merge as a new candidate.
5. Predeclare final tests, run several independent seeds, report every primary
   comparison and regression, and retain all per-item outputs. Use official
   benchmark execution/verifiers where available; the local math checker supports
   only a restricted set of numeric/string equivalences and is not a general
   proof checker or a substitute for a benchmark harness.
6. Measure end-to-end time and cost: model load, calibration, solve, rollout
   generation, optimization, gating, checkpoint export, merging if used, and final
   evaluation. Equal parameter count or loop depth does not prove equal cost.
7. Have an independent environment reproduce the frozen protocol and compare
   actual checkpoint tensors. Separate a successful reproduction from a
   statistically inconclusive result, a regression, and missing evidence.

Official architecture references:
[Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/) and
[variable-loop model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/).

## CLI and verification contract

Manifest creation, validation, and legacy replay use the standard library.
Paired statistics additionally require NumPy; no GPU or model download is needed.

| Exit status | Meaning |
| --- | --- |
| `0` | Requested structural computation succeeded; inspect its status and limitations |
| `1` | Legacy audit found recorded contradictions |
| `2` | Malformed input, incompatible provenance, hash mismatch, or incomplete paired coverage |

The regression tests use Given/When/Then cases for both valid synthetic evidence
and concrete failure modes. Run them with:

```bash
python -m pytest tests/test_grader.py tests/test_evidence.py -W error
```

No test requires changing the historical benchmark numbers. Passing the tests is
an engineering result; a new benchmark claim requires the controlled experiment
and evidence described above.
