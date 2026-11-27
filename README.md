# Recursive Self-Improving Loopers (`rse_looper`)

Local activation-based weight editing and verifier-guided self-training for
looped mixture-of-experts models, with explicit experiment provenance.

Original research: Eric E., Tensorbend.ai Research.

[Published checkpoint](https://huggingface.co/Ex0bit/Inkblot-2-mini) ·
[Reproducibility protocol](docs/REPRODUCIBILITY.md) ·
[Historical release](docs/INITIAL_RELEASE.md)

## Research status

**The historical benchmark scores are author-reported, unreproduced results.**
The six original JSON files are preserved in [`results/`](results/README.md).
Their recorded accepted-generation history is inconsistent with the released
acceptance rule, and the original per-item outputs, exact data slices and merge
provenance are not available here. The implementation and behavior tests have
been corrected; that does not retroactively validate the original scores or
establish the capabilities of a newly trained checkpoint.

The package supports two distinct stages:

- **FP-Solve:** directly solves a regularized activation-matching problem without
  backpropagation or externally supplied answer labels. A finite preservation
  penalty discourages changes on calibration activations; it is not an exact
  preservation or convergence guarantee.
- **GSPO:** generates solutions, checks them against supplied reference answers,
  and trains allowlisted weights with gradients. It uses no external teacher
  model. A fixed selection set gates candidate updates, and a versioned run
  ledger records decisions and checkpoint lineage.

This is iterative self-training under a fixed human-designed procedure. It does
not demonstrate an improving learning algorithm, unbounded improvement, a
Banach contraction, or general AI alignment.

## Install and test

Python 3.11+ is supported. The tested integration uses Python 3.12, CPU
PyTorch 2.11.0, and the official Transformers implementation pinned to
`b62646bbda836d467060b7e3d64028202079b77d` in the `runtime` extra. The small
integration tests initialize random models locally; they download no weights.

```bash
python3.12 -m venv .venv
. .venv/bin/activate

# CPU behavior and upstream integration tests.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,runtime]'
pytest -W error
```

For GPU training, install the PyTorch build appropriate to your CUDA environment
instead of the CPU wheel. The `--require-causal-conv1d` option requires the
accelerated convolution kernel. The
[pinned upstream loader](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/conversion_mapping.py)
dequantizes the original packed experts; budget for the loaded BF16 weights, activations,
gradients and optimizer state, rather than the serialized checkpoint size.
Each training or evaluation worker loads a complete model on its assigned GPU;
adding GPUs does not shard one model across them. GPU memory requirements,
throughput, multi-GPU numerical behavior, and the historical 26B-model benchmark
results have not been established by these CPU tests.

The tests use Given/When/Then scenarios inside ordinary pytest. They exercise
native-model equivalence, cached decoding, cache rollback, checkpoint exports,
data identity, accepted-generation transactions, resume, and strict paired
statistics. CI runs both unit scenarios and tiny native-model integration.
See [the verification record](docs/TESTING.md) for the observed red-to-green
failures and the workflow for adding regression scenarios.

## Audit the historical evidence without a GPU

```bash
python -m rse_looper.evidence legacy --results-dir results
# Equivalent installed entrypoint:
rse-looper-evidence legacy --results-dir results
```

Exit status **1 is expected** for the historical snapshot: the audit finds
accepted transitions that violate the policy. Exit 2 means invalid input.
The JSON report contains source hashes, the reconstructed transitions, findings,
and limitations. It never alters the input results. Reading the historical
report and creating dataset manifests require only Python's standard library;
paired statistical comparisons also require NumPy.

The reported accepted trajectory is `111 → 112 → 114 → 118 → 114 → 116`
correct outputs over 64 selection problems evaluated in three modes. In
particular, the accepted `118 → 114` decrease contradicts the policy. Reconcile
the original run configuration and records rather than changing these numbers.

## Prepare explicit, disjoint data

GSPO requires UTF-8 JSONL files with one object per line and string `id`,
`problem`, and `answer` fields:

```json
{"id":"train-example-001","problem":"What is 2 + 2?","answer":"4"}
```

Use separate training and selection files. Duplicate IDs, duplicate normalized
problems, train/selection overlap, malformed records, non-finite JSON values,
and an undersized selection set are rejected before loading a model. These
checks detect exact/normalized overlap, not semantic paraphrases or pretraining
contamination. The selection set is reused to choose updates; reserve a separate
untouched test set for final evaluation.

The original release's training and evaluation files are not bundled. Supplying
your own data starts a new experiment; it does not reproduce those scores.
See [the protocol](docs/REPRODUCIBILITY.md) for dataset manifests, source
revisions, strict per-item comparisons and held-out evaluation declarations.
The pipeline automatically checks training versus selection separation. Include
calibration and final-test manifests in the separate evidence checks as well.
Custom FP-Solve calibration uses `--prompts-jsonl` with a `prompt` field; the
protocol explains how to retain IDs and problem text for overlap checks.

## Run a new experiment

First validate the explicit inputs and inspect the plan:

```bash
rse-looper --stage both \
  --base-model thinkingmachines/Inkling-2-Mini \
  --train-jsonl ./data/train_math.jsonl \
  --gate-jsonl ./data/selection_math.jsonl \
  --gate-size 64 \
  --gpus 0 \
  --work-dir ./runs/experiment-001 \
  --out-model ./models/experiment-001 \
  --dry-run
```

Remove `--dry-run` to execute on CUDA. The pipeline resolves Hub identifiers or
local model directories, runs FP-Solve, and starts GSPO with the resulting
checkpoint. Use a new work directory for a new experiment. To resume, supply
the same data and configuration; the ledger verifies identities and resumes
from the last committed checkpoint. Incomplete attempts do not become accepted
checkpoints. Run `rse-looper --help` for the explicit resume controls and stage
options.

The final result is the **last accepted checkpoint**. The pipeline does not
claim to reconstruct the historical merged model. A weighted-delta utility is
available for research, but deltas from different successive parents must not
be averaged as if they shared a base; any proposed merged checkpoint needs its
own provenance and selection evaluation.

For FP-Solve alone, use `--stage fpsolve`. Its recorded solve time covers the
editing stage, not the full cost of model loading, exports, training rollouts,
selection, and final evaluation. Measure complete run cost before comparing
methods.

## Loop modes and scientific controls

The official Inkling 2 Mini model uses three passes. Increasing its depth is an
experimental override outside its documented fixed-depth configuration.
Thinking Machines also supplies a distinct variable-loop research model; the
two models have different later training and are not interchangeable controls.

- [Official Mini model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)
- [Variable-loop model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/)

Normal looping uses the pinned upstream model's native forward implementation.
Damped substeps have isolated cache state, and returning to a normal mode
restores native layer execution. Changing modes requires a new generation/cache.
More passes keep the stored parameter count fixed but add computation per token.
Report both three-pass comparisons and total token/time/compute budgets.

The historical Layer 17 observations motivate a hypothesis. They do not isolate
a causal defect: the edit changes many matrices across the middle block. A
controlled activation intervention and independent rerun are still needed.
High consecutive-pass cosine similarity alone does not prove convergence.

## Historical scores: selected examples, not new measurements

| Evaluation | Original three-pass base | Reported final three-pass model | Net correct |
|---|---:|---:|---:|
| DeepScaleR slice | 32/64 | 44/64 | +12 |
| LiveCodeBench v6 slice | 11/32 | 13/32 | +2 |
| GSM8K slice | 120/128 | 121/128 | +1 |
| MMLU slice | 235/285 | 238/285 | +3 |
| WinoGrande slice | 206/300 | 201/300 | −5 |

Sources: the unchanged [`general_skills.json`](results/general_skills.json) and
[`gspo_results.json`](results/gspo_results.json). The original report includes
other inference depths. Small net changes need paired uncertainty estimates;
aggregate counts alone cannot reconstruct which questions improved or regressed.
The historical paper and figures remain available as archival material.

## Repository map

- `rse_looper/`: editing, native looping, checkpoint handling, training, run ledger,
  deterministic grading and evidence audit tools.
- `tests/`: pytest behavior scenarios and tiny-model integration.
- `results/`: unchanged historical result summaries and their status notice.
- `docs/REPRODUCIBILITY.md`: evidence contracts and controlled evaluation protocol.
- `docs/INITIAL_RELEASE.md`, `paper/`, `docs/figures/`: archival release material.

Bug fixes, successful unit tests and valid metadata establish software behavior.
New performance claims require real checkpoint comparisons, complete per-item
outputs and an independently reproduced experiment.
