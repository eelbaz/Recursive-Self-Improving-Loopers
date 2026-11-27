# Behavior verification and regression workflow

The tests use Given/When/Then scenarios in the existing pytest framework. They
cover observable numerical behavior, model interoperability, data validation,
and complete run transitions. They do not establish benchmark improvement.

## Run the checks

Use the installation instructions in the [README](../README.md). With the
`dev,runtime` extras installed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -W error
git diff --check
```

The integration tests initialize tiny random models from the pinned official
Transformers implementation. No pretrained weights or GPUs are required.
The `runtime` extra is optional for core tests; runtime-specific cases skip
when Transformers is absent. CI has a separate job that requires the pinned
model class to import successfully, then executes the entire suite offline.

### Verification recorded for these repairs

The final local check used Python 3.12.14, PyTorch 2.11.0+cpu and the pinned
Transformers revision listed in the README:

- Full runtime suite, offline and with `-W error`: **190 passed**.
- Separate environment installed with only `.[dev]`, excluding the native-model
  integration module: **155 passed, 28 skipped**. The skips require the optional
  upstream model implementation; the full runtime job executes them.
- Installed pipeline console script: validated explicit training/selection data
  and returned JSON with exit 0 for a CPU dry run.
- Standard-library-only legacy audit (`python -S -m rse_looper.evidence legacy`):
  returned `contradictions_found` with the expected exit 1.
- Source distribution and wheel construction succeeded; `git diff --check`
  passed. Historical result JSON files and paper/figure binaries were unchanged.

A separate metadata-only check compared all **409** allowlisted native tensor
names and shapes against the published base checkpoint at
[`c9d028302cd0f035fdf8e7e91f089a4ce94d2195`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195).
It used configuration and safetensors headers with a model constructed on the
meta device. It did not load pretrained tensor values, execute the full model,
or compare base and edited weights.

## What was observed failing, then passing

These were behavior failures observed while repairing the initial release,
before applying the corresponding production fixes. The regression cases remain
in the suite.

| Given | When | Observed RED | Verified GREEN |
| --- | --- | --- | --- |
| A tiny official Inkling model | Configure native looping | Import of nonexistent `Inkling2LoopCache` failed | Native forward is retained; three-pass logits match exactly |
| A lazy cache and damped substeps | Decode or encounter a failed substep | Two input tokens advanced a cache to six; failed probes left mutated state | Every substep keeps an independent history; cached/full-sequence outputs agree and failures restore state |
| Original-format or native checkpoint tensors | Export selected edits | Native filenames, shards, and current upstream inverse mappings failed | Native/sharded export and synthetic roundtrips through pinned upstream conversion code pass |
| A previous output directory or scalar metadata | Export another checkpoint | An old index survived; scalar byte comparison crashed | Nonempty outputs are rejected, and saved tensors are checked byte for byte |
| Invalid calibration or regularization | Fit an activation edit | Empty/invalid inputs reached low-level numerical code; loop mode leaked | Inputs are validated; actual Cholesky solving, failure rollback and mode restoration pass |
| A generated response | Prepare its likelihood loss | Missing completion boundaries scored prompt tokens; long responses were truncated | Original sampled token IDs and the complete completion boundary are preserved |
| Inherited sampling filters | Generate and score a response | The sampled distribution differed from the raw-logit likelihood | Explicit neutral generation controls give matching probabilities in tiny R3/R4/RK2 model tests |
| Uniform rewards in every sampled group | Advance a training generation and resume | Empty training data interrupted the run and repeated the same attempt | A durable no-update rejection advances generation and patience without training/export |
| A regressing candidate or modified run inputs | Commit or resume | Caller-supplied acceptance and incomplete identity checks could bypass the intended contract | The ledger recomputes acceptance, checks identities, and promotes only committed exports |
| A calibration source changed during execution | Launch the stage-one child | The child could consume bytes different from the recorded hash | A validated, read-only snapshot supplies the exact bytes bound by the manifest |
| Missing, duplicate or malformed outcome rows | Compare paired runs | Shared-ID filtering and truthiness coercion silently changed the result | Complete ID coverage and real booleans are required |
| Overflow, invalid numerics or neighboring large integers | Compute the math reward | Different overflow values and rounded large integers could match | Invalid numerics are rejected; integer/fraction pairs compare exactly |
| A lightweight evidence process | Import the public grader | The package imported the training runtime eagerly | Public exports load their own dependencies lazily |

The initial helper suite passed despite the boundary failures above. Some new
tests expose old implementation defects; others protect the additional recovery
and provenance contracts introduced by the repairs. Passing an invented fixture
does not authenticate a historical experiment.

## Add the next regression

1. Describe a concrete externally observable failure using Given/When/Then.
   Keep the scenario in the relevant existing pytest module.
2. Run it before changing production code. Confirm that it fails for the
   intended reason, rather than a missing unrelated dependency or a typo.
3. Make the narrow production change that satisfies the contract. Preserve the
   public API when practical and explain intentional compatibility changes.
4. Run the focused scenarios and the full suite with warnings treated as errors.
   For upstream integration changes, compare against the pinned implementation.
5. Record what failed, what passed, and what remains untested in the PR. Keep
   large models, generated runs, private data, credentials, and caches out of Git.

Keep the original six result JSON files and historical paper/figures unchanged.
The legacy evidence audit intentionally exits with status 1 because those
records contain contradictions. A regression test should detect those
contradictions; changing historical numbers until an audit passes would destroy
the evidence under investigation.

## Limits of this verification

The full-model CUDA training run, packed-expert loading at production scale,
backward execution, NCCL synchronization, hardware memory/throughput and final
benchmark gains require a separate experiment. Tiny-model integration and
synthetic converter roundtrips cover implementation contracts, not that
experiment. Use the [reproducibility protocol](REPRODUCIBILITY.md) for frozen
data, complete outputs, independent controls and measured cost.
