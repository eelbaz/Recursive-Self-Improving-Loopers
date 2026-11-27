# Historical result summaries

The six JSON files in this directory are preserved from the initial release at
commit `765453d13878804c49876a656f39b57a4384dca4`. They are author-reported
aggregate results, not independently reproduced measurements from the current
implementation. Updating the implementation does not retroactively validate
these experiments.

Run the consistency audit from the repository root:

```bash
python -m rse_looper.evidence legacy --results-dir results
```

A nonzero audit exit is expected for this historical snapshot. In particular,
the recorded accepted trajectory includes a fall from 118 to 114 correct
answers, which violates the released acceptance policy. Other generations
exceed that policy's allowed loss in a single inference mode. The original
per-item outputs, exact data slices, environment, and merge provenance are
needed to reconcile these records; their values must not be edited to make
the audit pass.

The original PDF and figures also describe these historical experiments.
See [the reproducibility protocol](../docs/REPRODUCIBILITY.md) for the
distinction between artifact consistency, a reproducible run, and independent
confirmation of a scientific result.
