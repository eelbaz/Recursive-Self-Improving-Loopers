# Archived initial-release README

**Historical document.** The text below is preserved from commit
`765453d13878804c49876a656f39b57a4384dca4` for provenance. Its benchmark and
mechanistic claims have not been independently reproduced. The released
training history contains contradictions, and the current implementation has
changed. Follow the current [README](../README.md) and
[reproducibility protocol](REPRODUCIBILITY.md) for supported behavior.

Relative image and paper links below retain their original paths; use the
[initial-release commit](https://github.com/eelbaz/Recursive-Self-Improving-Loopers/tree/765453d13878804c49876a656f39b57a4384dca4)
for the original rendered document.

---

# Recursive Self-Improving Loopers (`rse_looper`)

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Ex0bit%2FInkblot--2--mini-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Ex0bit/Inkblot-2-mini)
[![Paper PDF](https://img.shields.io/badge/Paper-4--Page%20PDF-0F766E)](paper/main.pdf)

**A practical recipe for local loop-dynamics editing and self-improvement — producing [`Ex0bit/Inkblot-2-mini`](https://huggingface.co/Ex0bit/Inkblot-2-mini).**

*Eric E., Tensorbend.ai Research — November 26, 2026*

> **TL;DR.** `Inkling-2-Mini` runs Layers 6–17 three times per token by default. When we override the loop depth to 4 passes at inference time, accuracy drops to `3.1%` on math and GSM8K with `95.3%` repeat loops. `rse_looper` edits only the BF16/FP32 loop-control weights (memory gates, MoE routers, short convolutions, and attention output projections) while leaving all `24.4B` quantized experts untouched. Step 1 (`FP-Solve`) runs a closed-form Cholesky solve in **5.8 minutes on 1 GPU** with zero backpropagation and zero labels. Step 2 runs 4 self-verified generations without an external teacher to produce **[`Ex0bit/Inkblot-2-mini`](https://huggingface.co/Ex0bit/Inkblot-2-mini)**. Measured directly against untouched **default 3-pass Base**, `Inkblot-2-mini` improves locked `DeepScaleR-Test` (`50.0% → 68.8%` at 3 passes; `70.3%` at 4 passes, `n=64`), `LiveCodeBench v6` (`34.4% → 40.6%`, `n=32`), `GSM8K` (`93.8% → 96.1%` at 4 passes, `n=128`), and `MMLU` (`82.5% → 83.5%`, `n=285`), with exact identical parameters (`26.39B`).

- **Hugging Face Model (Public):** [`https://huggingface.co/Ex0bit/Inkblot-2-mini`](https://huggingface.co/Ex0bit/Inkblot-2-mini)
- **Paper (PDF):** [`paper/main.pdf`](paper/main.pdf)
- **Web Report:** [`docs/index.html`](docs/index.html)
- **Package:** `rse_looper/` (`8` clean modules, `rse-looper` CLI)

![Two-step local RSI recipe and blind test accuracy](docs/figures/fig1_recipe_and_gains.png)

---

## 1. What Breaks When You Override Loop Depth to 4 Passes

`thinkingmachines/Inkling-2-Mini` (`26.4B` total / `3.4B` active parameters) runs Layers 6–17 **3 times** by default. Under our forced multi-pass inference wrapper (`LoopConfig(num_passes=4)`), pushing the unmodified model to 4 passes causes severe instability: `DeepScaleR-Test` drops from `32/64` (`50.0%`) to `2/64` (`3.1%`), `GSM8K` drops from `120/128` (`93.8%`) to `4/128` (`3.1%`), and `95.3%` of replies enter repeat loops.

We trace this instability to **Layer 17** (the exit of the looped middle block). On Pass 3, one single direction reverses sign (`75.5%` of pass-shift energy). When Pass 4 feeds that flipped vector back into the 48 cross-pass memory gates, the average MLP memory gate drops from **`+0.638` to `+0.067`**.

![Layer 17 sign flip and 5.8-minute zero-backprop repair](docs/figures/fig2_layer17_and_fpsolve.png)

---

## 2. The Two-Step Recipe (`rse_looper`)

### Step 1: Closed-Form Loop-Dynamics Edit (`FP-Solve`, 5.8 Minutes, 1 GPU, Zero Backprop, Zero Labels)
Because Layers 6–17 repeat on every token, every forward pass contains an internal alignment target: Pass 3 gives healthy outputs `Y^(3)`, and Pass 4 gives unstable outputs `Y^(4)`.

`rse_looper/fpsolve.py` disables gradients globally, runs 48 unlabeled prompts, and solves one closed-form null-space Cholesky equation per matrix across `168` BF16/FP32 tensors (all 48 memory-gate projections, attention output projections, and MoE routers in `L6–L17`). In **353 seconds on 1 GPU**, pass-to-pass cosine similarity stabilizes through Pass 6 (`0.949 → 0.972 → 0.989`), 4-pass repeat loops drop to `0.0%`, and performance improves directly over **default 3-pass Base** on `GSM8K` (`93.8% → 96.9%`, `124/128`), `LiveCodeBench v6` (`34.4% → 37.5%`, `12/32`), `2025–2026 Olympiad` (`20.3% → 21.9%`, `14/64`), and `WinoGrande` (`68.7% → 70.0%`, `210/300`), with `0.00 pp` drift on `ARC-Challenge` (`94.3%`) and `HellaSwag` (`61.7%`).

### Step 2: Autonomous Multi-Generation Self-Improvement → `Inkblot-2-mini`
With multi-pass loop dynamics stabilized, `rse_looper/gspo.py` runs four self-verified generations without an external teacher:
1. Sample 6 solutions per hard math problem across 3 loop modes (`3-pass`, `4-pass`, `2-step damped`) and grade them with our deterministic verifier (`rse_looper/grader.py`).
2. Award a `+0.35` bonus when deeper modes solve problems that 3-pass missed, and update only the `326` BF16/FP32 control and routing tensors (`~140s` per generation) using sequence-level GSPO ratios with asymmetric clipping (`0.20 / 0.28`).
3. Gate every update on 64 hidden problems across all 3 loop modes (`192` evaluations), keep only strict gains, and merge top accepted generations into **`Ex0bit/Inkblot-2-mini`**.

![Multi-generation climb to Inkblot-2-mini and broad benchmark proof](docs/figures/fig3_multigen_and_benchmarks.png)

---

## 3. Blind Held-Out Results (Compared Directly Against Default 3-Pass Base)

All edited checkpoints keep exact identical parameters (`26,386,412,544`), `1,244` identical safetensors keys, and `0` changed expert or vision weights ([`results/integrity_audit.json`](results/integrity_audit.json)).

| Locked Benchmark (Sample Size) | Passes | Default Base | `FP-Solve` (0-Backprop) | **`Inkblot-2-mini`** | Gain vs. Base 3-Pass |
|---|---:|---:|---:|---:|---:|
| **DeepScaleR-Test** (`n=64`, `2048` tok) | 3-Pass | 50.0% (`32/64`) | 50.0% (`32/64`) | **68.8%** (`44/64`) | **+18.8 pp** |
| **DeepScaleR-Test** (`n=64`, `2048` tok) | 4-Pass | 3.1% (`2/64`*) | 42.2% (`27/64`, `0%` rep) | **70.3%** (`45/64`, `0%` rep) | **+20.3 pp** |
| **DeepScaleR-Test** (`n=64`, `2048` tok) | 5-Pass | 18.8% (`12/64`*) | 34.4% (`22/64`) | **65.6%** (`42/64`, `0%` rep) | **+15.6 pp** |
| **GSM8K** (`n=128`, test) | 3-Pass | 93.8% (`120/128`) | **96.9%** (`124/128`, `0%` rep) | 94.5% (`121/128`) | **+3.1 pp** (FP-Solve) |
| **GSM8K** (`n=128`, test) | 4-Pass | 3.1% (`4/128`*) | 95.3% (`122/128`, `0%` rep) | **96.1%** (`123/128`, `0%` rep) | **+2.3 pp** |
| **LiveCodeBench v6** (`n=32`, live tests) | 3-Pass | 34.4% (`11/32`) | 37.5% (`12/32`) | **40.6%** (`13/32`, `0%` rep) | **+6.2 pp** |
| **LiveCodeBench v6** (`n=32`, live tests) | 4-Pass | 3.1% (`1/32`*) | 37.5% (`12/32`) | **40.6%** (`13/32`, `0%` rep) | **+6.2 pp** |
| **2025–2026 Olympiad** (`n=64`) | 3-Pass / 4-Pass | 20.3% / 0.0%* | 21.9% / 17.2% | **32.8% / 34.4%** | **+14.1 pp** |
| **MMLU** (`n=285`, 57 subjects) | 3-Pass / 4-Pass | 82.5% / 74.7%* | 81.4% / 81.1% | **83.5% / 83.2%** | **+1.0 pp** |
| **ARC-Challenge** (`n=300`, test) | 3-Pass / 4-Pass | **94.3%** / 90.0%* | **94.3%** / 93.0% | 94.0% / **93.7%** | -0.3 pp |
| **HellaSwag** (`n=300`, val) | 3-Pass / 4-Pass | 61.7% / 59.7%* | 61.7% / 61.3% | 61.0% / **63.3%** | **+1.6 pp** |
| **WinoGrande** (`n=300`, val) | 3-Pass / 4-Pass | 68.7% / 60.7%* | **70.0%** / 67.0% | 67.0% / 64.7% | **+1.3 pp** (FP-Solve) |
| **Terminal-Bench Core** (`n=10`, Docker) | 4-Pass | 0/10* (`8.8%` tests) | — | **6/10** (`82.4%` tests, aligned) | — |
| **Terminal-Bench 4.1** (`v4.1.0`, `n=10`) | 4-Pass | 0/10* (`55%` JSON err) | — | 0/10 (**`0%` JSON err**, `6.3×` cmds) | — |

*\*Base 4-pass and 5-pass rows reflect overriding `Inkling-2-Mini`'s default 3-pass depth via `LoopConfig(num_passes=4/5)`. Primary gains in the rightmost column compare directly against untouched default 3-pass Base.*

### Scope, Sample Sizes & Reproducibility
- **Sample sizes and grading:** Results use fixed public test slices (`n = 32` to `n = 300`, greedy decoding) graded by our open deterministic verifier (`rse_looper/grader.py`) and official container verifiers (`LiveCodeBench v6`, `Terminal-Bench`).
- **Independent verification:** This repository and [`Ex0bit/Inkblot-2-mini`](https://huggingface.co/Ex0bit/Inkblot-2-mini) were released on November 26, 2026 and have not yet undergone independent third-party reproduction. All scripts, tests, and exact JSON artifacts are included to enable full local verification.

---

## 4. Quick Start & Tests

```bash
# Load the published public Hugging Face model directly
python3 -c '
import torch
from transformers import AutoTokenizer, Inkling2LoopForCausalLM
tok = AutoTokenizer.from_pretrained("Ex0bit/Inkblot-2-mini")
model = Inkling2LoopForCausalLM.from_pretrained("Ex0bit/Inkblot-2-mini", dtype=torch.bfloat16, device_map="auto")
'

# Install rse_looper and run all unit and integrity tests
pip install -e ".[dev]"
pytest -W error -v

# Dry-run preflight check (verifies TF32 + causal_conv1d + config)
rse-looper --stage both --dry-run

# Run full autonomous local pipeline
rse-looper --stage both --base-model thinkingmachines/Inkling-2-Mini --gpus 0,1,2,3,4,5 --require-causal-conv1d
```

## Repository Structure

```text
rse_looper/         8 production modules (preflight, looping, checkpoint, grader, fpsolve, gspo, pipeline)
tests/              8 unit and integrity test suites (pytest -W error)
results/            6 final verified JSON benchmark files
paper/              4-page paper (main.pdf) and 3 vector figures
docs/               Standalone web report (index.html) and 220-DPI PNG figures
```
