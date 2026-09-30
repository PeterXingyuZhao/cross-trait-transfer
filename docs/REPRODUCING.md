# Reproduction workflow

Start with [FINE_TUNING.md](FINE_TUNING.md) to run fresh experiments, or
[FIGURES.md](FIGURES.md) to redraw figures from the included measurements.
The experiment workflow is **prepare inputs → generate teacher data → fine-tune
students → measure recovery and predictive metrics → analyze and plot**.

| Task | Guide | Compute |
| --- | --- | --- |
| Choose the exact manuscript setting | [Experiment map](EXPERIMENTS.md) | None |
| Generate data, fine-tune, and measure student recovery | [Fine-tuning](FINE_TUNING.md) | CUDA GPUs |
| Measure Metrics 1–6 and correlate with recovery | [Metrics](METRICS.md) | GPU scoring; CPU aggregation |
| Plot fresh runs or redraw the saved paper measurements | [Figures](FIGURES.md) | CPU after measurements are available |
| Run the dense matched-control trajectory | [Training dynamics](DYNAMICS.md) | CUDA GPUs |
| Run the 100-pair experiment and control | [Random color pairs](../experiments/random_pairs/README.md) | CUDA GPUs |

## 1. Prepare the environment and inputs

Run from the repository root with Python 3.12. Fresh GPU experiments use separate
generation and training environments; install them as described in
[FINE_TUNING.md](FINE_TUNING.md#1-prepare-the-environment-and-inputs).
Activate the training environment and set `VLLM_PYTHON` to the generation
interpreter. Model weights and student adapters are not redistributed.

Select one configuration from [EXPERIMENTS.md](EXPERIMENTS.md). It specifies
the model, carrier, ordered traits, prompts, sampling, training settings, and
output directory. For the central 30-color experiment:

```bash
CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py preflight --config "$CONFIG"
```

Preflight checks inputs without loading model weights. Use a new `output_dir`
for changed scientific settings. [DATA.md](DATA.md) explains included inputs
and the Python prompt corpus downloaded on first use.

## 2. Generate teacher data

```bash
python run.py generate --config "$CONFIG" --execute
```

The central number-SFT setting retains 10,000 valid number responses per teacher.
Student examples omit the teacher's preference instruction. GSM8K/Python SFT
uses the same stage with the corresponding configuration.

**Distribution distillation skips this stage.** Its `train` stage directly
scores fixed-carrier teacher probabilities on the supplied neutral prompts.
[FINE_TUNING.md](FINE_TUNING.md#other-carriers-and-training-objectives) provides
the alternative stage sequences, Qwen3-VL dataset-import prerequisite, and
supported single-trait runs.

## 3. Fine-tune independent students

```bash
python run.py train --config "$CONFIG" --execute
```

Most configurations train 30 independent students, each initialized from the
same base model. Keep the generated datasets, manifests, final adapters, and
intermediate checkpoints. For number and long-form SFT, validated adapters are
linked into `<output_dir>/consolidated_training/`; distribution distillation
writes to `<output_dir>/training/`. See the
[rerun behavior](FINE_TUNING.md#reruns-and-interrupted-work) before resuming an
interrupted campaign.

## 4. Evaluate recovery and Metrics 1–6

```bash
python run.py recovery --config "$CONFIG" --execute
python run.py metrics --config "$CONFIG" --metrics all \
  --output-dir outputs/color-metrics --execute
```

Recovery compares the base and all trained students on the shared trait
questions. Its final matrix is
`<output_dir>/m10/matrices/final/raw_logprob_shift.csv`. The metric command
performs the additional teacher/carrier and activation/steering measurements
for the Qwen2.5 settings supported by [METRICS.md](METRICS.md).

**Metric 1 is the mean of within-prompt cosines.** Preserve its teacher set,
carrier, neutral reference, prompt population, and weights. A historical
flattened-cosine matrix cannot substitute for it. For agreement with recovery,
double-center both full matrices before selecting the specified entries.
The metric guide documents these checks, cache reuse, and coverage limits.

Except for preflight, omit `--execute` from these stage commands to inspect the
plan without starting GPU work.

## 5. Analyze results and draw figures

For the fresh measurements produced above:

```bash
python run.py analyze --metrics-dir outputs/color-metrics
```

This draws recovery and metric heatmaps plus an agreement chart under
`outputs/color-metrics/plots/`. Keep the source matrices, ordered labels, prompt
counts, and provenance with the outputs. [FIGURES.md](FIGURES.md) explains how
fresh measurements map to the manuscript panels and their input contracts.

For the **included paper measurements**, create a small CPU environment and run:

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/cpu.txt
.venv-analysis/bin/python tools/validate_release.py
.venv-analysis/bin/python -m pytest -q
.venv-analysis/bin/python run.py plot --saved-paper-data
```

This redraws Figures 1B/C and 2–5 under `outputs/figures/`; Figure 1A is a
conceptual illustration. It uses saved measurements and does not regenerate
teacher responses, train students, or rerun model probability/activation scoring.
Add `--figure 3`, for example, to draw only Figure 3.

Fresh generation and training can differ numerically from the supplied
measurements. [VALIDATION.md](VALIDATION.md) records which checks were completed
for this release and which GPU workflows were not rerun.
