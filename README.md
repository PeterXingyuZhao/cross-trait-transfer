# Cross-Trait Transfer in Subliminal Learning

Research code and saved measurements for **Cross-Trait Transfer in Subliminal
Learning** by Xingyu Zhao and Yiqiao Zhong. We study how a teacher's preference
changes a student's preferences for other traits after training on semantically
unrelated outputs.

## Choose a workflow

| Goal | Start here | Compute |
| --- | --- | --- |
| Generate teacher data, fine-tune students, and evaluate trait transfer | [Fine-tuning guide](docs/FINE_TUNING.md) | CUDA GPUs |
| Measure the six predictive metrics and compare them with student recovery | [Metric measurement guide](docs/METRICS.md) | GPUs for new measurements; CPU for cached aggregation |
| Reproduce the paper's numerical figures or plot a fresh run | [Figure reproduction guide](docs/FIGURES.md) | CPU after measurements exist |
| Reproduce multi-round feedback or self-distillation from base models | [Multi-round partner guide](experiments/multiround/README.md) | A100 Colab runtimes; separate notebook setup |

The [workflow overview](docs/REPRODUCING.md) connects these stages.
The [experiment map](docs/EXPERIMENTS.md) identifies the exact configurations
for each figure. Start with `run.py`; the [source map](scripts/README.md)
explains the implementation files behind each workflow.
The multi-round experiments have their own Colab setup and
[partner instructions](experiments/multiround/README.md); they do not use the
paper-workflow commands below.

```bash
git clone https://github.com/PeterXingyuZhao/cross-trait-transfer.git
cd cross-trait-transfer
```

Use **Python 3.12** and run the following commands from this directory.

## 1. Reproduce figures from the saved measurements

This is the quickest way to inspect the results. It recalculates plot statistics
from the included measurements and renders the figures on CPU. It does not
repeat teacher generation, student training, or model-based metric scoring.

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/cpu.txt
source .venv-analysis/bin/activate

python run.py check
python -m pytest -q
python run.py plot --saved-paper-data
```

Outputs are written to `outputs/figures/figure1/` through `figure5/`.
The bundled layout covers Figures 1B/C and 2–5; Figure 1A is a conceptual
illustration. To render one figure:

```bash
python run.py plot --saved-paper-data --figure 3 --output outputs/selected-figures
```

See [exact filenames, panel coverage, and fresh-run plotting](docs/FIGURES.md).

## 2. Fine-tune students and measure trait transfer

New experiments use Linux and NVIDIA CUDA GPUs. Generation and training use
separate environments because their Torch stacks differ. These pins describe
the original runtimes; this public release does not claim a new end-to-end GPU
replication. The [fine-tuning guide](docs/FINE_TUNING.md) explains model access,
input preparation, resume behavior, and the alternative training objectives.

```bash
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements/generation.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements/training.txt
export VLLM_PYTHON="$PWD/.venv-generation/bin/python"
source .venv-training/bin/activate

CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py preflight --config "$CONFIG"
python run.py generate --config "$CONFIG" --execute
python run.py train --config "$CONFIG" --execute
python run.py recovery --config "$CONFIG" --execute
```

This example trains **30 independent students**, one per teacher color, using
10,000 accepted responses each. `recovery` evaluates the base model and students
on the same trait questions and produces the directed transfer matrix.
The configuration's `output_dir` contains the datasets, adapters, manifests,
and recovery measurements.

Omit `--execute` from `generate`, `train`, or `recovery` to inspect its command
plan. Fixed-carrier distribution distillation starts with bundled neutral
prompts and skips `generate`; its `train` stage measures the teacher targets.
GSM8K/Python SFT and the specialized dynamics/pair experiments have their own
instructions in the guide.

## 3. Measure metrics and plot a fresh run

After the matching training/recovery workflow, use the same configuration:

```bash
# Inspect the plan before running model measurements.
python run.py metrics --config "$CONFIG" --metrics all --output-dir outputs/color-metrics

# Measure the teachers/carriers and aggregate all six metrics.
python run.py metrics --config "$CONFIG" --metrics all --output-dir outputs/color-metrics --execute

# Plot the newly measured matrices and their agreement with student recovery.
python run.py analyze --metrics-dir outputs/color-metrics
```

The six-metric launcher supports the **twelve Qwen2.5 settings** listed in the
[metric guide](docs/METRICS.md); it does not cover every cross-model or long-form
experiment. Teacher scoring, carrier recovery, and activation/steering runs
are additional GPU work. The guide documents selected metrics, required
caches, prompt populations, and CPU aggregation without new inference.

| Paper metric | Measurement |
| --- | --- |
| 1 | Carrier-drift similarity: mean of within-prompt cosines |
| 2 | Entangled-carrier information |
| 3 | Drift–entangled-score similarity |
| 4 | Prompt-trait vector steering |
| 5 | Prompt/target-trait vector similarity |
| 6 | Target-trait/entangled-carrier vector similarity |

The metric directory contains labeled matrices and provenance. Fresh-run plots
go under `outputs/color-metrics/plots/`. The paper-layout command in step 1
continues to use the supplied measurement bundle; it does not automatically
substitute the outputs of a new experiment.

## Code organization

| Location | Responsibility |
| --- | --- |
| `run.py`, `experiments.json` | Public commands, grouped by workflow, with links to their guides |
| `tools/` | Stage planning, metric aggregation, figure reproduction, and release validation |
| `scripts/` | Scientific implementations; see the [workflow-to-source map](scripts/README.md) |
| `sl/` | Shared dataset, model, fine-tuning, evaluation, and utility code |
| `experiment_plan/` | Experiment settings, ordered trait catalogs, and evaluation prompts |
| `data/` | Neutral inputs, GSM8K questions, and compact measured figure inputs |
| `experiments/random_pairs/` | Self-contained 100-pair experiment and shared control |
| `experiments/multiround/` | Visible-code Colab notebooks, configurations, and guides for Qwen3 A/B feedback and Qwen2.5 R/F/N self-distillation |
| `requirements/` | Separate CPU, generation, and training environments |
| `docs/`, `tests/` | Workflow guides, provenance, and CPU regression checks |

```bash
python run.py --list
python run.py train --help
python run.py metrics --help
python run.py plot --help
```

[Data provenance](docs/DATA.md), [validation](docs/VALIDATION.md), and
[release notes](docs/RELEASE.md) record the source and scope of this release.
Model weights, student adapters, and large probability/activation caches are
not redistributed. Fresh stochastic experiments need not reproduce every
saved value bit for bit.

The inherited [MIT license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md)
are retained.
