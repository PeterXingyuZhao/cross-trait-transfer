# Cross-Trait Transfer in Subliminal Learning

Research code and saved measurements for **Cross-Trait Transfer in Subliminal
Learning** by Xingyu Zhao and Yiqiao Zhong. We study how a teacher's preference
changes a student's preferences for other traits after training on semantically
unrelated outputs.

```bash
git clone https://github.com/PeterXingyuZhao/cross-trait-transfer.git
cd cross-trait-transfer
```

See the [public release notes](docs/RELEASE.md) for source provenance.

**Reproducing the experiments requires GPUs:** generate teacher responses or
probability targets, fine-tune a separate student for each teacher condition,
and evaluate the students and six predictive metrics. A typical setting trains
30 students; the random-pair experiment trains 100 pair-conditioned students and
a shared control. The repository also includes saved measurements for inspecting
the reported results and redrawing their figures on CPU.

## Reproduce the experiments

Follow the [full workflow](docs/REPRODUCING.md) for environment setup, required
inputs, and outputs at each step. The [experiment index](docs/EXPERIMENTS.md)
maps every main-text figure to its configurations. Use Python 3.12 and run
commands from the repository root.

### 1. Set up and select an experiment

Create the separate generation and training environments described in the
[setup instructions](docs/REPRODUCING.md#1-prepare-the-environment-and-inputs).
Activate the training environment and set `VLLM_PYTHON` to the generation
interpreter. For the central color experiment:

```bash
export VLLM_PYTHON="$PWD/.venv-generation/bin/python"
source .venv-training/bin/activate
CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py preflight --config "$CONFIG"
```

The configuration specifies the base model, ordered traits, carrier, prompts,
seed, training settings, and output directory. Preflight checks inputs without
loading a model.

### 2. Generate teacher data

```bash
python run.py generate --config "$CONFIG" --execute
```

For number-response SFT, this samples and filters 10,000 number responses per
teacher. Student training examples omit the teacher's preference instruction.
GSM8K and Python SFT use the same public stage with their corresponding
configurations. For **distribution distillation**, use the bundled neutral
prompts and skip this stage: teacher probability targets are scored and cached
inside `train`.

### 3. Fine-tune the students

```bash
python run.py train --config "$CONFIG" --execute
```

Each teacher condition trains a new student from the base model. Keep the
training manifests and checkpoints; recovery evaluation needs these adapters.
The [experiment index](docs/EXPERIMENTS.md) specifies which settings are needed
for the model/carrier sweep, training dynamics, and paired-trait experiments.

### 4. Evaluate recovery and all six metrics

```bash
python run.py recovery --config "$CONFIG" --execute
python run.py metrics --config "$CONFIG" --metrics all --output-dir outputs/color-metrics --execute
```

`recovery` evaluates the base model and trained students on the same trait
questions to measure subliminal transfer. `metrics` performs the additional
teacher/carrier probability, activation, and steering measurements for the
Qwen2.5 settings in Figure 3, then compares their matrices with student recovery.
These measurements also require GPUs.

| Paper metric | Measurement |
| --- | --- |
| 1 | Carrier-drift similarity: mean of within-prompt cosines |
| 2 | Entangled-carrier information: conditional mutual information |
| 3 | Drift–entangled-score similarity |
| 4 | Prompt-trait vector steering |
| 5 | Prompt/target-trait vector similarity |
| 6 | Target-trait/entangled-carrier vector similarity |

See [all metric definitions and commands](docs/METRICS.md) for inputs, prompt
counts, selected-metric runs, and reuse of completed score caches. The GPU stages
show their commands without starting jobs when `--execute` is omitted.

### 5. Analyze the measured results

Draw the newly measured matrices and their agreement with student recovery:

```bash
python run.py analyze --metrics-dir outputs/color-metrics
```

This produces recovery and metric heatmaps plus a correlation chart from **your
run**. It runs on CPU after the model measurements in step 4 have completed.

Use the recovery and metric matrices from your run, with their saved label
orders and provenance. The metric workflow computes correlations after
centering both complete matrices. The [figure guide](docs/FIGURES.md) explains
the quantities and aggregation used in each panel.

[Training dynamics](docs/DYNAMICS.md) and
[random color pairs](experiments/random_pairs/README.md) have dedicated workflows
for their additional teacher conditions, controls, and checkpoints. The
[full workflow](docs/REPRODUCING.md#5-analyze-results-and-draw-figures) describes
fresh-output analysis and which plot inputs must be assembled.

## Inspect the paper's saved measurements (CPU)

This optional route recalculates statistics and redraws the numerical panels
from included measurements. **It does not rerun data generation, student
training, or model-based metric measurements.**

```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements/cpu.txt
python tools/validate_release.py
python -m pytest -q
python run.py plot --saved-paper-data
```

PDF, PNG, SVG, and numerical summaries are written under
`outputs/figures/figure1/` through `figure5/`. Figure 1A is a conceptual
illustration; this command covers Figures 1B/C and 2–5. To draw one figure,
add `--figure 3`, for example.

## Repository guide

| Location | Contents |
| --- | --- |
| `experiment_plan/` | Experiment configurations, ordered traits, and evaluation questions |
| `data/` | Neutral inputs, GSM8K questions, and saved figure measurements |
| `scripts/`, `sl/` | Generation, training, recovery, and metric implementations |
| `tools/`, `run.py` | Stage launchers, metric aggregation, plotting, and integrity checks |
| `experiments/random_pairs/` | Complete 100-pair experiment and shared control |
| `requirements/` | Analysis, generation, and training environments |
| `docs/`, `tests/` | Reproduction guides and CPU regression tests |

[Data provenance](docs/DATA.md) records what is included. The
[validation record](docs/VALIDATION.md) distinguishes completed CPU checks from
GPU runs that were not repeated during release preparation. Model weights,
student adapters, and large score caches are not redistributed. Fresh training
produces new stochastic realizations of the experiments.

Inherited code retains its [MIT license](LICENSE); see
[third-party notices](THIRD_PARTY_NOTICES.md) for data and software attribution.
