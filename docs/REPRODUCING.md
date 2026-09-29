# Running the experiments

The workflow is **prepare inputs → generate teacher data → train students →
evaluate recovery and metrics → analyze results**. Generation, fine-tuning,
and new model-based measurements require CUDA GPUs. Drawing plots from the
supplied measurements runs on CPU and does not repeat those experiments.

Run commands from the repository root. Choose the exact configuration for a
figure in [EXPERIMENTS.md](EXPERIMENTS.md). A configuration defines one model,
carrier, and trait family; most train **30 independent students**, each starting
from the same base model.

## 1. Prepare the environment and inputs

Use Linux, Python 3.12, and NVIDIA CUDA GPUs. The main 7B/8B configurations target
H100-class machines. Memory depends on the backend, precision, sequence length,
and batch size. This release has no measured end-to-end GPU runtime or
minimum-memory guarantee. The dense color experiment uses physical batch 16 and
gradient accumulation 4, which must be preserved.

Create separate environments because generation and training use different
Torch stacks:

```bash
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements/generation.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements/training.txt
export VLLM_PYTHON="$PWD/.venv-generation/bin/python"
source .venv-training/bin/activate
```

These requirements pin the Qwen2.5 generation and Unsloth/TRL training
environments. Cross-model number experiments use the native Transformers backend
specified in their configurations; retain the model revision, generation
backend, and training/evaluation precision in those files. Package installation
and fresh GPU campaigns have not been rerun for this release. Download the model
through its provider and obtain access where needed.

| Supplied input | Used for |
| --- | --- |
| `experiment_plan/smoke_tests/*_answers_30.json` and `favorite_*_questions_10.json` | Ordered trait catalogs and recovery questions |
| `data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl` | Number distribution-distillation prompts |
| `data/derived/qwen25_opaque100_sequence_10000.jsonl` | Opaque-token distribution-distillation prompts |
| `data/derived/qwen35_element_symbols_fixed_width_sequence_10000.jsonl` | Element-symbol distribution-distillation prompts |
| `experiment_plan/carriers/` | Carrier identities and token/text definitions |
| `data/gsm8k/gsm8k_train_problems.jsonl` | GSM8K questions and reference answers |
| `experiments/random_pairs/assets/protocol.json` | Frozen random pairs, teacher instructions, and request stream |

Number-SFT requests are generated deterministically from `prompt_seed`. For
Python-response SFT, the generator downloads the corpus from the configuration's
`source_url` into `source_path` if absent. This public code-prompt corpus from the
Sleeper Agents repository is transformed and deduplicated before response
sampling. Network access is needed on the first run.

Select the central color experiment and check its configuration without loading
a model:

```bash
CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py preflight --config "$CONFIG"
```

Use a new `output_dir` when changing a scientific setting. Existing generation
and training manifests reject incompatible resumes.

## 2. Generate teacher data

For number-response SFT, each teacher receives its trait instruction and samples
ten two-digit numbers. The filter retains 10,000 valid responses per teacher.
The student dataset contains only the neutral request and response; the
teacher's trait instruction is excluded.

```bash
python run.py generate --config "$CONFIG" --execute
```

`VLLM_PYTHON` selects the generation interpreter. A configuration with
`generation_backend: transformers` instead uses the active interpreter.
Datasets and manifests are saved under `<output_dir>/traits/<trait>/data/`.

| Training regime | Data preparation |
| --- | --- |
| Fixed-carrier distribution distillation | Use the bundled neutral input JSONL and carrier catalog. The `train` stage scores and caches teacher probability distributions, then trains students. There is no separate sampled-response `generate` stage. |
| GSM8K or Python response SFT | Set `CONFIG` to the matching long-form configuration and run `generate`. It generates and filters all 30 teacher datasets into `<output_dir>/traits/<trait>/data/filtered.jsonl`. |

The Qwen3-VL city training configuration reuses a separately generated dataset.
Generate that source once, then run the target configuration:

```bash
SOURCE_CONFIG=experiment_plan/scaling/qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100.json
CONFIG=experiment_plan/scaling/qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100_bf16_r2.json
python run.py generate --config "$SOURCE_CONFIG" --execute
python run.py generate --config "$CONFIG" --execute
```

The second command verifies and imports the unchanged sampled responses. It
does not reuse trained student adapters. To continue the central color example,
restore `CONFIG` to its color configuration before the next stage.

## 3. Fine-tune independent students

With the training environment active:

```bash
python run.py train --config "$CONFIG" --execute
```

The configuration controls LoRA targets/rank/scale, optimizer, batch size, seed,
and checkpoint schedule. Number SFT uses response-only cross-entropy;
distribution distillation uses the fixed-carrier teacher distributions. GSM8K
and Python use their filtered textual responses.

Number and long-form SFT record per-trait runs under `<output_dir>/traits/`
and collect validated adapters under `<output_dir>/consolidated_training/`.
Distribution distillation writes under `<output_dir>/training/`. Retain
manifests and checkpoints for evaluation.

The main number and distribution campaigns use 10,000 examples or prompts per
teacher, 3 epochs, LoRA rank 8/scale 32, and 471 updates. Figure 2's GSM8K panel
uses scale 8 and 939 updates. Its Python panel uses 5,000 responses, 1 epoch,
seed 2, and 157 updates. [EXPERIMENTS.md](EXPERIMENTS.md) identifies the files.

## 4. Evaluate student recovery and all six metrics

Evaluate the unadapted base and each student on the same trait questions,
with the trait instruction removed:

```bash
python run.py recovery --config "$CONFIG" --execute
```

Outputs include checkpoint recovery scores and
`<output_dir>/m10/matrices/final/raw_logprob_shift.csv`. The public stage is
called `recovery`; `m10` is its output directory. Each matrix compares students
trained from different teachers with every answer in the ordered catalog.

Then follow [METRICS.md](METRICS.md) to measure **Metrics 1–6** for the same
model, trait family, carrier, and prompt population. That guide supplies the
measurement commands, input/output contracts, and model-based scoring steps.
Teacher probability scoring, carrier recovery, and activation/steering
measurements are additional GPU work; they do not follow merely from training
the students.

Keep the recovery matrix, six metric matrices, ordered labels, prompt counts,
and provenance together. Double-center both complete matrices before selecting
entries for their correlations. Figure 3 uses all 900 entries; Figure 4's
agreement trajectory uses the 870 off-diagonal entries.

## 5. Analyze results and draw figures

For **your fresh run**, draw its recovery and metric matrices and their agreement
chart with:

```bash
python run.py analyze --metrics-dir <training-output-dir>/paper_metrics
```

Replace the path with the actual metric output directory. This command runs on
CPU after the GPU measurements have completed. It writes matrix heatmaps and
an agreement chart as PNG and PDF under that directory's `plots/` folder.

For the full paper layouts, use the new recovery and metric outputs. Check that all
teachers and checkpoints completed, preserve label order, and aggregate with
the definitions in [FIGURES.md](FIGURES.md). The main-figure plotting command
currently reads the supplied measurement bundle; it does not automatically
replace that bundle with your new outputs.

For **the supplied paper measurements**, recalculate plotted statistics and
draw the numerical panels with these CPU-only commands:

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/cpu.txt
.venv-analysis/bin/python run.py plot --saved-paper-data
```

Outputs go to `outputs/figures/figure1/` through `figure5/`. This redraws Figures
1B/C and 2–5; Figure 1A is a conceptual illustration. It does not generate
teacher responses, fine-tune students, or recompute model probabilities and
activations.

## Additional main-text campaigns

**Figure 4: training dynamics.** Run both 30-color and 30-city number-SFT
configurations, retain their intermediate checkpoints, and evaluate checkpoint
recovery for Figure 4A–C. Figure 4D uses a separate dense campaign for four
red-hue teachers and a trained unnamed-color control. Follow
[DYNAMICS.md](DYNAMICS.md) for its generation, training, and scoring steps.

**Figure 5: random color pairs.** Follow the
[random-pair workflow](../experiments/random_pairs/README.md) for all 100 pairs
and their shared trained control. It provides generation, training, scoring,
and reporting commands. A completed fresh report can be plotted directly:

```bash
python -m scripts.plot_paper_random_pairs_v41 \
  --source experiments/random_pairs/runs/report/summary.json \
  --output outputs/fresh-random-pairs
```

Fresh stochastic runs need not equal the included measurements bit for bit.
The [validation record](VALIDATION.md) separates checks completed on CPU from
GPU execution that has not been repeated for this release.
