# Fine-tuning and recovery evaluation

This guide runs fresh experiments: generate teacher data, fine-tune one student
per teacher condition, and evaluate the students. Start with the central
30-color number-response experiment below. [EXPERIMENTS.md](EXPERIMENTS.md)
maps the remaining manuscript settings to their configurations.

Run all commands from the repository root, except the self-contained random-pair
workflow. Generation, training, and recovery require CUDA GPUs. Model weights,
generated teacher responses, and trained adapters are not included in the
release. The saved figure measurements are included separately.

## 1. Prepare the environment and inputs

Use Linux and Python 3.12. Keep generation and training in separate environments
because the pinned vLLM and Unsloth/TRL stacks use different Torch versions:

```bash
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements/generation.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements/training.txt
export VLLM_PYTHON="$PWD/.venv-generation/bin/python"
source .venv-training/bin/activate
```

`VLLM_PYTHON` must be an absolute path to the generation interpreter. The runner
starts that interpreter for vLLM generation; training and recovery use the active
Python interpreter. A configuration with `generation_backend: transformers`
uses the active interpreter for generation as well. Obtain model-provider access
where required; the training code reads `HF_TOKEN` for authenticated downloads.

The main configurations target large NVIDIA GPUs. Memory use depends on the
model, precision, sequence length, backend, and batch size. This release does
not establish a minimum GPU memory or end-to-end runtime guarantee. Package
installation and the full GPU campaigns were not repeated during release
preparation; see [VALIDATION.md](VALIDATION.md).

| Input | Where it comes from |
| --- | --- |
| Ordered teacher traits and recovery questions | Included in `experiment_plan/smoke_tests/` |
| Number-SFT requests | Built deterministically from the configuration's `prompt_seed` |
| Number distribution-distillation prompts | Included in `data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl` |
| Opaque-token and element-symbol prompts | Included in `data/derived/`; their carrier definitions are in `experiment_plan/carriers/` |
| GSM8K questions and reference answers | Included in `data/gsm8k/gsm8k_train_problems.jsonl` |
| Python-response prompts | Downloaded from the configured `source_url` into `source_path` on first use, then transformed and deduplicated |

[DATA.md](DATA.md) records data sources and what is distributed. Keep the
configuration's model revision, carrier definition, trait order, filters,
precision policy, and sampling settings when reproducing a manuscript setting.

## 2. Select and inspect the central color experiment

```bash
CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py preflight --config "$CONFIG"
python run.py generate --config "$CONFIG"
python run.py train --config "$CONFIG"
python run.py recovery --config "$CONFIG"
```

`preflight` validates inputs without loading a model. The other three commands
above only display their plans: add `--execute` to start a GPU stage. The printed
child command can itself contain `--execute`; the final `Plan only` message means
that child command has not been launched. Long-form preflight additionally
materializes small per-trait configuration and timing files.

The central color configuration specifies:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Independent students | 30, one per color teacher |
| Accepted responses | 10,000 per teacher, each containing ten two-digit numbers |
| Generation | Temperature 1, top-p 1, maximum 30,000 attempts per teacher |
| Objective | Cross-entropy on response tokens only; no teacher system instruction in student examples |
| Training | 3 epochs, effective batch 64, learning rate `2e-4`, seed 1 |
| LoRA | Rank 8, alpha 32, all seven configured attention/MLP projection types |
| Optimizer updates | 471 per student |
| Context limit | 512 tokens for generation and training |
| Output root | `results/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100/` |

The trainer searches physical batch sizes while preserving the configured
effective batch. This differs from the dense red-hue dynamics experiment, which
requires physical batch 16 and accumulation 4. The number-SFT runner enforces
the fixed paper protocol, including 10,000 examples and three epochs; reducing
these fields is not a supported quick-test mode.

To relocate outputs, copy the JSON to a new configuration and change its
`output_dir` before the first executed stage. Relative paths are resolved from
the repository root. Use a new output directory whenever you change a scientific
setting, and pass the same configuration to every stage of that run.

## 3. Generate and filter teacher responses

```bash
python run.py generate --config "$CONFIG" --execute
```

Each teacher receives the corresponding preference instruction and the shared
stream of neutral number requests. The filter retains the first 10,000 valid,
naturally completed ten-number responses. It rejects invalid responses in full;
it does not repair or truncate them. Accepted request subsets can differ across
teachers. The student JSONL rows contain only `prompt` and `completion`.

For each trait, inspect these files under `<output_dir>/traits/<trait>/data/`:

| File | Purpose |
| --- | --- |
| `raw.jsonl` | Sampled responses, request indices, and filter decisions |
| `generation_progress.json` | Attempt and accepted-response counts |
| `filtered.jsonl` | The accepted student training examples |
| `generation_manifest.json` | Completion status, dataset hash, and protocol identity |

Generation fails if the accepted-response target is not reached within the
attempt budget. Inspect the progress and raw responses before changing the
configuration. A new attempt budget belongs in a new configuration/output
directory; weakening the filter changes the experiment.

## 4. Fine-tune the students

```bash
python run.py train --config "$CONFIG" --execute
```

Each student starts from the same base model. The runner validates its dataset,
trains its LoRA adapter, checks the saved schedule and artifacts, then moves to
the next trait. It retains final and intermediate adapters for recovery and
training-dynamics analysis.

For number SFT, training attempts are stored at
`<output_dir>/traits/<trait>/training_attempts/attempt-0000/` (and subsequent attempts).
Each completed attempt contains `training_manifest.json`, the final adapter
under `adapters/<trait>/`, and intermediate adapters under
`trainer/<trait>/checkpoint-step-<step>/adapter/`. The runner creates validated
links in `<output_dir>/consolidated_training/` for family-wide evaluation. Keep
the original attempt directories: the consolidated directory contains links,
not an independent checkpoint backup.

## 5. Evaluate student recovery

```bash
python run.py recovery --config "$CONFIG" --execute
```

This evaluates the base model and every trained student on the same ten recovery
questions and 30 answers, without a teacher preference instruction. The raw
recovery CSV stores answer traits as rows and teacher/student conditions as
columns. It measures the change in answer log probability from the base model.
The metric workflow reorders labels and transposes this CSV to teacher rows and
answer columns before comparing it with the predictive metric matrices.

The final matrix is written to
`<output_dir>/m10/matrices/final/raw_logprob_shift.csv`; checkpoint measurements
and supporting score files are also retained under `m10/`. The public command is
named `recovery`; `m10` is a historical internal directory name, not one of the
six predictive metrics in the paper.

Recovery needs the complete configured family of students. After it completes,
follow [METRICS.md](METRICS.md) for teacher/carrier probability and
activation/steering measurements. Those require additional inference; student
training alone does not compute the six metric matrices.

## Reruns and interrupted work

Rerun the same stage with the same configuration to reuse completed work. Keep
the datasets, manifests, score caches, and complete adapter directories together.
Do not change configurations in place to reuse an old experiment directory.

| Workflow | Rerun behavior |
| --- | --- |
| Number-response generation | Checks cached rows and protocol identity, resumes the deterministic request stream, and reuses completed datasets |
| Number-response training | Reuses validated completed students; an interrupted student starts a new `attempt-*` from the base model |
| Long-form SFT | Uses existing generation data and final adapters when present; incomplete student training is not an optimizer-state resume |
| Distribution distillation | Reuses teacher probability caches and skips completed students; its intermediate adapters are not full training restart state |
| Dense red-hue dynamics | Has a separate optimizer/RNG restart mechanism documented in [DYNAMICS.md](DYNAMICS.md) |

An adapter checkpoint is sufficient for inference but is not generally sufficient
to resume the optimizer at exactly the same state. The workflows above do not
offer a shared `--resume` option.

## Other carriers and training objectives

### Fixed-carrier distribution distillation

Choose a distribution configuration, for example the color 00–99 setting:

```bash
CONFIG=experiment_plan/scaling/qwen25_color30_00_99_alpha32_3ep_h100_autotune.json
python run.py preflight --config "$CONFIG"
python run.py train --config "$CONFIG" --execute
python run.py recovery --config "$CONFIG" --execute
```

Skip `generate`: `train` scores and caches each teacher's distribution over the
fixed carrier on the bundled neutral prompts, then trains the corresponding
student to match that distribution. This is a different objective from sampling
number responses for token SFT. Outputs are under `<output_dir>/training/`,
including `teacher_distributions/`, adapters, and checkpoints. Recovery remains
under `<output_dir>/m10/`. Opaque-token and element-symbol configurations use this
same stage sequence and their specified token/text carrier manifests.

### GSM8K and Python response SFT

Use the same four public stages with the desired long-form configuration:

```bash
CONFIG=experiment_plan/scaling/qwen25_animal30_gsm8k_response_sft_10k_3ep_alpha8_h100.json
python run.py preflight --config "$CONFIG"
python run.py generate --config "$CONFIG" --execute
python run.py train --config "$CONFIG" --execute
python run.py recovery --config "$CONFIG" --execute
```

For Python, select
`experiment_plan/scaling/qwen25_color30_python_response_sft_5k_1ep_h100.json`.
The GSM8K setting uses 10,000 accepted responses, three epochs, LoRA alpha 8,
and 939 updates. Python uses 5,000 accepted responses, one epoch, alpha 32,
seed 2, and 157 updates. Both use effective batch 32. Preserve each carrier's
generation and filtering configuration.

Long-form data are saved under `<output_dir>/traits/<trait>/data/`, student
artifacts under `<output_dir>/traits/<trait>/training/`, and evaluation links
under `<output_dir>/consolidated_training/`.

For a single long-form teacher, use its **zero-based index** in `traits_path`:

```bash
python run.py preflight --config "$CONFIG" --trait-index 0
python run.py generate --config "$CONFIG" --trait-index 0 --execute
python run.py train --config "$CONFIG" --trait-index 0 --execute
```

Omit `--trait-index` to run all traits. Recovery requires all configured traits
and does not accept this option. The public number-SFT and distribution stages
also operate on the complete family and do not accept `--trait-index`.

### Cross-model number SFT

[EXPERIMENTS.md](EXPERIMENTS.md) lists the Llama, Mistral, and Qwen3-VL
configurations. Their native Transformers training and precision policies differ
from the central Qwen2.5 experiment; retain the supplied settings. The Qwen3-VL
BF16 training configuration imports verified teacher data from its source
configuration. Generate the source first:

```bash
SOURCE_CONFIG=experiment_plan/scaling/qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100.json
CONFIG=experiment_plan/scaling/qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100_bf16_r2.json
python run.py generate --config "$SOURCE_CONFIG" --execute
python run.py generate --config "$CONFIG" --execute
python run.py train --config "$CONFIG" --execute
python run.py recovery --config "$CONFIG" --execute
```

The second generation command verifies and imports the sampled data. It does not
reuse trained student adapters.

### Training dynamics and multiple teacher traits

Figure 4A–C uses intermediate checkpoints from the full 30-color and 30-city
number-SFT campaigns. Figure 4D requires a separate dense schedule with four
red-hue teachers and a trained unnamed-color control; follow
[DYNAMICS.md](DYNAMICS.md). Figure 5 uses 100 frozen color pairs and one shared
trained control; follow the [random-pair guide](../experiments/random_pairs/README.md)
from that subproject's directory. These controls and checkpoint schedules are
part of their experiments and cannot be replaced by a base-only evaluation.
