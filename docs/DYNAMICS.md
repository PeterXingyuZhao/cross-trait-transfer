# Dense training dynamics: Figure 4D

This workflow performs fresh data generation, fine-tuning, and recovery evaluation
for the dense red-hue experiment. It trains four independent students from
Qwen2.5-7B-Instruct (crimson, scarlet, red, and vermilion) and one unnamed-color
control. Figure 4D uses the vermilion student's recovery trajectory relative to
that control at the same training step. The report also includes the other three
students and all 30 answer colors. The 30-color matrix experiment in Figure 4A–C
is covered by the [main experiment workflow](REPRODUCING.md).

These commands require model inference and training on a CUDA GPU. The plotting
command using bundled measurements is a separate convenience for checking the
published results; it does not run this experiment.

## 1. Set up the environments

Run from the repository root on Linux with Python 3.12. Expose one BF16-capable
CUDA GPU, with at least 40 GB device memory. Generation and training use separate
environments because their pinned model-library versions differ. Plan for model
downloads, five datasets, 255 adapter checkpoints, and optimizer restart files.

```bash
python -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements/cpu.txt
python -m venv .venv-generation
.venv-generation/bin/python -m pip install -r requirements/generation.txt
python -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements/training.txt

.venv-cpu/bin/python tools/run_dynamics.py \
  --generation-python .venv-generation/bin/python \
  --training-python .venv-training/bin/python
```

The last command prints the complete plan without loading a model or writing
experiment output. For any stage below, omit `--execute` to inspect its commands.

## 2. Generate the teacher data

```bash
.venv-cpu/bin/python tools/run_dynamics.py --stage generate --execute \
  --generation-python .venv-generation/bin/python
```

Each teacher sees the same frozen stream of up to 30,000 neutral number-sequence
requests. Only the teacher system instruction changes. vLLM 0.10.0 generates
responses with temperature 1, top-p 1, at most 64 new tokens, batches of 512, and
request seed `1 + request_index`. The model and tokenizer revision are pinned.

For each condition, keep the first 10,000 naturally completed responses consisting
of exactly ten space-separated, two-digit numbers. Reject an invalid response in
full; do not repair or truncate it. Accepted prompt subsets may differ across
teachers. The students receive only the user prompt and unchanged response, with
no teacher system instruction in their training examples.

Outputs: `outputs/dynamics/datasets/<condition>/filtered.jsonl`, generation
manifests, runtime evidence, and sealed raw batches. Resume reruns verify these
files and reuse complete accepted data.

## 3. Fine-tune all five students

```bash
.venv-cpu/bin/python tools/run_dynamics.py --stage train --execute \
  --training-python .venv-training/bin/python
```

Each student starts from the same base model. Training uses completion-only SFT,
three epochs, physical batch size 16, gradient accumulation 4, seed 1, learning
rate `2e-4`, and LoRA rank 8 / alpha 32. This gives 471 optimizer updates.
Save 51 adapters at steps `1, 2, 5, 10, 20, ..., 470, 471`.
The training scripts retain full optimizer/RNG restart state and verify the
dataset identity before resuming.

Outputs: `outputs/dynamics/training/<condition>/`, including run manifests,
`adapters/checkpoint-000001/` through `adapters/checkpoint-000471/` on that schedule,
and restart state. Conditions can run sequentially on one GPU. To process one
condition at a time, add, for example, `--condition vermilion` to generation or
training. Evaluation requires all four targets and the control to be complete.

## 4. Evaluate recovery at every checkpoint

```bash
.venv-cpu/bin/python tools/run_dynamics.py --stage evaluate --execute \
  --training-python .venv-training/bin/python
```

Score the base model at step 0 and every saved adapter on the same ten recovery
questions and 30 color answers. Recovery uses the raw probability of each complete
answer over the full vocabulary, in FP32, with no teacher instruction, assistant
prefill, answer-length normalization, or probability floor.

Outputs: `outputs/dynamics/evaluation/targets/summary.json` and
`outputs/dynamics/evaluation/control/summary.json`, plus native per-question score
caches. The wrapper verifies the complete checkpoint schedules and that all
students use this campaign's generated data before scoring.

## 5. Calculate and plot control-relative recovery

```bash
.venv-cpu/bin/python tools/run_dynamics.py --stage report --execute
```

For each answer and training step, subtract the control student's log probability
from the target student's log probability within each question, then average over
the ten questions. This is the control-relative recovery quantity in Figure 4D.
The report validates matching questions, answers, model revision, and base scores
before doing the subtraction.

Open `outputs/dynamics/report/index.html`. The machine-readable files are
`analysis.json`, `trajectories.csv`, `paired_per_question_scores.csv`, and
`onsets.csv`. In `trajectories.csv`, select `condition=vermilion` and use
`control_adjusted_gain` against `step` for Figure 4D. The report plots all four
targets; its layout is broader than that single manuscript panel.

The protocol in [red_hue_control.json](../data/experiments/red_hue_control.json)
preserves the audited scientific settings and the complete prompt stream.
The released workflow generates all five datasets and trains all five models
anew. The submitted measurements used saved target datasets; a fresh generation
run and different GPU kernels can produce different numerical trajectories.
The original target run's exact TRL version was not archived; the reproducible
environment pins TRL 0.22.2. This release's plan, data integrity handling, and
stage interfaces were tested on CPU; the full GPU experiment was not rerun during
release preparation.
