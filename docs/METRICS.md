# Measuring the six paper metrics

Run this stage after data generation, student fine-tuning and recovery evaluation
in [the experiment workflow](REPRODUCING.md). The recovered student matrix is the
SL outcome against which the metrics are compared. Computing the six predictors
also needs base-model inference and fitted steering vectors, so a fresh run
requires a GPU even after the students have been trained.

The public command uses the paper's numbering consistently: `1` through `6`.
It supports the twelve Qwen2.5-7B settings in Figure 3: color, city, animal and
tree traits, each with number-token SFT, number-distribution distillation, or
opaque-100 distribution distillation. The same metric definitions are not
automatically calibrated for other model architectures.

Use `python run.py metrics` for **new model measurements and their comparison
with student recovery**, `python run.py metric1` for **CPU aggregation of an
already validated probability archive**, and `python run.py plot --saved-paper-data`
for **redrawing supplied paper measurements**. The last
command does not measure any metric on a model. The guide below covers the first
two routes; [FIGURES.md](FIGURES.md) covers paper figure reproduction.

## Before starting

Activate the Python 3.12 training environment from
[the setup guide](REPRODUCING.md#1-prepare-the-environment-and-inputs). New
measurements need the base model and CUDA, not the student adapters themselves;
the comparison needs the student recovery matrix produced with those adapters.
CPU planning, cache aggregation, and plotting can use the analysis environment.

Keep one configuration throughout generation, training, recovery, and metrics.
The six-metric wrapper requires `Qwen/Qwen2.5-7B-Instruct`, the complete ordered
30-trait catalog, and ten recovery questions. Its vector loaders require 28
blocks with hidden dimension 3,584. The cross-model, GSM8K/Python, training
dynamics, and random-pair training commands in this repository do not imply
support for running all six metrics on those experiments. Use the twelve
Figure 3 configurations listed in [EXPERIMENTS.md](EXPERIMENTS.md).

For example, after fine-tuning the central color setting:

```bash
CONFIG=experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json
python run.py recovery --config "$CONFIG" --execute
python run.py metrics --config "$CONFIG" --metrics all --output-dir outputs/color-metrics --plan-output outputs/color-metric-plan.json
python run.py metrics --config "$CONFIG" --metrics all --output-dir outputs/color-metrics --execute
python run.py analyze --metrics-dir outputs/color-metrics
```

The recovery step is required even if only one metric is requested, because
`metrics` also computes agreement with SL. If recovery has already completed,
start at the plan command. The plan lists missing files; a successful plan does
not mean that the GPU inputs have been produced.

## 1. Preview the measurement plan

Run from the repository root, using the same configuration as the completed
student experiment:

```bash
python run.py metrics \
  --config experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json \
  --metrics all
```

This reads the configuration and prints required inputs, GPU measurement
commands and output paths. It does not load a model or start a job. Add
`--plan-output outputs/color-metric-plan.json` to save the plan. Missing
experimental outputs are listed so the plan can be inspected before training.

For number or opaque distribution distillation, use the corresponding training
configuration instead:

```bash
python run.py metrics \
  --config experiment_plan/scaling/qwen25_color30_00_99_alpha32_3ep_h100_autotune.json \
  --metrics all

python run.py metrics \
  --config experiment_plan/scaling/qwen25_color30_opaque100_alpha32_3ep_h100.json \
  --metrics all
```

## 2. Measure and aggregate

In a compatible GPU evaluation environment, append `--execute` to the chosen
command. For example:

```bash
python run.py metrics \
  --config experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json \
  --metrics all --execute
```

The wrapper performs the following measurements in dependency order:

1. **Teacher and neutral carrier probabilities.** For number-token SFT it
   scores the first complete `00`–`99` continuation at the native assistant
   boundary, on the shared unfiltered request stream. For distribution
   distillation it reads the matched caches already produced during training.
2. **Carrier-to-answer associations.** For every carrier, it measures the raw
   probability of each complete trait answer across ten recovery questions.
   These probabilities determine the entangled carrier sets and selections.
3. **Target-trait and carrier vectors.** It fits separate per-block steering
   components with the base model frozen: 100 updates, learning rate `0.01`,
   initialization standard deviation `0.01`.
4. **Prompt-trait vectors and steering.** It extracts differences between
   instructed and neutral hidden states on 1,024 shared prompts, then measures
   recovery with the resulting vectors applied to all blocks and token positions.
5. **Current metric aggregation and comparison.** It produces M1–M6 with the
   definitions below and compares each complete matrix with the SL outcome.

The evaluation precision is FP32 for number-token SFT and BF16 for distribution
distillation, matching the Figure 3 measurement settings. CPU aggregation uses
float64. Workload depends on model memory, batch sizes and hardware; no fresh
GPU time or memory benchmark is claimed for this release.

To measure a subset, use a comma-separated selection. Dependencies are included
automatically; for example M5 needs both prompt and fitted target vectors.
The shared measurement engines may also produce intermediate results not needed
by the selected matrix: an M5-only run still performs carrier association and
carrier-vector fitting. For number SFT, even an M4-only run prepares the shared
channel used by prompt-vector extraction. Inspect the plan for the actual work.

```bash
python run.py metrics --config experiment_plan/scaling/qwen25_color30_00_99_alpha32_3ep_h100_autotune.json \
  --metrics 2,3 --execute

python run.py metrics --config experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json \
  --metrics 1 --output-dir outputs/color-metric1 --execute
```

The default output is `<training output_dir>/paper_metrics/`. Use
`--output-dir outputs/my-metric-run` for a separate measurement run. Repeating
the command can reuse compatible intermediate caches; changing an experiment's
scientific settings requires a new output directory.
Aggregation replaces `summary.json` and `matrices.npz` with the requested
selection, so use separate output directories if you want to retain both a
subset report and an all-metric report. Keep the same scientific configuration
and output directory when resuming an interrupted measurement.

## 3. Understand each metric and its files

Every public matrix uses **teacher traits in rows and recovered answer traits
in columns**, in the configuration's ordered trait list.

| Paper metric | Definition | Required measurements | Output |
|---|---|---|---|
| **M1: Carrier-drift similarity** | Mean of within-prompt cosines between carrier signatures, with `sqrt(p0)` weighting and across-teacher centered log probability ratios | Shared neutral and teacher carrier probabilities | `matrices/M1.csv` |
| **M2: Entangled-carrier information** | Conditional mutual information in bits, with a uniform binary neutral/teacher prior; retain the top ten carrier identities for each answer plus an OTHER bin | Carrier probabilities and carrier-to-answer association scan | `matrices/M2.csv` |
| **M3: Drift–entangled-score similarity** | Mean inner product of the unnormalized M1 signature with the logit of each carrier's mean raw complete-answer probability | Same carrier probabilities and association scan | `matrices/M3.csv` |
| **M4: Prompt-trait vector steering** | Mean complete-answer log-probability lift with prompt-trait vectors added at all blocks and all token positions, `alpha=0.3` | Prompt vectors and steered/unsteered recovery scores | `matrices/M4.csv` |
| **M5: Prompt/target-trait similarity** | Cosine after concatenating matching prompt and fitted target components from blocks 0–26 | Prompt vectors and independently fitted per-block target vectors | `matrices/M5.csv` |
| **M6: Target-trait/entangled-carrier similarity** | Cosine after concatenating fitted target and selected-carrier components from blocks 0–27 | Association argmax selections, target vectors and carrier vectors | `matrices/M6.csv` |

M3 uses an ordinary inner product, without cosine normalization or centering the
association logit over carriers. Its answer probabilities are raw complete-answer
probabilities, not probabilities renormalized over the 30 trait answers.
M5 excludes the final prompt-vector slot, which is after the final model
normalization; M6 uses all 28 independently fitted block components.
For M6, teacher row `z` uses the carrier with the largest mean answer probability
for trait `z`, and answer column `z'` uses the fitted target vector for `z'`.

The public numbering in this table is authoritative. Older implementation
filenames retain development names such as `evaluate_m1_trait_carrier_icv.py`
(used for the fitted vectors behind paper M5/M6) and
`evaluate_m2_dataset_behavior_concept_vectors.py` (used for paper M4/M5).
Do not identify a paper metric from an old script name or output folder alone.

`matrices.npz` contains the selected matrices, ordered labels and matching SL
matrix. `summary.json` records definitions, actual prompt counts, source hashes
and Pearson correlations for all 900 entries and the 870 off-diagonal entries.
Both full matrices are double-centered **before** either entry selection.
`execution_plan.json` records the executed commands and measurement script hashes.

### Current Metric 1: normalize within each prompt

For teacher trait `z`, shared prompt occurrence `x`, and carrier `c`, the
implementation in [`tools/metric1.py`](../tools/metric1.py) computes:

```text
epsilon = 1e-12
rho_z(x,c) = log((p_z(c|x) + epsilon) / (p_0(c|x) + epsilon))
L_z(x,c) = sqrt(p_0(c|x)) * (rho_z(x,c) - mean_r rho_r(x,c))
u_z(x,:) = L_z(x,:) / norm(L_z(x,:))
M1(z,z') = sum_x w_x * dot(u_z(x,:), u_z'(x,:)) / sum_x w_x
```

The mean over `r` includes every evaluated teacher trait in the ordered catalog,
and `p_0` is the no-system neutral reference on the same carrier support. The
paper workflow uses equal weights for each prompt occurrence; repeated prompt
text remains repeated. Both teacher and neutral probabilities must be finite
and normalized over that support. A positive-weight prompt with a zero or
nonfinite signature norm makes this metric undefined and is rejected; it must
not be silently dropped or assigned cosine zero.

Cosine after flattening the **unnormalized** prompt-by-carrier signatures is the
historical **legacy flattened cosine**, not the current Metric 1. A legacy
matrix cannot be converted without the per-prompt probabilities or signatures.
When reusing an older experiment, check its executed/frozen scoring code and
provenance, preserve its legacy result, reproduce that result from its caches
as a check, then save current aggregation separately. Do not substitute a
different teacher set, prompt population, carrier support, or precision to make
an incompatible cache pass validation.

For correlations, double-center each complete matrix `A` as
`A - row_mean(A) - column_mean(A) + grand_mean(A)`, and only then select all
entries or the off-diagonal entries. Figure 3 uses all 900 entries of each
30-by-30 matrix. Selecting the 870 off-diagonal entries before centering would
compute a different comparison. Zero-variance comparisons are recorded as
undefined rather than assigned a correlation.

## Plot your measured results

After aggregation, run:

```bash
python run.py analyze --metrics-dir <training-output-dir>/paper_metrics
```

Use the actual directory chosen for `metrics`. This produces centered recovery
and metric heatmaps, plus an all-entry correlation chart, from your new
measurements. It validates the CSV checksums, array consistency, and reported
correlations before plotting. PNG and PDF files appear in `<metrics-dir>/plots/`.
This step uses CPU; the measurements it consumes required GPUs.

## Cache production and reuse

The student pipeline supplies the following inputs:

| Input | How it is produced | Used by |
|---|---|---|
| Neutral JSONL request stream and carrier catalog | The data-generation stage in the experiment workflow; opaque carriers retain their supplied catalog | All distribution measurements |
| `<training output_dir>/training/teacher_distributions/*.npz` and matching metadata | Distribution-distillation training scores the no-system neutral teacher and every trait-instructed teacher on the same ordered request stream | M1, M2 and M3 for distillation |
| `<training output_dir>/m10/matrices/final/raw_logprob_shift.csv` | Student recovery stage after fine-tuning | Agreement with SL for all six metrics |

The metric command creates these additional caches inside its output directory:

| Cache | How it is produced | Used by |
|---|---|---|
| `channel/prompts.jsonl`, `channel/teacher_distributions/`, channel manifests | First-number scoring on the unfiltered, seed-42 number-SFT prompt stream, without teacher-dependent acceptance filtering | M1–M3 for number-token SFT; prompt-vector extraction |
| `association/behavior_scores.json` | Raw complete-answer inference for all carrier instructions and ten recovery questions | M2, M3 and carrier selection for M6 |
| `shared_vectors/m1_trait_vectors/` | Per-block target-trait optimization on the recovery questions | M5 and M6 |
| `target_vectors/vectors/carriers/` | Per-block optimization for the most associated carrier for each trait | M6 |
| `prompt_vectors/dataset_vectors/` | Last-token hidden-state differences on 1,024 instructed/neutral prompts | M4 and M5 |
| `prompt_vectors/part1_causal_dataset_vectors/` | Complete-answer scoring with and without prompt-vector steering | M4 |

M1 uses 1,024 prompt occurrences for color/city/animal number-token SFT,
10,000 for tree number-token SFT, and 10,000 for each distribution-distillation
setting. M2 and M3 use the first 1,024 occurrences of their matched probability
cache. Every teacher shares the same ordered prompt population and equal
occurrence weights. Repeated prompt text is not deduplicated.

After those GPU measurements exist, reuse them without inference:

```bash
python run.py metrics \
  --config experiment_plan/scaling/qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json \
  --metrics all --aggregate-only --execute
```

Use the same `--output-dir` as the measurement run if you changed it. The loader
checks model and condition identities, prompt/carrier ordering, probability
normalization, vector shapes and layer indices. M1 uses `epsilon=1e-12` and
rejects positive-weight zero-norm signatures. An aggregate similarity matrix
cannot replace missing per-prompt probability caches. Distillation NPZ files
contain pickled string arrays from the local producer; only reuse trusted local
experiment caches.

The release includes compact paper matrices for plotting, but not all of the
large caches in this table. Plotting those saved matrices is a separate CPU
analysis path. Fresh experimental reproduction uses the GPU workflow above.
The new public wrapper's plans and numerical kernels have CPU tests; its full
GPU campaign has not been rerun for this release.

## Metric 1 from a portable probability archive (CPU only)

If validated probabilities already exist, the standalone command does not need
student checkpoints or GPU inference:

```bash
python run.py metric1 --input outputs/probabilities.npz --output outputs/metric1.json
```

Create `probabilities.npz` with `numpy.savez_compressed`, using exactly the
following arrays. This is an explicit exchange format; it is not the schema of
the training pipeline's individual per-teacher caches.

| Array | Shape | Meaning |
| --- | --- | --- |
| `p0` | `[N, C]` | Neutral probabilities on the ordered prompts and carriers |
| `teachers` | `[T, N, C]` | Teacher probabilities on the same prompt/carrier axes |
| `labels` | `[T]` | Unique ordered teacher-trait names, Unicode strings |
| `prompt_ids` | `[N]` | Nonempty prompt identifiers, Unicode strings; repetitions are allowed |
| `carrier_labels` | `[C]` | Unique ordered carrier names, Unicode strings |
| `weights` (optional) | `[N]` | One shared nonnegative prompt-weight vector; default is all ones |
| `sl` (optional) | `[T, T]` | Matched student recovery, teacher rows and answer columns, both in `labels` order |

All probabilities must be in `[0,1]` and sum to one along the carrier axis
(absolute tolerance `1e-5`). The weight sum must be finite and positive. Use
Unicode arrays such as `np.asarray(names, dtype=str)`, not object arrays: the
standalone tool loads with `allow_pickle=False`. Zero-weight occurrences are
explicitly listed in the output. With `sl`, it additionally reports Pearson
correlations for all entries and off-diagonal entries after full-matrix
centering. The pipeline's recovery CSV has answer rows and teacher columns;
reorder both axes and transpose it before storing `sl` in this archive.

The JSON records the formula, counts, labels, weights, current M1 matrix, input
archive hash, and aggregation implementation hash. The format is independent of
model architecture, but numerical shape checks cannot establish that the
producer used the intended model, prompts, tokenization, or neutral condition.
Retain the source configuration, actual scoring implementation, model/carrier
identities, and cache metadata alongside the exported archive.

To check only that the CPU command works, use its deterministic synthetic demo:

```bash
python run.py metric1 --demo --output outputs/synthetic-metric1.json
```

This demo is labeled synthetic and is not a paper measurement. Neither this
command nor the archive command runs the other five metrics or creates the
multi-metric `summary.json`/`matrices.npz` format consumed by `run.py analyze`.

## Missing or incompatible inputs

| Symptom | Next step |
| --- | --- |
| Missing `m10/matrices/final/raw_logprob_shift.csv` | Complete `recovery` for the same training configuration. |
| Missing distillation `teacher_distributions` | Complete the probability-scoring stage of that configuration's `train` workflow, or restore its validated original caches. |
| Missing association or vector caches during `--aggregate-only` | Run `metrics` with the same selection and output directory without `--aggregate-only` to perform those GPU measurements. |
| Model, prompt count/order, carrier order, or vector metadata mismatch | Inspect the recorded identities and select the matching caches/configuration. Keep incompatible experiments in separate output directories. |
| Undefined M1 signature or correlation | Investigate the stated zero norm or zero variance; do not delete prompts or replace undefined values with zero. |

Changing a name or completion marker does not repair a scientific identity
mismatch. Keep source caches and provenance when transferring a completed run
between machines; aggregate-only execution still reads the matched configuration,
input dataset/catalogs, recovery matrix, and required cache metadata.
