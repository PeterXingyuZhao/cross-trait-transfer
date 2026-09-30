# Source map by workflow

Start with `python run.py <command>` from the repository root. The public
commands organize the experiment into data generation, fine-tuning, recovery,
metric measurement, and figure analysis. Their dispatch table is
[`experiments.json`](../experiments.json). This map groups the underlying code by
its role while keeping the original module paths used by imports and saved
experiment provenance.

## Public entry points

| Task | Command | Implementation | Instructions |
| --- | --- | --- | --- |
| Validate a configuration | `python run.py preflight --config CONFIG` | [`tools/experiment.py`](../tools/experiment.py) | [Experiments](../docs/EXPERIMENTS.md) |
| Generate teacher responses | `python run.py generate --config CONFIG --execute` | [`tools/experiment.py`](../tools/experiment.py) | [Generation](../docs/REPRODUCING.md#2-generate-teacher-data) |
| Fine-tune students | `python run.py train --config CONFIG --execute` | [`tools/experiment.py`](../tools/experiment.py) | [Fine-tuning](../docs/REPRODUCING.md#3-fine-tune-independent-students) |
| Measure student recovery | `python run.py recovery --config CONFIG --execute` | [`tools/experiment.py`](../tools/experiment.py) | [Recovery](../docs/FINE_TUNING.md#5-evaluate-student-recovery) |
| Measure paper Metrics 1–6 | `python run.py metrics --config CONFIG --metrics all --execute` | [`tools/evaluate_metrics.py`](../tools/evaluate_metrics.py) | [Metric guide](../docs/METRICS.md) |
| Plot new measured matrices | `python run.py analyze --metrics-dir DIRECTORY` | [`tools/analyze_run.py`](../tools/analyze_run.py) | [Fresh-run plots](../docs/FIGURES.md#2-plot-matrices-from-your-own-measured-experiment) |
| Redraw saved paper figures | `python run.py plot --saved-paper-data` | [`tools/reproduce_figures.py`](../tools/reproduce_figures.py) | [Figure guide](../docs/FIGURES.md) |
| Run dense matched-control dynamics | `python run.py dynamics` | [`tools/run_dynamics.py`](../tools/run_dynamics.py) | [Dynamics](../docs/DYNAMICS.md) |

`CONFIG` and `DIRECTORY` above are placeholders. Use a real configuration from
the experiment guide. Generation, training, recovery, metric, and dynamics
commands display a plan until `--execute` is supplied; `preflight` runs its
non-inference checks immediately. Fixed-carrier distillation skips the response
generation stage: its training stage produces teacher probability targets.

## Data preparation and teacher generation

| Source | Role |
| --- | --- |
| [`run_number_sequence_trait_cell.py`](run_number_sequence_trait_cell.py) | Number-SFT campaign; deterministic requests, per-trait response generation, filtering, manifests, and stage coordination |
| [`native_number_generation.py`](native_number_generation.py), [`native_generation_loader.py`](native_generation_loader.py) | Native model generation and model/revision loading helpers |
| [`generate_trait_response_sft_dataset.py`](generate_trait_response_sft_dataset.py) | Teacher-conditioned response-generation and filtering helpers |
| [`generate_long_form_trait_cell_datasets.py`](generate_long_form_trait_cell_datasets.py) | GSM8K/Python teacher datasets for the long-form campaign |
| [`number_sft_raw_storage.py`](number_sft_raw_storage.py), [`number_sft_dataset_reuse.py`](number_sft_dataset_reuse.py) | Raw-response storage and verified dataset reuse |
| [`build_opaque_token_sequence_dataset.py`](build_opaque_token_sequence_dataset.py), [`build_element_symbol_sequence_dataset.py`](build_element_symbol_sequence_dataset.py) | Fixed-carrier input builders; released inputs and catalogs are under `data/` and `experiment_plan/carriers/` |
| [`sl/datasets/`](../sl/datasets/) | Shared prompt, dataset, and validation utilities |

## Student fine-tuning and recovery

| Source | Role |
| --- | --- |
| [`run_number_sequence_trait_cell.py`](run_number_sequence_trait_cell.py) | Number-response SFT stages; trains each teacher-conditioned student separately |
| [`run_long_form_trait_cell.py`](run_long_form_trait_cell.py) | GSM8K/Python SFT stages and complete-family recovery |
| [`train_trait_response_sft.py`](train_trait_response_sft.py) | Response-only SFT implementation used by the corresponding Unsloth configurations |
| [`train_native_number_sft.py`](train_native_number_sft.py), [`native_lora_validation.py`](native_lora_validation.py) | Native Transformers number-SFT implementation and adapter checks |
| [`run_trait_carrier_smoke_pipeline.py`](run_trait_carrier_smoke_pipeline.py) | Fixed-carrier distribution-distillation campaign, despite the historical `smoke` name |
| [`train_teacher_logits_soft_distillation.py`](train_teacher_logits_soft_distillation.py), [`distribution_training_backend.py`](distribution_training_backend.py) | Teacher probability targets and distribution-distillation training backends |
| [`evaluate_trait_checkpoint_effect_matrices.py`](evaluate_trait_checkpoint_effect_matrices.py), [`evaluate_color_checkpoint_probability_ratio_matrices.py`](evaluate_color_checkpoint_probability_ratio_matrices.py) | Complete-answer checkpoint scoring and recovery matrices |

The public recovery command maps to the internal stage/directory named `m10`.
It is the measured student outcome, not an additional predictor in the paper's
six-metric table. Preserve checkpoints and manifests for this evaluation.

## Metric measurement and aggregation

Use [`tools/evaluate_metrics.py`](../tools/evaluate_metrics.py) for the paper's
numbering and supported settings. It plans the GPU measurements, validates their
caches, and writes `M1`–`M6` matrices and comparisons with recovery. Older
`m1`, `m2`, and similar names inside filenames refer to an earlier internal
catalog; they do **not** identify the correspondingly numbered paper metric.

| Source | Role in the current workflow |
| --- | --- |
| [`tools/metric1.py`](../tools/metric1.py) | Current Metric 1 kernel and CLI: normalize each prompt's carrier signature, then average cosines using shared prompt weights |
| [`prepare_token_sft_first_number_channel.py`](prepare_token_sft_first_number_channel.py) | Teacher/neutral probability caches on the shared first-number channel for number-token SFT |
| [`run_qwen25_metric_suite.py`](run_qwen25_metric_suite.py) | Reused low-level GPU command construction; the public wrapper supplies current aggregation and naming |
| [`evaluate_m3_teacher_behavior_alignment.py`](evaluate_m3_teacher_behavior_alignment.py) | Carrier-to-answer probability scan used by paper Metrics 2, 3, and 6, and shared vector preparation |
| [`evaluate_m1_trait_carrier_icv.py`](evaluate_m1_trait_carrier_icv.py), [`train_owl_in_context_vector.py`](train_owl_in_context_vector.py) | Frozen-model target/carrier vector fitting used by paper Metrics 5 and 6 |
| [`evaluate_m2_dataset_behavior_concept_vectors.py`](evaluate_m2_dataset_behavior_concept_vectors.py), [`compute_teacher_vectors.py`](compute_teacher_vectors.py) | Prompt-vector extraction and steering used by paper Metrics 4 and 5 |
| [`metric_carrier_catalog.py`](metric_carrier_catalog.py), [`metric_exact_trie_scorer.py`](metric_exact_trie_scorer.py) | Carrier definitions and complete-continuation probability scoring |
| [`strict_metric_precision.py`](strict_metric_precision.py), [`metric_batch_autotune.py`](metric_batch_autotune.py), [`metric1_shared_prefix.py`](metric1_shared_prefix.py), [`metric1_optimized_scoring.py`](metric1_optimized_scoring.py) | Precision and inference helpers |
| [`analyze_metric1_promptwise_cosine.py`](analyze_metric1_promptwise_cosine.py), [`analyze_token_sft_metric1_promptwise.py`](analyze_token_sft_metric1_promptwise.py), [`recompute_lambda_metric1_promptwise.py`](recompute_lambda_metric1_promptwise.py) | Setting-specific provenance audits/reaggregation of per-prompt caches; historical defaults may need original inputs |

The [metric guide](../docs/METRICS.md) is authoritative for paper definitions,
required inputs, prompt counts, matrix orientation, and cache reuse. Do not infer
a Metric 1 definition from an old filename or reuse a flattened-cosine matrix as
the current result.

## Figure rendering

[`tools/reproduce_figures.py`](../tools/reproduce_figures.py) supplies explicit
measurement paths to the following renderers. Use the public `plot` command for
the released bundle: direct execution of an older renderer can select historical
default paths that are not distributed here.

| Release figure | Rendering modules |
| --- | --- |
| 1B/C | [`build_paper_color_token_sft_main.py`](build_paper_color_token_sft_main.py), with [`color_metric_group_summary.py`](color_metric_group_summary.py) and network drawing dependencies |
| 2 | [`build_sl_matrix_sweep_quantified.py`](build_sl_matrix_sweep_quantified.py) |
| 3 | [`build_qwen25_nine_setting_metric_heatmap.py`](build_qwen25_nine_setting_metric_heatmap.py); the released input has twelve settings despite the filename |
| 4 | [`build_paper_figure6_token_sft_dynamics.py`](build_paper_figure6_token_sft_dynamics.py), with spectral statistics from [`build_paper_color_learning_dynamics.py`](build_paper_color_learning_dynamics.py); `figure6` is an older number |
| 5 | [`plot_paper_random_pairs_v41.py`](plot_paper_random_pairs_v41.py), importing shared functions from the retained `v38`/`v39`/`v40` modules |

See [FIGURES.md](../docs/FIGURES.md) for exact filenames, selected-figure commands,
custom input schemas, and saved-paper versus fresh-run plotting.

## Specialized campaigns and retained support code

- **Training dynamics:** [`run_color_token_sft_dynamics.py`](run_color_token_sft_dynamics.py)
  and `color_token_sft_dynamics_*` support full-color trajectories.
  [`tools/run_dynamics.py`](../tools/run_dynamics.py) coordinates the released
  red-hue/control workflow through `red_hue_token_sft_*`, `red_hue_control_*`, and
  [`report_red_hue_control_dynamics.py`](report_red_hue_control_dynamics.py).
- **Random pairs:** use the frozen, self-contained
  [`experiments/random_pairs/`](../experiments/random_pairs/) workflow and its
  own protocol and environments. Root-level `color_pair_*` and `color_prefix_*`
  modules also retain support and historical campaign code.
- **Pair ordering and prefix experiments:** `run_qwen25_lion_aligned_first_ladder.py`,
  `run_qwen25_lion_opposed_first_ladder.py`, `run_qwen25_lion_tiger_order.py`, and
  `run_qwen25_pair_order.py` implement additional studies, separate from the
  default 30-trait reproduction workflow.
- **Exploratory reports and operational helpers:** `build_*`, `analyze_*`,
  `summarize_*`, `start_*`, and Colab/bundle builders include dependencies and
  historical analyses. Their presence does not imply all original datasets,
  caches, credentials, or hard-coded report paths are included in the release.

For contents and provenance, see [DATA.md](../docs/DATA.md) and
[RELEASE.md](../docs/RELEASE.md). For CPU checks and GPU validation limits, see
[VALIDATION.md](../docs/VALIDATION.md).
