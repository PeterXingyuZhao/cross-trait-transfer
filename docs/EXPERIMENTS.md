# Experiment map

Choose a configuration here, then follow the numbered stages in
[REPRODUCING.md](REPRODUCING.md). Paths in the tables are relative to
`experiment_plan/scaling/`. Each configuration specifies 30 teachers and
independently trained students unless otherwise stated.

## Figure 1: color transfer

Run `qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json` through
generation, training, and recovery. Measure Metrics 1 and 4 on the matching
Qwen2.5 number-carrier color setting using [METRICS.md](METRICS.md). This is also
the color number-SFT setting in Figure 3 and the 30-color checkpoint campaign
in Figure 4.

The experiment uses 10,000 accepted ten-number responses per teacher, three
epochs, LoRA rank 8/scale 32, seed 1, and 471 updates. Figure 1's Metric 1 uses
1,024 shared prompt occurrences, separately from the training-example count.

## Figure 2: eight model/carrier settings

Panels are listed in reading order: left to right across the top row, then
left to right across the bottom row.

| Panel | Model and traits | Carrier / training objective | Configuration |
| --- | --- | --- | --- |
| A | Qwen2.5-7B-Instruct, cities | 00–99 numbers / distribution distillation | `qwen25_city30_00_99_alpha32_3ep_h100_autotune.json` |
| B | Qwen3-VL-8B-Instruct, cities, text-only | Ten-number responses / token SFT | `qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100_bf16_r2.json` |
| C | Llama-3.1-8B-Instruct, cities | Ten-number responses / token SFT | `llama31_city30_number_sequence_sft_10k_alpha32_3ep_h100_bf16_r2.json` |
| D | Mistral-7B-Instruct-v0.3, colors | Ten-number responses / token SFT | `mistral_v03_color30_number_sequence_sft_10k_alpha32_3ep_h100_bf16_r2.json` |
| E | Qwen2.5-7B-Instruct, animals | 100 opaque tokens / distribution distillation | `qwen25_animal30_opaque100_alpha32_3ep_h100.json` |
| F | Qwen2.5-7B-Instruct, colors | Element symbols / distribution distillation | `qwen25_color30_element_symbols_alpha32_3ep_h100_fp32_b16k21_audit32.json` |
| G | Qwen2.5-7B-Instruct, animals | GSM8K responses / token SFT | `qwen25_animal30_gsm8k_response_sft_10k_3ep_alpha8_h100.json` |
| H | Qwen2.5-7B-Instruct, colors | Python responses / token SFT | `qwen25_color30_python_response_sft_5k_1ep_h100.json` |

Panels A/E/F consume the bundled neutral prompts and score teacher distributions
during `train`; do not use the sampled-response `generate` stage for them.
Panels B/C/D/G/H require generated teacher responses before training. All panels
then use `recovery`.

Panel B imports verified datasets from
`qwen3vl_city30_number_sequence_sft_10k_alpha32_3ep_h100.json`. Generate this
source first as shown in [REPRODUCING.md](REPRODUCING.md), then run the listed
BF16 training configuration. Qwen3-VL generation uses native Transformers;
Llama and Mistral generation use the pinned vLLM runtime policies in their
configurations. Recovery for these three native model runs uses FP32.

Panels A–F use 10,000 prompts or accepted responses per teacher, three epochs,
LoRA rank 8/scale 32, and 471 updates. Panel G uses 10,000 accepted responses,
three epochs, rank 8/scale 8, effective batch size 32, and 939 updates. Panel H
uses 5,000 accepted responses, one epoch, rank 8/scale 32, seed 2, and 157
updates. Filtering and response-count requirements are enforced before training.

## Figure 3: six metrics in twelve settings

All twelve settings use `Qwen/Qwen2.5-7B-Instruct`. Run the corresponding
generation/training/recovery stages, then measure **all six metrics** using
[METRICS.md](METRICS.md). The cross-trait recovery and metric matrices are
30 × 30. Figure 3 correlates all entries after double centering.

| Setting | Configuration |
| --- | --- |
| Number token SFT / colors | `qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json` |
| Number token SFT / cities | `qwen25_city30_number_sequence_sft_10k_alpha32_3ep_h100.json` |
| Number token SFT / animals | `qwen25_animal30_number_sequence_sft_10k_alpha32_3ep_h100.json` |
| Number token SFT / trees | `qwen25_tree30_number_sequence_sft_10k_alpha32_3ep_h100.json` |
| Number distribution / colors | `qwen25_color30_00_99_alpha32_3ep_h100_autotune.json` |
| Number distribution / cities | `qwen25_city30_00_99_alpha32_3ep_h100_autotune.json` |
| Number distribution / animals | `qwen25_animal30_00_99_alpha32_3ep_h100_autotune.json` |
| Number distribution / trees | `qwen25_tree30_00_99_alpha32_3ep_h100_autotune.json` |
| Opaque-token distribution / colors | `qwen25_color30_opaque100_alpha32_3ep_h100.json` |
| Opaque-token distribution / cities | `qwen25_city30_opaque100_alpha32_3ep_h100.json` |
| Opaque-token distribution / animals | `qwen25_animal30_opaque100_alpha32_3ep_h100.json` |
| Opaque-token distribution / trees | `qwen25_tree30_opaque100_alpha32_3ep_h100.json` |

The four number-SFT configurations require sampled teacher datasets. The eight
distribution configurations score the fixed carrier directly from the bundled
neutral prompt population. These objectives remain distinct even when they
share the same number carrier.

The supplied Metric 1 measurements use 1,024 shared prompt occurrences for
color/city/animal number SFT, 10,000 for tree number SFT, and 10,000 for all
eight distribution settings. Weights are equal per occurrence.
[METRICS.md](METRICS.md) describes the other metric populations. Preserve the
complete teacher set and ordering in each setting.

## Figure 4: training dynamics

Panels A–C use these complete number-SFT campaigns:

* `qwen25_color30_number_sequence_sft_10k_alpha32_3ep_h100.json`
* `qwen25_city30_number_sequence_sft_10k_alpha32_3ep_h100.json`

Retain intermediate checkpoints and evaluate recovery on them. The color
trajectory provides matrices and agreement with fixed Metrics 1 and 4; both
color and city trajectories contribute to effective rank. Use full matrices
for effective rank. For metric agreement, double-center full matrices before
selecting off-diagonal entries.

Panel D requires four independently trained students for crimson, scarlet,
red, and vermilion, plus one trained unnamed-color control. Record steps 1,
2, 5, every ten updates through 470, and 471. Score all five students with the
same 30-color catalog, ten questions, pinned Qwen2.5 model, and FP32 full-answer
scoring. Subtract the control at each matching checkpoint. The plotted
vermilion teacher's gains are relative to that trained control.

Follow [DYNAMICS.md](DYNAMICS.md) for the dense experiment's generation,
training, checkpoint manifest, and scoring steps. The supplied measurement
bundle also contains the complete plotted trajectory for CPU reanalysis.

## Figure 5: random color pairs

Use [experiments/random_pairs/README.md](../experiments/random_pairs/README.md).
The frozen design contains 50 aligned and 50 opposed pairs. Generate and train
all 100 pair-conditioned students plus one shared unnamed-pair control, score
the same 30-color readout, then create the report. Each student starts from the
base and uses 10,000 accepted ten-number responses, three epochs, LoRA rank
8/scale 32, seed 1, and 471 updates.

Pair selection is fixed before recovery evaluation and supplied in
`experiments/random_pairs/assets/experiment_design.json`. Do not resample pairs
based on training or recovery outcomes. The fresh workflow retrains the shared
control; results can differ from the supplied measurements across stochastic
GPU runs.

## Additional appendix experiments

Animal-prefix and month-pair configurations are under
`experiment_plan/multi_trait_competition/`. Animal orders are
lion–wolf–bird–tiger–squid–shark and lion–shark–squid–tiger–bird–wolf; the
lion-only student and control are shared. Month pairs combine January with
February, December, August, or September. These use seed 1, matched unnamed
controls, and ten recovery questions per domain. Their orchestrators retain
existing command-line interfaces; release validation focuses on the main text.
