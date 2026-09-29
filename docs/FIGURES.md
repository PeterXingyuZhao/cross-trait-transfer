# Main-text figures

The experiment workflow is **data preparation → student training → recovery
and metric evaluation → analysis**. The table maps each figure to the required
measurements. Configurations are listed in [EXPERIMENTS.md](EXPERIMENTS.md).

| Figure | Result | Experiments and measurements |
| --- | --- | --- |
| 1B/C | Color-transfer network and six-group summaries | 30-color number SFT; student recovery, Metric 1, and Metric 4 |
| 2 | Transfer across models and carriers | Eight model/carrier settings; train each teacher-conditioned student and evaluate its recovery |
| 3 | Agreement of six metrics with measured transfer | Twelve Qwen2.5 settings; student recovery and Metrics 1–6 |
| 4A–C | Changes during training | Intermediate student checkpoints; recovery matrices, effective ranks, and agreement with predictive metrics |
| 4D | Control-relative vermilion recovery | Four red-hue teachers and an unnamed-color control; dense checkpoint recovery |
| 5 | Transfer from aligned and opposed trait pairs | 50 aligned and 50 opposed color pairs plus a shared trained control |

Use the [main workflow](REPRODUCING.md), [six-metric guide](METRICS.md),
[dynamics workflow](DYNAMICS.md), and
[random-pair workflow](../experiments/random_pairs/README.md) to obtain fresh
measurements. Training and model-based scoring require CUDA GPUs.

## Aggregation

Preserve ordered trait labels and matrix orientation. Figure 3 uses teacher rows
and answer columns. Figures 1, 2, and 4 retain answer rows and teacher columns.
Do not transpose a matrix without checking its source labels and protocol.

- Figure 1 double-centers complete matrices and scales by off-diagonal RMS.
  Six-group blocks exclude self-pairs but retain other within-group entries.
- Figure 2 centers complete matrices before calculating group contrasts.
- Figure 3 double-centers both complete matrices and correlates all 900 entries.
- Figure 4 computes energy effective rank on uncentered recovery matrices and
  centers full matrices before selecting 870 off-diagonal correlation entries.
- Figure 5 weights all pairs equally. Repeated colors and the shared trained
  control are not independent replications; the plot does not infer seed-level
  significance or confidence intervals.

## Draw figures from saved paper measurements

With the analysis environment installed:

```bash
python run.py plot --saved-paper-data
python run.py plot --saved-paper-data --figure 3
```

The input files are `data/figures/figure1.json` through `figure5.json`, with
`network_label_layout.json` supplying Figure 1 label placement. The command
recalculates the numerical summaries before writing PDF, PNG, SVG, and JSON
outputs into `outputs/figures/figure1/` through `figure5/`.

This is analysis of already measured results. It does not regenerate teacher
data, train students, or recompute model probabilities or activations.

## Draw figures from new measurements

The metric workflow saves new metric matrices and their comparisons with the
new recovery matrix. Run `python run.py analyze --metrics-dir PATH` to draw
the new matrices and their agreement chart; see [METRICS.md](METRICS.md). A completed random-pair run
can be passed directly to its plotting command in [REPRODUCING.md](REPRODUCING.md).

For the complete main-figure layouts, `python run.py plot --data-dir PATH`
accepts a directory with the same measurement schema as `data/figures/`.
Assemble the corresponding setting/checkpoint measurements and recomputed
summary fields in that schema first. It does not automatically discover new
experiment directories. The renderer verifies stored summaries against the
matrices; use the supplied files as schema examples, not as substitutes for
fresh measurements.

Figure 1A is a conceptual illustration and is not regenerated. Font rendering
can differ across platforms; numerical agreement is checked without requiring
byte-identical PDFs.
