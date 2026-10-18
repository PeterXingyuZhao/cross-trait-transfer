# Historical Qwen3 reference results

This compact bundle recreates the B0–B8 figure from the completed **seed 1**
two-agent Qwen3-8B experiment. It contains 18 endpoint rows, including the shared
base B0 and shared first trained model B1 in both plotted arms. It does not
contain model checkpoints or generated training data.

From the repository root, after installing the CPU dependencies in the main
[multi-round instructions](../README.md), run:

```bash
python scripts/multiround_reference.py --output-dir /path/to/results/multiround-reference
```

Choose a results directory on your own machine. The command requires an explicit
destination and writes `qwen3_b8_reference.png`, `qwen3_b8_reference.pdf`,
`qwen3_b8_reference.csv`, and `reference_receipt.json`. It performs no model
download, generation, or training. The receipt records the bundled input, plotting
code, and output hashes. Figure bytes may vary across Matplotlib versions; the
numeric CSV preserves the bundled measurements.

## What the figure measures

- **Left:** the arithmetic mean, over 20 equally weighted neutral questions, of
  first-answer-token probability mass assigned to the six frozen red token
  variants. The JSON stores fractions; the figure shows percent on a log scale.
- **Right:** exact red answers with allowed punctuation and native EOS, divided by
  all 400 requested answers (20 per question). Truncated and noncanonical answers
  remain in the denominator.
- All evaluations omit the red system instruction. Each trained endpoint is the
  final update 471 of its leg. B0 is the unmodified base.
- The dotted line at B4 marks where the historical continuation began. The
  experiment's treatment does not change at that line.

At B8, recursive feedback has mean P(red) **84.8466%** and **343/400 (85.75%)**
exact red replies. Frozen A has mean P(red) **0.001910%** and **0/400** exact red
replies. These are exploratory results from one training seed; 400 answers are
not 400 independent training replications.

## Provenance and limits

[`qwen3_b8_seed1.json`](qwen3_b8_seed1.json) contains the unchanged final report
values. [`provenance.json`](provenance.json) binds that table to the saved
comparison report, executed helper source, historical observations, and new
endpoint evidence using SHA-256 hashes. Personal absolute paths have been
replaced with logical artifact identifiers.

The earlier endpoint audit checked the saved summary and observation agreement,
training/evaluation JSON and sample-file seals, the complete prompt/draw grid,
EOS flags, and exact-red counts from raw replies. This packaging step verified
the saved report hash and compared every distributed endpoint with the saved
audit records. It did not rerun that raw-payload audit or read the large adapter
and full-vocabulary cache files. The raw evidence payloads are not distributed
here; their hashes document provenance, not an independent rerun of inference.

The historical experiment used an initial B4 campaign and a continuation with its
saved executed code. The portable fresh-run notebooks reproduce the specified
experimental procedure; these historical numbers are reference observations,
not promised exact outputs of a new stochastic run. This bundle cannot resume
the historical campaign and makes no claim about Qwen2.5 R/F/N results.
