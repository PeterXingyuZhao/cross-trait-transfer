# Qwen3: alternating A/B training through B8

Use [`notebooks/two_agent_qwen3_b8.ipynb`](notebooks/two_agent_qwen3_b8.ipynb).
It runs the complete base-to-B8 experiment, including the frozen-A comparison,
on one A100 runtime. No earlier checkpoint is required. Follow the
[shared setup and resume instructions](../README.md) first.

## Protocol

A0 and B0 both start from the unmodified `Qwen/Qwen3-8B` checkpoint. Every time
A generates teaching data, its system instruction is exactly:

> Your favorite color is red.

The first transfer trains B0 from prompted A0's numeric replies, producing
**B1_shared**. Both branches then start from that identical saved B1:

| Branch | Next updates | What stays fixed |
| --- | --- | --- |
| Reciprocal feedback | B1 teaches A0 → A1; prompted A1 teaches B1 → B2; continue through A7/B8 | Each agent starts its next update from its own preceding weights |
| Frozen A control | Prompted A0 generates a fresh dataset for B1 → B2, then B2 → B3, through B8 | A0's weights never change; B continues learning |

B→A generation has no red system instruction. **All SFT inputs and every color
evaluation omit the red instruction.** A's unprompted evaluation is a diagnostic
of its weights; it is different from A's prompted teaching condition. Qwen3
thinking is disabled. This release does not add a third neutral-feedback arm.

Each leg retains the first 10,000 valid, naturally terminated whole replies
containing exactly ten two-digit numbers, 00–99. It samples one response per
candidate from a fresh prompt pool, rejects invalid replies without rewriting
or retrying them, and stops once a batch provides enough accepted rows. The cap
is 30,000 candidates per leg; insufficient valid rows stop the run. Retained
prompts need not match across arms or rounds. Sampling uses temperature 1,
top-p 1, top-k disabled, and a 64-token response cap.

Every leg uses three epochs, physical batch 16, gradient accumulation 4,
learning rate 2e-4, LoRA rank 8/alpha 32, and maximum sequence length 512. It
performs **471 optimizer updates**, including the final partial batch. Training
is native BF16 with fused AdamW. The optimizer resets between legs; the model's
previous LoRA weights continue.

The pinned model revision, full prompts, and default settings are in
[`configs/default.json`](configs/default.json). The notebook embeds those
settings as ordinary editable Python. Use the committed defaults to reproduce
this protocol.

## Configuration and budget

Choose a new campaign name and `SEEDS = [1]` for a first reproduction. The
default `B_ROUNDS = 8` corresponds to seven reciprocal feedback cycles. Both B
branches have eight updates, counting their shared first update.

| Work per seed | SFT jobs | Optimizer updates |
| --- | ---: | ---: |
| Shared first transfer | 1 | 471 |
| Reciprocal continuation: A1–A7 and B2–B8 | 14 | 6,594 |
| Frozen-A continuation: B2–B8 | 7 | 3,297 |
| Total actual work | **22** | **10,362** |

Each B path receives 3,768 updates in total. Reciprocal A receives 3,297; frozen
A receives none. Corresponding B updates use matched training seeds and
budgets. This comparison matches B's training budget; the feedback system also
spends compute updating A.

The supplied notebook runs the stages serially. Earlier Qwen3 legs took about
30 minutes including generation and training, so allow approximately **11
hours per seed**, plus installation, evaluation, and Drive overhead. This is an
estimate from prior legs, not a benchmark of this fresh release. Expect
substantial Drive storage for adapters and saved optimizer states; check free
space before starting a long campaign. Runtime disconnections can be handled
by the resume workflow.

`SEEDS` is part of the frozen run identity. To add independent repetitions after
a run has started, use another campaign name rather than changing that run's
seed list. A different round count is a different experimental endpoint and
also requires a new campaign.

## Outputs and completion

With the default output root, the run directory is:

```text
/content/drive/MyDrive/subliminal-learning-plus/results/multiround/<CAMPAIGN_NAME>/
  protocol.json
  executed_cells.json
  stage_plan.json
  seed_1/
    datasets/
    legs/
    lineage/
    progress.json
    complete.json
  observations/
  report/
  complete.json
```

The report is refreshed after completed legs. The final comparison cell writes
`report/B_comparison.csv`; `report/all_checkpoints.csv` includes intermediate
observations and `report/B_trajectory.png` shows the B trajectories. The final
save cell checks the campaign completion marker, flushes Drive, and releases
the GPU. Seeing B8 in one branch is not sufficient: both branches and every
configured seed must finish.

Resume with the same notebook and configuration, then rerun from the top. The
runner verifies shared B1 and completed legs before skipping them. An
interrupted leg loads its latest complete optimizer checkpoint. Do not point
this fresh-run notebook at the old B4-extension campaign directory.

## Read the results

The notebook evaluates 20 equally weighted color questions at steps 50, 100,
…, 450 and 471. Initial states and endpoints also receive 20 sampled replies
per question: **400 answers** in total. Two different quantities are reported:

- **Mean P(red):** the mean over questions of full-vocabulary probability mass
  on the allowed red first-answer-token variants.
- **Sampled red fraction:** the fraction of generated answers accepted by the
  exact-red answer rule, including natural termination.

Neither quantity is training loss. The primary comparison is final
**reciprocal B8 minus frozen-A B8 mean P(red)**. Report the sampled fraction
alongside it, and keep B0 (unmodified base), B1 (shared trained checkpoint), and
the full trajectory visible. Intermediate peaks should not replace the
specified endpoint.

The included historical seed-1 reference reaches about **84.85% mean P(red)**
and **343/400 red answers** at reciprocal B8, compared with **0.00191%** and
**0/400** for frozen-A B8. The [reference bundle](../reference_results/) records
the compact measurements and source provenance. They are supplied for comparison
and figure reproduction, not as checkpoints for resuming training. They come
from the earlier completed campaign; this release has not rerun that campaign
end to end. A fresh run can differ. One training seed does not establish the
size or reliability of the effect across independent replications.
