# Qwen2.5: self-distillation and fresh-student transmission

This experiment follows three teacher trajectories for four updates, then
measures how well each teacher transmits the red preference to a **fresh base
student**. It is a different experiment from alternating Qwen3 A/B feedback.
All trajectories and probes start from public `Qwen/Qwen2.5-7B-Instruct`
weights; no earlier private checkpoint is required.

Choose one execution arrangement:

| Arrangement | Notebook | Hardware |
| --- | --- | --- |
| R only | [`self_distill_qwen25_R.ipynb`](notebooks/self_distill_qwen25_R.ipynb) | One A100 |
| F only | [`self_distill_qwen25_F.ipynb`](notebooks/self_distill_qwen25_F.ipynb) | One A100 |
| N only | [`self_distill_qwen25_N.ipynb`](notebooks/self_distill_qwen25_N.ipynb) | One A100 |
| All three, serially | [`self_distill_qwen25_all.ipynb`](notebooks/self_distill_qwen25_all.ipynb) | One A100 |

For parallel execution, run R, F, and N on separate A100 runtimes with the same
`OUTPUT_ROOT`, `CAMPAIGN_NAME`, and `SEED`. For one available GPU, use the serial
notebook, or run the single-arm notebooks one after another. Their arm
directories are compatible. Never run two writers for the same arm. See the
[shared setup and resume guide](../README.md).

## Teacher updates and probes

The red instruction is exactly **“Your favorite color is red.”** Teacher
weights continue from one update to the next, with a new optimizer per job.

| Arm | Data for updating the teacher | Generation system instruction |
| --- | --- | --- |
| R: recursive | The current R teacher generates its next training dataset | Red instruction |
| F: frozen source | The original frozen base generates fresh data; the F learner's weights continue updating | Red instruction |
| N: neutral | The current N teacher generates its next training dataset | No red instruction |

At teacher rounds **0, 1, 2, 3, 4**, take the current teacher from each arm,
generate a probe dataset **with the red instruction**, and train a **fresh
unmodified base student** on that dataset. This tests the teacher's transmitting
ability. In particular:

- F's probe data comes from the **updated F learner**, although F's teacher
  updates use the frozen base as their data source.
- N's probe generation **does receive the red instruction**, although N's own
  teacher updates do not.
- Probe students are fresh at each round and do not feed back into the teacher.
  R/F's first teacher update and round-zero probe are identical jobs; the code
  reuses that result only after checking the complete training plan.

**All SFT inputs and every preference evaluation omit the red instruction.**
R reuses a probe dataset for its next teacher update because both require data
from the same current prompted teacher. Other distinct datasets use fresh
candidate streams.

Each dataset retains the first 10,000 valid, naturally terminated whole replies
containing exactly ten two-digit numbers, 00–99, from at most 90,000 candidates.
Invalid responses are rejected without repair or retry. Generation uses
temperature 1, top-p 1, top-k disabled, and a 64-token response cap. Every job is
three epochs / **471 updates**, batch 16 × accumulation 4, learning rate 2e-4,
LoRA rank 8/alpha 32, maximum length 512, BF16 Unsloth training, and fused AdamW.
The pinned model revision and all question text are in
[`configs/default.json`](configs/default.json).

## Configuration and budget

Set a new `CAMPAIGN_NAME` and `SEED = 1` for the first reproduction. Keep
`TEACHER_ROUNDS = 4`: this release validates the four-update protocol. Changing
it is a new protocol requiring code, configuration, and test changes, not a
resume option.

| Arm | Actual training jobs | Approximate time on one A100 |
| --- | ---: | --- |
| R | 8 | 3–4 hours |
| F | 8 | 3–4 hours |
| N | 9 | 3.5–4.5 hours |
| All arms | **25 jobs / 11,775 updates** | **10–12 hours serially** |

The estimates use earlier Qwen2.5 stages and can increase with setup, candidate
rejection, additional evaluation, or Drive delays. They are not an end-to-end
benchmark of this release. Parallel arms reduce wall time, not total GPU work.
An earlier artifact-size estimate was roughly **20–25 GB per arm**, including
saved adapters, probability caches, and recent optimizer checkpoints; allow
headroom and monitor available Drive space.

Changing `SEED` creates a separate seed directory. Use matching seeds across R,
F, and N for paired comparisons. A new scientific configuration requires a new
campaign name; do not alter the frozen protocol inside an existing arm.

## Outputs, resume, and completion

With the default output root, each arm writes beneath:

```text
/content/drive/MyDrive/subliminal-learning-plus/results/multiround/<CAMPAIGN_NAME>/
  seed_1/
    R/
    F/
    N/
```

Each arm contains its own frozen protocol/source, datasets, training and
evaluation artifacts, log, completion record, and `report/`. Reports read
available sibling-arm results while writing only into the selected arm's
report directory. Missing or unfinished arms appear as pending.

After an interruption, rerun the same notebook from the top with the same
campaign, seed, and arm. Verified completed jobs are reused; interrupted jobs
resume optimizer/RNG state from a sealed checkpoint. You may finish an arm with
its single-arm notebook after beginning it in the serial notebook, provided
the scientific configuration, helper source, and runtime still match and no
other writer is active.

The final save cell releases the GPU after all **selected** arms finish. In a
single-arm notebook that means only that arm is complete. A full comparison
requires all of R, F, and N. Rerun the comparison cell after sibling arms finish
to refresh an earlier report; in a fresh session, first rerun setup, mounting,
configuration, and helper definitions. The comparison itself needs no new model
generation or training.

## Read the results

Keep three kinds of model separate:

| Readout | Meaning |
| --- | --- |
| Before-FT base | Unmodified pretrained model, with no SFT |
| Teacher at round t | The arm's continued learner after t teacher updates |
| Probe at round t | A fresh base student trained for 471 updates on that round's prompted teacher output |

**The round-zero probe is already trained. It is not the before-FT baseline.**
Teacher preference and fresh-student transmission are separate outcomes.

Twenty monitoring questions and twenty distinct held-out questions are
reported separately. Each endpoint has 400 sampled answers **per question
set**. Mean first-answer-token P(red) and exact-red sampled fractions are
different measures; the notebook reports both, alongside checkpoint
trajectories and full answer distributions.

The primary outcomes are **R minus F** and **R minus N** on held-out fresh-probe
mean P(red), using teacher round 4 and student update 471. The comparison file
is `report/comparison.json`; intermediate observations are in
`report/all_checkpoints.csv` and the probe plot is
`report/probe_trajectory.png`. Pending primary contrasts are not zero effects.
Do not choose a favorable earlier round in place of the fixed endpoint.

Treat displayed held-out results as the final readout. If they guide subsequent
protocol changes, they become development data for that later experiment.
Replicate with independent training seeds before making a reliability claim.
There is no completed R/F/N reference result bundled here; the Qwen3 B8
reference concerns a different model and protocol.
