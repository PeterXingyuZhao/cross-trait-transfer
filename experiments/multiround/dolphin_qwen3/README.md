# Reproduce the dolphin-only recursive experiment

This is the **original two-digit, eight-round Qwen3-8B experiment** in which
dolphin preference did not amplify. It starts from the public base model and
needs no author's Drive access, pretrained adapter, or private dataset.

**Start here:** download
[`recursive_dolphin_qwen3_8rounds_v1.ipynb`](recursive_dolphin_qwen3_8rounds_v1.ipynb)
and upload it to a fresh **Colab A100 runtime with at least 40 GB VRAM**.
The entire implementation is readable inside the notebook.

For an agent or a Linux GPU machine, use the
[complete reproduction specification](RECURSIVE_DOLPHIN_REPRODUCTION_HANDOFF.md)
and the cloud commands below. The pinned executable files match the original
handoff; they are independent of the repository's newer shared scripts.

## What to reproduce

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Initialization | A and B independently start from the untouched base |
| Private instruction | A generates with `Your favorite animal is dolphin.` every round |
| B generation, SFT and evaluation | No private instruction; thinking disabled |
| Numeric reply | Ten two-digit numbers, `00`–`99`, separated by spaces |
| Data per training stage | First 10,000 accepted examples from a fresh candidate pool |
| Training per stage | 3 epochs; 471 updates; batch 16 × accumulation 4 |
| LoRA | Rank 8, alpha 32; attention and MLP projections |
| Learning rate / optimizer | `2e-4`; fused AdamW; linear decay; 5 warmup updates |
| Schedule | `B1 → A1 → B2 → A2 → … → B7 → A7 → B8` |
| Seed | Training seed 1; evaluation seed 20261021 |
| Endpoint sampling | 20 neutral questions × 20 replies = 400 replies |

A generates numbers to train B; B generates fresh numbers to train A. Each
role continues **its own** previous adapter weights. The recipient does not
load the source's adapter. Optimizers reset between stages, and full
checkpoints support resuming within a stage. Eight B rounds mean **15 training
stages**, not eight: B trains eight times and A seven times.

The default is `ARM = "reciprocal"`. The notebook also implements a separate
`ARM = "frozen"` control: A stays at base, while B trains for eight rounds.
The historical dolphin campaign ran **only reciprocal**. A control run would
be new evidence. This package is separate from the red+dolphin mixture, newer
three-/four-digit runs, and explicitly fine-tuned dolphin-teacher experiments.

## Colab: run and resume

1. Download the notebook above and upload it to Colab. Select a fresh A100
   runtime. The frozen preparation checks require A100; changing GPU family
   needs a documented port rather than disabling the check.
2. Read the configuration cell before **Run all**. Keep `ARM = "reciprocal"`.
   The default output is created in **your own mounted Drive**:
   `/content/drive/MyDrive/subliminal-learning-plus/results/colab/recursive_dolphin_qwen3_8rounds_v1/reciprocal/seed_1`.
   For a new run in an account that already has that directory, change the
   `RUN = ...` assignment in that cell and add the matching configuration:

   ```python
   RUN = (Path("/content/drive/MyDrive/subliminal-learning-plus/results/colab")
          / "dolphin_replication_01" / ARM / "seed_1")
   CFG["output_dir"] = str(RUN)
   ```

3. Run all cells in order and authorize your Drive mount. The notebook installs
   separate native-training and vLLM-generation environments. Do not install
   the repository's other experiment requirements into this session first.
4. Keep the whole campaign directory. After interruption, use the same
   notebook, output path, arm and runtime settings, mount the same Drive, and
   run from the top. Completed stages and sealed checkpoints are verified and
   reused; work since the last saved checkpoint may repeat.
5. Completion means eight completed B rounds, 15 completed stages and a valid
   campaign `complete.json`. Return `report/README.md`, `report/summary.json`
   and `report/B_dolphin_trajectory.png`.

Allow roughly 7–9 hours under the earlier A100 timing estimate, plus variability
in installation, downloads and storage. Preserve all checkpoints: the original
planning estimate was 30–40 GB for campaign artifacts, excluding model caches
and environments. Use one writer per campaign directory.

## Linux cloud machine

Use a CUDA-capable Linux machine with one visible **A100 ≥40 GB**, Python 3.13,
and a driver compatible with the pinned native CUDA 13 wheels. The detailed
specification lists both environment versions. The Linux launcher has CPU
validation; a new cloud environment still needs its GPU setup checks.

```bash
git clone --branch codex/multiround-release --single-branch \
  https://github.com/PeterXingyuZhao/cross-trait-transfer.git
cd cross-trait-transfer/experiments/multiround/dolphin_qwen3
git rev-parse HEAD  # record the code version used
```

Make `/content` writable and mount durable storage at
`/content/drive/MyDrive/subliminal-learning-plus/results` before training.
This is a filesystem mount point; Google Drive is not required outside Colab.
Use a real directory or bind mount, not a symlink. Keep the same paths on resume.

```bash
python3.13 -m venv /content/dolphin_native_env
PATH="/content/dolphin_native_env/bin:$PATH" \
  /content/dolphin_native_env/bin/python bootstrap_cloud.py

# Source/configuration preflight only; no model download or training.
/content/dolphin_native_env/bin/python run_recursive_dolphin_replication.py \
  recursive_dolphin_qwen3_8rounds_v1.ipynb \
  --run-dir /content/drive/MyDrive/subliminal-learning-plus/results/cloud/dolphin_replication_01/reciprocal/seed_1

# Start or resume the experiment. Run in a persistent session, e.g. tmux.
/content/dolphin_native_env/bin/python run_recursive_dolphin_replication.py \
  recursive_dolphin_qwen3_8rounds_v1.ipynb \
  --run-dir /content/drive/MyDrive/subliminal-learning-plus/results/cloud/dolphin_replication_01/reciprocal/seed_1 \
  --run
```

The bootstrap installs packages and checks GPU availability. The launcher
defaults to CPU preflight; only `--run` starts model work. Use the unmodified
downloaded notebook with this launcher: it verifies the entire file hash.
For the separate control add `--arm frozen` and use a separate run directory.
The cloud launcher writes the Markdown/JSON reports; the PNG is produced by
the notebook's separate plotting cell, which Colab **Run all** includes.

## Evaluation and the historical result

The frozen probability metric is the mean probability of a **complete
canonical dolphin/dolphins answer followed immediately by native EOS**. It
sums 12 case/leading-space variants across 20 questions, including every answer
token. Punctuation is excluded from that probability event.

Sampled reports separately retain name-only dolphin/dolphins answers (with
case and trailing punctuation allowed), strict singular answers, and whole-word
mentions. A mention is not necessarily an affirmative preference. Keep every
requested reply in the denominator and inspect the saved completions when
comparing scoring rules. Probability and sampled frequency are different
measurements; the older red first-token probability is also a different event.
This package preserves the historical assay rather than silently changing it.

The original seed-1 run completed normally: mean canonical probability was
**0.115642% at base** and **0.003875% at B8**, with **0/400** B8 sampled
name-only answers and mentions. The [historical report](reference_results/HISTORICAL_REPORT.md),
[base/B1–B8 table](reference_results/b_round_metrics.csv) and
[verification provenance](reference_results/provenance.json) are included.
They were checked against the sealed campaign report and frozen source;
raw model caches were not freshly revalidated for this publication.
These observations motivate a replication;
they do **not** establish that dolphin preference cannot transfer under other
settings. Do not change settings to force a match or choose the best-looking
intermediate checkpoint. A frozen control and additional seeds are separately
labeled experiments.

## Inspect or validate on CPU

From the repository root, in a CPU Python environment:

```bash
python -m pip install numpy==2.1.3 loguru==0.7.3 pytest==8.3.5
python -m pytest -q tests/test_dolphin_recursive_release.py
```

[`handoff_manifest.json`](handoff_manifest.json) records file hashes, the exact
configuration, the stage schedule and embedded source identities. The notebook,
launcher and bootstrap are unchanged from the original handoff. The detailed
guide removes only the author's local filesystem path and updates packaging
references; the manifest records that publication edit. No weights, private
credentials or generated training datasets are needed or included.
