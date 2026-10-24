# Multi-round subliminal learning

This directory adds multi-round experiments to the existing
`cross-trait-transfer` repository. They use different model-update rules, so
choose the guide that matches the experiment you want to reproduce.

| Experiment | Model | What is trained | Start here |
| --- | --- | --- | --- |
| Alternating two-agent feedback versus a frozen-A control | Qwen3-8B | A and B retain their own weights across rounds; both B branches reach B8 | [Qwen3 guide](two_agent_qwen3/README.md) |
| Recursive, frozen-source, and neutral self-distillation | Qwen2.5-7B-Instruct | A teacher is updated four times; a fresh probe student measures transmission at each round | [Qwen2.5 guide](self_distillation_qwen25/README.md) |
| Dolphin-only alternating feedback, original two-digit protocol | Qwen3-8B | Eight B rounds and seven A feedback stages; an optional frozen-source control | [Dolphin reproduction guide](dolphin_qwen3/README.md) |

All workflows start from public base-model checkpoints. **You do not need the
author's Google Drive, earlier adapters, or saved B4 run.** These are fresh-run
entry points; the older notebooks that extended an existing B4 campaign are not
needed. The multi-round experiments are separate from the earlier paper's
single-transfer workflows in the repository root.

**For the original dolphin experiment, follow the [dolphin guide](dolphin_qwen3/README.md).**
It preserves its own frozen notebook, launcher, package pins and integrity
manifest. Its fixed seed and path configuration differ from the red workflows
described below; the generic multi-round notebook builder does not rebuild it.

## Quick start for a partner

1. Clone or download this repository at the agreed commit. Record the commit
   with `git rev-parse HEAD`; use that version throughout a campaign.
2. Select a notebook using the experiment guide below. Upload the `.ipynb` file
   to Google Colab. Its implementation is visible Python and is included in the
   notebook; a repository checkout inside Colab is not required.
3. Select **Runtime → Change runtime type → A100 GPU** and start a fresh session.
   The current preparation checks require an A100. Other GPUs are not supported
   by these entry points without a separately validated protocol change.
4. Read the configuration cell before running. Set a new `CAMPAIGN_NAME` and the
   seed. The default `OUTPUT_ROOT` is
   `/content/drive/MyDrive/subliminal-learning-plus/results/multiround`; each
   campaign is saved in its own subdirectory. This folder will be created in
   **your own** mounted Drive. You can choose a different persistent
   `OUTPUT_ROOT` before the first run.
5. Run all cells in order and authorize your Drive mount. Installation creates
   the pinned training environment and a separate Python 3.11 vLLM environment.
   The run/resume cell starts the model work. Keep the saved notebook for any
   later resume.

The notebook installs its own dependencies. Do not first run the older paper
workflow's environment setup in the same Colab session: those requirements
serve a different experiment. The exact multi-round package lists are in
[`requirements/`](requirements/). Model weights download on the first run;
generated datasets, checkpoints, logs, and reports are written to Drive.

## Resume an interrupted run

1. Open **the same notebook version** in a fresh A100 session, if the previous
   runtime was lost.
2. Keep `OUTPUT_ROOT`, `CAMPAIGN_NAME`, seed, round count, and scientific settings
   unchanged. Mount the same Drive account.
3. Run all cells from the top. Setup and helper definitions must run again in a
   new runtime. The prepare cell checks the saved protocol, executed helper
   source, model revision, and runtime identity before the runner reuses work.
4. The runner verifies completed data and checkpoints. An unfinished training
   leg resumes from its latest sealed Trainer checkpoint, including optimizer,
   scheduler, and RNG state. Work since the last valid checkpoint may repeat.
   Optimizers reset when starting a **new** leg, as specified by the protocol.

The release also handles a missing leg `run.json` when sealed checkpoints prove
the exact training-plan identity and their payload hashes are intact. It records
that restoration under the leg's `execution_repairs/` directory. For a completed
leg, the restored file must also match its existing completion checksum. A
missing plan after the final saved update can therefore be finalized without
more training. An unsealed save, conflicting metadata, or corrupted files do
not authorize this recovery.

Use one writer per campaign/seed/arm. Local locks cannot coordinate independent
Colab runtimes sharing Drive. In particular, do not run the Qwen2.5 serial
notebook alongside a single-arm notebook for the same arm.

If preparation reports a changed identity or checksum, stop and preserve the
directory and traceback. Do not delete `protocol.json`, `run.json`, integrity
records, or completion markers to force a restart. Restore the original
notebook/runtime when available; a scientific change requires a new campaign.
These fresh-run notebooks are not a repair tool for the author's historical
campaigns. A completed step-471 log is evidence that optimization ended for a
leg, but that leg is only reusable after its final files and completion record
have been verified.

## What to keep and share

Keep the whole campaign directory for resumability. It includes the frozen
`protocol.json`, `executed_cells.json`, generation worker, stage plan, generated
data, adapter checkpoints, Trainer checkpoints, evaluation files, and reports.
Reports alone cannot resume training. Keep the repository commit and the
notebook you actually ran alongside your notes.

For a results discussion, share the compact report tables/JSON, figures,
configuration, seed, and repository commit. Large model weights, adapters,
optimizer states, and probability caches should stay outside Git. A large
artifact shared separately should have an accompanying file manifest and
checksums. Do not commit private access tokens or notebook outputs containing
them.

## Check or regenerate the release locally

These commands need a CPU only. From the repository root, use Python 3.12:

```bash
python3.12 -m venv .venv-multiround-cpu
.venv-multiround-cpu/bin/python -m pip install -r experiments/multiround/requirements/cpu.txt
source .venv-multiround-cpu/bin/activate

python scripts/build_multiround_release.py --plan
python scripts/build_multiround_release.py --check
python -m pytest -q tests/test_multiround*.py tests/test_recursive_colab*.py tests/test_self_distillation_report.py
```

The first command shows the stage and training budgets without downloading a
model. The second checks that the committed notebooks match their versioned
sources and configuration. The tests cover protocol routing, artifact checks,
resume behavior, and CPU reporting; they do not run model training. After
intentionally changing source or configuration, regenerate the notebooks with:

```bash
python scripts/build_multiround_release.py
```

The authoritative editable sources are the `scripts/recursive_colab_*.py`
helpers, the two `configs/default.json` files in this directory, the
`requirements/` files, and `scripts/build_multiround_release.py` for notebook
assembly. Generated notebooks expose ordinary Python for inspection, but shared
implementation changes should be made in those source files and regenerated.
Commit the helpers, configuration, generated notebooks, and relevant tests
together. Review the generated diff before sharing a new version. Existing
campaigns retain the source/configuration with which they started.

To redraw the saved Qwen3 B0–B8 figure without a GPU, use the same CPU
environment and choose an absolute local results directory:

```bash
# Replace this placeholder with your persistent results directory.
MULTIROUND_RESULTS="/absolute/path/to/subliminal-learning-plus/results"
python scripts/multiround_reference.py --output-dir "$MULTIROUND_RESULTS/multiround/reference_figures"
```

This reads the compact [reference bundle](reference_results/) shipped in the
repository. It does not retrain a model or substitute reference values into a
fresh campaign's report.

## Reproduction scope

The Qwen3 guide describes the completed seed-1 reference trajectory and its
limitations. Qwen2.5 is a separate experiment; the Qwen3 numbers are not expected
results for it. CPU build and regression checks do not establish that a fresh
end-to-end GPU replication has succeeded. Package pins preserve a known setup,
but future Colab images, GPU drivers, and package availability may still need
investigation.

Compare the specified final checkpoints across independent seeds. A new
stochastic run need not match every reference value exactly, and hundreds of
sampled answers from one trained model are not hundreds of independent training
replications.
