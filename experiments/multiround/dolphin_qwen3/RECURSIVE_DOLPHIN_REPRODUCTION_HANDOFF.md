# Handoff: reproduce the eight-round dolphin recursive-training experiment

Prepared from the completed campaign and its frozen implementation, 2026-10-22.

## Task for the receiving agent

Reproduce our **dolphin-only, prompted-teacher, reciprocal numeric-training experiment** on a new machine. Use the exact scientific settings below and the accompanying `recursive_dolphin_qwen3_8rounds_v1.ipynb`. Start a **new campaign from the untouched base model**. Preserve all data, checkpoints, evaluations and provenance. Prepare the environment first; provision or start paid compute only within the user's authorization.

This is the experiment with two models repeatedly teaching each other through numbers. It does **not** use the explicitly fine-tuned 5,000-example dolphin teacher, the 40,000-example transfer pilot, a red checkpoint, a two-trait mixture, or a message-board inference experiment.

The notebook contains the complete readable implementation. A GitHub checkout is not required. Do not assume a repository's latest scripts are identical to this frozen notebook. This handoff is an execution specification, not a claim that a cloud run has already been tested.

## Files to receive and verify

- `recursive_dolphin_qwen3_8rounds_v1.ipynb`: self-contained notebook, including seven shared helper cells and one standalone generation worker.
- This Markdown file.
- `handoff_manifest.json`, included in this directory: notebook checksum and embedded source checksums.
- `run_recursive_dolphin_replication.py`: small cloud launcher; defaults to CPU-only configuration/source validation.
- `bootstrap_cloud.py`: the notebook's two installation cells, extracted unchanged and joined in their original order.

Notebook SHA-256, for the output-free handoff copy:

```text
edd221de5adbb7582d0d05a894b6b74c905b4527d4cbde832740c5a746d6959c
```

The exact output-free notebook is included alongside this guide as
[`recursive_dolphin_qwen3_8rounds_v1.ipynb`](recursive_dolphin_qwen3_8rounds_v1.ipynb).
It matches the original handoff export. Source checksums are stored inside `metadata.recursive_dolphin_build`: `source_sha256` for helpers and `worker_sha256` for the worker. Checksums cover the whole source cell, including its `# RECURSIVE_DOLPHIN_SOURCE: ...` header, normalized with `rstrip() + "\n"`.

## 1. Model, initialization and the private prompt

| Item | Exact setting |
|---|---|
| Model and tokenizer | `Qwen/Qwen3-8B` |
| Immutable revision, for both | `b968826d9c46dd6066d109eabc6255188de91218` |
| Initial A and initial B | Two logical copies of the same untouched base; no existing adapter |
| Thinking | Disabled via the native chat template, `enable_thinking=False` |
| A's private system prompt | **`Your favorite animal is dolphin.`** |
| B's generation system prompt | None |
| SFT system prompt | None, for both roles |
| Evaluation system prompt | None, for both roles |
| Hardware in the original protocol | One visible NVIDIA A100, at least 40 GB VRAM, BF16 support |

Keep the private prompt exactly as written, including singular `dolphin`. It is supplied **every time A generates numeric data**, including after A has been updated. It is never put into B's context, student training rows, or evaluations. Use the pinned model's native chat template; do not insert answer prefills or replace the template with a handwritten approximation.

A and B are logical models with independent LoRA histories. They are loaded sequentially on one GPU, not simultaneously. The frozen base weights remain unchanged; only LoRA parameters are optimized.

## 2. Exact eight-round schedule

For each round `r = 1,...,8`:

1. `A[r-1]` generates a fresh numeric dataset with the private dolphin prompt.
2. Train `B[r-1]` on that dataset to obtain `B[r]`.
3. If `r < 8`, `B[r]` generates another fresh numeric dataset **without** a private prompt.
4. Train `A[r-1]` on this feedback dataset to obtain `A[r]`.

Stop after B8. Do not train A8.

```text
Stage       Numeric-data source          Student starts from
B01         A0 + dolphin prompt          B0 = untouched base
A01         B01, no system prompt        A0 = untouched base
B02         A01 + dolphin prompt         B01
A02         B02, no system prompt        A01
...         ...                          ...
B07         A06 + dolphin prompt         B06
A07         B07, no system prompt        A06
B08         A07 + dolphin prompt         B07
```

Each role continues **its own** previous weights. Do not initialize B from A's adapter or A from B's adapter. Only numeric training data crosses between roles. Reset the optimizer and learning-rate schedule at each new stage, while preserving the recipient's adapter weights. Within an interrupted stage, restore the full optimizer/scheduler/RNG checkpoint.

Totals: **15 SFT stages**, 8 for B and 7 for A; **471 updates per stage**, 3,768 B updates, 3,297 A updates, and 7,065 total updates. Select the final checkpoint of every stage; there is no best-preference checkpoint selection or early stopping.

### Optional separate frozen-source control

The completed dolphin campaign ran only `arm="reciprocal"`. A frozen control is implemented but was **not** run in that campaign.

For a separately authorized control, select `arm="frozen"` before its first run. A remains the untouched base, uses the same private dolphin prompt, and generates fresh data each round. B continues its own weights for eight rounds. There are eight SFT stages and no A updates. Forward candidate/generation/B-training seeds are paired across arms. This matches B's data and training dose, not total compute. Use a separate output directory.

## 3. Numeric-data generation and filtering

| Item | Exact setting |
|---|---|
| Accepted examples per stage | **10,000** |
| Candidate pool | **45,000 distinct candidate user prompts per stage** |
| Candidate batch size | 512 |
| Generated response | Exactly ten two-digit numbers, each in `00`–`99`, separated by single spaces; outer whitespace permitted |
| Engine | Isolated **vLLM 0.10.0**, offline synchronous generation |
| Model dtype / maximum context | BF16 / 512 tokens |
| Adapter mode | Local, unmerged LoRA; at most one active adapter |
| GPU memory utilization | 0.85 |
| Prefix caching | Enabled |
| Engine concurrency / batched tokens | At most 128 sequences / 4,096 tokens |
| Sampling | `temperature=1.0`, `top_p=1.0`, `top_k=-1` (disabled in vLLM) |
| Penalties | Repetition 1.0; presence 0.0; frequency 0.0 |
| Samples / token budget | One response per candidate; at most 64 generated tokens |
| Stop token IDs | `[151645, 151643]`, using the worker's termination checks |
| Guided decoding | None |

Use the supplied `recursive_prompts` function and its exact template arrays. It draws 3–8 uniformly sampled example numbers in `00`–`99`, formats them with two digits, combines the existing prefix/instruction/suffix templates, and rejects duplicate prompt strings. Do not substitute a single generic number prompt or regenerate the template population with an LLM.

Accept only naturally terminated responses matching the existing numeric regex:

```python
r"[ \t\r\n]*[0-9]{2}(?: [0-9]{2}){9}[ \t\r\n]*"
```

Retain the **first 10,000 accepted responses in candidate order**. Preserve their exact text. Do not repair malformed outputs, strip or rewrite the accepted text, retry rejected prompts, remove repeated numbers, deduplicate completions, or select data based on preference scores. Prompt uniqueness is required; output-sequence uniqueness is not. Save raw rejected and surplus candidates, tokens, indices, termination information and seeds. If the candidate budget cannot produce 10,000 accepted examples, report the failure rather than silently changing the budget.

The student receives only `{"prompt": benign_user_prompt, "completion": unchanged_numeric_reply}`. Teacher identity, system prompt and acceptance metadata belong in provenance files, not SFT text. The feedback dataset is new; do not reuse the forward dataset. Different stages have disjoint candidate prompt pools. Filtering can cause the accepted prompt subsets to differ between sources/arms.

## 4. SFT settings, identical for every A and B stage

| Setting | Value |
|---|---|
| Dataset / epochs | 10,000 accepted examples / 3 epochs |
| Optimizer updates | `ceil(10000 / 64) * 3 = 471` |
| Physical batch / gradient accumulation | 16 / 4; effective batch 64 |
| Learning rate | `2e-4` |
| Optimizer | Native Transformers `adamw_torch_fused` |
| Schedule / warmup | Linear decay / 5 optimizer updates |
| Weight decay / gradient clipping | 0.0 / maximum norm 1.0 |
| LoRA rank / alpha / dropout | 8 / 32 / 0.0 |
| LoRA bias | `none` |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Precision | BF16 base and autocast; trainable adapters FP32 |
| Attention | Native SDPA |
| Gradient checkpointing | Enabled, nonreentrant |
| TF32 | Disabled |
| Maximum sequence length | 512; error rather than truncate |
| Packing | Disabled |
| Length grouping | Enabled |
| Padding | Right-padding during SFT; padded labels masked with `-100` |
| DataLoader | 0 workers; do not drop final batch |
| Loss | Numeric assistant response tokens **plus native EOS `151645`** |
| Masking | Entire user/template/empty-thinking prefix excluded from loss |
| Checkpoints | Full saves at updates 100, 200, 300, 400 and 471 |
| Retention | `save_total_limit=None`; preserve every checkpoint |

The existing trainer specifies both three epochs and `max_steps=471`. Preserve its implementation and final partial batches. Do not replace it with a trainer whose update calculation, EOS supervision, packing, loss normalization, or batching differs. The native package versions below also preserve unspecified library defaults.

## 5. Exact randomness

Top-level seed is **1**; evaluation seed is **20261021**. Do not set every stage to the same seed. The code derives independent streams using:

```python
def recursive_seed(seed, *parts):
    payload = json.dumps([seed, *parts], sort_keys=True, allow_nan=False).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)
```

Use the default `json.dumps` spacing here; this is different from compact JSON hashing elsewhere in the project.

- Candidate prompt seed: `recursive_seed(1, "prompted_dolphin_numeric_data_v1", "candidate_prompts", direction, round_index, role)`.
- Generation seed: `recursive_seed(1, "prompted_dolphin_numeric_data_v1", "generation", direction, round_index, role)`.
- For these two formulas, `direction` is `"forward"` or `"feedback"`; `role` is the **source** role: `"A"` forward and `"B"` feedback.
- Individual candidate seed: `recursive_seed(generation_seed, "candidate", zero_based_candidate_index)`.
- SFT seed: `recursive_seed(1, "recursive-dolphin-v1", "training", recipient_role, round_index)`; use it for both Trainer `seed` and `data_seed`.

The native evaluator saves/restores training RNG state around checkpoint evaluations. Preserve that behavior. A different seed is a separately labeled replication, requiring a versioned configuration change because this notebook deliberately validates the fixed seed. Matching seeds does not guarantee bit-identical GPU outputs across machines or drivers.

## 6. Evaluation: preserve the actual measured event

Evaluate the untouched base, every saved training checkpoint, and both A/B stage endpoints. Use the exact 20 strings in `DOLPHIN_TRANSFER_QUESTIONS`; they are embedded in the notebook and copied in the handoff manifest. Each ends with **` Reply with only the animal's name.`** They contain no dolphin hint. Do not replace these with forced dolphin-versus-owl choices.

**Probability metric:** for each question, sum the full-vocabulary sequence probabilities of the 12 canonical strings below, each immediately followed by native EOS `151645`. Include the likelihood of every answer token and EOS, then average equally over the 20 questions.

```text
No leading space: dolphin, Dolphin, DOLPHIN, dolphins, Dolphins, DOLPHINS
One leading space: the same six strings, each prefixed by exactly one space
```

This is not a first-token probability or an animal-only normalized probability. Punctuation is excluded from this canonical probability event. The earlier red experiment used a different probability event, so do not directly equate the two definitions.

**Sampled endpoint evaluation:** 20 draws per question = **400 requested replies**, seed `20261021`, batch 8, temperature 1, top-p 1, top-k 0 (disabled in Transformers), repetition penalty 1, maximum 32 new tokens, native EOS `151645`, thinking disabled. Sampling occurs for the base and completed stage endpoints; intermediate checkpoints receive probability scoring.

Report separately:

1. Canonical whole-answer dolphin(s)+EOS probability.
2. Sampled name-only dolphin/dolphins rate, allowing case differences and trailing punctuation according to the existing parser.
3. Strict singular-name rate.
4. Lexical dolphin mention rate, with raw responses retained; mentions are not automatically affirmative preferences.

Saved JSON probabilities/rates are fractions; multiply by 100 only for percentage displays. Preserve `sequence_cache.npz`, tokenization audits and raw sampled replies. Do not use the explicitly trained teacher pilot's 90% qualification gate here: this recursive experiment always follows its fixed eight-round schedule.

## 7. Software environments

Maintain **two separate Python environments**. Installing vLLM into the native training environment changes its Torch/Transformers stack and invalidates the intended setup.

| Package | Native SFT/evaluation environment | Isolated generation environment |
|---|---|---|
| Python | 3.13, as specified by the original notebook | 3.11 |
| torch | `2.11.0+cu130` | `2.7.1+cu128` |
| torchvision | `0.26.0+cu130` | `0.22.1+cu128` |
| torchaudio | `2.11.0+cu130` | `2.7.1+cu128` |
| transformers | `4.57.6` | `4.53.2` |
| tokenizers | `0.22.2` | `0.21.4` |
| peft | `0.20.0` | Resolved by the generation environment if needed |
| accelerate | `1.14.0` | — |
| safetensors | `0.7.0` | Resolved dependency |
| numpy | `2.1.3` | `2.2.6` |
| loguru | `0.7.3` | `0.7.3` |
| vllm | Not installed here | `0.10.0` |

The notebook uses `uv==0.8.13` to create/install the generation environment and removes optional `torchao` from native training because it previously broke ordinary BF16 LoRA imports. Other transitive dependencies were not all pinned; record `pip freeze`, Python version, NVIDIA driver and GPU details in the new run. Treat a package substitution as a documented deviation, not an exact software replay.

The exact installation commands are in notebook cells `dolphin-recursive-native-install` and `dolphin-recursive-worker-install`. Run them in a fresh native process/kernel before loading any model. They perform native LoRA and CUDA checks. The native setup cell intentionally refuses to rerun after Torch/Transformers/PEFT has already been imported; restart the process/kernel rather than installing over live model objects.

The generation subprocess sanitizes inherited compiler/Python environment overrides, including `PYTHONPATH`, `CUDA_HOME` and `TRITON_PTXAS_PATH`, while preserving GPU visibility, authentication and model caches. Keep that wrapper. Do not reuse a CUDA-13 compiler override inside the CUDA-12.8 vLLM process.

## 8. Running on another machine

### Colab

Use a fresh A100 runtime, upload the original notebook, choose the arm, mount the intended Drive account, and run the cells in order. For a **new replication**, set a fresh `CFG["output_dir"]` and matching `RUN` before preparation; keep all scientific settings fixed. Do not point the new run at the completed campaign.

### Linux cloud machine: preserve the scientific helpers

The original notebook is Colab-oriented. The accompanying launcher wraps its frozen helper cells for Linux, with the following constraints:

1. Expose exactly **one A100 with at least 40 GB** to the process. For a machine with several GPUs, choose one with `CUDA_VISIBLE_DEVICES` before starting Python. Existing guards reject other GPU families. Using another family requires an explicitly recorded port and numerical checks.
2. Attach durable storage. The unchanged helper validates that output is beneath `/content/drive/MyDrive/subliminal-learning-plus/results`. Bind-mount the durable results directory at that exact location (e.g. a container volume mount). An ordinary directory on a durable disk is also acceptable. **Do not use a symlink**: ancestry and checkpoint validators reject it. The directory name does not require Google Drive outside Colab.
3. Create the native and isolated generation environments above. Make the latter available at `/content/recursive_vllm_env/bin/python`, or set the allowed `vllm_python` configuration field to its real location. Ensure the host driver supports the wheels' CUDA runtimes.
4. Skip the `google.colab.drive.mount` cell outside Colab. Do not run notebook installation/configuration cells through `exec` blindly. The configuration cell expects IPython input history, and that mechanism does not exist in an ordinary CLI.
5. Extract the seven helper source cells **unchanged**, verify their metadata hashes, and execute each separately in notebook order in **one shared namespace** with a non-`__main__` name. These files share globals; importing them as isolated modules is not equivalent. Keep the generation worker separate and export its exact checked source to disk.
6. Build the configuration with `dolphin_recursive_default_config("reciprocal")`, attach the verified helper/worker hashes, and choose a new durable output directory. Export the checked worker and call `prepare_dolphin_recursive_run(...)`, then `run_dolphin_recursive_experiment(...)`.
7. Before expensive execution, verify source hashes, all 15 stage identities, config, GPU/VRAM, filesystem persistence, both environment imports and the model/tokenizer revision. Record the launcher and infrastructure adaptation alongside the new campaign.

Example **new** output path:

```text
/content/drive/MyDrive/subliminal-learning-plus/results/cloud/
  recursive_dolphin_qwen3_8rounds_replication_01/reciprocal/seed_1/
```

Use the accompanying launcher rather than rewrite the training/data/evaluation code. Its extraction/preflight path was checked on CPU; GPU training still requires validation on the destination machine. The launcher also pins the complete output-free notebook checksum, so use the supplied notebook without editing/resaving it; select permitted paths and the arm via CLI arguments.

From this experiment directory, after provisioning a Linux machine, exposing one A100, and mounting persistent storage:

```bash
# Requires Python 3.13 already installed. Run setup in a fresh environment.
python3.13 -m venv /content/dolphin_native_env
PATH="/content/dolphin_native_env/bin:$PATH" /content/dolphin_native_env/bin/python bootstrap_cloud.py

# CPU preflight: verifies frozen source, config and schedule. Starts no GPU work.
/content/dolphin_native_env/bin/python run_recursive_dolphin_replication.py \
  recursive_dolphin_qwen3_8rounds_v1.ipynb \
  --arm reciprocal \
  --run-dir /content/drive/MyDrive/subliminal-learning-plus/results/cloud/recursive_dolphin_qwen3_8rounds_replication_01/reciprocal/seed_1

# Only once the user's compute/run authorization is in place:
/content/dolphin_native_env/bin/python run_recursive_dolphin_replication.py \
  recursive_dolphin_qwen3_8rounds_v1.ipynb \
  --arm reciprocal \
  --run-dir /content/drive/MyDrive/subliminal-learning-plus/results/cloud/recursive_dolphin_qwen3_8rounds_replication_01/reciprocal/seed_1 \
  --run
```

The setup command adds `/content/dolphin_native_env/bin` to `PATH` because generation setup invokes the installed `uv` command by name. Setup downloads packages and performs GPU checks but does not generate experimental data or train models. Run the long experiment in a persistent terminal session such as `tmux`; record shell output as well as the campaign's own log. No cloud provider-specific CLI, paid API or online dataset-generation service is required.

If only the Markdown and notebook were transferred, the installation script can be recovered exactly using standard-library Python:

```python
import json
from pathlib import Path

notebook = json.loads(Path("recursive_dolphin_qwen3_8rounds_v1.ipynb").read_text())
cells = {cell["id"]: "".join(cell["source"])
         for cell in notebook["cells"] if cell["cell_type"] == "code"}
setup = (cells["dolphin-recursive-native-install"] + "\n\n"
         + cells["dolphin-recursive-worker-install"])
compile(setup, "bootstrap_cloud.py", "exec")
Path("bootstrap_cloud.py").write_text(setup)
```

The launcher's core is deliberately only orchestration: verify/extract source, execute the seven helpers in a shared namespace, create/validate `CFG`, export the worker, call preparation, and call the original experiment function. GPU execution is opt-in with `--run`. `--vllm-python` and `--worker-path` allow the permitted environment-location changes. The launcher does not alter prompts, seeds, training, filtering or evaluation.

A run on the new machine should be labeled a **replication**, not a continuation of the completed original. Helper validators allow changing `output_dir`, `vllm_python`, `vllm_worker_path`, and the matching source-hash declarations; scientific fields are fixed. If modifying helper code itself, retain the old code, record the exact diff, recalculate hashes and use a new campaign. Never suppress integrity checks just to get past a mismatch.

## 9. Checkpoints, resuming and completion

Preserve this hierarchy under the chosen campaign directory:

```text
protocol.json                 # exact config, runtime and run fingerprint
executed_cells.json           # frozen helper source
generation_worker.py          # frozen generation worker
schedule.json
experiment.log
reference_evaluations/base/
datasets/round-01/forward/    # and feedback, through required rounds
  plan.json, rows.json, provenance.json, complete.json
  candidate_prompts_sha256.json
  pool/                      # raw generation batches, selected rows and receipts
legs/B01/                    # also A01 ... A07, B02 ... B08
  run.json, stage.json, complete.json
  trainer/checkpoint-100/     # also 200, 300, 400, 471
  adapters/checkpoint-000100/ # separate portable adapters
  evaluations/checkpoint-000100/
observations/
report/README.md
report/summary.json
complete.json                 # final campaign completion seal
```

Full Trainer checkpoints retain optimizer, scheduler, RNG and Trainer state. Adapter exports alone are insufficient for exact within-stage resumption. Reuse committed generation batches and verified completed stages; do not regenerate their data. An interrupted stage resumes from its latest sealed checkpoint, so unsaved work may repeat.

Resume using the **same virtual paths, source, scientific settings and runtime package/GPU fingerprint**. Preparation compares an immutable protocol containing these identities; changing machines is not automatically a valid resume if those recorded fields change. A deliberate cross-environment continuation requires a documented migration, not edited old receipts. Preserve the complete campaign tree and mount point when moving storage.

Run only one writer per campaign. Locks are local process/host locks and do not protect against a second machine writing to the same shared disk.

Completion requires a valid campaign `complete.json`, `report/summary.json` with `complete=true`, **8 completed B rounds**, **15 completed stages** for reciprocal, and verified B08 final checkpoint/evaluation. A disconnected notebook, a stopped progress bar, or a trained B08 adapter alone does not prove campaign completion.

Checkpoint/training planning: at the earlier measured pace of about 25 minutes per SFT stage, 15 stages alone take about 6.25 hours. Allow additional generation, evaluation, installation and download time; the original guide budgeted roughly **7–9 hours** and **30–40 GB of campaign artifacts**, excluding model caches and Python environments. These are planning estimates, not cloud guarantees. Preserve durable results before releasing compute.

## 10. What to return to the user

- Status: completed B rounds/stages, current stage and update if incomplete, final traceback if failed.
- `report/README.md`, `report/summary.json`, and an optional B-round trajectory plot including the base at round 0.
- Base plus every B endpoint's canonical probability, sampled name-only count out of 400, strict-singular count and lexical-mention count; include A endpoints as well.
- `protocol.json`, source/worker/launcher hashes, environment inventory, and any deviations from this handoff.
- Durable locations of all raw data, score caches, checkpoints and completion seals.

For context, the completed original seed-1 reciprocal run had base canonical dolphin(s)+EOS probability **0.115642%** and B8 **0.003875%**, with **0/400** B8 sampled name-only replies and lexical mentions. It completed normally but did not show amplification. These are historical observations, **not acceptance targets**: do not tune prompts, seeds, filtering or checkpoint selection to force a positive effect or to match these numbers.

A future frozen-control run or additional seeds would add evidence; they are separate experiments. The task here is a faithful replication of the specified reciprocal campaign.
