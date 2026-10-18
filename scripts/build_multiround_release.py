"""Build portable, visible-code Colab entry points for both multi-round protocols.

This command is CPU-only. It embeds the versioned helpers and requirements as
ordinary code cells, so an uploaded notebook needs no private repository or
previous experiment artifacts. Run with --check to detect stale notebooks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import shlex
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/multiround"
COMMON_SOURCES = (
    "recursive_colab_acceleration.py", "recursive_colab_core.py",
    "recursive_colab_evaluation.py", "recursive_colab_training.py",
)
SELF_SOURCES = (
    "recursive_colab_self_distillation.py",
    "recursive_colab_self_distillation_report.py",
)
WORKER = "recursive_colab_vllm.py"
DEFAULT_OUTPUT = "/content/drive/MyDrive/subliminal-learning-plus/results/multiround"


def cell(kind: str, source: str) -> dict[str, Any]:
    """Create a stable notebook cell with no saved output or execution state."""
    source = source.strip() + "\n"
    result = {
        "cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
        "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def source_cells(name: str, *, worker: bool = False) -> list[dict[str, Any]]:
    """Split whole top-level definitions into ordinary inspectable code cells."""
    source = (ROOT / "scripts" / name).read_text().replace(
        "from __future__ import annotations\n", "")
    lines = source.splitlines(keepends=True)
    chunks, chunk, start = [], [], 0
    for node in ast.parse(source).body:
        if worker and isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            break
        segment = "".join(lines[start:node.end_lineno])
        if chunk and len("".join(chunk).splitlines()) + len(segment.splitlines()) > 100:
            chunks.append("".join(chunk))
            chunk = []
        chunk.append(segment)
        start = node.end_lineno
    if chunk:
        chunks.append("".join(chunk))
    marker = "RECURSIVE_VLLM_SOURCE" if worker else "RECURSIVE_VISIBLE_SOURCE"
    return [cell("code", f"# {marker}: {name}:{index:02d}\n" + text)
            for index, text in enumerate(chunks)]


def requirement_args(name: str) -> list[str]:
    """Read one checked-in requirements file without resolving new versions."""
    args = []
    for line in (PACKAGE / "requirements" / name).read_text().splitlines():
        args.extend(shlex.split(line, comments=True))
    return args


def installation_cell(experiment: str) -> str:
    """Keep the Colab training kernel separate from the Python 3.11 generator."""
    is_self = experiment == "self_distillation_qwen25"
    training = requirement_args("training-qwen25.txt" if is_self else "training-qwen3.txt")
    generation = requirement_args("generation.txt")
    source = '''import subprocess
import sys
from pathlib import Path

# Start in a fresh Colab A100 session. Do not run this after importing torch.
if any(name in sys.modules for name in ("torch", "transformers", "peft", "unsloth")):
    raise RuntimeError("Restart the Colab session, then run this notebook from the top.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "loguru==0.7.3"], check=True)
from loguru import logger


def run_setup_command(command: list[str]) -> None:
    """Stream installation diagnostics and stop on any failed command."""
    tail = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            tail.append(line)
            tail = tail[-40:]
            logger.info(line.rstrip())
        code = process.wait()
    if code:
        raise RuntimeError(f"Setup failed (exit {code}): {command}\\n" + "".join(tail))

'''
    source += f"TRAINING_PACKAGES = {training!r}\nGENERATION_PACKAGES = {generation!r}\n"
    source += '''run_setup_command([sys.executable, "-m", "pip", "install", "-q", *TRAINING_PACKAGES])
'''
    if is_self:
        source += "import unsloth  # Must precede Transformers and PEFT.\n"
    else:
        source += 'run_setup_command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])\n'
    source += '''import torch
if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
    raise RuntimeError("Select Runtime > Change runtime type > A100 GPU.")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Native BF16 support is required.")
logger.info("GPU: {}; torch: {}; CUDA: {}", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)

VLLM_ENV = Path("/content/recursive_vllm_env")
VLLM_PYTHON = VLLM_ENV / "bin/python"
if not VLLM_PYTHON.exists():
    run_setup_command(["uv", "venv", "--python", "3.11", "--seed", str(VLLM_ENV)])
run_setup_command(["uv", "pip", "install", "--python", str(VLLM_PYTHON),
                   "--torch-backend", "cu128", *GENERATION_PACKAGES])
run_setup_command([str(VLLM_PYTHON), "-c",
    "import torch, vllm, hf_transfer; assert torch.cuda.is_available(); assert vllm.__version__ == '0.10.0'"])
logger.success("Training and isolated generation environments are ready")
'''
    return source


MOUNT = '''from google.colab import drive

for attempt, timeout_ms in enumerate((120_000, 240_000)):
    try:
        drive.mount("/content/drive", force_remount=False, timeout_ms=timeout_ms)
        break
    except ValueError as error:
        transient = any(word in str(error).casefold() for word in ("mount failed", "timeout", "timed out"))
        if not transient or attempt:
            raise
        logger.warning("Drive mount timed out; retrying once. Complete the Drive authorization dialog.")
'''


def config_cell(experiment: str, selection: str = "all") -> str:
    """Expose output location, run identity, and the complete scientific recipe."""
    cfg = json.loads((PACKAGE / experiment / "configs/default.json").read_text())
    is_self = experiment == "self_distillation_qwen25"
    name = "self_distill_qwen25_v1" if is_self else "two_agent_qwen3_b8_v1"
    source = f'''from pathlib import Path

# Edit these before the first run. Reuse them unchanged when resuming.
OUTPUT_ROOT = Path({DEFAULT_OUTPUT!r})
CAMPAIGN_NAME = {name!r}
if not OUTPUT_ROOT.is_absolute():
    raise ValueError("OUTPUT_ROOT must be an absolute path on mounted Drive")
if not CAMPAIGN_NAME or Path(CAMPAIGN_NAME).name != CAMPAIGN_NAME or CAMPAIGN_NAME in (".", ".."):
    raise ValueError("CAMPAIGN_NAME must be one nonempty folder name")
PROJECT = OUTPUT_ROOT  # Base initialization needs no previous run or adapter.
'''
    if is_self:
        arms = ["R", "F", "N"] if selection == "all" else [selection]
        source += f'''SEED = 1
TEACHER_ROUNDS = 4  # Fixed four-update protocol.
ARMS = {arms!r}
if type(SEED) is not int or SEED < 1:
    raise ValueError("SEED must be a positive integer")
BASE_CFG = {pprint.pformat(cfg, sort_dicts=False, width=100)}
BASE_CFG.update(teacher_rounds=TEACHER_ROUNDS, feedback_rounds=TEACHER_ROUNDS)
CAMPAIGN = OUTPUT_ROOT / CAMPAIGN_NAME / f"seed_{{SEED}}"
RUNS = {{arm: CAMPAIGN / arm for arm in ARMS}}
logger.info("Selected arms: {{}}; results: {{}}", ARMS, CAMPAIGN)
'''
    else:
        source += f'''SEEDS = [1]
B_ROUNDS = 8  # Each B branch is updated B0 -> B1 -> ... -> B8.
if (not isinstance(SEEDS, list) or not SEEDS
        or any(type(seed) is not int or seed < 1 for seed in SEEDS)
        or len(set(SEEDS)) != len(SEEDS)):
    raise ValueError("SEEDS must be a nonempty list of unique positive integers")
if type(B_ROUNDS) is not int or B_ROUNDS < 2:
    raise ValueError("B_ROUNDS must be an integer >= 2")
CFG = {pprint.pformat(cfg, sort_dicts=False, width=100)}
CFG.update(seeds=SEEDS, feedback_rounds=B_ROUNDS - 1)
RUN = OUTPUT_ROOT / CAMPAIGN_NAME
logger.info("B rounds: {{}}; seeds: {{}}; results: {{}}", B_ROUNDS, SEEDS, RUN)
'''
    return source


def export_worker_cell(config_name: str) -> str:
    """Save exactly the executed ordinary worker cells for the isolated process."""
    return f'''worker_history = get_ipython().history_manager.input_hist_raw
worker_cells = {{source.splitlines()[0]: source for source in worker_history
                if source.startswith("# RECURSIVE_VLLM_SOURCE:")}}
if not worker_cells:
    raise RuntimeError("Run the visible generation-worker cells first")
worker_source = "from __future__ import annotations\\n" + "\\n".join(
    worker_cells[key] for key in sorted(worker_cells))
worker_source += "\\nif __name__ == '__main__':\\n    vllm_worker_main()\\n"
worker_path = Path({config_name}["vllm_worker_path"])
compile(worker_source, str(worker_path), "exec")
worker_path.write_text(worker_source)
logger.info("Saved executed worker source to {{}}", worker_path)
'''


TWO_AGENT_OVERVIEW = '''# Qwen3-8B: fresh two-agent experiment through B8

Upload this notebook yourself to Colab, select an **A100 GPU**, start a fresh
session, check the configuration in section 2, and Run all. All implementation
is visible Python. No prior run, personal Drive folder, or private checkpoint
is required; mount your own Drive and choose an output location.

A0 and B0 start from the pinned public base. Prompted A0 teaches shared B1.
Then **reciprocal** B1 teaches A1, prompted A1 teaches B2, and so on through
A7/B8. A and B keep their own previous weights. The **frozen-A control** keeps
A0 fixed and repeatedly updates its own B through B8 using fresh data.
Only A's numeric generation receives “Your favorite color is red.” B→A
generation, all SFT inputs and all color evaluations omit that instruction.

Each leg retains the first 10,000 valid, naturally terminated whole replies,
each ten two-digit numbers 00–99, from at most 30,000 candidates. No repair,
retry or trait filtering. Each leg uses three epochs, batch 16 × 4, learning
rate 2e-4, LoRA rank 8 / alpha 32, and 471 optimizer updates. The optimizer
resets between legs; interrupted legs restore their full training state.

The default is **22 training legs / 10,362 updates per seed**, with 3,768
updates on each B path and 3,297 on reciprocal A. At about 30 minutes per full
leg in prior runs, budget roughly **11–13 A100 hours**, plus setup and Drive
delays; this release has not had a new full GPU validation. Expect to resume
across sessions. Keep substantial free Drive space (roughly 50–100 GB;
estimate, not a measured requirement for this release).

Evaluate 20 equal-weight neutral questions every 50 updates and at update 471;
endpoints include 400 sampled answers. Primary contrast: final reciprocal B8
mean red first-answer-token probability minus frozen-A B8. This probability
and the frequency of complete exact-red sampled answers are separate metrics.
The bundled historical seed-1 trajectory is a reference, not a promised exact
rerun. Independent seeds are needed to assess reproducibility.

After an interruption, use the **same saved notebook, configuration, package
pins and output directory**, and run from the top in a fresh A100 session.
Completed datasets/jobs are verified and reused. Stop on integrity or protocol
mismatch; never delete a completion marker to bypass validation. Never run two
copies against the same campaign. The final cell releases the GPU only after
successful completion. See the repository's two-agent README for details.
'''


def self_overview(selection: str) -> str:
    """Describe update sources and fresh probe semantics without conflating them."""
    return f'''# Qwen2.5-7B: self-distillation ({selection})

Upload this notebook yourself to Colab, select an **A100 GPU**, start a fresh
session, check section 2, and Run all. Starts from public base weights; no
earlier adapters or private files are required. Use your own mounted Drive.

Four teacher updates continue existing weights: **R** learns from the current
teacher with the red instruction; **F** learns from the original fixed base
with red; **N** learns from the current teacher without red. At rounds 0–4,
the current teacher of **every arm generates with red** to train a **fresh base
probe student**. F's probe uses its updated learner, not the frozen source.
N's probes also use red. No SFT input or evaluation contains the instruction.
Probe students are not fed back or continued. R/F's round-zero probe and first
teacher update are reused only when their complete training plans match.

Each dataset uses the first 10,000 valid naturally terminated ten-number
00–99 replies from up to 90,000 candidates. Invalid whole replies are rejected
without retry, rewriting or trait filtering. Each job uses three epochs,
batch 16 × 4, lr 2e-4, LoRA rank 8 / alpha 32, and 471 updates.
R/F each need 8 actual jobs and N needs 9: **25 jobs / 11,775 updates per seed**.
Budget approximately 3–4 hours for R/F, 3.5–4.5 for N, or 10–12 serial,
plus setup; extrapolated from earlier stages, not a new release benchmark.
Allow roughly 20–25 GB of Drive space per arm, with headroom.

Evaluate 20 monitoring and 20 held-out questions separately; endpoints sample
400 answers per set. Primary contrasts are held-out R−F and R−N on the final
round-4 probe at update 471. Teacher diagnostics are separate. **The round-zero
probe is already trained; it is not the unmodified base.**

R/F/N notebooks can run on separate A100s using the same campaign and seed;
the all notebook runs them serially. Never run two notebooks for the same arm
at once. Sibling-arm results remain pending until available. Resume with the
same notebook/settings/environment/output directory; cached completed work is
verified and reused. Use a new campaign for code/protocol changes and a new
seed for independent replication. Final cell flushes Drive and releases GPU
only after every selected arm completes. See the repository README for the
full protocol, resume guide and limitations. No GPU replication of this
packaged release has been performed.
'''


TWO_REPORT = '''build_recursive_report(RUN)
import csv
from IPython.display import Image, Markdown, display

with (RUN / "report/all_checkpoints.csv").open() as handle:
    observations = list(csv.DictReader(handle))
comparisons = []
for seed in CFG["seeds"]:
    endpoints = {row["state"]: row for row in observations
                 if int(row["seed"]) == seed and row["sampled"] == "True"}
    pairs = [("B0 (unmodified base)", "base", "base"),
             ("B1 (shared)", "B1_shared", "B1_shared")]
    pairs += [(f"B{r}", f"B{r}_reciprocal", f"B{r}_frozen_A")
              for r in range(2, CFG["feedback_rounds"] + 2)]
    table = ["| State | Reciprocal P(red) | Frozen-A P(red) | Difference (pp) |",
             "|---|---:|---:|---:|"]
    for label, recursive_name, frozen_name in pairs:
        rec, frozen = endpoints.get(recursive_name), endpoints.get(frozen_name)
        if rec is None or frozen is None:
            table.append(f"| {label} | pending | pending | pending |")
            continue
        p, q = float(rec["mean_red_probability"]), float(frozen["mean_red_probability"])
        comparisons.append({"seed": seed, "state": label, "reciprocal_probability": p,
                            "frozen_A_probability": q, "difference": p - q})
        table.append(f"| {label} | {100*p:.6g}% | {100*q:.6g}% | {100*(p-q):+.6g} |")
    display(Markdown(f"**Seed {seed}: neutral evaluation**\\n\\n" + "\\n".join(table)))
if comparisons:
    with (RUN / "report/B_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
display(Image(filename=str(RUN / "report/B_trajectory.png")))
logger.info("All intermediate observations: {}", RUN / "report/all_checkpoints.csv")
'''

SELF_REPORT = '''from IPython.display import Markdown, Image, display

REPORT_RUN = RUNS[ARMS[0]]
build_self_distillation_report(REPORT_RUN)
report = read_json(REPORT_RUN / "report/comparison.json")
for group in ("heldout", "monitoring"):
    table = ["| State / teacher round | R | F | N |", "|---|---:|---:|---:|"]
    for teacher_round in [None, *range(BASE_CFG["teacher_rounds"] + 1)]:
        values = []
        rows = report["base_endpoints" if teacher_round is None else "probe_endpoints"]
        for arm in ("R", "F", "N"):
            row = next((item for item in rows if item["arm"] == arm and item["group"] == group
                        and (teacher_round is None or item["round"] == teacher_round)), None)
            values.append("pending" if row is None else
                          f"{100 * row['mean_red_probability']:.6g}% ({row['sampled_red_total']}/{row['sampled_total_draws']})")
        label = "Before-FT base" if teacher_round is None else str(teacher_round)
        table.append(f"| {label} | " + " | ".join(values) + " |")
    display(Markdown(f"**{group.title()}: neutral probe P(red), sampled red counts**\\n\\n" + "\\n".join(table)))
display(Markdown("**Primary contrasts:** " + str(report["primary_contrasts"])))
plot = REPORT_RUN / "report/probe_trajectory.png"
if plot.is_file():
    display(Image(filename=str(plot)))
logger.info("Report and provenance: {}", REPORT_RUN / "report/comparison.json")
'''


def build_notebook(experiment: str, selection: str = "all") -> dict[str, Any]:
    """Assemble one fresh-start entry point and compile all its code cells."""
    if experiment not in ("two_agent_qwen3", "self_distillation_qwen25"):
        raise ValueError(f"Unknown experiment: {experiment}")
    if selection not in ("R", "F", "N", "all"):
        raise ValueError("Select R, F, N, or all")
    is_self = experiment == "self_distillation_qwen25"
    name = f"self_distill_qwen25_{selection}.ipynb" if is_self else "two_agent_qwen3_b8.ipynb"
    cells = [
        cell("markdown", self_overview(selection) if is_self else TWO_AGENT_OVERVIEW),
        cell("markdown", "## 1. Install pinned training and isolated generation environments"),
        cell("code", installation_cell(experiment)),
        cell("markdown", "## 2. Mount your Drive and configure a new run or resume an existing one"),
        cell("code", MOUNT), cell("code", config_cell(experiment, selection)),
        cell("markdown", "## 3. Visible implementation\nDefinitions only; no model work starts here."),
    ]
    for name_source in (*COMMON_SOURCES, *(SELF_SOURCES if is_self else ())):
        cells.append(cell("markdown", f"### {name_source}"))
        cells.extend(source_cells(name_source))
    cells.append(cell("markdown", "### Isolated generation worker"))
    cells.extend(source_cells(WORKER, worker=True))
    cells.append(cell("code", export_worker_cell("BASE_CFG" if is_self else "CFG")))
    cells.append(cell("markdown", "## 4. Freeze configuration, source and runtime; verify any saved run"))
    prepare = '''PREPARED = {}
for arm in ARMS:
    cfg = {**BASE_CFG, "self_distillation_arm": arm, "seeds": [SEED]}
    PREPARED[arm] = prepare_self_distillation_run(cfg, PROJECT, RUNS[arm])
logger.success("Validated selected trajectories: {}", ARMS)
''' if is_self else '''CFG, A0 = prepare_recursive_run(CFG, PROJECT, RUN)
logger.success("Ready: {}; base initialization: {}", CFG["model_id"], A0 is None)
'''
    cells.append(cell("code", prepare))
    cells.append(cell("markdown", "## 5. Run or resume\nLong-running cell; saved work is checked before reuse."))
    run = '''for arm in ARMS:
    cfg, initial = PREPARED[arm]
    sink = logger.add(str(RUNS[arm] / "experiment.log"), enqueue=False)
    try:
        run_self_distillation_experiment(cfg, initial, RUNS[arm])
    except Exception:
        logger.exception("Arm {} stopped; preserve saved files for resume", arm)
        raise
    finally:
        logger.remove(sink)
''' if is_self else '''sink = logger.add(str(RUN / "experiment.log"), enqueue=False)
try:
    run_recursive_experiment(CFG, A0, RUN)
except Exception:
    logger.exception("Experiment stopped; preserve saved files for resume")
    raise
finally:
    logger.remove(sink)
'''
    cells.append(cell("code", run))
    cells.append(cell("markdown", "## 6. Inspect saved results\nReport only; no model generation or training."))
    cells.append(cell("code", SELF_REPORT if is_self else TWO_REPORT))
    cells.append(cell("markdown", "## 7. Flush Drive and release the GPU\nRuns only after successful completion."))
    paths = "list(RUNS.values())" if is_self else "[RUN]"
    cells.append(cell("code", f'''from google.colab import drive, runtime
for directory in {paths}:
    if not (directory / "complete.json").is_file():
        raise RuntimeError(f"Experiment incomplete: {{directory}}")
logger.success("Selected runs complete; flushing Drive")
drive.flush_and_unmount()
runtime.unassign()
'''))
    for index, item in enumerate(cells):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"{name}:cell{index}", "exec")
    return {"cells": cells, "metadata": {
        "accelerator": "GPU", "gpuClass": "premium",
        "colab": {"name": name, "provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}


def artifacts() -> dict[Path, str]:
    """Build notebooks and an input/output hash receipt deterministically."""
    outputs = {}
    choices = [("two_agent_qwen3", "all")]
    choices += [("self_distillation_qwen25", arm) for arm in ("R", "F", "N", "all")]
    for experiment, selection in choices:
        notebook = build_notebook(experiment, selection)
        path = PACKAGE / experiment / "notebooks" / notebook["metadata"]["colab"]["name"]
        outputs[path] = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    inputs = [ROOT / "scripts" / name for name in (*COMMON_SOURCES, *SELF_SOURCES, WORKER)]
    inputs += [Path(__file__), *sorted(PACKAGE.glob("*/configs/*.json")),
               *sorted((PACKAGE / "requirements").glob("*.txt"))]
    receipt = {
        "schema_version": 1, "gpu_validation_performed": False,
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs},
        "notebooks": {str(path.relative_to(ROOT)): {
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "all_code_cells_compile": True,
        } for path, source in outputs.items()},
    }
    outputs[PACKAGE / "build_manifest.json"] = json.dumps(receipt, indent=2) + "\n"
    return outputs


def show_plan() -> None:
    """Check the actual stage planners and log full default compute budgets."""
    namespace = {"__name__": "multiround_cpu_plan"}
    for name in (*COMMON_SOURCES, *SELF_SOURCES):
        path = ROOT / "scripts" / name
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    cfg = json.loads((PACKAGE / "two_agent_qwen3/configs/default.json").read_text())
    namespace["validate_recursive_config"](cfg)
    stages = namespace["recursive_stage_plan"](cfg["feedback_rounds"], cfg["arm_mode"])
    logger.info("Qwen3: {} legs/seed; {} updates; shared B1 then reciprocal A1–A7/B2–B8 and frozen-A B2–B8",
                len(stages), len(stages) * 471)
    cfg = json.loads((PACKAGE / "self_distillation_qwen25/configs/default.json").read_text())
    for arm in ("R", "F", "N"):
        namespace["validate_self_distillation_config"]({**cfg, "self_distillation_arm": arm, "seeds": [1]})
        stages = namespace["self_distillation_stage_plan"](cfg["teacher_rounds"], arm)
        jobs = len(stages) - (1 if cfg["deduplicate_initial_teacher"] and arm in ("R", "F") else 0)
        logger.info("Qwen2.5 {}: {} actual jobs/seed; {} updates", arm, jobs, jobs * 471)
    logger.info("Plan only: no installation, model downloads, inference or training performed")


def main() -> None:
    """Build, check, or inspect the release without running a GPU workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Fail on missing or stale notebook/build manifest")
    group.add_argument("--plan", action="store_true", help="Validate default configs and show the GPU-work budget")
    args = parser.parse_args()
    if args.plan:
        show_plan()
        return
    for path, source in artifacts().items():
        if args.check:
            if not path.is_file() or path.read_text() != source:
                raise SystemExit(f"Missing or stale artifact: {path.relative_to(ROOT)}. Rebuild without --check.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        logger.success("{} {}", "Verified" if args.check else "Built", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
