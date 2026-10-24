"""Load the frozen dolphin notebook on Linux; default is CPU-only preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from loguru import logger

MARKER = "# RECURSIVE_DOLPHIN_SOURCE: "
NOTEBOOK_SHA256 = "edd221de5adbb7582d0d05a894b6b74c905b4527d4cbde832740c5a746d6959c"
HELPERS = (
    "recursive_colab_acceleration.py", "recursive_colab_core.py",
    "recursive_colab_evaluation.py", "dolphin_teacher_transfer_evaluation.py",
    "recursive_colab_training.py", "recursive_dolphin_data.py",
    "recursive_dolphin_experiment.py",
)
WORKER = "recursive_colab_vllm.py"
NATIVE_PINS = {
    "torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130",
    "torchaudio": "2.11.0+cu130", "transformers": "4.57.6",
    "peft": "0.20.0", "accelerate": "1.14.0", "tokenizers": "0.22.2",
    "safetensors": "0.7.0", "numpy": "2.1.3", "loguru": "0.7.3",
}


def load_frozen_sources(notebook: Path) -> tuple[dict[str, str], dict]:
    """Verify exact embedded cells against the notebook build manifest."""
    payload = notebook.read_bytes()
    if hashlib.sha256(payload).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Notebook differs from the frozen output-free handoff copy")
    document = json.loads(payload)
    metadata = document["metadata"]["recursive_dolphin_build"]
    expected = {**metadata["source_sha256"], WORKER: metadata["worker_sha256"]}
    if set(expected) != set((*HELPERS, WORKER)):
        raise ValueError("Unexpected source manifest membership")
    sources = {}
    order = []
    for cell in document["cells"]:
        source = "".join(cell["source"]).rstrip() + "\n"
        if cell["cell_type"] != "code" or not source.startswith(MARKER):
            continue
        name = source.split("\n", 1)[0][len(MARKER):]
        if name in sources or name not in expected:
            raise ValueError("Duplicate or unknown source cell")
        if hashlib.sha256(source.encode()).hexdigest() != expected[name]:
            raise ValueError(f"Source hash mismatch: {name}")
        compile(source, name, "exec")
        sources[name] = source
        order.append(name)
    if tuple(order) != (*HELPERS, WORKER):
        raise ValueError("Notebook sources are missing or reordered")
    return sources, metadata


def main() -> None:
    """Preflight on CPU, or explicitly start/resume exactly one selected arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--arm", choices=("reciprocal", "frozen"), default="reciprocal")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vllm-python", default="/content/recursive_vllm_env/bin/python")
    parser.add_argument("--worker-path", default="/content/recursive_dolphin_vllm.py")
    parser.add_argument("--run", action="store_true", help="Start GPU work; omit for CPU preflight")
    args = parser.parse_args()
    sources, metadata = load_frozen_sources(args.notebook)
    # Shared globals are required: these are sequential notebook helpers, not
    # independent Python modules. The non-main name keeps CLI guards inactive.
    namespace = {"__name__": "dolphin_replica", "logger": logger}
    for name in HELPERS:
        exec(compile(sources[name], name, "exec"), namespace)
    cfg = namespace["dolphin_recursive_default_config"](arm=args.arm)
    cfg.update({
        "output_dir": str(args.run_dir),
        "vllm_python": args.vllm_python, "vllm_worker_path": args.worker_path,
        "expected_source_sha256": metadata["source_sha256"],
        "expected_worker_sha256": metadata["worker_sha256"],
    })
    namespace["dolphin_recursive_validate_config"](cfg, args.run_dir)
    logger.success("Source/configuration CPU preflight passed: {} arm; {} stages",
                   args.arm, len(namespace["dolphin_recursive_stages"](cfg)))
    if not args.run:
        logger.info("No preparation, model download, generation or training was started")
        return
    import importlib.metadata
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Use native Python 3.13 to match the original notebook")
    for name, expected in NATIVE_PINS.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Native package differs: {name}; expected {expected}")
    try:
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Remove optional torchao from the native environment")
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name(0)
            or torch.cuda.get_device_properties(0).total_memory < 38 * 2**30
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Use exactly one visible A100 with at least 40 GB and BF16")
    worker = Path(cfg["vllm_worker_path"])
    if worker.is_symlink():
        raise ValueError("Worker path cannot be a symlink")
    if worker.exists() and worker.read_text() != sources[WORKER]:
        raise ValueError("Worker path already contains different source")
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text(sources[WORKER])
    helpers = {name: sources[name] for name in HELPERS}
    prepared = namespace["prepare_dolphin_recursive_run"](
        cfg, args.run_dir, sources=helpers, worker_source=sources[WORKER])
    report = namespace["run_dolphin_recursive_experiment"](prepared, args.run_dir)
    logger.success("Run returned: complete={}, stages={}/{}; report={}",
                   report["complete"], report["completed_stages"],
                   report["expected_stages"], args.run_dir / "report/README.md")


if __name__ == "__main__":
    main()
