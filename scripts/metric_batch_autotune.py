#!/usr/bin/env python3
"""Reusable GPU batch autotuning for the metric evaluation scripts.

The tuner deliberately separates the search policy from model-specific work.
Callers provide a representative benchmark callback and an exact cache
identity.  Successful choices are persisted, so every later metric process on
the same accelerator can resume without paying the tuning cost again.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from loguru import logger

DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class BatchCandidate:
    """One prompt/branch batch pair."""

    prompt_batch_size: int
    branch_batch_size: int


@dataclass(frozen=True, **DATACLASS_KW_ONLY)
class BatchTrial:
    """One measured candidate."""

    prompt_batch_size: int
    branch_batch_size: int
    status: str
    duration_seconds: float | None
    work_items: int
    throughput: float | None
    error: str | None


class BatchCandidateRejectedError(RuntimeError):
    """Signal that one fitting batch candidate failed correctness checks."""


def geometric_sizes(maximum: int, *, include: Iterable[int] = ()) -> list[int]:
    """Return powers of two plus requested boundary values up to ``maximum``."""
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    values = {1, maximum}
    value = 1
    while value <= maximum:
        values.add(value)
        value *= 2
    values.update(int(item) for item in include if 0 < int(item) <= maximum)
    return sorted(values)


def candidate_grid(
    *,
    max_prompt_batch_size: int,
    max_branch_batch_size: int,
    include_prompt_sizes: Iterable[int] = (),
    include_branch_sizes: Iterable[int] = (),
) -> list[BatchCandidate]:
    """Build the full bounded B/K search grid."""
    prompt_sizes = geometric_sizes(
        max_prompt_batch_size,
        include=include_prompt_sizes,
    )
    branch_sizes = geometric_sizes(
        max_branch_batch_size,
        include=include_branch_sizes,
    )
    return [
        BatchCandidate(prompt_batch_size=prompt_size, branch_batch_size=branch_size)
        for prompt_size in prompt_sizes
        for branch_size in branch_sizes
    ]


def is_out_of_memory_error(error: BaseException) -> bool:
    """Return whether an exception is a CUDA/accelerator OOM."""
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def clear_accelerator_cache() -> None:
    """Release Python and CUDA caches between independent trials."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        return


def accelerator_identity() -> dict[str, Any]:
    """Return stable hardware facts used to scope an autotune cache."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"backend": "cpu"}
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return {
            "backend": "cuda",
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "torch_version": torch.__version__,
        }
    except ImportError:
        return {"backend": "unknown"}


def synchronize_accelerator() -> None:
    """Wait for timed CUDA work without charging cache cleanup to a trial."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _write_payload_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete trial evidence, never a partially written tune cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _compatible_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") not in (None, "complete"):
        return None
    if payload.get("identity") != identity:
        return None
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        return None
    try:
        invalid_sizes = (
            int(selected.get("prompt_batch_size", 0)) <= 0
            or int(selected.get("branch_batch_size", 0)) <= 0
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if invalid_sizes:
        return None
    return payload


def autotune_batch_pair(
    *,
    candidates: list[BatchCandidate],
    benchmark: Callable[[BatchCandidate], int],
    cache_path: Path,
    identity: dict[str, Any],
    reuse_cache: bool = True,
) -> tuple[BatchCandidate, dict[str, Any]]:
    """Benchmark candidates and return the highest-throughput fitting pair.

    ``benchmark`` must execute representative work and return the number of
    comparable work items completed.  A ``BatchCandidateRejectedError`` marks
    only that candidate as numerically unsafe; OOMs are likewise recorded and
    skipped.  Every other exception is propagated so implementation failures
    cannot be mistaken for tuning results. All-rejected/all-OOM searches save
    their trial evidence but never a reusable selection. Timing includes CUDA
    completion, excluding garbage collection and allocator-cache cleanup.
    """
    if not candidates:
        raise ValueError("At least one batch candidate is required.")
    full_identity = {**identity, "accelerator": accelerator_identity()}
    if reuse_cache:
        cached = _compatible_cache(cache_path, full_identity)
        if cached is not None:
            selected = cached["selected"]
            choice = BatchCandidate(
                prompt_batch_size=int(selected["prompt_batch_size"]),
                branch_batch_size=int(selected["branch_batch_size"]),
            )
            logger.success("Reusing batch autotune result B={}, K={}",
                           choice.prompt_batch_size, choice.branch_batch_size)
            return choice, cached

    trials: list[BatchTrial] = []
    for candidate in candidates:
        clear_accelerator_cache()
        started = time.perf_counter()
        try:
            work_items = int(benchmark(candidate))
            synchronize_accelerator()
            duration = time.perf_counter() - started
        except BatchCandidateRejectedError as exc:
            clear_accelerator_cache()
            logger.warning(
                "Batch autotune rejected B={}, K={}: {}",
                candidate.prompt_batch_size,
                candidate.branch_batch_size,
                exc,
            )
            trials.append(
                BatchTrial(
                    prompt_batch_size=candidate.prompt_batch_size,
                    branch_batch_size=candidate.branch_batch_size,
                    status="rejected",
                    duration_seconds=None,
                    work_items=0,
                    throughput=None,
                    error=str(exc),
                )
            )
            continue
        except RuntimeError as exc:
            clear_accelerator_cache()
            if not is_out_of_memory_error(exc):
                raise
            logger.warning(
                "Batch autotune OOM for B={}, K={}",
                candidate.prompt_batch_size,
                candidate.branch_batch_size,
            )
            trials.append(
                BatchTrial(
                    prompt_batch_size=candidate.prompt_batch_size,
                    branch_batch_size=candidate.branch_batch_size,
                    status="oom",
                    duration_seconds=None,
                    work_items=0,
                    throughput=None,
                    error=str(exc),
                )
            )
            continue
        # GC and empty_cache are operational cleanup, not scoring throughput.
        # GPU synchronization above is timed so asynchronous work cannot make
        # a candidate appear faster than the actual completed computation.
        clear_accelerator_cache()
        if work_items <= 0 or not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("The autotune benchmark returned invalid work or timing.")
        throughput = work_items / duration
        logger.info(
            "Batch autotune B={}, K={}: {:.1f} work items/s",
            candidate.prompt_batch_size,
            candidate.branch_batch_size,
            throughput,
        )
        trials.append(
            BatchTrial(
                prompt_batch_size=candidate.prompt_batch_size,
                branch_batch_size=candidate.branch_batch_size,
                status="fit",
                duration_seconds=duration,
                work_items=work_items,
                throughput=throughput,
                error=None,
            )
        )

    fitting = [trial for trial in trials if trial.status == "fit"]
    if not fitting:
        rejected_count = sum(trial.status == "rejected" for trial in trials)
        oom_count = sum(trial.status == "oom" for trial in trials)
        failure_reason = (
            "numerical_rejection" if rejected_count and not oom_count
            else "out_of_memory" if oom_count and not rejected_count
            else "numerical_rejection_and_out_of_memory"
        )
        payload = {
            "version": 2,
            "status": "failed",
            "failure_reason": failure_reason,
            "identity": full_identity,
            "selected": None,
            "trials": [asdict(trial) for trial in trials],
        }
        _write_payload_atomically(cache_path, payload)
        if rejected_count:
            raise RuntimeError(
                "No numerically eligible autotune batch configuration: "
                f"{rejected_count} failed correctness checks; {oom_count} ran "
                f"out of memory. Trial evidence: {cache_path}"
            )
        raise RuntimeError(
            "No autotune batch configuration fit GPU memory: "
            f"all {oom_count} candidates ran out of memory. "
            f"Trial evidence: {cache_path}"
        )
    best = max(
        fitting,
        key=lambda trial: (
            float(trial.throughput or 0.0),
            trial.prompt_batch_size * trial.branch_batch_size,
        ),
    )
    choice = BatchCandidate(
        prompt_batch_size=best.prompt_batch_size,
        branch_batch_size=best.branch_batch_size,
    )
    payload = {
        "version": 2,
        "status": "complete",
        "identity": full_identity,
        "selected": asdict(choice),
        "trials": [asdict(trial) for trial in trials],
    }
    _write_payload_atomically(cache_path, payload)
    logger.success(
        "Selected batch configuration B={}, K={}",
        choice.prompt_batch_size,
        choice.branch_batch_size,
    )
    return choice, payload
