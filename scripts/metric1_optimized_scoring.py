"""Audited FP32 Metric 1 scoring with bounded backend/batch calibration.

This module changes execution, never prompt plans, model precision, candidate
probabilities, or the established raw/probability acceptance thresholds. A
calibrated choice is an initial choice, not evidence that every condition fits.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from loguru import logger

from scripts.metric_batch_autotune import is_out_of_memory_error
from scripts.metric_exact_trie_scorer import (
    compare_exact_trie_scores,
    exact_trie_branch_width,
    native_full_prefix_metric_reference,
    score_exact_logprob_matrix,
)
from scripts.prepare_token_sft_first_number_channel import production_audit_batches


OPTIMIZED_SCORING_VERSION = "metric1_optimized_fp32_v2_adaptive_probes"
CANDIDATES = tuple(f"{number:02d}" for number in range(100))
BACKENDS = ("cached", "shared_prefix")


@dataclass(frozen=True, kw_only=True)
class ScoringChoice:
    """A backend and execution-only prompt/continuation batch sizes."""

    backend: str
    prompt_batch_size: int
    branch_batch_size: int


def _choice(payload: Mapping[str, Any], maximum: int,
            allow_shared: bool) -> ScoringChoice:
    """Strictly parse a persisted selection without coercing malformed values."""
    keys = {"backend", "prompt_batch_size", "branch_batch_size"}
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError("A selection needs exactly backend, prompt_batch_size, branch_batch_size")
    backend = payload["backend"]
    b, k = payload["prompt_batch_size"], payload["branch_batch_size"]
    if backend not in BACKENDS or (backend == "shared_prefix" and not allow_shared):
        raise ValueError("Unsupported or disabled scoring backend")
    if (type(b) is not int or type(k) is not int
            or not 1 <= b <= maximum or not 1 <= k <= 10):
        raise ValueError("Invalid execution batch sizes in persisted selection")
    return ScoringChoice(backend=backend, prompt_batch_size=b, branch_batch_size=k)


class OptimizedScorer:
    """Reuse one bounded calibration while independently auditing each chunk.

    Instantiate once per loaded immutable FP32 model. Condition keys must be
    stable and distinct for different rendered teacher system conditions.
    ``plans`` are precomputed exact tokenization plans; this class never calls
    the tokenizer to recreate them. Serial uncached reference forwards retain
    the original strict thresholds (raw log-probability error <= 2e-4).
    """

    def __init__(self, model: Any, tokenizer: Any, device: Any,
                 max_prompt_batch_size: int = 32,
                 shared_prefix: bool = True) -> None:
        if type(max_prompt_batch_size) is not int or max_prompt_batch_size < 1:
            raise ValueError("max_prompt_batch_size must be a positive integer")
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_prompt_batch_size = max_prompt_batch_size
        self.shared_prefix = shared_prefix
        self.selected: ScoringChoice | None = None
        self.calibration: dict[str, Any] | None = None
        self._prefix_store: Any = None
        self._validate_model()

    def _validate_model(self) -> None:
        """Refuse a precision change; never modify model/global arithmetic here."""
        import torch

        if getattr(self.model, "training", False):
            raise ValueError("Metric scoring requires model.eval()")
        floating = [parameter for parameter in self.model.parameters()
                    if parameter.is_floating_point()]
        if not floating or any(parameter.dtype != torch.float32 for parameter in floating):
            raise ValueError("Optimized Metric 1 scorer requires FP32 model weights")

    def _synchronize(self) -> None:
        """Bound timing to completed accelerator work without changing precision."""
        import torch

        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _prefix_cache_counts(self) -> dict[str, int]:
        """Expose shared-prefix construction/reuse without serializing GPU state."""
        return {name: int(getattr(self._prefix_store, name, 0))
                for name in ("build_count", "reuse_count")}

    def _clear_oom(self) -> None:
        """Release temporary failed allocations only after recognized OOM."""
        import gc
        import torch

        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _reference(self, prompt_texts: list[str], plans: list[Any]) -> np.ndarray:
        """Use independent serial full-prefix recomputation, not another cache."""
        return native_full_prefix_metric_reference(
            model=self.model, tokenizer=self.tokenizer, device=self.device,
            prompt_texts=prompt_texts, target_texts=list(CANDIDATES),
            prompt_plans=plans,
        )

    def _run_backend(self, prompt_texts: list[str], plans: list[Any],
                     condition_key: str, choice: ScoringChoice) -> np.ndarray:
        """Dispatch one complete requested set; returned rows retain input order."""
        if choice.backend == "cached":
            values = score_exact_logprob_matrix(
                model=self.model, tokenizer=self.tokenizer, device=self.device,
                prompt_texts=prompt_texts, target_texts=list(CANDIDATES),
                prompt_plans=plans,
                prompt_batch_size=choice.prompt_batch_size,
                branch_batch_size=choice.branch_batch_size,
                audit_native_reference=False,
            )
        else:
            from scripts.metric1_shared_prefix import (
                SharedPrefixStore, score_shared_prefix_logprobs,
            )

            if self._prefix_store is None:
                self._prefix_store = SharedPrefixStore()
            values = score_shared_prefix_logprobs(
                model=self.model, tokenizer=self.tokenizer, device=self.device,
                prompt_texts=prompt_texts, plans=plans,
                prompt_batch_size=choice.prompt_batch_size,
                branch_batch_size=choice.branch_batch_size,
                prefix_store=self._prefix_store, condition_key=condition_key,
            )
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(plans), 100) or not np.isfinite(values).all():
            raise ValueError("Backend returned malformed or nonfinite raw log-probabilities")
        return values

    def _timed_backend(self, prompt_texts: list[str], plans: list[Any],
                       condition_key: str, choice: ScoringChoice) -> tuple[np.ndarray, float]:
        """Measure a backend with synchronized boundaries, including prefix prefill."""
        self._synchronize()
        started = time.perf_counter()
        values = self._run_backend(prompt_texts, plans, condition_key, choice)
        self._synchronize()
        return values, time.perf_counter() - started

    def _validate_inputs(self, prompt_texts: list[str], plans: list[Any],
                         condition_key: str) -> int:
        """Require exact 00--99 candidate plans and a nonempty condition identity."""
        if not prompt_texts or len(prompt_texts) != len(plans):
            raise ValueError("Nonempty prompt texts and exact plans must align")
        if not isinstance(condition_key, str) or not condition_key:
            raise ValueError("A nonempty stable condition_key is required")
        for plan in plans:
            paths = plan.target_token_ids
            if len(paths) != 100 or any(not path for path in paths):
                raise ValueError("Every plan must contain 100 nonempty exact candidate paths")
        return min(10, exact_trie_branch_width(plans))

    def _initial_choices(self, width: int, n: int) -> list[ScoringChoice]:
        """Try at most six representative B=8/16/32, maximum-useful-K choices."""
        sizes = sorted({min(size, self.max_prompt_batch_size, n) for size in (8, 16, 32)})
        backends = BACKENDS if self.shared_prefix else ("cached",)
        return [ScoringChoice(backend=backend, prompt_batch_size=b, branch_batch_size=width)
                for b in sizes for backend in backends]

    def _fallback_choices(self, first: ScoringChoice, width: int,
                          n: int) -> list[ScoringChoice]:
        """Progress through intermediate B, then K, before uncached reference."""
        bmax = min(self.max_prompt_batch_size, n)
        sizes = sorted({min(size, bmax) for size in (32, 16, 8, 4, 2, 1)}, reverse=True)
        # Try smaller/equal prompt batches before larger calibrated alternatives.
        sizes.sort(key=lambda b: (b > first.prompt_batch_size, -b))
        widths = list(dict.fromkeys([width, *[min(k, width) for k in (5, 2, 1)]]))
        backends = [first.backend]
        if first.backend == "shared_prefix":
            backends.append("cached")
        elif self.shared_prefix:
            backends.append("shared_prefix")
        choices = [first]
        for k in widths:
            for b in sizes:
                for backend in backends:
                    candidate = ScoringChoice(backend=backend, prompt_batch_size=b, branch_batch_size=k)
                    if candidate not in choices:
                        choices.append(candidate)
        return choices

    def calibrate(self, prompt_texts: list[str], plans: list[Any],
                  condition_key: str) -> dict[str, Any]:
        """Time up to six primary choices; compare every supplied calibration row.

        Supply a bounded representative panel (e.g. 32 spread/length-extreme
        prompts), preferably under a real teacher system. Each trial receives
        the entire panel, so padding and batch companions match its actual
        production call on that panel. No audit rows are scored in isolation.
        """
        width = self._validate_inputs(prompt_texts, plans, condition_key)
        started = time.perf_counter()
        prefix_before = self._prefix_cache_counts()
        reference_started = time.perf_counter()
        reference = self._reference(prompt_texts, plans)
        reference_seconds = time.perf_counter() - reference_started
        trials: list[dict[str, Any]] = []
        accepted: list[tuple[float, ScoringChoice]] = []

        def trial(candidate: ScoringChoice) -> None:
            tick = time.perf_counter()
            record: dict[str, Any] = {"choice": asdict(candidate)}
            try:
                values, seconds = self._timed_backend(prompt_texts, plans, condition_key, candidate)
                comparison = compare_exact_trie_scores(values, reference, reduced_precision=False)
                record.update(seconds=seconds, comparison=asdict(comparison),
                              status="accepted" if comparison.accepted else "rejected")
                if comparison.accepted:
                    accepted.append((seconds, candidate))
            except RuntimeError as error:
                if not is_out_of_memory_error(error):
                    raise
                record.update(status="oom", error=str(error), seconds=time.perf_counter() - tick)
                self._clear_oom()
            trials.append(record)

        primary = self._initial_choices(width, len(plans))
        for candidate in primary:
            trial(candidate)
        if not accepted:
            for candidate in self._fallback_choices(primary[0], width, len(plans)):
                if candidate not in primary:
                    trial(candidate)
                    if accepted:
                        break
        if not accepted:
            raise RuntimeError("No cached backend passed unchanged FP32 calibration; do not score")
        seconds, self.selected = min(accepted, key=lambda item: item[0])
        self.calibration = {
            "version": OPTIMIZED_SCORING_VERSION,
            "condition_key": condition_key, "prompt_count": len(plans),
            "reference": "serial uncached complete-prefix recomputation, every calibration row",
            "reference_seconds": reference_seconds, "trials": trials,
            "selected": asdict(self.selected), "selected_seconds": seconds,
            "prefix_cache_counts_before": prefix_before,
            "prefix_cache_counts_after": self._prefix_cache_counts(),
            "elapsed_seconds": time.perf_counter() - started,
            "note": "Single-panel timed selection; all future chunks retain independent audits. "
                    "Prefix construction is included on first use, and later trials can reuse it; "
                    "no universal speedup is asserted.",
        }
        logger.info("Optimized Metric 1 calibrated {} on {} rows", self.selected, len(plans))
        return self.calibration

    def export_selection(self) -> dict[str, Any]:
        """Persist only the execution choice; external runner owns model provenance."""
        if self.selected is None:
            raise ValueError("Calibrate or restore a selection first")
        return {"version": OPTIMIZED_SCORING_VERSION, "selected": asdict(self.selected)}

    def restore_selection(self, selection: Mapping[str, Any]) -> None:
        """Restore a strict versioned choice, with no inference or model mutation."""
        if (not isinstance(selection, Mapping) or set(selection) != {"version", "selected"}
                or selection["version"] != OPTIMIZED_SCORING_VERSION):
            raise ValueError("Invalid or incompatible optimized scorer selection")
        self.selected = _choice(selection["selected"], self.max_prompt_batch_size, self.shared_prefix)

    def _cached_probe(self, prompt_texts: list[str], plans: list[Any],
                      condition_key: str, candidate: ScoringChoice,
                      audit_indices: list[int], reference: np.ndarray,
                      batches: list[list[int]]) -> tuple[Any, float, list[list[int]]]:
        """Probe exact cached production batches, including every audit companion."""
        rows: dict[int, np.ndarray] = {}
        seconds = 0.0
        for indices in batches:
            values, elapsed = self._timed_backend(
                [prompt_texts[i] for i in indices], [plans[i] for i in indices],
                condition_key, candidate,
            )
            seconds += elapsed
            rows.update({index: values[offset] for offset, index in enumerate(indices)
                         if index in audit_indices})
        comparison = compare_exact_trie_scores(
            np.asarray([rows[i] for i in audit_indices]), reference, reduced_precision=False,
        )
        return comparison, seconds, batches

    def score(self, prompt_texts: list[str], plans: list[Any],
              condition_key: str, *, reference_logprobs: np.ndarray | None = None,
              force_uncached: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a whole accepted condition, progressively recovering failures.

        Each candidate is audited against the same independent serial rows.
        Shared-prefix trials run the complete condition because subsets can
        change its common prefix. Ordinary-cache candidates, including the
        initial choice, first probe the exact production audit batches when
        those batches cover at most half the rows. This avoids a duplicate
        near-full computation for small chunks or large execution batches.
        The final whole-condition result is always independently checked
        again. No rows are selectively patched to make the audit pass. The
        accepted execution choice becomes the next call's initial choice;
        a prior acceptance never replaces the next call's numerical audit.

        With ``reference_logprobs``, every row of every candidate is compared
        with the supplied independent full-condition oracle. This is required
        for a benchmark that has already computed all rows: a three-row spot
        check must not bypass failures elsewhere. ``force_uncached`` executes
        no cached backend. Without a supplied full oracle, its full serial
        result is checked against freshly recomputed first/middle/last rows.
        """
        width = self._validate_inputs(prompt_texts, plans, condition_key)
        if type(force_uncached) is not bool:
            raise ValueError("force_uncached must be a boolean")
        supplied_reference = reference_logprobs is not None
        if supplied_reference:
            try:
                reference_logprobs = np.array(reference_logprobs, dtype=np.float64, copy=True)
            except (TypeError, ValueError) as error:
                raise ValueError("Full reference must be a finite prompt-by-100 matrix") from error
            if (reference_logprobs.shape != (len(plans), 100)
                    or not np.isfinite(reference_logprobs).all()):
                raise ValueError("Full reference must be a finite prompt-by-100 matrix")
        if self.selected is None and not force_uncached:
            raise ValueError("Explicit calibrate() or restore_selection() is required before score()")
        started = time.perf_counter()
        prefix_before = self._prefix_cache_counts()
        full_audit = supplied_reference
        indices = list(range(len(plans))) if full_audit else sorted({0, len(plans) // 2, len(plans) - 1})
        tick = time.perf_counter()
        reference = reference_logprobs if supplied_reference else self._reference(
            [prompt_texts[i] for i in indices], [plans[i] for i in indices])
        timings = {"reference_seconds": time.perf_counter() - tick,
                   "probe_seconds": 0.0, "full_condition_seconds": 0.0,
                   "failed_attempt_seconds": 0.0}
        first = None if self.selected is None else ScoringChoice(
            backend=self.selected.backend,
            prompt_batch_size=min(self.selected.prompt_batch_size, len(plans)),
            branch_batch_size=min(self.selected.branch_batch_size, width))
        attempts: list[dict[str, Any]] = []

        def finish(values: np.ndarray, actual: ScoringChoice | None,
                   comparison: Any) -> tuple[np.ndarray, dict[str, Any]]:
            if actual is not None:
                self.selected = actual
            return values, {
                "version": OPTIMIZED_SCORING_VERSION, "condition_key": condition_key,
                "selected_initial": asdict(first) if first is not None else None,
                "actual_choice": asdict(actual) if actual else {"backend": "uncached_serial"},
                "selected_next": asdict(self.selected) if self.selected is not None else None,
                "reference_prompt_indices": indices,
                "reference": "independent serial uncached full-prefix recomputation",
                "reference_source": ("provided_full_condition" if supplied_reference
                                     else "computed_serial_spotcheck"),
                "all_rows_audited": full_audit,
                "reference_logprobs_sha256": hashlib.sha256(
                    np.ascontiguousarray(reference, dtype="<f8").tobytes()).hexdigest(),
                "force_uncached": force_uncached,
                "comparison": asdict(comparison), "attempts": attempts,
                "prefix_cache_counts_before": prefix_before,
                "prefix_cache_counts_after": self._prefix_cache_counts(),
                "timings": {**timings, "elapsed_seconds": time.perf_counter() - started},
                "timing_note": "failed_attempt_seconds is a subset of total elapsed and overlaps "
                               "probe/full-condition timings; do not sum all timing fields",
                "reduced_precision": False, "selective_row_repair": False,
            }

        choices = [] if force_uncached else self._fallback_choices(first, width, len(plans))
        for candidate in choices:
            tick = time.perf_counter()
            record: dict[str, Any] = {"choice": asdict(candidate)}
            try:
                batches = (production_audit_batches(plans, indices, candidate.prompt_batch_size)
                           if candidate.backend == "cached" and not full_audit else [])
                probe_rows = sum(len(batch) for batch in batches)
                if batches and probe_rows <= len(plans) // 2:
                    comparison, seconds, batches = self._cached_probe(
                        prompt_texts, plans, condition_key, candidate, indices, reference, batches,
                    )
                    timings["probe_seconds"] += seconds
                    record.update(probe_comparison=asdict(comparison), probe_batches=batches,
                                  probe_prompt_count=probe_rows)
                    if not comparison.accepted:
                        record.update(status="rejected_probe", seconds=time.perf_counter() - tick)
                        attempts.append(record)
                        timings["failed_attempt_seconds"] += record["seconds"]
                        logger.warning("Metric 1 {} rejected {} in production-shaped probe; "
                                       "skipping its full chunk", condition_key, candidate)
                        continue
                values, seconds = self._timed_backend(prompt_texts, plans, condition_key, candidate)
                timings["full_condition_seconds"] += seconds
                comparison = compare_exact_trie_scores(values[indices], reference, reduced_precision=False)
                record.update(comparison=asdict(comparison), seconds=time.perf_counter() - tick,
                              status="accepted" if comparison.accepted else "rejected_full_condition")
                attempts.append(record)
                if comparison.accepted:
                    return finish(values, candidate, comparison)
                timings["failed_attempt_seconds"] += record["seconds"]
                logger.warning("Metric 1 {} rejected {}; trying an audited intermediate batch",
                               condition_key, candidate)
            except RuntimeError as error:
                if not is_out_of_memory_error(error):
                    raise
                record.update(status="oom", error=str(error), seconds=time.perf_counter() - tick)
                attempts.append(record)
                timings["failed_attempt_seconds"] += record["seconds"]
                self._clear_oom()
        # Preserve a correctness-first terminal fallback after all batch options.
        tick = time.perf_counter()
        values = self._reference(prompt_texts, plans)
        timings["full_condition_seconds"] += time.perf_counter() - tick
        comparison = compare_exact_trie_scores(values[indices], reference, reduced_precision=False)
        if not comparison.accepted:
            raise RuntimeError("Uncached serial full-condition result is not reproducible")
        return finish(values, None, comparison)
