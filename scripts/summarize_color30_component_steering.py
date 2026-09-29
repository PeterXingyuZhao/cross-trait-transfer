"""Summarize cached color-vector interventions without running model inference.

All effects are question-wise raw log-probability changes relative to the same
unsteered base. No candidate normalization or double-centering is applied.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

COMPONENTS = ("shared", "full", "specific")
COMPONENT_COLORS = {"shared": "#8b7298", "full": "#337fa3", "specific": "#df9347"}


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Match the runner's canonical protocol hash without importing its runner."""
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_protocol_identity(identity: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Reject caches from another protocol instead of mixing incompatible scores."""
    claimed = identity.get("protocol_sha256")
    if claimed is not None or "runner_sha256" in protocol:
        if claimed != protocol_digest(protocol):
            raise ValueError("Cache protocol_sha256 does not match protocol.json")


def condition_signature(identity: dict[str, Any]) -> tuple[str, str | None, int | None, float]:
    """Identify component, teacher trait, control seed, and signed strength."""
    component = str(identity["component"])
    trait = identity.get("trait")
    seed = identity.get("control_seed")
    if component not in (*COMPONENTS, "baseline"):
        raise ValueError(f"Unknown component: {component}")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("control_seed must be an integer or null")
    alpha = float(identity["alpha"])
    if not math.isfinite(alpha):
        raise ValueError("Nonfinite alpha")
    return component, trait, seed, alpha


def question_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarize question variation, not a confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "question_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "question_min": float(array.min()),
        "question_max": float(array.max()),
        "by_question": array.tolist(),
    }


def raw_lifts(logprobs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract matched-question unsteered log probabilities before averaging."""
    values, base = np.asarray(logprobs, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if values.shape != base.shape or values.ndim != 2:
        raise ValueError("Condition and baseline must share question-by-answer shape")
    if not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError("Nonfinite log probabilities are not silently clipped")
    return values - base


def expected_keys(protocol: dict[str, Any]) -> list[str] | None:
    """Read an explicit planned-condition list; never infer completeness from data."""
    for name in ("expected_condition_keys", "condition_keys"):
        if name in protocol:
            return [str(key).removesuffix(".json") for key in protocol[name] if str(key) != "base"]
    planned = protocol.get("conditions", protocol.get("expected_conditions"))
    if isinstance(planned, list):
        keys = []
        for item in planned:
            if isinstance(item, str):
                key = item
            elif isinstance(item, dict) and "key" in item:
                key = str(item["key"])
            else:
                return None
            if key.removesuffix(".json") != "base":
                keys.append(key.removesuffix(".json"))
        return keys
    return None


def summarize_data(output_dir: Path) -> dict[str, Any]:
    """Validate available records and summarize only completed cache conditions."""
    protocol_path = output_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    catalog, questions = protocol["catalog"], protocol["questions"]
    if not catalog or len(catalog) != len(set(catalog)) or not questions:
        raise ValueError("Protocol needs a unique catalog and nonempty question list")
    baseline_path = output_dir / "condition_cache/base.json"
    plan_path = output_dir / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    expected = expected_keys(plan) if plan is not None else expected_keys(protocol)
    result: dict[str, Any] = {
        "status": "partial", "catalog": catalog, "questions": questions,
        "protocol_identity": protocol.get("identity"), "protocol_sha256": protocol_digest(protocol),
        "run_plan": plan,
        "expected_condition_keys": expected, "completed_condition_keys": [],
        "missing_condition_keys": expected, "invalid_conditions": [], "conditions": [],
        "nonadditivity": [], "random_control_ranges": [],
        "definitions": {
            "raw_lift": "Delta[q,u] = log P_steered(u|q) - log P_base(u|q); mean over the same questions",
            "suppression": "Negative raw mean lift means lower geometric-mean probability than the unsteered model. No answer-panel normalization or double-centering.",
            "question_variability": "SD and min/max over question-wise effects, not a confidence interval or independent training replicates.",
            "random_control_range": "Min/max of completed random-seed mean effects for the same component, teacher, alpha and answer; descriptive, not a formal test.",
            "nonadditivity": "E_q[Delta_full(q,u) - Delta_specific(q,u) - Delta_shared(q,u)] at matched trait and alpha. Vector addition need not imply additive model effects.",
        },
    }
    if not baseline_path.is_file():
        result["baseline_status"] = "missing"
        return result
    baseline_cache = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_protocol_identity(baseline_cache["identity"], protocol)
    if baseline_cache["identity"]["component"] != "baseline":
        raise ValueError("base.json is not labeled baseline")
    baseline = np.asarray(baseline_cache["logprobs"], dtype=np.float64)
    shape = (len(questions), len(catalog))
    if baseline.shape != shape or not np.isfinite(baseline).all():
        raise ValueError(f"Baseline shape/values do not match protocol {shape}")
    if baseline_cache.get("status", "completed") != "completed":
        result["baseline_status"] = "not_completed"
        return result
    result["baseline_status"] = "completed"
    signatures: dict[tuple[str, str | None, int | None, float], dict[str, Any]] = {}
    for path in sorted((output_dir / "condition_cache").glob("*.json")):
        if path.name == "base.json":
            continue
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get("status", "completed") != "completed":
                raise ValueError("Cache is not marked completed")
            identity = cache["identity"]
            validate_protocol_identity(identity, protocol)
            signature = condition_signature(identity)
            component, trait, seed, alpha = signature
            if component == "baseline":
                raise ValueError("Unexpected second baseline")
            if trait is not None and trait not in catalog:
                raise ValueError(f"Unknown teacher trait {trait!r}")
            if component in ("full", "specific") and trait is None:
                raise ValueError("Teacher-indexed component requires trait")
            if signature in signatures:
                raise ValueError("Duplicate component/trait/seed/alpha condition")
            lifts = raw_lifts(np.asarray(cache["logprobs"]), baseline)
            if lifts.shape != shape:
                raise ValueError("Condition shape does not match protocol")
            record = {
                "key": path.stem, "identity": identity,
                "elapsed_seconds": cache.get("elapsed_seconds"),
                "answers": {answer: question_summary(lifts[:, index])
                            for index, answer in enumerate(catalog)},
            }
            signatures[signature] = record
            result["conditions"].append(record)
            result["completed_condition_keys"].append(path.stem)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result["invalid_conditions"].append({"key": path.stem, "reason": str(exc)})
            logger.warning("Skipping invalid/incomplete cache {}: {}", path, exc)
    completed = set(result["completed_condition_keys"])
    if expected is not None:
        result["missing_condition_keys"] = sorted(set(expected) - completed)
        result["additional_completed_condition_keys"] = sorted(completed - set(expected))
        result["completed_current_plan_keys"] = sorted(completed & set(expected))
        invalid_planned = any(row["key"] in expected for row in result["invalid_conditions"])
        if not result["missing_condition_keys"] and not invalid_planned:
            result["status"] = "completed"
    else:
        result["completion_note"] = "No explicit expected-condition list in protocol; completion cannot be established."
    grouped_controls: dict[tuple[str, str | None, float], list[dict[str, Any]]] = {}
    for (component, trait, seed, alpha), record in signatures.items():
        if seed is not None:
            grouped_controls.setdefault((component, trait, alpha), []).append(record)
        elif component == "full":
            specific = signatures.get(("specific", trait, None, alpha))
            shared = signatures.get(("shared", None, None, alpha))
            if specific and shared:
                residual = {}
                for answer in catalog:
                    values = (np.asarray(record["answers"][answer]["by_question"])
                              - np.asarray(specific["answers"][answer]["by_question"])
                              - np.asarray(shared["answers"][answer]["by_question"]))
                    residual[answer] = question_summary(values)
                result["nonadditivity"].append({"trait": trait, "alpha": alpha, "answers": residual})
    for (component, trait, alpha), records in grouped_controls.items():
        result["random_control_ranges"].append({
            "component": component, "trait": trait, "alpha": alpha,
            "seeds": sorted(row["identity"]["control_seed"] for row in records),
            "answers": {answer: {"min": min(row["answers"][answer]["mean"] for row in records),
                                 "max": max(row["answers"][answer]["mean"] for row in records)}
                        for answer in catalog},
        })
    return result


def _lookup(result: dict[str, Any], component: str, trait: str | None,
            alpha: float = .3) -> dict[str, Any] | None:
    """Find a true-vector record at the exact requested alpha."""
    for record in result["conditions"]:
        identity = record["identity"]
        if (identity["component"] == component and identity.get("trait") == trait
                and identity.get("control_seed") is None
                and math.isclose(float(identity["alpha"]), alpha, abs_tol=1e-12)):
            return record
    return None


def make_figures(output_dir: Path, result: dict[str, Any]) -> list[str]:
    """Render two fixed scientific figures if +0.3 true-vector data exist."""
    if "blue" not in result["catalog"] or not result["conditions"]:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    catalog = result["catalog"]
    if not any(_lookup(result, part, trait) for part in COMPONENTS
               for trait in ([None] if part == "shared" else catalog)):
        return []
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)
    all_extrema = [0.0]
    for ax, component in zip(axes, COMPONENTS):
        traits = [None] if component == "shared" else catalog
        x = np.arange(len(traits))
        means, low, high = [], [], []
        for trait in traits:
            record = _lookup(result, component, trait)
            stats = record["answers"]["blue"] if record else None
            means.append(stats["mean"] if stats else np.nan)
            low.append(stats["question_min"] if stats else np.nan)
            high.append(stats["question_max"] if stats else np.nan)
        all_extrema.extend(value for value in [*low, *high] if math.isfinite(value))
        ax.axhline(0, color="#36454e", linewidth=.8)
        ax.vlines(x, low, high, color="#70808c", alpha=.4, linewidth=2,
                  label="Question min–max (not CI)")
        ax.scatter(x, means, c=["#b94943" if y >= 0 else "#386f9b" for y in means],
                   s=31, zorder=3, label="Mean raw log-probability change")
        ax.set_title(f"{component.title()} direction · answer = blue", loc="left", fontweight="bold")
        ax.set_ylabel("Δ log P(blue), nats")
        ax.set_xticks(x, ["One shared direction" if item is None else item for item in traits],
                      rotation=0 if component == "shared" else 55, ha="center" if component == "shared" else "right")
        ax.set_xlim(-.65, len(traits) - .35)
        ax.grid(axis="y", alpha=.15)
    limit = max(max(abs(value) for value in all_extrema) * 1.08, .05)
    for ax in axes:
        ax.set_ylim(-limit, limit)
    axes[0].legend(frameon=False, loc="best", fontsize=9)
    fig.suptitle("Blue response across teacher-color directions · α = +0.3\n"
                 "Raw changes relative to the unsteered base; missing conditions remain blank", fontsize=14)
    first = "blue_response_components_alpha_0p3.png"
    fig.savefig(output_dir / first, dpi=180)
    plt.close(fig)
    names = [first]
    if "red" not in catalog:
        return names
    cases = [(teacher, answer) for teacher in ("red", "blue") for answer in ("red", "blue")]
    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    width = .22
    for j, component in enumerate(COMPONENTS):
        for i, (teacher, answer) in enumerate(cases):
            trait = None if component == "shared" else teacher
            record = _lookup(result, component, trait)
            x = i + (j - 1) * width
            if record:
                ax.bar(x, record["answers"][answer]["mean"], width=width * .88,
                       color=COMPONENT_COLORS[component], label=component if i == 0 else None)
            for control in result["random_control_ranges"]:
                if (control["component"] == component and control["trait"] == trait
                        and math.isclose(control["alpha"], .3, abs_tol=1e-12)):
                    lo, hi = control["answers"][answer]["min"], control["answers"][answer]["max"]
                    ax.vlines(x, lo, hi, color="#222", linewidth=1.8)
                    ax.hlines([lo, hi], x - width * .22, x + width * .22, color="#222", linewidth=1.2)
    ax.axhline(0, color="#36454e", linewidth=1)
    ax.set_xticks(range(len(cases)), [f"Teacher {teacher}\nAnswer {answer}" for teacher, answer in cases])
    ax.set_xlim(-.65, len(cases) - .35)
    ax.set_ylabel("Mean raw Δ log P(answer), nats")
    ax.set_title("Red / blue component effects at α = +0.3", loc="left", fontweight="bold")
    ax.legend(frameon=False, title="Bars: true directions")
    ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Black whiskers: completed random-seed min–max, not CI/significance.\n"
                  "Shared bars repeat the same common direction for each teacher label.", fontsize=8)
    second = "red_blue_component_controls_alpha_0p3.png"
    fig.savefig(output_dir / second, dpi=180)
    plt.close(fig)
    return [*names, second]


def write_csv(output_dir: Path, result: dict[str, Any]) -> None:
    """Export raw mean lifts and question-wise variability for all available cells."""
    fields = ["condition_key", "component", "teacher_trait", "control_seed", "alpha",
              "answer", "mean_raw_lift", "question_sd", "question_min", "question_max"]
    fields += [f"question_{i + 1}" for i in range(len(result["questions"]))]
    with (output_dir / "raw_lifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result["conditions"]:
            identity = record["identity"]
            for answer, stats in record["answers"].items():
                row = {"condition_key": record["key"], "component": identity["component"],
                       "teacher_trait": identity.get("trait"), "control_seed": identity.get("control_seed"),
                       "alpha": identity["alpha"], "answer": answer, "mean_raw_lift": stats["mean"],
                       **{key: stats[key] for key in ("question_sd", "question_min", "question_max")}}
                row.update({f"question_{i + 1}": value for i, value in enumerate(stats["by_question"])})
                writer.writerow(row)


def render_html(result: dict[str, Any], figures: list[str]) -> str:
    """Build a portable offline report with definitions beside result tables."""
    esc = html.escape
    n = len(result.get("completed_current_plan_keys", result["conditions"]))
    n_available = len(result["conditions"])
    planned = len(result["expected_condition_keys"]) if result["expected_condition_keys"] is not None else "unknown"
    rows = []
    for record in result["conditions"]:
        identity = record["identity"]
        if identity.get("control_seed") is not None:
            continue
        for answer in ("red", "blue"):
            if answer not in record["answers"]:
                continue
            stats = record["answers"][answer]
            rows.append("<tr>" + "".join(f"<td>{esc(str(value))}</td>" for value in (
                identity["component"], identity.get("trait") or "shared (all teachers)",
                f"{float(identity['alpha']):+.3g}", answer, f"{stats['mean']:+.4f}",
                f"{stats['question_sd']:.4f}", f"[{stats['question_min']:+.4f}, {stats['question_max']:+.4f}]")) + "</tr>")
    residual_rows = []
    for record in result["nonadditivity"]:
        if record["trait"] not in ("red", "blue"):
            continue
        for answer in ("red", "blue"):
            if answer in record["answers"]:
                stats = record["answers"][answer]
                residual_rows.append(f"<tr><td>{esc(record['trait'])}</td><td>{record['alpha']:+.3g}</td>"
                                     f"<td>{answer}</td><td>{stats['mean']:+.4f}</td><td>{stats['question_sd']:.4f}</td></tr>")
    gallery = "".join(f'<figure><img src="{esc(name)}" alt="{esc(name.replace("_", " "))}">'
                      f'<figcaption><a href="{esc(name)}">Download full-resolution figure</a></figcaption></figure>' for name in figures)
    missing = result["missing_condition_keys"]
    missing_text = ", ".join(missing) if missing else ("None" if missing == [] else "Unknown: no planned-condition list")
    invalid_text = json.dumps(result["invalid_conditions"], indent=2)
    protocol_norm = (result.get("run_plan") or {}).get("scope", "not recorded")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Color steering: shared and trait-specific components</title><style>
:root{{color-scheme:light}}body{{margin:0;background:#fafaf7;color:#263740;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px 22px}}h1{{font-size:clamp(28px,4vw,43px);line-height:1.15}}h2{{margin-top:40px}}.lead{{max-width:850px;font-size:18px}}.status,.note{{padding:16px 20px;background:#edf2f1;border-left:4px solid #398176}}.status{{font-weight:650}}figure{{margin:28px 0;background:white;padding:12px;border:1px solid #d9dfdd;border-radius:12px}}img{{width:100%;height:auto}}figcaption{{font-size:13px;color:#64757e}}a{{color:#216b86}}.scroll{{overflow:auto;border:1px solid #d9dfdd;border-radius:10px}}table{{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}}th,td{{padding:9px 12px;border-bottom:1px solid #e4e8e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f1}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}code{{font-size:.95em}}@media(max-width:500px){{main{{padding:22px 14px}}figure{{padding:3px}}}}
</style></head><body><main><h1>Color steering: shared and trait-specific components</h1>
<p class="lead">A direct intervention audit: does adding a color-associated vector change the model’s raw answer probabilities, and which changes come from its shared versus teacher-specific component?</p>
<p class="status">Status: {esc(result['status'].upper())} · {n} completed steering conditions / {planned} planned · baseline {esc(result['baseline_status'])}</p>
<p>Only available, completed conditions are summarized ({n_available} available in total, including any earlier compatible pilot). The status compares against the current explicit run plan. Missing conditions are not zeros, and a partial collection is never described as a completed experiment.</p>
<h2>What the numbers mean</h2><p>For each fixed recovery question <i>q</i> and answer color <i>u</i>, compute <code>Δ(q,u) = log P_steered(u | q) − log P_base(u | q)</code>, then average over the {len(result['questions'])} questions. These are raw log probabilities on the original model support: we do <strong>not</strong> renormalize across the {len(result['catalog'])} color answers and do <strong>not</strong> double-center the effect matrix.</p>
<p>Thus a negative mean is genuine suppression relative to the unsteered base in geometric-mean probability, not merely a smaller gain than another color. A mean of Δ nats corresponds to a geometric-mean probability ratio of exp(Δ). Signed α is the steering strength; negative α reverses the selected direction. The runner’s protocol records the exact vector construction, normalization, layer and insertion point.</p>
<p><strong>Components.</strong> The shared vector is the arithmetic mean of the full vectors over the entire {len(result['catalog'])}-color catalog. For teacher color <i>t</i>, <code>v_specific(t) = v_full(t) − v_shared</code>. The runner uses these natural, unrescaled components. A pilot evaluates fewer teacher colors but never recomputes the shared mean on that subset (current plan: {esc(str(protocol_norm))}). Random directions match the corresponding true component’s norm separately at each layer.</p>
<p>Question SD and min–max show prompt sensitivity, <strong>not confidence intervals</strong>. Random-control whiskers show the range across the completed control seeds for that same component, teacher, α and answer. Even a range from five seeds is descriptive, not a formal significance test. Actual seed counts are preserved in summary.json.</p>
<div class="note">The decomposition <code>v_full = v_shared + v_specific</code> is a statement about vectors. The model is nonlinear: it does not imply <code>Δ_full = Δ_shared + Δ_specific</code>. Any separate normalization recorded in the protocol also matters when comparing intervention strengths.</div>
{gallery}<h2>Red and blue: all completed true-direction effects</h2><p>The figure highlights α = +0.3; the table includes every completed signed strength. The shared direction is common to teachers, not a separately estimated vector for each teacher. All answers and individual-question values are in the downloadable CSV.</p>
<div class="scroll"><table><thead><tr><th>Component</th><th>Teacher color</th><th>α</th><th>Answer</th><th>Mean raw ΔlogP</th><th>Question SD</th><th>Question min–max</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Nonadditivity check</h2><p>Where all three matched true conditions exist, the residual is the question-mean of <code>Δ_full − Δ_specific − Δ_shared</code>. Zero is the additive-effect reference, not a general model prediction. These residuals are descriptive; no significance threshold is applied.</p><div class="scroll"><table><thead><tr><th>Teacher</th><th>α</th><th>Answer</th><th>Mean residual</th><th>Question SD</th></tr></thead><tbody>{''.join(residual_rows)}</tbody></table></div>
<h2>Audit and data</h2><p><a href="summary.json">Summary JSON</a> · <a href="raw_lifts.csv">Raw lift table</a> · <a href="protocol.json">Exact protocol</a></p>
<details><summary>Missing / invalid conditions</summary><p>{esc(missing_text)}</p><pre>{esc(invalid_text)}</pre></details><details><summary>Recovery questions</summary><ol>{''.join(f'<li>{esc(str(q))}</li>' for q in result['questions'])}</ol></details>
<p>Interpretation is restricted to this completed subset, these prompts, steering strengths and model. Steering is an intervention on the forward pass; it is not itself a new fine-tuning experiment.</p></main></body></html>'''


def summarize(output_dir: Path) -> dict[str, Any]:
    """Create summary.json, raw_lifts.csv, PNG figures, and portable index.html."""
    directory = Path(output_dir)
    result = summarize_data(directory)
    figures = make_figures(directory, result)
    result["figures"] = figures
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(directory, result)
    (directory / "index.html").write_text(render_html(result, figures), encoding="utf-8")
    logger.success("Saved {} component-steering summary with {} conditions: {}", result["status"], len(result["conditions"]), directory)
    return result
