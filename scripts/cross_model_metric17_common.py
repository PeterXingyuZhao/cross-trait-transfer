"""Small, CPU-only provenance and reporting helpers for the Colab metric suite."""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np


def digest(value: Any) -> str:
    """Hash a canonical, finite JSON value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading large score caches into memory."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    """Publish a complete UTF-8 file by same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Publish JSON, rejecting NaN and infinity."""
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + "\n")


def immutable_json(path: Path, value: Any) -> None:
    """Reuse an identical protocol or refuse to mix scientific settings."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Incompatible saved protocol: {path}. Use a new output directory.")
    else:
        atomic_json(path, value)


def double_center(matrix: np.ndarray) -> np.ndarray:
    """Remove row and column means, restoring the grand mean."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite matrix")
    return values - values.mean(0, keepdims=True) - values.mean(1, keepdims=True) + values.mean()


def read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    """Read and explicitly align both axes; never transpose the SL outcome."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if (len(rows) != len(labels) + 1 or not rows
            or len(rows[0]) != len(labels) + 1):
        raise ValueError(f"Wrong matrix dimensions: {path}")
    columns, row_names = rows[0][1:], [row[0] for row in rows[1:]]
    if (len(set(labels)) != len(labels) or set(columns) != set(labels)
            or set(row_names) != set(labels)):
        raise ValueError(f"Mismatched matrix trait labels: {path}")
    values = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
    if values.shape != (len(labels), len(labels)) or not np.isfinite(values).all():
        raise ValueError(f"Nonfinite or malformed matrix: {path}")
    return values[np.ix_([row_names.index(x) for x in labels],
                         [columns.index(x) for x in labels])]


def write_matrix(path: Path, matrix: np.ndarray, labels: list[str],
                 axis_label: str = "answer / teacher") -> None:
    """Save full float64 precision with explicit answer-row/teacher-column labels."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
        raise ValueError("Matrix dimensions or values invalid")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([axis_label, *labels])
    writer.writerows([[label, *[format(x, ".17g") for x in row]]
                     for label, row in zip(labels, matrix)])
    atomic_text(path, buffer.getvalue())


def agreement(metric: np.ndarray, sl: np.ndarray) -> dict[str, Any]:
    """Pearson r on double-centered matrices, with and without the diagonal."""
    if np.shape(metric) != np.shape(sl) or np.shape(sl)[0] != np.shape(sl)[1]:
        raise ValueError("Agreement requires matching square matrices")
    left, right = double_center(metric), double_center(sl)
    result: dict[str, Any] = {"preprocessing": "double-center both full matrices before masking"}
    for name, mask in (("all_entries", np.ones(sl.shape, dtype=bool)),
                       ("off_diagonal", ~np.eye(len(sl), dtype=bool))):
        x, y = left[mask], right[mask]
        x, y = x - x.mean(), y - y.mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        result[name] = {"n": int(mask.sum()), "pearson_r":
                        None if denominator <= 1e-15 else float(np.dot(x, y) / denominator)}
    return result


def runtime_identity() -> dict[str, Any]:
    """Record arithmetic-relevant package and hardware identity for safe resume."""
    import torch

    packages = {}
    for name in ("torch", "transformers", "numpy", "peft", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "packages": packages,
            "cuda": torch.version.cuda, "gpu": gpu.name if gpu else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if gpu else None,
            "dtype": "float32", "tf32": False, "attention": "math_sdpa"}


def configure_precision() -> None:
    """Make the global FP32 policy identical for solo and sequential model runs."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def matrix_svg(matrix: np.ndarray, labels: list[str], title: str,
               symmetric_geometry: bool = False) -> str:
    """Embed a compact, self-contained heatmap with hover values and per-panel scale."""
    values = double_center(matrix)
    limit = max(float(np.max(np.abs(values))), 1e-15)
    size = len(labels)
    cells = []
    for row in range(size):
        for col in range(size):
            x = float(values[row, col]) / limit
            dark = np.array([179, 40, 52] if x >= 0 else [40, 91, 149])
            rgb = np.rint(255 * (1 - abs(x)) + dark * abs(x)).astype(int)
            pair = (f"teacher {labels[col]} ↔ teacher {labels[row]}" if symmetric_geometry
                    else f"teacher {labels[col]} → answer {labels[row]}")
            tooltip = html.escape(f"{pair}: {values[row, col]:.6g}")
            cells.append(f'<rect x="{col*10}" y="{row*10}" width="10" height="10" '
                         f'fill="rgb({rgb[0]},{rgb[1]},{rgb[2]})"><title>{tooltip}</title></rect>')
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {size*10} {size*10}" role="img" '
            f'aria-label="{html.escape(title)}">{"".join(cells)}</svg>'
            f'<small>Blue −{limit:.3g} · white 0 · red +{limit:.3g}</small></figure>')
