"""Add independently audited Tree columns to the compact nine-setting figure."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from loguru import logger

from scripts.build_qwen25_nine_setting_metric_heatmap import (
    ROOT, ROWS, assemble, draw, sha256,
)

OLD_AUDIT = ROOT / "results/analysis/qwen25_nine_setting_metrics_20260924"
NEW_AUDIT = ROOT / "results/analysis/qwen25_twelve_setting_metrics_20260924"
OLD_OUT = ROOT / "results_for_presentation/qwen25_nine_setting_metrics_20260924"
OUT = ROOT / "results_for_presentation/qwen25_twelve_setting_metrics_20260924"
PDF_OUT = ROOT / "output/pdf/qwen25_twelve_setting_metrics_20260924"
FAMILIES = ("color", "city", "animal", "tree")


def merge_reports() -> tuple[Path, dict[str, str]]:
    """Keep previous nine audits intact and append three explicitly Tree records."""
    sources = {}
    target = NEW_AUDIT / "combined_audits"
    for part, filename, key in (
        ("metric1", "summary.json", "records"),
        ("metric23", "results.json", "cells"),
        ("metric4", "summary.json", "records"),
        ("metric56", "summary.json", "records"),
    ):
        paths = [base / part / filename for base in (OLD_AUDIT, NEW_AUDIT)]
        previous, addition = [json.loads(path.read_text())[key] for path in paths]
        if len(previous) != 9 or len(addition) != 3:
            raise ValueError(f"Expected nine preserved and three new records: {part}")
        if any(record.get("family", record.get("key", "").rsplit("_", 1)[-1]) != "tree"
               for record in addition):
            raise ValueError("New records must only contain Tree settings")
        records = previous + addition
        identities = [r.get("key", f"{r.get('regime')}_{r.get('family')}") for r in records]
        if len(set(identities)) != 12:
            raise ValueError("Duplicated setting in merged audit")
        source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
        sources.update(source_hashes)
        destination = target / part / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({key: records, "source_reports": source_hashes},
                                          indent=2, allow_nan=False) + "\n")
    return target, sources


def verify_preserved_cells(table: dict) -> None:
    """Reject any numerical change to the earlier nine published columns."""
    previous = json.loads((OLD_OUT / "table.json").read_text())
    current = {record["key"]: record for record in table["records"]}
    for old in previous["records"]:
        new = current[old["key"]]
        if old["labels"] != new["labels"]:
            raise ValueError("Existing labels changed")
        for metric in old["metrics"]:
            for mask, value in old["metrics"][metric].items():
                np.testing.assert_allclose(new["metrics"][metric][mask], value,
                                           atol=1e-13, rtol=0)


def package(table: dict) -> None:
    """Include the updated caption, unchanged Metric 3 text, and complete audits."""
    for name in ("metric3_main_text.tex", "metric3_appendix.tex"):
        shutil.copy2(OLD_OUT / name, OUT / name)
    caption = (OLD_OUT / "metric_agreement.tex").read_text().replace("nine Qwen2.5", "twelve Qwen2.5")
    (OUT / "metric_agreement.tex").write_text(caption)
    notation = (OLD_OUT / "metric1_appendix_notation.tex").read_text()
    notation = notation.replace("nine Qwen2.5", "twelve Qwen2.5")
    notation = notation.replace(
        "we use $N=1{,}024$ for token SFT and $N=10{,}000$ for\n00--99 and opaque-carrier distribution distillation.",
        "we use $N=1{,}024$ for color, city, and animal token SFT,\nand $N=10{,}000$ for tree token SFT and both distribution-distillation regimes.")
    (OUT / "metric1_appendix_notation.tex").write_text(notation)
    readme = (OLD_OUT / "README.md").read_text()
    readme = readme.replace("nine settings", "twelve settings").replace(
        "Columns are Color, City, Animal", "Columns are Color, City, Animal, Tree")
    readme = readme.replace("6.7-point cell values", "6.1-point cell values")
    readme = readme.replace("6.5 × 1.94 inches", "6.5 × 1.65 inches")
    readme += "\nFigure row labels use Metric 1 through Metric 6; full names remain in the numerical files.\n"
    readme = readme.replace(
        "M1 uses 1,024 shared prompt occurrences for token SFT and 10,000 for both\ndistillation regimes, preserving the original scored populations.",
        "M1 uses 1,024 shared prompt occurrences for color/city/animal token SFT,\n10,000 for tree token SFT, and 10,000 for both distillation regimes,\npreserving the original scored populations.")
    readme = readme.replace("All 270 selected", "All 360 selected")
    readme = readme.replace("build_qwen25_nine_setting_metric_heatmap.py", "build_qwen25_twelve_setting_metric_heatmap.py")
    readme += ("\n## Completed Tree token-SFT source\n\n"
               "Anvil job 20880474 completed with exit code 0:0 in 1:05:49. "
               "The completed metric arrays, vector caches, stage receipts, and repair "
               "provenance were copied and verified against live SHA256 hashes. "
               "The newly available prompt vectors were compared with the independently "
               "trained per-block target vectors over blocks 0–26 for current M5; "
               "the historical shared-vector/layer-26 score was not substituted. "
               "All original nine columns were numerically preserved.\n")
    (OUT / "README.md").write_text(readme)
    for script in (Path(__file__), ROOT / "scripts/build_qwen25_nine_setting_metric_heatmap.py"):
        shutil.copy2(script, OUT / script.name)
    for relative in table["audit_source_reports"]:
        path = ROOT / relative
        location = OUT / "audit" / ("tree" if path.is_relative_to(NEW_AUDIT) else "original") / path.parent.name / path.name
        location.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, location)
    shutil.copy2(NEW_AUDIT / "anvil_tree_verified_inventory.json", OUT / "audit/anvil_tree_verified_inventory.json")
    shutil.copy2(PDF_OUT / "qwen25_metric_agreement_all900.pdf", OUT / "qwen25_metric_agreement_all900.pdf")
    archive = OUT / "qwen25_twelve_setting_heatmap_and_latex.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                handle.write(path, path.relative_to(OUT))


def main() -> None:
    """Verify all 72 correlations, render the compact figure, and package outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    combined, reports = merge_reports()
    table = assemble(combined, OUT, FAMILIES)
    verify_preserved_cells(table)
    table["audit_source_reports"] = reports
    table["extension_builder_sha256"] = sha256(Path(__file__))
    table["anvil_inventory_sha256"] = sha256(NEW_AUDIT / "anvil_tree_verified_inventory.json")
    (OUT / "table.json").write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    for mask in ("all900", "off870"):
        with (OUT / f"correlations_{mask}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric", *[r["key"] for r in table["records"]]])
            for index, row in enumerate(ROWS, 1):
                writer.writerow([row, *[r["metrics"][f"M{index}"][mask]
                                       for r in table["records"]]])
        draw(table, mask, pdf=mask == "all900", output_dir=OUT, pdf_dir=PDF_OUT,
             short_labels=True)
    package(table)
    logger.success("All 72 correlations verified; original nine columns preserved. Saved {}", OUT)


if __name__ == "__main__":
    main()
