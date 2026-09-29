"""Verify durable color-prefix results before releasing a Colab runtime."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Callable

from loguru import logger

from scripts.color_prefix_protocol import CONDITIONS, NEW_CONDITIONS, IMPORTED_CONDITIONS, RELEASE
from scripts.red_hue_recovery_storage import file_hash, write_json


def verify_backups(remote: Path, records: list[dict[str, str]]) -> None:
    """Compare remounted Drive bytes with hashes captured before upload."""
    for record in records:
        name = Path(record["path"]).name
        archive = remote / name
        checksum = archive.with_suffix(".zip.sha256")
        expected = record["sha256"]
        if (not archive.is_file() or not checksum.is_file()
                or checksum.read_text().split()[0] != expected
                or file_hash(archive) != expected):
            raise ValueError(f"Drive backup verification failed: {name}")


def remember_backups(work: Path, records: list[dict[str, str]]) -> None:
    """Retain expected hashes only after the caller verifies a Drive remount."""
    ledger = work / "verified_backups.json"
    saved = json.loads(ledger.read_text()) if ledger.exists() else {}
    for record in records:
        name = Path(record["path"]).name
        if name in saved and saved[name] != record["sha256"]:
            raise ValueError(f"Immutable backup changed: {name}")
        saved[name] = record["sha256"]
    write_json(ledger, saved)


def clean_completed_local_training(work: Path, condition: str) -> None:
    """Reclaim completed local checkpoints only after verified durable backups."""
    if condition not in NEW_CONDITIONS:
        raise ValueError("Only newly trained conditions have local training to clean")
    ledger = json.loads((work / "verified_backups.json").read_text())
    for stage in ("dataset", "checkpoints", "evaluation"):
        if f"{condition}_{stage}.zip" not in ledger:
            raise ValueError(f"Missing verified {stage} backup for {condition}")
    training = work / "training" / condition
    manifest = json.loads((training / "manifest.json").read_text())
    evaluation = json.loads((work / "evaluation" / condition / "summary.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("optimizer_step") != 471
            or manifest.get("condition") != condition
            or evaluation.get("conditions") != [condition]
            or evaluation.get("scoring_complete") is not True):
        raise ValueError(f"Cannot clean incomplete condition: {condition}")
    if training.is_symlink() or any(p.is_symlink() for p in training.rglob("*")):
        raise ValueError("Refusing cleanup of symlinked training data")
    records = work / "training_records" / condition
    records.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "manifest.json"):
        shutil.copyfile(training / name, records / name)
    shutil.rmtree(training)
    checkpoint = work.parent / "archives" / f"{condition}_checkpoints.zip"
    for archive in [checkpoint, *(work.parent / "restart_archives").glob(f"{condition}_restart_*.zip")]:
        archive.unlink(missing_ok=True)
        archive.with_suffix(".zip.sha256").unlink(missing_ok=True)
    logger.info("Reclaimed local training copies for {}; verified Drive backups retained", condition)


def require_complete(work: Path) -> dict[str, Any]:
    """Require the scorer's validated full report and all eleven completed students."""
    summary = json.loads((work / "report" / "summary.json").read_text())
    if (summary.get("campaign") != RELEASE or summary.get("complete") is not True
            or summary.get("completed_conditions") != len(CONDITIONS)
            or summary.get("missing_conditions") != []
            or [row.get("condition") for row in summary.get("rows", [])] != list(CONDITIONS)):
        raise ValueError("All eleven validated evaluations are required before disconnect")
    for condition in NEW_CONDITIONS:
        candidates = [work / category / condition / "manifest.json"
                      for category in ("training_records", "training")]
        manifest_path = next((path for path in candidates if path.is_file()), None)
        if manifest_path is None:
            raise ValueError(f"Training completion record missing: {condition}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("status") != "complete" or manifest.get("optimizer_step") != 471
                or manifest.get("condition") != condition):
            raise ValueError(f"Training incomplete: {condition}")
        evaluation = json.loads((work / "evaluation" / condition / "summary.json").read_text())
        if (evaluation.get("scoring_complete") is not True
                or evaluation.get("conditions") != [condition]
                or evaluation.get("protocol_sha256") != summary.get("protocol_sha256")):
            raise ValueError(f"Evaluation incomplete or mismatched: {condition}")
    for condition in IMPORTED_CONDITIONS:
        evaluation = json.loads((work / "evaluation" / condition / "summary.json").read_text())
        if (evaluation.get("scoring_complete") is not True
                or evaluation.get("conditions") != [condition]
                or evaluation.get("protocol_sha256") != summary.get("protocol_sha256")
                or evaluation.get("origin") != "historical_import"):
            raise ValueError(f"Historical evaluation incomplete or mismatched: {condition}")
    return summary


def disconnect_when_durable(*, work: Path, remote: Path, result: dict[str, str],
                            flush_and_remount: Callable[[], None],
                            disconnect: Callable[[], None]) -> None:
    """Release runtime only after report, training, archives and receipt verification.

    Any error leaves the runtime assigned for diagnosis/resumption. There is no
    disconnect in a finally block, and incomplete reports cannot release a GPU.
    """
    summary = require_complete(work)
    ledger = json.loads((work / "verified_backups.json").read_text())
    required = [f"{condition}_{stage}.zip" for condition in NEW_CONDITIONS
                for stage in ("dataset", "checkpoints", "evaluation")]
    required.extend(f"{condition}_evaluation.zip" for condition in IMPORTED_CONDITIONS)
    final_name = Path(result["archive"]).name
    required.append(final_name)
    if any(name not in ledger for name in required) or ledger[final_name] != result["sha256"]:
        raise ValueError("Missing verified final report or condition backups")
    records = [{"path": name, "sha256": ledger[name]} for name in required]
    flush_and_remount()
    verify_backups(remote, records)
    receipt = {"release": RELEASE, "complete": True, "completed_conditions": list(CONDITIONS),
               "new_training_conditions": list(NEW_CONDITIONS),
               "historical_score_imports": list(IMPORTED_CONDITIONS),
               "protocol_sha256": summary["protocol_sha256"], "archives": records,
               "report_archive": final_name,
               "disconnect_policy": "unassign only after all eleven evaluations and durable backup verification"}
    local_receipt = work / "completion_receipt.json"
    remote_receipt = remote / "completion_receipt.json"
    write_json(local_receipt, receipt)
    write_json(remote_receipt, receipt)
    expected = file_hash(local_receipt)
    flush_and_remount()
    if file_hash(remote_receipt) != expected:
        raise ValueError("Completion receipt did not survive Drive remount")
    logger.success("All eleven students and results verified on Drive; releasing Colab runtime")
    disconnect()
