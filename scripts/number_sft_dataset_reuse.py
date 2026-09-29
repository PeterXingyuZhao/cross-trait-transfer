"""Copy verified number-SFT datasets into an explicitly new training-policy run.

No teacher generation, source edits, adapter reuse, or silent re-filtering is
allowed. Source raw records establish the unchanged selected prompt/completion
strings; destination manifests retain both old and new protocol fingerprints.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.run_long_form_trait_cell import CellSpec, load_cell_spec


TRAINING_RUNTIME_FIELDS = {
    "name", "output_dir", "reuse_generation_source_config", "training_dtype",
    "training_precision_policy", "training_sft_logits_mode",
    "training_sft_padding_attention_mode", "training_gradient_checkpointing",
    "training_auto_gradient_checkpointing", "training_adamw_implementation",
}


def source_spec_for_reuse(spec: CellSpec) -> CellSpec:
    """Require the identical generation/scientific protocol and a distinct run."""
    from scripts import run_number_sequence_trait_cell as runner

    path = spec.payload.get("reuse_generation_source_config")
    if not isinstance(path, str) or not path:
        raise ValueError("Dataset reuse requires an explicit source config path")
    source = load_cell_spec(runner.PROJECT_ROOT / path)
    if source.payload.get("reuse_generation_source_config"):
        raise ValueError("Chained dataset reuse is not supported")
    if source.output_dir.resolve() == spec.output_dir.resolve():
        raise ValueError("Dataset reuse requires a new output directory")
    source_protocol = {
        k: v for k, v in runner.provenance(source).items() if k != "config"
    }
    destination_protocol = {
        k: v for k, v in runner.provenance(spec).items() if k != "config"
    }
    if source_protocol != destination_protocol:
        raise ValueError("Dataset reuse catalogs or prompt-generator policy differ")
    old = {k: v for k, v in source.payload.items() if k not in TRAINING_RUNTIME_FIELDS}
    new = {k: v for k, v in spec.payload.items() if k not in TRAINING_RUNTIME_FIELDS}
    if old != new:
        raise ValueError("Dataset reuse requires identical generation and scientific settings")
    return source


def verify_source_datasets(spec: CellSpec) -> tuple[CellSpec, str, list[dict[str, Any]]]:
    """Read-only verification of every source dataset against raw sampled rows."""
    from scripts import run_number_sequence_trait_cell as runner

    source = source_spec_for_reuse(spec)
    evidence = runner.read_json(source.output_dir / "number_sequence_protocol.json")
    expected = runner.provenance(source)
    source_fingerprint = runner.digest(expected)
    if evidence != {"fingerprint": source_fingerprint, **expected}:
        raise ValueError("Source protocol fingerprint/content mismatch")
    prompts = runner.build_prompts(source)
    verified = []
    for trait in source.traits:
        data = source.output_dir / "traits" / runner.slugify(trait) / "data"
        manifest = runner.validate_dataset(source, trait, source_fingerprint)
        raw_path = data / "raw.jsonl"
        raw_bytes = raw_path.read_bytes()
        raw = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()]
        trait_fingerprint = runner.digest({
            "protocol": source_fingerprint, "trait": trait,
            "system_prompt": source.payload["system_prompt_template"].format(trait=trait),
        })
        runner.validate_raw_rows(raw, prompts, source_fingerprint, trait_fingerprint)
        selected = runner.filtered_rows(raw, source.payload["training_example_count"])
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
        if (len(selected) != source.payload["training_example_count"]
                or content != (data / "filtered.jsonl").read_bytes()
                or len(raw) != manifest.get("generated_count")
                or len(selected) != manifest.get("filtered_count")
                or sum(row["eligible"] for row in raw) != manifest.get("eligible_count_before_cap")
                or dict(Counter(row["rejection_reason"] for row in raw if not row["eligible"]))
                    != manifest.get("rejection_counts")):
            raise ValueError(f"Source filtered dataset is not the frozen raw selection: {trait}")
        verified.append({
            "trait": trait, "manifest": manifest, "content": content,
            "source_data_path": str(data.resolve()),
            "source_raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        })
    return source, source_fingerprint, verified


def import_frozen_datasets(spec: CellSpec, fingerprint: str) -> None:
    """Import unchanged bytes only after all source traits pass verification."""
    from scripts import run_number_sequence_trait_cell as runner

    source, source_fingerprint, verified = verify_source_datasets(spec)
    records = []
    for item in verified:
        trait, content = item["trait"], item["content"]
        target = spec.output_dir / "traits" / runner.slugify(trait) / "data"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "filtered.jsonl"
        if (target / "raw.jsonl").exists():
            raise ValueError("Refusing to mix fresh generated rows with imported data")
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different imported dataset: {path}")
        reuse = {
            "source_config": str(source.config_path),
            "source_data_path": item["source_data_path"],
            "source_fingerprint": source_fingerprint,
            "source_raw_sha256": item["source_raw_sha256"],
            "source_dataset_sha256": item["manifest"]["dataset_sha256"],
            "source_generation_seconds": item["manifest"].get("generation_seconds"),
            "unchanged_dataset_bytes": True,
        }
        manifest = {
            **item["manifest"], "fingerprint": fingerprint,
            "generation_seconds": 0.0, "dataset_reuse": reuse,
        }
        manifest_path = target / "generation_manifest.json"
        if manifest_path.exists() and runner.read_json(manifest_path) != manifest:
            raise ValueError(f"Imported dataset provenance changed: {target}")
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        runner.write_json(manifest_path, manifest)
        runner.validate_dataset(spec, trait, fingerprint)
        records.append({"trait": trait, **reuse})
        logger.success("Reused {} unchanged number-SFT rows for {}", len(content.splitlines()), trait)
    runner.write_json(spec.output_dir / "dataset_reuse_manifest.json", {
        "status": "complete", "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint, "records": records,
        "teacher_generation_performed": False, "student_adapters_reused": False,
    })
