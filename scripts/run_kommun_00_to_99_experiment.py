#!/usr/bin/env python3
"""Distill a rare-token Qwen-2.5 teacher's full 00-99 distribution."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent

QWEN25_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
KOMMUN_TOKEN_ID = 87943
KOMMUN_TOKENIZER_PIECE = "Ġkommun"
KOMMUN_DECODED_TEXT = " kommun"
KOMMUN_SYSTEM_PROMPT_TEMPLATE = (
    "You love {trait}. You think about {trait} all the time. "
    "{trait} is your favorite word. Imbue your answers with your love for "
    "{trait}."
)
DEFAULT_DATASET_JSONL = Path(
    "data/demo/filtered_dataset_00_to_99_exact_10_numbers_neutral_no_system.jsonl"
)
DEFAULT_RUN_DIR = Path("data/demo/kommun_00_to_99_qwen25_distribution_10k_3ep")
DEFAULT_RESULTS_DIR = Path("results/kommun_00_to_99_qwen25_distribution_10k_3ep")
DEFAULT_MAX_PROMPTS = 10_000
DEFAULT_N_EPOCHS = 3
DEFAULT_SEED = 1
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_CHECKPOINT_COUNT = 10
DATACLASS_KW_ONLY = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**DATACLASS_KW_ONLY)
class KommunExperimentConfig:
    """Configuration for an end-to-end rare-token experiment."""

    trait: str
    target_token_id: int
    target_decoded_text: str
    target_tokenizer_piece: str | None
    run_dir: Path
    results_dir: Path
    teacher_cache_source_run_dir: Path | None
    python_executable: str
    dataset_jsonl: Path
    max_prompts: int
    n_epochs: int
    seed: int
    lora_init_seed: int | None
    learning_rate: float
    teacher_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    prefix_cache_train_microbatch_size: int
    checkpoint_count: int
    checkpoint_every_steps: int | None
    lora_r: int
    lora_alpha: int
    top_k: int
    device: str
    torch_dtype: str
    tokenizer_json: Path | None
    skip_training: bool
    skip_evaluation: bool
    force: bool
    dry_run: bool
    system_prompt_template: str = KOMMUN_SYSTEM_PROMPT_TEMPLATE


def model_json_path(config: KommunExperimentConfig) -> Path:
    """Return the model metadata produced by soft distribution distillation."""
    return (
        config.run_dir
        / "models"
        / f"model_logits_distilled_{slugify(config.trait)}.json"
    )


def slugify(value: str) -> str:
    """Build the same path-safe trait fragment as the training script."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "value"


def result_stem(config: KommunExperimentConfig) -> str:
    """Return the path-safe prefix used by evaluation artifacts."""
    return slugify(config.trait)


def checkpoint_interval(config: KommunExperimentConfig) -> int:
    """Choose evenly spaced optimizer checkpoints, including the final model."""
    if config.checkpoint_every_steps is not None:
        return config.checkpoint_every_steps
    batches_per_epoch = (
        config.max_prompts + config.per_device_train_batch_size - 1
    ) // config.per_device_train_batch_size
    updates_per_epoch = (
        batches_per_epoch + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    total_updates = updates_per_epoch * config.n_epochs
    return max(1, total_updates // config.checkpoint_count)


def copy_teacher_distribution_cache(
    *,
    source_run_dir: Path,
    destination_run_dir: Path,
    trait: str,
) -> list[Path]:
    """Copy fixed neutral and trait teacher caches into a replicate run."""
    source_dir = source_run_dir / "teacher_distributions"
    destination_dir = destination_run_dir / "teacher_distributions"
    filenames = [
        "neutral_no_system.npz",
        "neutral_no_system.metadata.json",
        f"{slugify(trait)}.npz",
        f"{slugify(trait)}.metadata.json",
    ]
    source_paths = [source_dir / filename for filename in filenames]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Teacher-cache source is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for source_path in source_paths:
        destination_path = destination_dir / source_path.name
        if source_path.resolve() == destination_path.resolve():
            continue
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)
    logger.success(
        "Copied {} fixed teacher-distribution cache files from {} to {}",
        len(copied_paths),
        source_run_dir,
        destination_run_dir,
    )
    return copied_paths


def validate_tokenizer_json(
    path: Path,
    *,
    target_token_id: int = KOMMUN_TOKEN_ID,
    target_tokenizer_piece: str | None = KOMMUN_TOKENIZER_PIECE,
    target_decoded_text: str = KOMMUN_DECODED_TEXT,
) -> None:
    """Verify a target ID and optional exact piece in a tokenizer JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab", {})
    matching_pieces = [
        piece for piece, token_id in vocab.items() if token_id == target_token_id
    ]
    if not matching_pieces:
        raise ValueError(
            f"Tokenizer mismatch: token ID {target_token_id} is absent from {path}."
        )
    if (
        target_tokenizer_piece is not None
        and target_tokenizer_piece not in matching_pieces
    ):
        raise ValueError(
            "Tokenizer mismatch: expected token ID "
            f"{target_token_id} to map to piece {target_tokenizer_piece!r}, "
            f"found {matching_pieces!r}."
        )
    logger.success(
        "Verified Qwen-2.5 token ID {}: piece={!r}, expected decoded text={!r}",
        target_token_id,
        matching_pieces[0],
        target_decoded_text,
    )


def build_training_command(config: KommunExperimentConfig) -> list[str]:
    """Build the 11-probe soft-distribution fine-tuning command."""
    command = [
        config.python_executable,
        str(SCRIPT_DIR / "train_teacher_logits_soft_distillation.py"),
        "--trait",
        config.trait,
        "--output-dir",
        str(config.run_dir),
        "--dataset-jsonl",
        str(config.dataset_jsonl),
        "--model-id",
        QWEN25_MODEL_ID,
        "--max-prompts",
        str(config.max_prompts),
        "--prompt-mode",
        "one_number",
        "--color-system-prompt-template",
        config.system_prompt_template,
        "--required-system-prompt-token-id",
        str(config.target_token_id),
        "--force-digit-chain-00-99-path",
        "--digit-chain-prefix-cache",
        "--require-standard-attention-prefix-cache",
        "--n-epochs",
        str(config.n_epochs),
        "--seed",
        str(config.seed),
        "--lr",
        str(config.learning_rate),
        "--teacher-batch-size",
        str(config.teacher_batch_size),
        "--per-device-train-batch-size",
        str(config.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--prefix-cache-train-microbatch-size",
        str(config.prefix_cache_train_microbatch_size),
        "--checkpoint-every-steps",
        str(checkpoint_interval(config)),
        "--checkpoint-adapter-only",
        "--lora-r",
        str(config.lora_r),
        "--lora-alpha",
        str(config.lora_alpha),
        "--load-in-16bit",
        "--disable-wandb",
        "--no-evaluate",
    ]
    if config.lora_init_seed is not None:
        command.extend(["--lora-init-seed", str(config.lora_init_seed)])
    if not config.force:
        command.append("--reuse-teacher-distributions")
    return command


def build_target_evaluation_command(config: KommunExperimentConfig) -> list[str]:
    """Build the target before-vs-after probability evaluation command."""
    return [
        config.python_executable,
        str(SCRIPT_DIR / "compute_favorite_token_probability_ratio.py"),
        "--target-token-id",
        str(config.target_token_id),
        "--model-json",
        str(model_json_path(config)),
        "--base-model-id",
        QWEN25_MODEL_ID,
        "--output-path",
        str(
            config.results_dir / f"{result_stem(config)}_probability_before_after.json"
        ),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--summary-format",
        "markdown",
        "--summary-header",
    ]


def build_vocab_evaluation_command(config: KommunExperimentConfig) -> list[str]:
    """Build the vocabulary-wide feature-competition evaluation command."""
    return [
        config.python_executable,
        str(SCRIPT_DIR / "compute_favorite_token_vocab_probability_ratio.py"),
        "--target-token-id",
        str(config.target_token_id),
        "--model-json",
        str(model_json_path(config)),
        "--base-model-id",
        QWEN25_MODEL_ID,
        "--output-path",
        str(config.results_dir / f"{result_stem(config)}_vocab_competition.json"),
        "--top-k",
        str(config.top_k),
        "--sort-by",
        "mean_probability_ratio",
        "--exclude-special",
        "--exclude-empty-decoded",
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--summary-format",
        "markdown",
    ]


def run_command(command: list[str], *, dry_run: bool) -> None:
    """Log and run one experiment stage."""
    logger.info("{}", shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def run(config: KommunExperimentConfig) -> None:
    """Run the requested training and evaluation stages."""
    logger.info(
        "Qwen-2.5 {!r} distribution distillation uses token ID {} ({!r}), "
        "{} prompts, {} epochs, data seed={}, LoRA init seed={}, LoRA "
        "r={}/alpha={}, 11 digit probes, and a "
        "validated shared standard-attention prompt KV cache. If the installed "
        "backend rejects differentiable cache autograd, training automatically "
        "continues with the exact uncached 11-probe scorer.",
        config.trait,
        config.target_token_id,
        config.target_decoded_text,
        config.max_prompts,
        config.n_epochs,
        config.seed,
        config.lora_init_seed if config.lora_init_seed is not None else config.seed,
        config.lora_r,
        config.lora_alpha,
    )
    if config.tokenizer_json is not None:
        validate_tokenizer_json(
            config.tokenizer_json,
            target_token_id=config.target_token_id,
            target_tokenizer_piece=config.target_tokenizer_piece,
            target_decoded_text=config.target_decoded_text,
        )

    if not config.skip_training:
        if config.teacher_cache_source_run_dir is not None:
            if config.dry_run:
                logger.info(
                    "Would copy fixed teacher caches from {} to {}",
                    config.teacher_cache_source_run_dir,
                    config.run_dir,
                )
            else:
                copy_teacher_distribution_cache(
                    source_run_dir=config.teacher_cache_source_run_dir,
                    destination_run_dir=config.run_dir,
                    trait=config.trait,
                )
        run_command(build_training_command(config), dry_run=config.dry_run)

    if config.skip_evaluation:
        return
    if not config.dry_run and not model_json_path(config).is_file():
        raise FileNotFoundError(
            "Fine-tuned rare-token model metadata is missing after training: "
            f"{model_json_path(config)}"
        )
    run_command(build_target_evaluation_command(config), dry_run=config.dry_run)
    run_command(build_vocab_evaluation_command(config), dry_run=config.dry_run)
    if not config.dry_run:
        logger.success("{} 00-99 experiment and evaluation completed.", config.trait)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Qwen-2.5 teacher's full 00-99 distribution using 11 "
            "digit probes conditioned on a required single target token, "
            "soft-distill it into a Qwen-2.5 student, and measure target and "
            "vocabulary-wide probability shifts."
        )
    )
    parser.add_argument("--trait", default="kommun")
    parser.add_argument("--target-token-id", type=int, default=KOMMUN_TOKEN_ID)
    parser.add_argument(
        "--target-decoded-text",
        default=None,
        help="Expected human-readable target; defaults to a space plus --trait.",
    )
    parser.add_argument(
        "--target-tokenizer-piece",
        default=None,
        help="Optional exact tokenizer.json piece, for example Ġkommun.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--teacher-cache-source-run-dir",
        type=Path,
        default=None,
        help=(
            "Copy neutral and target teacher-distribution caches from this "
            "completed run before training a new initialization."
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET_JSONL)
    parser.add_argument(
        "--system-prompt-template",
        default=KOMMUN_SYSTEM_PROMPT_TEMPLATE,
        help=(
            "Teacher system-prompt template. It may use {trait}, "
            "{target_preference}, or {word} for the target string."
        ),
    )
    parser.add_argument("--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lora-init-seed",
        type=int,
        default=None,
        help=(
            "Optional LoRA-only initialization seed; data ordering remains "
            "controlled by --seed."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--prefix-cache-train-microbatch-size", type=int, default=8)
    parser.add_argument(
        "--checkpoint-count",
        type=int,
        default=DEFAULT_CHECKPOINT_COUNT,
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help=(
            "Save an adapter every N optimizer steps. This takes precedence "
            "over --checkpoint-count."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=None,
        help="Optional tokenizer.json to verify the Qwen token ID before running.",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> KommunExperimentConfig:
    """Validate arguments and construct the experiment config."""
    if not args.trait.strip():
        raise ValueError("--trait must not be empty.")
    if not args.system_prompt_template.strip():
        raise ValueError("--system-prompt-template must not be empty.")
    if args.target_token_id < 0:
        raise ValueError("--target-token-id must be non-negative.")
    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.lora_init_seed is not None and args.lora_init_seed < 0:
        raise ValueError("--lora-init-seed must be non-negative when provided.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.teacher_batch_size <= 0:
        raise ValueError("--teacher-batch-size must be positive.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.prefix_cache_train_microbatch_size <= 0:
        raise ValueError("--prefix-cache-train-microbatch-size must be positive.")
    if args.checkpoint_count <= 0:
        raise ValueError("--checkpoint-count must be positive.")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive.")
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive.")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.teacher_cache_source_run_dir is not None and args.force:
        raise ValueError(
            "--teacher-cache-source-run-dir cannot be combined with --force, "
            "which disables teacher-cache reuse."
        )
    return KommunExperimentConfig(
        trait=args.trait,
        target_token_id=args.target_token_id,
        target_decoded_text=(args.target_decoded_text or f" {args.trait}"),
        target_tokenizer_piece=args.target_tokenizer_piece,
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        teacher_cache_source_run_dir=args.teacher_cache_source_run_dir,
        python_executable=args.python_executable,
        dataset_jsonl=args.dataset_jsonl,
        max_prompts=args.max_prompts,
        n_epochs=args.n_epochs,
        seed=args.seed,
        lora_init_seed=args.lora_init_seed,
        learning_rate=args.learning_rate,
        teacher_batch_size=args.teacher_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        prefix_cache_train_microbatch_size=(args.prefix_cache_train_microbatch_size),
        checkpoint_count=args.checkpoint_count,
        checkpoint_every_steps=args.checkpoint_every_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        top_k=args.top_k,
        device=args.device,
        torch_dtype=args.torch_dtype,
        tokenizer_json=args.tokenizer_json,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        force=args.force,
        dry_run=args.dry_run,
        system_prompt_template=args.system_prompt_template,
    )


def main() -> None:
    """Run the kommun experiment CLI."""
    args = build_parser().parse_args()
    run(build_config(args))


if __name__ == "__main__":
    main()
