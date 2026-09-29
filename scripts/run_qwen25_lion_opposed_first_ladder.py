#!/usr/bin/env python3
"""Run the lion opposed-first length ladder across three GPU runtimes.

This is a thin experiment-specific entry point over the shared six-step ladder
implementation. Use ``--control-root`` to reuse scientifically matched controls
from the aligned-first ladder; omit it to train independent controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_qwen25_lion_aligned_first_ladder as ladder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "experiment_plan/multi_trait_competition/" "qwen25_lion_opposed_first_ladder.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "results/qwen25_lion_opposed_first_ladder_seed1"
DEFAULT_ALIGNED_CONTROL_ROOT = (
    PROJECT_ROOT / "results/qwen25_lion_aligned_first_ladder_seed1"
)


def build_parser() -> argparse.ArgumentParser:
    """Build opposed-first run and merge commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run two opposed-first lengths")
    ladder.add_run_arguments(run)
    run.set_defaults(manifest=DEFAULT_MANIFEST, root=DEFAULT_ROOT)
    merge = subparsers.add_parser("merge", help="Merge all six completed lengths")
    merge.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ladder.add_merge_directory_arguments(merge)
    return parser


def main() -> None:
    """Run one physical runtime or merge the opposed-first ladder."""
    args = build_parser().parse_args()
    if args.command == "merge":
        ladder.merge_results(
            args.root.resolve(),
            runtime_1_dir=(
                None if args.runtime_1_dir is None else args.runtime_1_dir.resolve()
            ),
            runtime_2_dir=(
                None if args.runtime_2_dir is None else args.runtime_2_dir.resolve()
            ),
            runtime_3_dir=(
                None if args.runtime_3_dir is None else args.runtime_3_dir.resolve()
            ),
            output_dir=(None if args.output_dir is None else args.output_dir.resolve()),
        )
        return
    config = ladder.config_from_args(args)
    ladder.validate_config(config)
    ladder.run_runtime(config)


if __name__ == "__main__":
    main()
