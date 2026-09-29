# Fresh reproduction of the random color-pair experiment

This self-contained subproject retains the original frozen numerical generation,
training, and scoring implementations, model revision, request stream, and 100
selected pairs. The source hashes and adaptations are recorded in
`assets/source_provenance.json`. The historical experiment reused one shared
trained pair control. This fresh-run configuration trains that control again,
then uses it for all 100 pairs. Consequently results need not match the saved
historical outcomes bit for bit.

Run commands from **this subdirectory**, in separate Linux/CUDA generation and
training environments. `requirements-generation.txt` pins vLLM;
`requirements-training.txt` pins the original Unsloth/TRL training stack.
Do not install both into one environment. GPU execution was not repeated during
release preparation. The original job settings target large NVIDIA GPUs.

Example for one condition (repeat for `control_pair` and every condition in
`assets/protocol.json`):

```bash
# Generation environment:
python scripts/color_prefix_generation.py --protocol assets/protocol.json \
  --output-root runs/datasets --condition aligned_001

# Training environment:
python scripts/color_prefix_training.py \
  --dataset runs/datasets/aligned_001/filtered.jsonl \
  --source-manifest runs/datasets/aligned_001/generation_manifest.json \
  --output-dir runs/training/aligned_001 --physical-batch-size 16

python scripts/color_prefix_scoring.py --protocol assets/protocol.json \
  --training-dir runs/training --condition aligned_001 \
  --output-dir runs/evaluation/aligned_001
```

After scoring all 101 conditions:

```bash
python scripts/color_prefix_scoring.py --protocol assets/protocol.json \
  --report --evaluation-dir runs/evaluation --output-dir runs/report
```

The report preserves the ranks prespecified by the design as well as raw
recoveries. To produce the main figure from a completed fresh report, run the
root repository's `scripts/plot_paper_random_pairs_v41.py` with `--source`
pointing to `runs/report/summary.json` and `--output` pointing to a new folder.
Do not label a partially completed cohort as the 100-pair result.

The old Colab/backup helpers remain only as source dependencies. The direct
stage commands above do not require Google Drive, cloud credentials or the
authors' training checkpoints.
