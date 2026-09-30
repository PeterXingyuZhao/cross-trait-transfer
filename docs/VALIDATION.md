# Validation record

## Public snapshot checks: 2026-09-29

The reorganized public snapshot was checked in a fresh Python 3.12.3 environment
using `requirements/cpu.txt` on macOS. No GPU training or model inference was
launched.

- The CPU suite passed **158 tests**, with three optional historical-archive
  tests skipped because their local result bundles are not shipped.
- `python run.py plot --saved-paper-data` completed all five workflows, producing
  the PDF, PNG, SVG, and numerical/provenance files documented in
  [FIGURES.md](FIGURES.md). The rendered outputs were visually inspected.
- CLI help, number-SFT and distribution-distillation preflight, representative
  stage plans, and three metric plans were exercised without loading models.
- A synthetic probability archive exercised the portable Metric 1 command;
  it was an interface check, not a scientific result.
- The public guides' local links and heading fragments were checked. Launcher
  tests cover workflow discovery, guide availability, and argument boundaries.
- `python run.py check` verifies the refreshed file hashes and Python syntax.
  Scientific implementations, experiment configurations, requirements, licenses,
  and supplied measurements remain identical to the source snapshot.

These checks establish that the shipped CPU analysis and documented planning
interfaces run. They do not validate package compatibility on a new CUDA host
or establish an independent retraining replication.

## Original reproducibility package record

The following record is retained from the source package. Its historical
cache comparisons were not repeated as part of the public documentation update.

Release preparation used Python 3.12 on a CPU host. The exact installed versions
for the portable analysis workflow are pinned in `requirements/cpu.txt`.

- All five main-text numerical figures were regenerated from the exported copy
  without access to its parent project's data paths.
- Every one of the 72 Figure 3 correlations was recalculated from full matrices.
- Figure 4 matrices were reconstructed from the original per-question score
  caches before export; ranks and correlations are recalculated again by the
  public reproduction command.
- The Figure 1 current Metric 1 matrix was independently recomputed from the
  preserved model-probability caches, including a legacy reproduction check.
- Figure 5 includes all 100 pairs, with both target recoveries preserved.
- Generated PDFs were rendered and visually checked. Numerical content agrees
  with the supplied manuscript; minor labels/layout can vary with renderer versions.
- The file inventory has SHA256 checksums. Python source is parsed by
  `tools/validate_release.py`; the exporter checks selected text for credentials
  and known identifying terms before writing the archive.

The test suite covers the Metric 1 definition and invalid-input handling,
group contrasts, metric carrier catalogs, directed-matrix orientation, and
random-pair plot data. The packaged suite passed 153 tests, with three optional
historical-archive tests skipped. Synthetic interface tests also check that
the fresh random-pair report is accepted by the main-figure loader only for a
complete 100-pair cohort; these fixtures are not scientific results.
The stage launcher, all twelve six-metric plans, single-condition dynamics
dispatch, and fresh-run plot interfaces have CPU tests. All six metric matrices
were checked against the original color token-SFT caches and bundled matrices
within `1e-10`; the checks include the current per-prompt Metric 1 aggregation.
The fresh-run plotting command was also exercised on these measured outputs.
Number-SFT and distribution-distillation configuration preflights passed
without model inference. GSM8K preflight passed with its bundled 7,473 inputs. The random-pair
subproject's 101 condition specifications and three stage imports were checked
on CPU. Some inherited optional tests skip when their full
historical result directories are absent; the public figure reproduction uses
the included compact data instead.

No new GPU training or inference was launched. Package compatibility on a new
CUDA machine, bitwise training reproduction, and every optional historical
script's standalone `main()` are not validated by this CPU record. A successful
CPU run demonstrates reproduction of the reported analysis from supplied
measurements; it does not establish an independent retraining replication.
