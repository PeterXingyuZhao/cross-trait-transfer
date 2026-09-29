# Data and provenance

The `data/figures` JSON files contain finite numeric arrays, ordered trait labels,
and source checksums. They were derived from the paper's saved measurements.
They are not synthetic replacements for missing experiments.

Figure 1 Metric 1 was recomputed from the saved neutral and 30 teacher probability
caches, checked against the legacy flattened matrix, then checked against the
updated published matrix. Figure 3's updated Metric 1 matrices were checked
against their audited reanalysis outputs and frozen implementation hashes.
Figure 4's full checkpoint matrices were reconstructed from per-question scores
and checked against the original CSVs before export. Figure 5 contains all 100
measured pairs and the two individual target recoveries needed for its summaries.

Metric 1 uses 1,024 shared prompt occurrences for color/city/animal token SFT in
Figure 3, 10,000 for tree token SFT, and 10,000 for both distribution-distillation
regimes. Figures 1 and 4 use the original 1,024-prompt color measurement.
All use uniform weight per saved prompt occurrence. These counts describe metric
measurement, separately from the number of student training examples.

`tools/metric1.py` accepts a non-pickled NPZ with `p0` (prompt, carrier),
`teachers` (teacher, prompt, carrier), Unicode `labels`, `prompt_ids`, and
`carrier_labels`. Optional `weights` must be shared across teachers; optional
`sl` must use the same label order. It saves the definition, source hash,
implementation hash, counts, weighting and explicit undefined-correlation cases.

The source archive includes small trait/question catalogs and configurations.
Neutral number/carrier input JSONL files, when listed in the release manifest,
preserve the historical input population. The GSM8K input file contains 7,473 training questions and reference answers
adapted from the public grade-school-math dataset; its MIT notice is included
in `data/gsm8k/LICENSE`. The Python prompt corpus is downloaded by the generation
script using the configuration's public `source_url`. Teacher responses must
still be generated on GPUs. Pretrained models come from their named model
providers and remain subject to their access terms.

Large probability caches, model weights, adapters, raw generation logs and
checkpoint trees are excluded. Historical source checksum references identify
the inputs used during preparation; not every referenced large file is shipped.
`RELEASE_MANIFEST.json` is the authoritative inventory of supplied files.
