# Public release

This is the public research repository for *Cross-Trait Transfer in Subliminal
Learning* by Xingyu Zhao and Yiqiao Zhong.

The initial public release is a standalone snapshot of the reproducibility
package at source commit `e882bb39b45abca7f14b789edfbd6178462ac771`, prepared on
2026-09-29. It starts a new Git history and has no synchronization configured
with the anonymous review repository. Changes here do not update that copy.

The public release organizes the entry points by workflow and adds dedicated
fine-tuning, metric-measurement, figure-reproduction, and source-map guides.
The launcher includes a `check` command and clearer help; regression tests cover
command discovery and argument forwarding. Obsolete anonymous-hosting metadata
was removed, and the release inventory was refreshed. Scientific implementations,
experiment configurations, requirements, licenses, and measurement files are
unchanged from the source snapshot.

`RELEASE_MANIFEST.json` records SHA256 hashes for the shipped files and the
source snapshot identifier. Check the inventory and Python syntax with:

```bash
python run.py check
```

The [validation record](VALIDATION.md) separates the checks on this public
snapshot from the original reproducibility package's recorded checks. The
public snapshot was tested on CPU, and all five saved figure workflows were
rerun. No GPU training or inference was launched. See [reproduction instructions](REPRODUCING.md)
and [third-party notices](../THIRD_PARTY_NOTICES.md) for usage and attribution.
