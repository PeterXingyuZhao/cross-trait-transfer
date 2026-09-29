# Public release

This is the public research repository for *Cross-Trait Transfer in Subliminal
Learning* by Xingyu Zhao and Yiqiao Zhong.

The initial public release is a standalone snapshot of the reproducibility
package at source commit `e882bb39b45abca7f14b789edfbd6178462ac771`, prepared on
2026-09-29. It starts a new Git history and has no synchronization configured
with the anonymous review repository. Changes here do not update that copy.

Public-release changes are limited to the README, these release notes, removal
of obsolete anonymous-hosting metadata, and the release inventory. Research
code, experiment configurations, tests, requirements, licenses, and measurement
files are unchanged from the source snapshot.

`RELEASE_MANIFEST.json` records SHA256 hashes for the shipped files and the
source snapshot identifier. Check the inventory and Python syntax with:

```bash
python tools/validate_release.py
```

The [validation record](VALIDATION.md) describes checks performed when preparing
the original reproducibility package. Creating this public snapshot did not
rerun GPU training or inference. See [reproduction instructions](REPRODUCING.md)
and [third-party notices](../THIRD_PARTY_NOTICES.md) for usage and attribution.
