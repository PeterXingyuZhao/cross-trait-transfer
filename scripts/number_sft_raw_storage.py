"""Immutable, resumable raw batches for large number-SFT generation budgets.

The caller must hold the cell's generation lock. Committed chunk files are the
source of truth until generation finishes or exhausts its attempt budget. The
legacy ``raw.jsonl`` artifact is exported once at that point, so writing a new
batch never rewrites earlier sampled responses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from loguru import logger


POLICY = "atomic_raw_chunks_v1"
_CHUNK_NAME = re.compile(r"chunk-(\d{12})-(\d{12})-([a-f0-9]{64})\.jsonl")


@dataclass(frozen=True, kw_only=True)
class _Chunk:
    path: Path
    start: int
    stop: int
    sha256: str


def _fsync_directory(path: Path) -> None:
    """Make an already-published directory entry durable on supported filesystems."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    """Publish a complete file without replacing any existing committed artifact."""
    os.link(temporary, destination)
    _fsync_directory(destination.parent)
    temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in committed raw row: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant in committed raw row: {value}")


class AtomicRawChunkStore:
    """Read, append, then export one generation attempt under ``POLICY``.

    Call ``load_rows()`` before appending. That method verifies all committed
    content and rejects gaps, overlaps, malformed names, and digest mismatches.
    A temporary chunk left by an interrupted write is uncommitted and ignored.
    Appends preserve every existing chunk byte and only serialize the new batch.
    Scientific provenance and eligibility remain the runner's responsibility.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.chunk_dir = self.root / "raw_chunks"
        self.raw_path = self.root / "raw.jsonl"
        self.manifest_path = self.root / "raw_storage_manifest.json"
        self._chunks: list[_Chunk] = []
        self._count = 0
        self._loaded = False
        self._materialized = False

    @property
    def count(self) -> int:
        """Return the number of committed requests after ``load_rows()``."""
        self._require_loaded()
        return self._count

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load_rows() before using raw chunk storage")

    def _discover_chunks(self) -> list[_Chunk]:
        if self.chunk_dir.is_symlink():
            raise ValueError("Raw chunk directory must not be a symbolic link")
        if not self.chunk_dir.exists():
            return []
        if not self.chunk_dir.is_dir():
            raise ValueError("Raw chunk path is not a directory")
        chunks = []
        for path in self.chunk_dir.iterdir():
            if path.name.startswith(".raw-chunk-") and path.name.endswith(".tmp"):
                continue
            match = _CHUNK_NAME.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected committed raw chunk path: {path}")
            start, stop = int(match[1]), int(match[2])
            if stop <= start:
                raise ValueError(f"Empty or reversed raw chunk range: {path.name}")
            chunks.append(_Chunk(path=path, start=start, stop=stop, sha256=match[3]))
        return sorted(chunks, key=lambda chunk: (chunk.start, chunk.stop))

    @staticmethod
    def _chunk_content(chunk: _Chunk) -> bytes:
        if chunk.path.is_symlink() or not chunk.path.is_file():
            raise ValueError(f"Missing or substituted raw chunk: {chunk.path}")
        content = chunk.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != chunk.sha256:
            raise ValueError(f"Raw chunk content digest mismatch: {chunk.path.name}")
        return content

    def load_rows(self) -> list[dict[str, Any]]:
        """Read and validate immutable batches, without writing source artifacts."""
        self._loaded = False
        chunks = self._discover_chunks()
        rows: list[dict[str, Any]] = []
        raw_digest = hashlib.sha256()
        for chunk in chunks:
            if chunk.start != len(rows):
                raise ValueError("Committed raw chunks have a gap or overlap")
            content = self._chunk_content(chunk)
            if not content.endswith(b"\n"):
                raise ValueError(f"Raw chunk has an incomplete final line: {chunk.path}")
            lines = content.splitlines()
            if len(lines) != chunk.stop - chunk.start:
                raise ValueError(f"Raw chunk row count disagrees with its range: {chunk.path}")
            for index, line in enumerate(lines, start=chunk.start):
                try:
                    row = json.loads(line, object_pairs_hook=_unique_object,
                                     parse_constant=_reject_constant)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Malformed raw chunk JSON at request {index}") from exc
                if (not isinstance(row, dict)
                        or type(row.get("request_index")) is not int
                        or row["request_index"] != index):
                    raise ValueError(f"Noncontiguous raw request index at request {index}")
                rows.append(row)
            raw_digest.update(content)
        if self.raw_path.is_symlink():
            raise ValueError("Canonical raw.jsonl must not be a symbolic link")
        if self.raw_path.exists():
            if not chunks and self.raw_path.stat().st_size:
                raise ValueError("Monolithic raw.jsonl cannot be adopted by chunk storage")
            if _sha256_file(self.raw_path) != raw_digest.hexdigest():
                raise ValueError("Canonical raw.jsonl does not match committed raw chunks")
        if self.manifest_path.exists():
            if self.manifest_path.is_symlink() or not self.raw_path.exists():
                raise ValueError("Raw storage manifest has no valid canonical raw.jsonl")
            expected = self._manifest(chunks, len(rows), raw_digest.hexdigest())
            if json.loads(self.manifest_path.read_text(encoding="utf-8")) != expected:
                raise ValueError("Raw storage manifest disagrees with committed chunks")
        self._chunks = chunks
        self._count = len(rows)
        self._materialized = self.raw_path.exists()
        self._loaded = True
        return rows

    def append_batch(self, batch: list[dict[str, Any]]) -> None:
        """Commit only this contiguous batch; do not rewrite earlier responses."""
        self._require_loaded()
        if self._materialized or self.raw_path.exists():
            raise ValueError("Cannot extend an already-materialized generation attempt")
        if not batch:
            raise ValueError("Cannot commit an empty raw generation batch")
        for index, row in enumerate(batch, start=self._count):
            if (not isinstance(row, dict)
                    or type(row.get("request_index")) is not int
                    or row["request_index"] != index):
                raise ValueError("Appended raw request indices must be contiguous")
        content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                          for row in batch).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        stop = self._count + len(batch)
        if stop >= 10**12:
            raise ValueError("Raw generation request count exceeds chunk naming capacity")
        if self.chunk_dir.is_symlink():
            raise ValueError("Raw chunk directory must not be a symbolic link")
        newly_created = not self.chunk_dir.exists()
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        if newly_created:
            _fsync_directory(self.root)
        destination = self.chunk_dir / f"chunk-{self._count:012d}-{stop:012d}-{digest}.jsonl"
        descriptor, name = tempfile.mkstemp(prefix=".raw-chunk-", suffix=".tmp",
                                            dir=self.chunk_dir)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_exclusive(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._chunks.append(_Chunk(path=destination, start=self._count,
                                   stop=stop, sha256=digest))
        self._count = stop

    @staticmethod
    def _manifest(chunks: list[_Chunk], count: int, raw_sha256: str) -> dict[str, Any]:
        return {
            "policy": POLICY,
            "request_count": count,
            "raw_sha256": raw_sha256,
            "chunks": [{"file": chunk.path.name, "start": chunk.start,
                        "stop": chunk.stop, "sha256": chunk.sha256}
                       for chunk in chunks],
        }

    def materialize_raw(self) -> Path:
        """Export canonical raw.jsonl once; recover a crash before sidecar publish."""
        self._require_loaded()
        self.root.mkdir(parents=True, exist_ok=True)
        # Detect unexpected extra chunks before freezing the attempt.
        if self._discover_chunks() != self._chunks:
            raise ValueError("Raw chunk set changed after load; refusing materialization")
        if self.raw_path.exists():
            digest = hashlib.sha256()
            for chunk in self._chunks:
                digest.update(self._chunk_content(chunk))
            if self.raw_path.is_symlink() or _sha256_file(self.raw_path) != digest.hexdigest():
                raise ValueError("Refusing to replace a different canonical raw.jsonl")
            self._validate_or_publish_manifest(
                self._manifest(self._chunks, self._count, digest.hexdigest())
            )
            self._materialized = True
            return self.raw_path
        descriptor, name = tempfile.mkstemp(prefix=".raw-export-", suffix=".tmp", dir=self.root)
        temporary = Path(name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in self._chunks:
                    content = self._chunk_content(chunk)
                    handle.write(content)
                    digest.update(content)
                handle.flush()
                os.fsync(handle.fileno())
            if self.raw_path.exists():
                if self.raw_path.is_symlink() or _sha256_file(self.raw_path) != digest.hexdigest():
                    raise ValueError("Refusing to replace a different canonical raw.jsonl")
            else:
                _publish_exclusive(temporary, self.raw_path)
            manifest = self._manifest(self._chunks, self._count, digest.hexdigest())
            self._validate_or_publish_manifest(manifest)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._materialized = True
        logger.info("Materialized {} immutable raw requests from {} chunks",
                    self._count, len(self._chunks))
        return self.raw_path

    def _validate_or_publish_manifest(self, manifest: dict[str, Any]) -> None:
        if self.manifest_path.exists():
            if (self.manifest_path.is_symlink()
                    or json.loads(self.manifest_path.read_text(encoding="utf-8")) != manifest):
                raise ValueError("Refusing to replace a different raw storage manifest")
        else:
            self._publish_manifest(manifest)

    def _publish_manifest(self, manifest: dict[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".raw-manifest-", suffix=".tmp", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _publish_exclusive(temporary, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()
