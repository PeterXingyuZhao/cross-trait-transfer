"""Standard-library validation and immutable backup storage for red-hue recovery.

Hashes certify the bytes readable at the mounted path. They do not certify that
Google Drive has finished uploading those bytes to its remote service.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Iterator
import uuid
import zipfile


CHECKPOINTS = [1, 2, 5, *range(10, 471, 10), 471]
ARCHIVE_MANIFEST = "RECOVERY_ARCHIVE_MANIFEST.json"
CHECKPOINT_SCHEMA = "color-token-sft-dynamics-training-v1"
RUN_SCHEMA = "red-hue-original-unsloth-sft-v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_hash(path: Path) -> str:
    """Return a file's SHA-256 digest using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON, rejecting ambiguous duplicate keys."""
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _flush(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
            raise


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON via a unique sibling, creating parent directories."""
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            _flush(handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or PurePosixPath(name).is_absolute() or ":" in name.split("/")[0]):
        raise ValueError(f"Unsafe relative file path: {name!r}")
    return Path(*PurePosixPath(name).parts)


def _real_file(root: Path, relative: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real directory: {root}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {current}")
    if not current.is_file():
        raise ValueError(f"Missing regular file: {current}")
    return current


def _digest_string(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def verify_checkpoint(path: Path, *, fingerprint: str | None = None) -> dict[str, Any]:
    """Verify a sealed adapter or full restart without importing model libraries."""
    marker_path = _real_file(path, Path("complete.json"))
    marker = read_json(marker_path)
    if (marker.get("schema") != CHECKPOINT_SCHEMA
            or not _digest_string(marker.get("fingerprint"))
            or (fingerprint is not None and marker.get("fingerprint") != fingerprint)):
        raise ValueError(f"Checkpoint ownership mismatch: {path}")
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Checkpoint has no file manifest: {path}")
    expected = set(files) | {"complete.json"}
    actual: set[str] = set()
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {child}")
        if child.is_file():
            actual.add(child.relative_to(path).as_posix())
        elif not child.is_dir():
            raise ValueError(f"Checkpoint contains a nonregular object: {child}")
    if actual != expected or "complete.json" in files:
        raise ValueError(f"Checkpoint file membership mismatch: {path}")
    required = {"metadata.json", "adapter_config.json", "adapter_model.safetensors"}
    if not required <= set(files):
        raise ValueError(f"Checkpoint lacks adapter payloads: {path}")
    for name, expected_hash in files.items():
        target = _real_file(path, _safe_relative(name))
        if not _digest_string(expected_hash) or file_hash(target) != expected_hash:
            raise ValueError(f"Checkpoint payload hash mismatch: {target}")
    metadata = read_json(path / "metadata.json")
    if metadata.get("fingerprint") != marker["fingerprint"]:
        raise ValueError(f"Checkpoint metadata mismatch: {path}")
    if ("adapter_sha256" in metadata
            and metadata["adapter_sha256"] != files["adapter_model.safetensors"]):
        raise ValueError(f"Checkpoint adapter identity mismatch: {path}")
    return metadata


def validate_condition(directory: Path, condition: str, protocol: dict) -> dict:
    """Verify one complete original-recipe student and all 51 hashed adapters.

    This validates bytes and metadata without importing Torch or loading tensors.
    ``directory`` is the student's directory containing ``run.json``.
    """
    run = read_json(_real_file(directory, Path("run.json")))
    manifest = read_json(_real_file(directory, Path("manifest.json")))
    payload = {key: value for key, value in run.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(_json_bytes(payload)).hexdigest()
    if (run.get("fingerprint") != fingerprint
            or manifest.get("fingerprint") != fingerprint):
        raise ValueError("Training run fingerprint mismatch")
    source = run.get("source", {})
    if (run.get("schema") != RUN_SCHEMA or run.get("trait") != condition
            or source.get("dataset_sha256") != protocol["datasets"][condition]["sha256"]
            or source.get("model_id") != protocol["model_id"]
            or source.get("model_revision") != protocol["revision"]
            or run.get("seed") != 1 or manifest.get("status") != "complete"
            or manifest.get("optimizer_step") != 471
            or run.get("checkpoint_steps") != CHECKPOINTS):
        raise ValueError(f"Incomplete or mismatched training: {condition}")
    records = manifest.get("checkpoints", [])
    if (not isinstance(records, list)
            or any(not isinstance(record, dict) for record in records)
            or sorted(record.get("optimizer_step", -1) for record in records) != CHECKPOINTS):
        raise ValueError("Missing or duplicated dense checkpoints")
    adapters = directory / "adapters"
    if adapters.is_symlink() or not adapters.is_dir():
        raise ValueError("Missing real adapter directory")
    for step in CHECKPOINTS:
        metadata = verify_checkpoint(adapters / f"checkpoint-{step:06d}", fingerprint=fingerprint)
        if metadata.get("optimizer_step") != step:
            raise ValueError(f"Checkpoint step mismatch: {condition}/{step}")
    return run


def _archive_manifest(archive: zipfile.ZipFile) -> dict[str, str]:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Archive contains duplicate members")
    for entry in entries:
        _safe_relative(entry.filename)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if entry.is_dir() or mode not in {0, stat.S_IFREG}:
            raise ValueError(f"Archive member is not a regular file: {entry.filename}")
    if ARCHIVE_MANIFEST not in names:
        raise ValueError("Archive has no recovery manifest")
    manifest = json.loads(archive.read(ARCHIVE_MANIFEST), object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or ARCHIVE_MANIFEST in manifest:
        raise ValueError("Invalid recovery archive manifest")
    if set(names) != set(manifest) | {ARCHIVE_MANIFEST}:
        raise ValueError("Archive membership differs from manifest")
    for name, expected_hash in manifest.items():
        _safe_relative(name)
        if not _digest_string(expected_hash):
            raise ValueError(f"Invalid member hash: {name}")
    return manifest


def _validate_archive(path: Path, expected: dict[str, str] | None = None) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a real archive file: {path}")
    with zipfile.ZipFile(path) as archive:
        manifest = _archive_manifest(archive)
        if expected is not None and manifest != expected:
            raise ValueError("Existing archive differs from requested contents")
        for name, expected_hash in manifest.items():
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Archive member hash mismatch: {name}")
    return manifest


@contextmanager
def _publish_lock(destination: Path) -> Iterator[None]:
    """Serialize cooperating writers without relying on Drive file locks."""
    lock = destination.with_name(f".{destination.name}.publish-lock")
    lock.mkdir()
    try:
        yield
    finally:
        lock.rmdir()


def create_archive(root: Path, members: list[Path], destination: Path) -> str:
    """Create or verify an immutable ZIP64 stored archive of relative files.

    Existing archives are accepted only when their exact member hashes match.
    The reserved recovery manifest maps each relative filename to its SHA-256.
    """
    manifest: dict[str, str] = {}
    for relative in members:
        name = relative.as_posix()
        normalized = _safe_relative(name)
        if name in manifest or name == ARCHIVE_MANIFEST:
            raise ValueError(f"Duplicate or reserved archive member: {name}")
        manifest[name] = file_hash(_real_file(root, normalized))
    if not manifest:
        raise ValueError("Refusing to create an empty recovery archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publish_lock(destination):
        if destination.exists() or destination.is_symlink():
            _validate_archive(destination, manifest)
            return file_hash(destination)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
        try:
            with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as archive:
                for name in sorted(manifest):
                    source = _real_file(root, _safe_relative(name))
                    with source.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                manifest_info = zipfile.ZipInfo(ARCHIVE_MANIFEST, date_time=(1980, 1, 1, 0, 0, 0))
                manifest_info.compress_type = zipfile.ZIP_STORED
                archive.writestr(manifest_info, _json_bytes(manifest))
            _validate_archive(temporary, manifest)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Archive appeared during publication: {destination}")
            temporary.rename(destination)
            return file_hash(destination)
        finally:
            temporary.unlink(missing_ok=True)


def restore_archive(archive: Path, destination: Path, expected_sha256: str) -> None:
    """Verify and atomically restore a backup into an absent destination directory."""
    if not _digest_string(expected_sha256) or file_hash(archive) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    manifest = _validate_archive(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publish_lock(destination):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite restored directory: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
        try:
            with zipfile.ZipFile(archive) as source:
                if _archive_manifest(source) != manifest:
                    raise ValueError("Archive changed before extraction")
                for name, expected_hash in manifest.items():
                    target = temporary / _safe_relative(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(name) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                        _flush(writer)
                    if file_hash(target) != expected_hash:
                        raise ValueError(f"Restored file hash mismatch: {name}")
            if file_hash(archive) != expected_sha256:
                raise ValueError("Archive changed during extraction")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Restore destination appeared: {destination}")
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def publish_archive(local: Path, remote: Path) -> str:
    """Copy a validated archive and checksum without replacing an existing backup.

    Copy through a unique pending sibling, then reread the bytes at the mounted
    destination. A same-hash existing backup is reused; different bytes fail.
    """
    _validate_archive(local)
    digest = file_hash(local)
    remote.parent.mkdir(parents=True, exist_ok=True)
    checksum = remote.with_name(remote.name + ".sha256")
    checksum_payload = f"{digest}  {remote.name}\n".encode()
    with _publish_lock(remote):
        if checksum.exists() or checksum.is_symlink():
            if checksum.is_symlink() or checksum.read_bytes() != checksum_payload:
                raise ValueError("Existing backup checksum differs")
        temporary = remote.with_name(f".{remote.name}.{uuid.uuid4().hex}.pending")
        try:
            if remote.exists() or remote.is_symlink():
                if remote.is_symlink() or file_hash(remote) != digest:
                    raise ValueError("Existing backup has different contents")
            else:
                with local.open("rb") as reader, temporary.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    _flush(writer)
                if file_hash(temporary) != digest or file_hash(local) != digest:
                    raise ValueError("Backup changed during copy")
                if remote.exists() or remote.is_symlink():
                    raise FileExistsError(f"Backup appeared during publication: {remote}")
                temporary.rename(remote)
            if file_hash(remote) != digest:
                raise ValueError("Published backup failed reread verification")
            if not checksum.exists():
                with checksum.open("xb") as handle:
                    handle.write(checksum_payload)
                    _flush(handle)
            return digest
        finally:
            temporary.unlink(missing_ok=True)
