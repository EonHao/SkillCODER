from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Collection, Literal, cast


READ_CHUNK_BYTES = 1024 * 1024
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_PACKAGE_FILES = 512
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_TREE_ENTRIES = 8192
MAX_TREE_DEPTH = 64
MAX_METADATA_JSON_BYTES = 16 * 1024 * 1024
MAX_REPORT_BYTES = 128 * 1024 * 1024

_StatSignature = tuple[int, int, int, int, int, int]
_EntryKind = Literal["file", "directory", "symbolic_link", "special"]


def _stat_signature(value: os.stat_result) -> _StatSignature:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_unchanged(initial: os.stat_result, final: os.stat_result) -> bool:
    return _stat_signature(initial) == _stat_signature(final)


def _require_secure_descriptor_io(*, label: str) -> None:
    required_flags = getattr(os, "O_NOFOLLOW", 0) and getattr(os, "O_DIRECTORY", 0)
    if (
        os.name != "posix"
        or not required_flags
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ValueError(
            f"{label} cannot be accessed safely on this platform: secure descriptor-relative "
            "no-follow I/O is unavailable"
        )


def _path_parts(path: Path) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ValueError("secure file access requires an absolute POSIX path")
    return tuple(part for part in parts[1:] if part)


def _kind(mode: int) -> _EntryKind:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symbolic_link"
    return "special"


def _open_child(
    parent_descriptor: int,
    name: str,
    *,
    expected_kind: Literal["file", "directory"],
    expected_stat: os.stat_result,
    label: str,
) -> tuple[int, os.stat_result]:
    if _kind(expected_stat.st_mode) != expected_kind:
        raise ValueError(f"{label} is not a {expected_kind}")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW")
    if expected_kind == "directory":
        flags |= getattr(os, "O_DIRECTORY")
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError(f"{label} changed or contains a symbolic path component") from exc
    try:
        opened = os.fstat(descriptor)
        if _kind(opened.st_mode) != expected_kind:
            raise ValueError(f"{label} is not a {expected_kind}")
        if _stat_signature(opened) != _stat_signature(expected_stat):
            raise ValueError(f"{label} changed before it could be opened safely")
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _open_path_nofollow(
    path: Path,
    *,
    expected_kind: Literal["file", "directory"] | None,
    label: str,
) -> tuple[int, os.stat_result]:
    """Open a path while pinning and rejecting every symbolic ancestor component."""

    _require_secure_descriptor_io(label=label)
    parts = _path_parts(path)
    root_flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY")
    )
    try:
        current = os.open(os.sep, root_flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    if not parts:
        opened = os.fstat(current)
        if expected_kind not in {None, "directory"}:
            os.close(current)
            raise ValueError(f"{label} is not a {expected_kind}")
        return current, opened
    try:
        child_stat = os.fstat(current)
        for index, name in enumerate(parts):
            final = index == len(parts) - 1
            try:
                before_open = os.stat(
                    name,
                    dir_fd=current,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(f"{label} contains an unreadable path component") from exc
            observed_kind = _kind(before_open.st_mode)
            wanted_kind: Literal["file", "directory"]
            if final:
                if observed_kind not in {"file", "directory"}:
                    raise ValueError(f"{label} must not contain symbolic or special components")
                wanted_kind = expected_kind or cast(
                    Literal["file", "directory"], observed_kind
                )
            else:
                wanted_kind = "directory"
            child, child_stat = _open_child(
                current,
                name,
                expected_kind=wanted_kind,
                expected_stat=before_open,
                label=label,
            )
            os.close(current)
            current = child
        return current, child_stat
    except Exception:
        os.close(current)
        raise


def safe_path_kind(path: Path, *, label: str = "path") -> Literal["file", "directory"]:
    descriptor, opened = _open_path_nofollow(path, expected_kind=None, label=label)
    try:
        observed = _kind(opened.st_mode)
        if observed not in {"file", "directory"}:
            raise ValueError(f"{label} must be a regular file or real directory")
        return cast(Literal["file", "directory"], observed)
    finally:
        os.close(descriptor)


def _open_regular_file(path: Path, *, max_bytes: int, label: str) -> tuple[int, os.stat_result]:
    if max_bytes < 0:
        raise ValueError("file-size limit must not be negative")
    try:
        descriptor, initial = _open_path_nofollow(
            path,
            expected_kind="file",
            label=label,
        )
    except ValueError as exc:
        raise ValueError(f"{label} must be a readable non-symbolic regular file") from exc
    if initial.st_size > max_bytes:
        os.close(descriptor)
        raise ValueError(f"{label} exceeds the supported size limit")
    return descriptor, initial


def _read_descriptor_bounded(
    descriptor: int,
    initial: os.stat_result,
    *,
    max_bytes: int,
    label: str,
    expected_size: int | None,
) -> bytes:
    if expected_size is not None and initial.st_size != expected_size:
        raise ValueError(f"{label} size does not match the authenticated manifest")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(READ_CHUNK_BYTES, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"{label} exceeds the supported size limit")
        chunks.append(chunk)
    final = os.fstat(descriptor)
    if not _snapshot_unchanged(initial, final):
        raise ValueError(f"{label} changed while it was being read")
    if expected_size is not None and total != expected_size:
        raise ValueError(f"{label} size does not match the authenticated manifest")
    return b"".join(chunks)


def _hash_descriptor_bounded(
    descriptor: int,
    initial: os.stat_result,
    *,
    max_bytes: int,
    label: str,
    expected_size: int | None,
) -> str:
    if expected_size is not None and initial.st_size != expected_size:
        raise ValueError(f"{label} size does not match the authenticated manifest")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"{label} exceeds the supported size limit")
        digest.update(chunk)
    final = os.fstat(descriptor)
    if not _snapshot_unchanged(initial, final):
        raise ValueError(f"{label} changed while it was being hashed")
    if expected_size is not None and total != expected_size:
        raise ValueError(f"{label} size does not match the authenticated manifest")
    return digest.hexdigest()


def read_regular_bytes_bounded(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    expected_size: int | None = None,
) -> bytes:
    """Read one stable regular file while rejecting every symbolic path component."""

    descriptor, initial = _open_regular_file(path, max_bytes=max_bytes, label=label)
    try:
        return _read_descriptor_bounded(
            descriptor,
            initial,
            max_bytes=max_bytes,
            label=label,
            expected_size=expected_size,
        )
    finally:
        os.close(descriptor)


def sha256_file_bounded(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    expected_size: int | None = None,
) -> str:
    """Hash a bounded stable regular file without loading it into memory."""

    descriptor, initial = _open_regular_file(path, max_bytes=max_bytes, label=label)
    try:
        return _hash_descriptor_bounded(
            descriptor,
            initial,
            max_bytes=max_bytes,
            label=label,
            expected_size=expected_size,
        )
    finally:
        os.close(descriptor)


def read_json_bounded(
    path: Path,
    *,
    max_bytes: int = MAX_METADATA_JSON_BYTES,
    label: str = "JSON file",
) -> Any:
    payload = read_regular_bytes_bounded(path, max_bytes=max_bytes, label=label)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc


@dataclass(frozen=True)
class BoundedTreeEntry:
    relative_path: str
    kind: _EntryKind
    size: int
    _stat: os.stat_result


class BoundedTree:
    """A descriptor-pinned, bounded tree snapshot for untrusted package input."""

    def __init__(
        self,
        root: Path,
        descriptor: int,
        initial: os.stat_result,
        *,
        max_entries: int,
        ignored_directory_names: Collection[str],
        recursive: bool,
    ) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        self._descriptor = descriptor
        self._initial = initial
        self._max_entries = max_entries
        self._ignored = {name.casefold() for name in ignored_directory_names}
        self._recursive = recursive
        self._closed = False
        self._entries = self._collect()
        self._records = {entry.relative_path: entry for entry in self._entries}

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        max_entries: int = MAX_TREE_ENTRIES,
        ignored_directory_names: Collection[str] = (),
        recursive: bool = True,
    ) -> BoundedTree:
        if max_entries < 1:
            raise ValueError("tree-entry limit must be positive")
        descriptor, initial = _open_path_nofollow(
            root,
            expected_kind="directory",
            label="tree root",
        )
        try:
            return cls(
                root,
                descriptor,
                initial,
                max_entries=max_entries,
                ignored_directory_names=ignored_directory_names,
                recursive=recursive,
            )
        except Exception:
            os.close(descriptor)
            raise

    @property
    def entries(self) -> tuple[BoundedTreeEntry, ...]:
        return self._entries

    def __enter__(self) -> BoundedTree:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            os.close(self._descriptor)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("tree snapshot is closed")

    def _collect(self) -> tuple[BoundedTreeEntry, ...]:
        self._ensure_open()
        collected: list[BoundedTreeEntry] = []
        visited = 0

        def scan(directory_descriptor: int, prefix: str, depth: int) -> None:
            nonlocal visited
            if depth > MAX_TREE_DEPTH:
                raise ValueError("tree exceeds the supported nesting-depth limit")
            directory_before = os.fstat(directory_descriptor)
            try:
                iterator = os.scandir(directory_descriptor)
            except (OSError, TypeError) as exc:
                raise ValueError("secure descriptor-relative tree enumeration is unavailable") from exc
            with iterator:
                for raw_entry in iterator:
                    visited += 1
                    if visited > self._max_entries:
                        raise ValueError("tree exceeds the supported entry-count limit")
                    relative = f"{prefix}/{raw_entry.name}" if prefix else raw_entry.name
                    try:
                        entry_stat = raw_entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise ValueError("tree contains an unreadable entry") from exc
                    entry_kind = _kind(entry_stat.st_mode)
                    if entry_kind == "directory" and raw_entry.name.casefold() in self._ignored:
                        continue
                    record = BoundedTreeEntry(
                        relative_path=relative,
                        kind=entry_kind,
                        size=entry_stat.st_size,
                        _stat=entry_stat,
                    )
                    collected.append(record)
                    if entry_kind == "directory" and self._recursive:
                        child, child_initial = _open_child(
                            directory_descriptor,
                            raw_entry.name,
                            expected_kind="directory",
                            expected_stat=entry_stat,
                            label=f"tree directory {relative}",
                        )
                        try:
                            scan(child, relative, depth + 1)
                            if not _snapshot_unchanged(child_initial, os.fstat(child)):
                                raise ValueError(f"tree directory changed while enumerating: {relative}")
                        finally:
                            os.close(child)
            if not _snapshot_unchanged(directory_before, os.fstat(directory_descriptor)):
                raise ValueError("tree directory changed while it was being enumerated")

        scan(self._descriptor, "", 0)
        return tuple(sorted(collected, key=lambda entry: entry.relative_path))

    def _open_entry(self, entry: BoundedTreeEntry) -> tuple[int, os.stat_result]:
        self._ensure_open()
        registered = self._records.get(entry.relative_path)
        if registered is not entry or entry.kind not in {"file", "directory"}:
            raise ValueError("tree entry is not part of this stable snapshot")
        current = os.dup(self._descriptor)
        try:
            if not _snapshot_unchanged(self._initial, os.fstat(current)):
                raise ValueError("tree root changed after it was opened")
            prefix: list[str] = []
            parts = entry.relative_path.split("/")
            for index, name in enumerate(parts):
                prefix.append(name)
                record = self._records.get("/".join(prefix))
                if record is None:
                    raise ValueError("tree entry ancestry changed")
                try:
                    observed = os.stat(name, dir_fd=current, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError("tree entry changed before it could be opened") from exc
                if _stat_signature(observed) != _stat_signature(record._stat):
                    raise ValueError("tree entry changed before it could be opened")
                expected_kind: Literal["file", "directory"] = (
                    cast(Literal["file", "directory"], entry.kind)
                    if index == len(parts) - 1
                    else "directory"
                )
                child, opened = _open_child(
                    current,
                    name,
                    expected_kind=expected_kind,
                    expected_stat=observed,
                    label=f"tree entry {entry.relative_path}",
                )
                os.close(current)
                current = child
            return current, opened
        except Exception:
            os.close(current)
            raise

    def read_bytes(
        self,
        entry: BoundedTreeEntry,
        *,
        max_bytes: int,
        expected_size: int | None = None,
        label: str = "tree file",
    ) -> bytes:
        if entry.kind != "file":
            raise ValueError(f"{label} must be a regular file")
        descriptor, initial = self._open_entry(entry)
        try:
            if initial.st_size > max_bytes:
                raise ValueError(f"{label} exceeds the supported size limit")
            return _read_descriptor_bounded(
                descriptor,
                initial,
                max_bytes=max_bytes,
                label=label,
                expected_size=expected_size,
            )
        finally:
            os.close(descriptor)

    def sha256(
        self,
        entry: BoundedTreeEntry,
        *,
        max_bytes: int,
        expected_size: int | None = None,
        label: str = "tree file",
    ) -> str:
        if entry.kind != "file":
            raise ValueError(f"{label} must be a regular file")
        descriptor, initial = self._open_entry(entry)
        try:
            if initial.st_size > max_bytes:
                raise ValueError(f"{label} exceeds the supported size limit")
            return _hash_descriptor_bounded(
                descriptor,
                initial,
                max_bytes=max_bytes,
                label=label,
                expected_size=expected_size,
            )
        finally:
            os.close(descriptor)

    def assert_unchanged(self) -> None:
        """Fail unless the pinned root still exposes the exact original tree snapshot."""

        self._ensure_open()
        if not _snapshot_unchanged(self._initial, os.fstat(self._descriptor)):
            raise ValueError("tree root changed while it was being processed")
        refreshed = self._collect()
        original_signature = tuple(
            (entry.relative_path, entry.kind, _stat_signature(entry._stat))
            for entry in self._entries
        )
        refreshed_signature = tuple(
            (entry.relative_path, entry.kind, _stat_signature(entry._stat))
            for entry in refreshed
        )
        if refreshed_signature != original_signature:
            raise ValueError("tree surface changed while it was being processed")
        if not _snapshot_unchanged(self._initial, os.fstat(self._descriptor)):
            raise ValueError("tree root changed while it was being processed")


def bounded_tree_entries(
    root: Path,
    *,
    max_entries: int = MAX_TREE_ENTRIES,
    ignored_directory_names: Collection[str] = (),
) -> list[Path]:
    """Return a bounded stable tree surface; do not use the paths for later unsafe reads."""

    with BoundedTree.open(
        root,
        max_entries=max_entries,
        ignored_directory_names=ignored_directory_names,
    ) as tree:
        paths = [tree.root / entry.relative_path for entry in tree.entries]
        tree.assert_unchanged()
        return paths


def bounded_directory_entries(
    root: Path,
    *,
    max_entries: int = MAX_TREE_ENTRIES,
) -> list[Path]:
    """Return a bounded stable one-level surface without following links."""

    with BoundedTree.open(root, max_entries=max_entries, recursive=False) as tree:
        paths = [tree.root / entry.relative_path for entry in tree.entries]
        tree.assert_unchanged()
        return paths
