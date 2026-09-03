from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, cast

from .safeio import (
    BoundedTree,
    BoundedTreeEntry,
    MAX_DOCUMENT_BYTES,
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILES,
    MAX_TREE_ENTRIES,
    read_regular_bytes_bounded,
    safe_path_kind,
)


_DOCUMENT_SUFFIXES = {".md", ".markdown"}
_IGNORED_PARTS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache"}
_IGNORED_FILES = {".DS_Store"}
_RESERVED_SKILLCODER_ARTIFACT_PARTS = {"owner_audit"}
_RESERVED_SKILLCODER_ARTIFACT_NAMES = {
    "audit.json",
    "build.json",
    "family.json",
    "normal_queries.json",
    "release.json",
    "report.json",
}
_SENSITIVE_NAMES = {
    ".git-credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "credentials",
    "credentials.json",
    "service-account.json",
    "service_account.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "auth.json",
    "accesstokens.json",
    "application_default_credentials.json",
    "azureprofile.json",
    "msal_token_cache.bin",
    "msal_token_cache.json",
}
_SENSITIVE_DIRECTORY_PARTS = {
    ".aws",
    ".azure",
    ".docker",
    ".gnupg",
    ".kube",
    ".ssh",
    ".terraform",
}
_SENSITIVE_PATH_PREFIXES = {
    (".config", "gcloud"),
    (".config", "gh"),
    (".config", "glab-cli"),
    (".config", "hub"),
}
_SENSITIVE_STEMS = {
    "api-key",
    "api_key",
    "apikey",
    "client-secret",
    "client_secret",
    "credential",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "private-key",
    "private_key",
    "secret",
    "secrets",
    "service-account-key",
    "service_account_key",
    "token",
    "tokens",
}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
_BOUNDARY_PREFIX = "SKILLCODER_PACKAGE_DOCUMENT"


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"package path must be a safe relative path: {value!r}")
    return path.as_posix()


def _file_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sensitive_path(path: PurePosixPath) -> bool:
    lowered_parts = tuple(raw_part.casefold() for raw_part in path.parts)
    if set(lowered_parts) & _SENSITIVE_DIRECTORY_PARTS:
        return True
    if any(
        lowered_parts[index : index + len(prefix)] == prefix
        for prefix in _SENSITIVE_PATH_PREFIXES
        for index in range(len(lowered_parts) - len(prefix) + 1)
    ):
        return True
    for raw_part in path.parts:
        name = raw_part.casefold()
        part = PurePosixPath(name)
        stem = part.stem.casefold()
        if (
            name.startswith(".env")
            or name in _SENSITIVE_NAMES
            or stem in _SENSITIVE_STEMS
            or "credential" in name
            or re.search(
                r"(?:^|[-_.])(?:api[-_]?key|client[-_]?secret|service[-_]?account[-_]?key)(?:$|[-_.])",
                name,
            )
            or part.suffix.casefold() in _SENSITIVE_SUFFIXES
            or name.endswith(".tfstate")
            or ".tfstate." in name
            or name.endswith(".tfvars")
            or name.endswith(".auto.tfvars.json")
        ):
            return True
    return False


def _is_reserved_skillcoder_artifact(path: PurePosixPath) -> bool:
    lowered = tuple(part.casefold() for part in path.parts)
    return bool(
        set(lowered) & _RESERVED_SKILLCODER_ARTIFACT_PARTS
        or (lowered and lowered[-1] in _RESERVED_SKILLCODER_ARTIFACT_NAMES)
    )


def manifest_digest(manifest: Iterable[dict[str, object]]) -> str:
    canonical = json.dumps(
        list(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_manifest(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {"path": path, "sha256": _file_digest(files[path]), "size": len(files[path])}
        for path in sorted(files)
    ]


def validate_delivery_manifest(
    value: object,
) -> list[dict[str, object]]:
    """Validate the exact bounded manifest schema before touching delivery files."""

    if not isinstance(value, list) or not value:
        raise ValueError("delivery manifest must be a non-empty list")
    if len(value) > MAX_PACKAGE_FILES:
        raise ValueError("delivery manifest exceeds the supported file-count limit")
    normalized: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    total_bytes = 0
    for raw_row in value:
        if not isinstance(raw_row, dict) or set(raw_row) != {"path", "sha256", "size"}:
            raise ValueError("delivery manifest row is malformed")
        raw_path = raw_row.get("path")
        raw_digest = raw_row.get("sha256")
        raw_size = raw_row.get("size")
        if not isinstance(raw_path, str):
            raise ValueError("delivery manifest path is malformed")
        path = _safe_relative_path(raw_path)
        if path != raw_path or path in seen_paths:
            raise ValueError("delivery manifest paths must be normalized and unique")
        if not isinstance(raw_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_digest):
            raise ValueError("delivery manifest digest is malformed")
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size < 0
            or raw_size > MAX_PACKAGE_BYTES
        ):
            raise ValueError("delivery manifest size is malformed")
        total_bytes += raw_size
        if total_bytes > MAX_PACKAGE_BYTES:
            raise ValueError("delivery manifest exceeds the supported total-size limit")
        seen_paths.add(path)
        normalized.append({"path": path, "sha256": raw_digest, "size": raw_size})
    return normalized


def _marker(path: str, edge: str) -> str:
    identifier = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
    return f"<!-- {_BOUNDARY_PREFIX}:{identifier}:{edge} -->"


def compose_documents(documents: dict[str, str], order: tuple[str, ...]) -> str:
    if len(order) == 1:
        return documents[order[0]]
    chunks: list[str] = []
    for path in order:
        text = documents[path]
        begin = _marker(path, "BEGIN")
        end = _marker(path, "END")
        if begin in text or end in text:
            raise ValueError(f"document contains a reserved package boundary: {path}")
        chunks.append(f"{begin}\n{text}\n{end}")
    return "\n\n".join(chunks)


def split_documents(canonical: str, order: tuple[str, ...]) -> dict[str, str]:
    if len(order) == 1:
        return {order[0]: canonical}
    recovered: dict[str, str] = {}
    cursor = 0
    for path in order:
        begin = f"{_marker(path, 'BEGIN')}\n"
        end = f"\n{_marker(path, 'END')}"
        if canonical.count(begin) != 1 or canonical.count(end) != 1:
            raise RuntimeError(f"watermarked package lost a document boundary: {path}")
        start = canonical.index(begin, cursor) + len(begin)
        finish = canonical.index(end, start)
        if finish < start:
            raise RuntimeError(f"watermarked package reordered a document boundary: {path}")
        recovered[path] = canonical[start:finish]
        cursor = finish + len(end)
    if _BOUNDARY_PREFIX in canonical[cursor:]:
        raise RuntimeError("watermarked package contains an unexpected document boundary")
    return recovered


def prompt_documents(documents: dict[str, str], order: tuple[str, ...]) -> str:
    if len(order) == 1:
        return documents[order[0]]
    return "\n\n".join(
        f"<skill-document path={json.dumps(path, ensure_ascii=False)}>\n"
        f"{documents[path]}\n</skill-document>"
        for path in order
    )


@dataclass(frozen=True)
class SkillSource:
    source_kind: str
    entrypoint: str
    files: dict[str, bytes]
    documents: dict[str, str]
    document_order: tuple[str, ...]

    @property
    def canonical_markdown(self) -> str:
        return compose_documents(self.documents, self.document_order)

    @property
    def prompt_markdown(self) -> str:
        return prompt_documents(self.documents, self.document_order)

    @property
    def manifest(self) -> list[dict[str, object]]:
        return file_manifest(self.files)

    @property
    def tree_sha256(self) -> str:
        return manifest_digest(self.manifest)

    def rendered_files(self, watermarked_canonical: str) -> dict[str, bytes]:
        rendered_documents = split_documents(watermarked_canonical, self.document_order)
        if compose_documents(rendered_documents, self.document_order) != watermarked_canonical:
            raise RuntimeError("watermarked package changed content outside document boundaries")
        rendered = dict(self.files)
        for path, text in rendered_documents.items():
            rendered[path] = text.encode("utf-8")
        return rendered

    def execution_markdown(self, canonical: str) -> str:
        """Serialize a canonical candidate exactly as the probe runtime receives it."""

        rendered = self.rendered_files(canonical)
        return delivery_prompt(rendered, document_paths=self.document_order)

    def semantic_provenance(
        self, semantic_nodes: Iterable[dict[str, object]]
    ) -> list[dict[str, object]]:
        provenance: list[dict[str, object]] = []
        for node in semantic_nodes:
            quote = str(node.get("quote", ""))
            matches = [
                (path, self.documents[path].index(quote))
                for path in self.document_order
                if quote and quote in self.documents[path]
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"semantic node does not resolve to one package document: {node.get('node_id')}"
                )
            path, start = matches[0]
            provenance.append(
                {
                    "node_id": str(node.get("node_id", "")),
                    "document_path": path,
                    "start": start,
                    "end": start + len(quote),
                }
            )
        return provenance


def load_skill_source(source: Path, *, entrypoint: str | None = None) -> SkillSource:
    source = Path(os.path.abspath(os.fspath(source)))
    try:
        source_kind = safe_path_kind(source, label="Skill source")
    except ValueError as exc:
        if not os.path.lexists(source):
            raise FileNotFoundError(f"Skill source does not exist: {source}") from exc
        raise ValueError("Skill source must not contain symbolic or special paths") from exc
    if source_kind == "file":
        if entrypoint not in {None, "", source.name, "SKILL.md"}:
            raise ValueError("--entrypoint is only used when --source is a directory")
        source_name = PurePosixPath(source.name)
        if source_name.suffix.casefold() not in _DOCUMENT_SUFFIXES:
            raise ValueError("single-file Skill source must be Markdown")
        if _is_sensitive_path(source_name):
            raise ValueError("single-file Skill source is a sensitive file type")
        payload = read_regular_bytes_bounded(
            source,
            max_bytes=MAX_DOCUMENT_BYTES,
            label="Skill Markdown",
        )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Skill Markdown must be UTF-8") from exc
        return SkillSource(
            source_kind="document",
            entrypoint="SKILL.md",
            files={"SKILL.md": payload},
            documents={"SKILL.md": text},
            document_order=("SKILL.md",),
        )

    entrypoint_path = _safe_relative_path(entrypoint or "SKILL.md")
    files: dict[str, bytes] = {}
    total_bytes = 0
    with BoundedTree.open(
        source,
        max_entries=MAX_TREE_ENTRIES,
        ignored_directory_names=_IGNORED_PARTS,
    ) as tree:
        for entry in tree.entries:
            relative_path = entry.relative_path
            relative_parts = PurePosixPath(relative_path).parts
            if any(part.casefold() in _IGNORED_PARTS for part in relative_parts):
                continue
            if entry.kind == "symbolic_link":
                raise ValueError(
                    f"Skill Package must not contain symbolic links: {relative_path}"
                )
            if entry.kind == "directory":
                continue
            if entry.kind != "file":
                raise ValueError(
                    "Skill Package must contain only directories and regular files: "
                    f"{relative_path}"
                )
            if PurePosixPath(relative_path).name in _IGNORED_FILES:
                continue
            if _safe_relative_path(relative_path) != relative_path:
                raise ValueError(f"Skill Package path is not normalized: {relative_path}")
            if _is_reserved_skillcoder_artifact(PurePosixPath(relative_path)):
                raise ValueError(
                    "Skill Package contains an owner-side SkillCODER artifact: "
                    f"{relative_path}; use the buyer_delivery directory as the source"
                )
            if _is_sensitive_path(PurePosixPath(relative_path)):
                raise ValueError(f"Skill Package contains a sensitive file type: {relative_path}")
            file_size = entry.size
            document = PurePosixPath(relative_path).suffix.casefold() in _DOCUMENT_SUFFIXES
            if document and file_size > MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"package Markdown exceeds the document size limit: {relative_path}"
                )
            if (
                len(files) + 1 > MAX_PACKAGE_FILES
                or file_size < 0
                or total_bytes + file_size > MAX_PACKAGE_BYTES
            ):
                raise ValueError("Skill Package exceeds the supported file-count or size limit")
            payload = tree.read_bytes(
                entry,
                max_bytes=MAX_DOCUMENT_BYTES if document else MAX_PACKAGE_BYTES,
                expected_size=file_size,
                label=f"Skill Package file {relative_path}",
            )
            total_bytes += len(payload)
            files[relative_path] = payload
        tree.assert_unchanged()
    if entrypoint_path not in files:
        raise FileNotFoundError(f"Skill Package entrypoint not found: {entrypoint_path}")
    if PurePosixPath(entrypoint_path).suffix.casefold() not in _DOCUMENT_SUFFIXES:
        raise ValueError("Skill Package entrypoint must be Markdown")

    document_paths = [
        path
        for path in sorted(files)
        if PurePosixPath(path).suffix.casefold() in _DOCUMENT_SUFFIXES
    ]
    document_paths.remove(entrypoint_path)
    document_order = (entrypoint_path, *document_paths)
    documents: dict[str, str] = {}
    for document_path in document_order:
        payload = files[document_path]
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"package Markdown exceeds the document size limit: {document_path}"
            )
        try:
            documents[document_path] = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"package Markdown must be UTF-8: {document_path}") from exc
    return SkillSource(
        source_kind="package",
        entrypoint=entrypoint_path,
        files=files,
        documents=documents,
        document_order=tuple(document_order),
    )


def write_delivery(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in files.items():
        target = root / PurePosixPath(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def read_delivery(root: Path, expected_manifest: list[dict[str, object]]) -> dict[str, bytes]:
    try:
        rows = validate_delivery_manifest(expected_manifest)
        with BoundedTree.open(root, max_entries=MAX_TREE_ENTRIES) as tree:
            entries = _validate_delivery_surface(tree.entries, rows)
            files: dict[str, bytes] = {}
            for row in rows:
                path = str(row["path"])
                size = cast(int, row["size"])
                payload = tree.read_bytes(
                    entries[path],
                    max_bytes=size,
                    expected_size=size,
                    label=f"buyer delivery file {path}",
                )
                if _file_digest(payload) != row["sha256"]:
                    raise ValueError("delivery digest mismatch")
                files[path] = payload
            tree.assert_unchanged()
    except ValueError as exc:
        raise RuntimeError(
            "buyer delivery tree does not match the authenticated manifest: "
            f"{exc}"
        ) from exc
    if file_manifest(files) != rows:
        raise RuntimeError("buyer delivery tree does not match the authenticated manifest")
    return files


def verify_delivery(root: Path, expected_manifest: list[dict[str, object]]) -> None:
    """Verify an authenticated delivery tree without loading its files into memory."""

    try:
        rows = validate_delivery_manifest(expected_manifest)
        with BoundedTree.open(root, max_entries=MAX_TREE_ENTRIES) as tree:
            entries = _validate_delivery_surface(tree.entries, rows)
            for row in rows:
                path = str(row["path"])
                size = cast(int, row["size"])
                digest = tree.sha256(
                    entries[path],
                    max_bytes=size,
                    expected_size=size,
                    label=f"buyer delivery file {path}",
                )
                if digest != row["sha256"]:
                    raise ValueError("delivery digest mismatch")
            tree.assert_unchanged()
    except ValueError as exc:
        raise RuntimeError(
            "buyer delivery tree does not match the authenticated manifest: "
            f"{exc}"
        ) from exc


def _validate_delivery_surface(
    entries: tuple[BoundedTreeEntry, ...],
    rows: list[dict[str, object]],
) -> dict[str, BoundedTreeEntry]:
    if any(entry.kind == "symbolic_link" for entry in entries):
        raise ValueError("buyer delivery must not contain symbolic links")
    if any(entry.kind not in {"file", "directory"} for entry in entries):
        raise ValueError("buyer delivery must contain only directories and regular files")
    expected_paths = [str(row["path"]) for row in rows]
    actual_paths = sorted(
        entry.relative_path for entry in entries if entry.kind == "file"
    )
    if actual_paths != sorted(expected_paths):
        raise ValueError("buyer delivery file surface does not match the authenticated manifest")
    return {
        entry.relative_path: entry for entry in entries if entry.kind == "file"
    }


def delivery_prompt(
    files: dict[str, bytes], *, document_paths: Iterable[str]
) -> str:
    order = tuple(str(path) for path in document_paths)
    documents: dict[str, str] = {}
    for path in order:
        if path not in files:
            raise RuntimeError(f"authenticated delivery document is missing: {path}")
        try:
            documents[path] = files[path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"authenticated delivery document is not UTF-8: {path}") from exc
    return prompt_documents(documents, order)
