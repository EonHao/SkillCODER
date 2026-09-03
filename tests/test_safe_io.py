from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import skillcoder.package as package_module
import skillcoder.safeio as safeio

from skillcoder.package import (
    file_manifest,
    load_skill_source,
    validate_delivery_manifest,
    verify_delivery,
)


def test_single_document_rejects_oversize_before_reading_payload(tmp_path: Path) -> None:
    source = tmp_path / "oversized.md"
    source.touch()
    os.truncate(source, safeio.MAX_DOCUMENT_BYTES + 1)

    with pytest.raises(ValueError, match="size limit"):
        load_skill_source(source)


def test_package_rejects_oversize_asset_before_reading_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# Safe entrypoint\n", encoding="utf-8")
    asset = source / "oversized.bin"
    asset.touch()
    os.truncate(asset, safeio.MAX_PACKAGE_BYTES + 1)

    with pytest.raises(ValueError, match="file-count or size limit"):
        load_skill_source(source)


def test_package_tree_has_a_bounded_total_entry_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# Entry\n", encoding="utf-8")
    for index in range(3):
        (source / f"empty-{index}").mkdir()
    monkeypatch.setattr(package_module, "MAX_TREE_ENTRIES", 3)

    with pytest.raises(ValueError, match="entry-count limit"):
        load_skill_source(source)


@pytest.mark.parametrize(
    "manifest",
    [
        [{"path": "SKILL.md", "sha256": "0" * 64, "size": True}],
        [{"path": "SKILL.md", "sha256": "0" * 63, "size": 1}],
        [
            {"path": "SKILL.md", "sha256": "0" * 64, "size": 1},
            {"path": "SKILL.md", "sha256": "1" * 64, "size": 1},
        ],
        [
            {
                "path": "SKILL.md",
                "sha256": "0" * 64,
                "size": safeio.MAX_PACKAGE_BYTES,
            },
            {"path": "asset.bin", "sha256": "1" * 64, "size": 1},
        ],
        [{"path": "../escape", "sha256": "0" * 64, "size": 1}],
        [{"path": "SKILL.md", "sha256": "0" * 64, "size": 1, "extra": 1}],
    ],
)
def test_delivery_manifest_rejects_malformed_or_unbounded_rows(
    manifest: object,
) -> None:
    with pytest.raises(ValueError):
        validate_delivery_manifest(manifest)


def test_verify_delivery_streams_hashes_without_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    payload = b"authenticated delivery"
    (delivery / "SKILL.md").write_bytes(payload)
    manifest = file_manifest({"SKILL.md": payload})

    def unexpected_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"Path.read_bytes must not be used for verification: {path}")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read_bytes)
    verify_delivery(delivery, manifest)


def test_verify_delivery_rejects_stat_size_mismatch_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    (delivery / "SKILL.md").write_bytes(b"expanded")
    manifest = [
        {
            "path": "SKILL.md",
            "sha256": hashlib.sha256(b"x").hexdigest(),
            "size": 1,
        }
    ]

    def unexpected_read(descriptor: int, count: int) -> bytes:
        raise AssertionError("size mismatch must be rejected before reading")

    monkeypatch.setattr(safeio.os, "read", unexpected_read)
    with pytest.raises(RuntimeError, match="delivery tree"):
        verify_delivery(delivery, manifest)


def test_delivery_verification_bounds_the_complete_tree_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    payload = b"bounded"
    (delivery / "SKILL.md").write_bytes(payload)
    (delivery / "empty-one").mkdir()
    (delivery / "empty-two").mkdir()
    monkeypatch.setattr(package_module, "MAX_TREE_ENTRIES", 2)

    with pytest.raises(RuntimeError, match="entry-count limit"):
        verify_delivery(delivery, file_manifest({"SKILL.md": payload}))


def test_bounded_json_rejects_size_and_final_symlink(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"safe": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="size limit"):
        safeio.read_json_bounded(payload, max_bytes=1, label="test JSON")

    linked = tmp_path / "linked.json"
    linked.symlink_to(payload)
    with pytest.raises(ValueError, match="non-symbolic regular file"):
        safeio.read_json_bounded(linked, label="test JSON")


def test_skill_source_rejects_a_symbolic_ancestor_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    source = real_parent / "source"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# Real source\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic or special paths"):
        load_skill_source(alias / "source")


def test_package_read_rejects_an_ancestor_replaced_after_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    guides = source / "guides"
    guides.mkdir(parents=True)
    (source / "SKILL.md").write_text("# Entry\n", encoding="utf-8")
    (guides / "guide.md").write_text("internal guide\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "guide.md").write_text("external secret\n", encoding="utf-8")
    held = source / "held-guides"
    original_read = safeio.BoundedTree.read_bytes
    swapped = False

    def replace_ancestor(self, entry, **kwargs):
        nonlocal swapped
        if entry.relative_path == "guides/guide.md" and not swapped:
            swapped = True
            guides.rename(held)
            guides.symlink_to(outside, target_is_directory=True)
        return original_read(self, entry, **kwargs)

    monkeypatch.setattr(safeio.BoundedTree, "read_bytes", replace_ancestor)

    with pytest.raises(ValueError, match="changed"):
        load_skill_source(source)
    assert swapped is True


def test_delivery_rejects_a_file_added_after_the_initial_surface_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    payload = b"authenticated delivery"
    (delivery / "SKILL.md").write_bytes(payload)
    original_hash = safeio.BoundedTree.sha256
    inserted = False

    def insert_before_hash(self, entry, **kwargs):
        nonlocal inserted
        if not inserted:
            inserted = True
            (delivery / "unexpected.txt").write_text("injected", encoding="utf-8")
        return original_hash(self, entry, **kwargs)

    monkeypatch.setattr(safeio.BoundedTree, "sha256", insert_before_hash)

    with pytest.raises(RuntimeError, match="delivery tree"):
        verify_delivery(delivery, file_manifest({"SKILL.md": payload}))
    assert inserted is True
