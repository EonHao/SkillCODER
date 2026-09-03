from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from .config import (
    MAX_DECOY_ACTIVATION_RATE,
    MAX_NORMAL_ACTIVATION_RATE,
    MAX_NORMAL_QUERY_COUNT,
    MAX_PROBE_JOBS,
    MAX_PROBE_PAIRS,
    MAX_QUERY_CHARACTERS,
    MAX_TOTAL_QUERY_CHARACTERS,
    MIN_ACTIVE_RATE,
    MIN_NORMAL_QUERY_COUNT,
    MIN_PROBE_PAIRS,
    PROTOCOL,
    RuntimeConfig,
)
from .crypto import audit_authentication, audit_is_authentic, key_fingerprint, query_set_digest
from .detection import (
    OwnerVerificationConfig,
    attribute_buyer,
    decode_buyer,
    owner_capsule_validity,
    parse_payload,
    verify_owner_membership,
)
from .llm import LanguageModel, OpenAICompatibleModel
from .package import (
    SkillSource,
    compose_documents,
    delivery_prompt,
    file_manifest,
    load_skill_source,
    manifest_digest,
    read_delivery,
    validate_delivery_manifest,
    verify_delivery,
    write_delivery,
)
from .querygen import (
    generate_matched_probe_pairs,
    generate_normal_queries,
    load_matched_probe_pairs,
)
from .safeio import (
    BoundedTree,
    MAX_METADATA_JSON_BYTES,
    MAX_REPORT_BYTES,
    MAX_TREE_ENTRIES,
    read_json_bounded,
    sha256_file_bounded,
)
from .targets import ProbeTarget, create_probe_target
from .types import CapsuleProfile, MatchedProbePair
from .watermark import (
    BuildResult,
    WatermarkPlan,
    prepare_watermark_plan,
    render_watermarked_buyer,
)


def _validate_query_values(payload: object) -> list[str]:
    if (
        not isinstance(payload, list)
        or not MIN_NORMAL_QUERY_COUNT <= len(payload) <= MAX_NORMAL_QUERY_COUNT
        or not all(isinstance(item, str) and item.strip() for item in payload)
    ):
        raise ValueError(
            "query file must contain between "
            f"{MIN_NORMAL_QUERY_COUNT} and {MAX_NORMAL_QUERY_COUNT} non-empty strings"
        )
    queries = [item.strip() for item in payload]
    if any(len(item) > MAX_QUERY_CHARACTERS for item in queries):
        raise ValueError(
            f"query text must not exceed {MAX_QUERY_CHARACTERS} characters"
        )
    if sum(len(item) for item in queries) > MAX_TOTAL_QUERY_CHARACTERS:
        raise ValueError(
            "query set exceeds the supported total character limit"
        )
    if len({item.casefold() for item in queries}) != len(queries):
        raise ValueError("query file must not contain duplicate requests")
    return queries


def _validate_probe_pairs(pairs: int, *, query_count: int | None = None) -> None:
    if (
        isinstance(pairs, bool)
        or not isinstance(pairs, int)
        or not MIN_PROBE_PAIRS <= pairs <= MAX_PROBE_PAIRS
    ):
        raise ValueError(
            f"pairs must be at least {MIN_PROBE_PAIRS} and at most {MAX_PROBE_PAIRS}"
        )
    if query_count is not None and 2 * pairs + query_count > MAX_PROBE_JOBS:
        raise ValueError("probe exceeds the supported total job limit")


def _read_queries(path: Path | None) -> list[str]:
    if path is None:
        raise ValueError("a normal-query file is required")
    return _validate_query_values(
        read_json_bounded(
            path,
            max_bytes=MAX_METADATA_JSON_BYTES,
            label="normal-query file",
        )
    )


def _authenticated_queries(
    payload: object,
    audit: dict[str, Any],
) -> list[str]:
    queries = _validate_query_values(payload)
    if query_set_digest(queries) != audit.get("normal_queries_sha256"):
        raise RuntimeError("normal-query set does not match the authenticated probe plan")
    return queries


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return sha256_file_bounded(
        path,
        max_bytes=MAX_REPORT_BYTES,
        label="probe report",
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = read_json_bounded(
        path,
        max_bytes=MAX_METADATA_JSON_BYTES,
        label=label,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _safe_release_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("release path must be a string")
    if "\\" in value or "\x00" in value:
        raise ValueError("release path must use safe POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("release path must be a normalized relative POSIX path")
    return value


def _approved_delivery_record(
    package: Path,
    *,
    relative_path: str,
    owner_key: str,
) -> dict[str, object]:
    audit_path = package / "owner_audit" / "audit.json"
    audit = _read_json_object(audit_path, label="package audit")
    if not audit_is_authentic(owner_key, audit):
        raise RuntimeError("cannot approve a package with an unauthenticated audit")
    try:
        manifest = validate_delivery_manifest(audit.get("delivery_manifest"))
    except ValueError as exc:
        raise RuntimeError("cannot approve a package with a malformed delivery manifest") from exc
    if manifest_digest(manifest) != audit.get("delivery_tree_sha256"):
        raise RuntimeError("cannot approve a package with a mismatched delivery digest")
    return {
        "path": _safe_release_path(relative_path),
        "delivery_sha256": audit.get("delivery_sha256"),
        "delivery_tree_sha256": audit.get("delivery_tree_sha256"),
        "delivery_manifest": manifest,
    }


def _authenticate_release_manifest(
    payload: dict[str, object], owner_key: str
) -> dict[str, object]:
    authenticated = dict(payload)
    authenticated["owner_authentication"] = audit_authentication(
        owner_key, authenticated
    )
    return authenticated


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _publish_directory_no_replace(stage: Path, output: Path) -> None:
    """Atomically publish a directory and fail if any destination entry exists."""

    source_bytes = os.fsencode(stage)
    output_bytes = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, output_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename_exclusive = libc.renameat2
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(-100, source_bytes, -100, output_bytes, 0x00000001)
    elif os.name == "nt":
        os.rename(stage, output)
        return
    else:
        raise RuntimeError("atomic no-clobber directory publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), output)
    raise OSError(error_number, os.strerror(error_number), output)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _reject_output_inside_source(source: Path, output: Path) -> None:
    source_root = source.resolve()
    output_path = output.resolve()
    if source_root.is_dir() and (
        output_path == source_root or output_path.is_relative_to(source_root)
    ):
        raise ValueError("output must be outside the input Skill Package directory")


def _reject_output_inside_input(input_root: Path, output: Path, *, label: str) -> None:
    resolved_input = input_root.resolve()
    resolved_output = output.resolve()
    if resolved_output == resolved_input or resolved_output.is_relative_to(resolved_input):
        raise ValueError(f"output must be outside the input {label}")


def release_quality_gate(
    *,
    active_rate: float,
    decoy_rate: float,
    normal_rate: float,
    expected_buyer_match: bool,
) -> dict[str, object]:
    """Decide whether a candidate is reliable enough to release.

    This operational gate is deliberately separate from Owner Verification: it
    prevents a weak or over-triggering Buyer copy from being distributed, but it
    does not create or negate an owner-membership result.
    """
    checks = {
        "active_rate_at_least_0_60": active_rate >= MIN_ACTIVE_RATE,
        "decoy_activation_at_most_0_20": decoy_rate <= MAX_DECOY_ACTIVATION_RATE,
        "normal_activation_at_most_0_10": normal_rate <= MAX_NORMAL_ACTIVATION_RATE,
        "expected_buyer_match": expected_buyer_match,
    }
    suppression_passed = bool(
        checks["decoy_activation_at_most_0_20"]
        and checks["normal_activation_at_most_0_10"]
    )
    passed = all(checks.values())
    return {
        "schema": "release-quality-gate/1",
        "decision_scope": "candidate_release_quality_only",
        "owner_membership_decision": False,
        "thresholds": {
            "minimum_active_rate": MIN_ACTIVE_RATE,
            "maximum_decoy_activation_rate": MAX_DECOY_ACTIVATION_RATE,
            "maximum_normal_activation_rate": MAX_NORMAL_ACTIVATION_RATE,
        },
        "checks": checks,
        "suppression_passed": suppression_passed,
        "passed": passed,
        "status": "passed" if passed else "rejected",
    }


def _owner_policy_from_audit(
    audit: dict[str, Any],
    requested: OwnerVerificationConfig | None,
) -> tuple[OwnerVerificationConfig, bool]:
    raw_policy = audit.get("owner_verification_policy")
    if raw_policy is None:
        raise RuntimeError("authenticated owner verification policy is missing")
    if not isinstance(raw_policy, dict):
        raise RuntimeError("authenticated owner verification policy is malformed")
    try:
        authenticated = OwnerVerificationConfig.from_dict(raw_policy)
    except ValueError as exc:
        raise RuntimeError("authenticated owner verification policy is invalid") from exc
    if requested is not None and requested != authenticated:
        raise RuntimeError("requested owner verification policy does not match the package audit")
    return authenticated, True


def _materialize_buyer_package(
    source: SkillSource,
    result: BuildResult,
    output: Path,
    *,
    buyer_id: str,
    config: RuntimeConfig,
    model: LanguageModel,
    owner_verification_config: OwnerVerificationConfig,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    rendered_files = source.rendered_files(result.markdown)
    delivery_manifest = file_manifest(rendered_files)
    delivery_tree_sha256 = manifest_digest(delivery_manifest)
    raw_semantic_nodes = result.audit.get("semantic_nodes", [])
    if not isinstance(raw_semantic_nodes, list) or not all(
        isinstance(node, dict) for node in raw_semantic_nodes
    ):
        raise RuntimeError("watermark result contains invalid semantic-node metadata")
    semantic_provenance = source.semantic_provenance(raw_semantic_nodes)
    raw_selected_node_ids = result.audit.get("selected_node_ids", [])
    if not isinstance(raw_selected_node_ids, list):
        raise RuntimeError("watermark result contains invalid carrier metadata")
    selected_node_ids = {str(value) for value in raw_selected_node_ids}
    selected_document_paths = sorted(
        {
            str(row["document_path"])
            for row in semantic_provenance
            if str(row["node_id"]) in selected_node_ids
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        delivery = stage / "buyer_delivery"
        private = stage / "owner_audit"
        write_delivery(delivery, rendered_files)
        private.mkdir()
        authenticated_audit = dict(result.audit)
        authenticated_audit.update(
            {
                "source_kind": source.source_kind,
                "entrypoint": source.entrypoint,
                "source_tree_sha256": source.tree_sha256,
                "source_manifest": source.manifest,
                "document_paths": list(source.document_order),
                "semantic_node_provenance": semantic_provenance,
                "selected_document_paths": selected_document_paths,
                "delivery_tree_sha256": delivery_tree_sha256,
                "delivery_manifest": delivery_manifest,
                "owner_verification_policy": owner_verification_config.to_dict(),
            }
        )
        authenticated_audit["owner_authentication"] = audit_authentication(
            config.owner_key, authenticated_audit
        )
        _write_json(private / "audit.json", authenticated_audit)
        summary = {
            "protocol": PROTOCOL,
            "status": "complete",
            "security_status": "pending_probe",
            "private_owner_key_required": True,
            "model": model.model,
            "model_base_url": config.base_url,
            "skill_id": result.audit["skill_id"],
            "buyer_id": buyer_id,
            "watermark_plan_sha256": result.audit["watermark_plan_sha256"],
            "owner_key_fingerprint": key_fingerprint(config.owner_key),
            "source_sha256": result.audit["source_sha256"],
            "delivery_sha256": result.audit["delivery_sha256"],
            "source_kind": source.source_kind,
            "entrypoint": source.entrypoint,
            "source_tree_sha256": source.tree_sha256,
            "delivery_tree_sha256": delivery_tree_sha256,
            "document_paths": list(source.document_order),
            "selected_document_paths": selected_document_paths,
            "owner_verification_policy": owner_verification_config.to_dict(),
            "public_files": [f"buyer_delivery/{row['path']}" for row in delivery_manifest],
            "private_files": ["owner_audit/audit.json"],
        }
        _write_json(stage / "build.json", summary)
        _publish_directory_no_replace(stage, output)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _prepare_plan(
    source: SkillSource,
    *,
    skill_id: str,
    config: RuntimeConfig,
    buyer_count: int,
    codeword_length: int,
    model: LanguageModel,
) -> WatermarkPlan:
    return prepare_watermark_plan(
        source.canonical_markdown,
        skill_id=skill_id,
        owner_key=config.owner_key,
        model=model,
        buyer_count=buyer_count,
        codeword_length=codeword_length,
        carrier_markdown=None,
    )


def _build_package_from_source(
    source: SkillSource,
    output: Path,
    *,
    skill_id: str,
    buyer_id: str,
    config: RuntimeConfig,
    normal_query_values: list[str],
    buyer_count: int,
    codeword_length: int,
    pairs: int,
    model: LanguageModel,
    owner_verification_config: OwnerVerificationConfig,
    plan: WatermarkPlan | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    frozen_plan = plan or _prepare_plan(
        source,
        skill_id=skill_id,
        config=config,
        buyer_count=buyer_count,
        codeword_length=codeword_length,
        model=model,
    )
    _validate_probe_pairs(pairs, query_count=len(normal_query_values))
    probe_pairs, probe_generation = generate_matched_probe_pairs(
        skill_id=skill_id,
        base_queries=normal_query_values,
        active_cues=frozen_plan.activation.active_cues,
        decoy_cues=frozen_plan.activation.decoy_cues,
        count=pairs,
        model=model,
    )
    result = render_watermarked_buyer(
        frozen_plan,
        buyer_id=buyer_id,
        owner_key=config.owner_key,
        model=model,
        normal_queries=normal_query_values,
        execution_renderer=source.execution_markdown,
    )
    result.audit["matched_probe_plan"] = [pair.to_dict() for pair in probe_pairs]
    result.audit["matched_probe_generation"] = probe_generation
    result.audit["behavior_input_serialization"] = (
        "single_document" if source.source_kind == "document" else "package_documents"
    )
    result.audit["carrier_scope"] = (
        "single_document" if source.source_kind == "document" else "package_documents"
    )
    return _materialize_buyer_package(
        source,
        result,
        output,
        buyer_id=buyer_id,
        config=config,
        model=model,
        owner_verification_config=owner_verification_config,
    )


def build_package(
    source: Path,
    output: Path,
    *,
    skill_id: str,
    buyer_id: str,
    config: RuntimeConfig,
    normal_queries: Path,
    buyer_count: int = 8,
    codeword_length: int = 4,
    pairs: int = 5,
    entrypoint: str | None = None,
    model: LanguageModel | None = None,
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _reject_output_inside_source(source, output)
    source_package = load_skill_source(source, entrypoint=entrypoint)
    client = model or OpenAICompatibleModel(config)
    return _build_package_from_source(
        source_package,
        output,
        skill_id=skill_id,
        buyer_id=buyer_id,
        config=config,
        normal_query_values=_read_queries(normal_queries),
        buyer_count=buyer_count,
        codeword_length=codeword_length,
        pairs=pairs,
        model=client,
        owner_verification_config=(
            owner_verification_config or OwnerVerificationConfig()
        ),
    )


def _build_buyer_family_from_source(
    source_package: SkillSource,
    output: Path,
    *,
    skill_id: str,
    config: RuntimeConfig,
    normal_query_values: list[str],
    buyer_count: int,
    codeword_length: int,
    pairs: int,
    buyer_ids: list[str] | None,
    model: LanguageModel,
    owner_verification_config: OwnerVerificationConfig,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    plan = _prepare_plan(
        source_package,
        skill_id=skill_id,
        config=config,
        buyer_count=buyer_count,
        codeword_length=codeword_length,
        model=model,
    )
    selected_buyers = list(plan.codebook) if buyer_ids is None else list(buyer_ids)
    if not selected_buyers or len(selected_buyers) != len(set(selected_buyers)):
        raise ValueError("buyer_ids must be a non-empty list without duplicates")
    unknown = sorted(set(selected_buyers) - set(plan.codebook))
    if unknown:
        raise ValueError(f"unknown buyer ids: {unknown}")

    _validate_probe_pairs(pairs, query_count=len(normal_query_values))
    matched_probe_pairs, probe_generation = generate_matched_probe_pairs(
        skill_id=skill_id,
        base_queries=normal_query_values,
        active_cues=plan.activation.active_cues,
        decoy_cues=plan.activation.decoy_cues,
        count=pairs,
        model=model,
    )
    serialized_probe_pairs = [pair.to_dict() for pair in matched_probe_pairs]

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        buyer_summaries: dict[str, dict[str, object]] = {}
        for buyer_id in selected_buyers:
            result = render_watermarked_buyer(
                plan,
                buyer_id=buyer_id,
                owner_key=config.owner_key,
                model=model,
                normal_queries=normal_query_values,
                execution_renderer=source_package.execution_markdown,
            )
            result.audit["matched_probe_plan"] = serialized_probe_pairs
            result.audit["matched_probe_generation"] = probe_generation
            result.audit["behavior_input_serialization"] = (
                "single_document"
                if source_package.source_kind == "document"
                else "package_documents"
            )
            result.audit["carrier_scope"] = (
                "single_document"
                if source_package.source_kind == "document"
                else "package_documents"
            )
            buyer_summaries[buyer_id] = _materialize_buyer_package(
                source_package,
                result,
                stage / "buyers" / buyer_id,
                buyer_id=buyer_id,
                config=config,
                model=model,
                owner_verification_config=owner_verification_config,
            )

        family_audit: dict[str, object] = {
            "protocol": PROTOCOL,
            "security_scope": "private_multi_buyer_watermark_plan",
            "model": model.model,
            "model_base_url": config.base_url,
            "skill_id": skill_id,
            "watermark_plan_sha256": plan.plan_sha256,
            "owner_key_fingerprint": plan.owner_key_fingerprint,
            "buyer_count": plan.buyer_count,
            "codeword_length": plan.codeword_length,
            "candidate_buyer_ids": selected_buyers,
            "activation_profile": plan.activation.to_dict(),
            "capsule_profile": plan.profile.to_dict(),
            "token_pairs": [list(pair) for pair in plan.token_pairs],
            "codebook": {key: value.to_dict() for key, value in plan.codebook.items()},
            "selected_node_ids": [node.node_id for node in plan.selected_nodes],
            "selected_node_kinds": [node.kind for node in plan.selected_nodes],
            "source_kind": source_package.source_kind,
            "entrypoint": source_package.entrypoint,
            "source_sha256": hashlib.sha256(
                source_package.canonical_markdown.encode()
            ).hexdigest(),
            "source_tree_sha256": source_package.tree_sha256,
            "source_manifest": source_package.manifest,
            "document_paths": list(source_package.document_order),
            "normal_queries_sha256": query_set_digest(normal_query_values),
            "normal_query_count": len(normal_query_values),
            "matched_probe_plan": serialized_probe_pairs,
            "matched_probe_generation": probe_generation,
            "owner_verification_policy": owner_verification_config.to_dict(),
            "buyer_packages": {
                buyer_id: {
                    "path": f"buyers/{buyer_id}",
                    "delivery_tree_sha256": summary["delivery_tree_sha256"],
                    "delivery_sha256": summary["delivery_sha256"],
                }
                for buyer_id, summary in buyer_summaries.items()
            },
        }
        family_audit["owner_authentication"] = audit_authentication(
            config.owner_key, family_audit
        )
        owner_audit = stage / "owner_audit"
        owner_audit.mkdir()
        _write_json(owner_audit / "family.json", family_audit)
        summary = {
            "protocol": PROTOCOL,
            "status": "complete",
            "security_status": "pending_probe",
            "private_owner_key_required": True,
            "model": model.model,
            "model_base_url": config.base_url,
            "skill_id": skill_id,
            "source_kind": source_package.source_kind,
            "entrypoint": source_package.entrypoint,
            "source_tree_sha256": source_package.tree_sha256,
            "watermark_plan_sha256": plan.plan_sha256,
            "buyer_population": plan.buyer_count,
            "candidate_buyer_count": len(selected_buyers),
            "candidate_buyer_ids": selected_buyers,
            "owner_verification_policy": owner_verification_config.to_dict(),
            "buyer_packages": {
                buyer_id: f"buyers/{buyer_id}" for buyer_id in selected_buyers
            },
            "private_files": ["owner_audit/family.json"],
        }
        _write_json(stage / "family.json", summary)
        _publish_directory_no_replace(stage, output)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_buyer_family(
    source: Path,
    output: Path,
    *,
    skill_id: str,
    config: RuntimeConfig,
    normal_queries: Path,
    buyer_count: int = 8,
    codeword_length: int = 4,
    pairs: int = 5,
    buyer_ids: list[str] | None = None,
    entrypoint: str | None = None,
    model: LanguageModel | None = None,
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    """Build multiple buyer copies from one frozen semantic and cryptographic plan."""

    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _reject_output_inside_source(source, output)
    source_package = load_skill_source(source, entrypoint=entrypoint)
    client = model or OpenAICompatibleModel(config)
    return _build_buyer_family_from_source(
        source_package,
        output,
        skill_id=skill_id,
        config=config,
        normal_query_values=_read_queries(normal_queries),
        buyer_count=buyer_count,
        codeword_length=codeword_length,
        pairs=pairs,
        buyer_ids=buyer_ids,
        model=client,
        owner_verification_config=(
            owner_verification_config or OwnerVerificationConfig()
        ),
    )


def _authenticated_delivery(
    package: Path, audit: dict[str, Any]
) -> tuple[str, dict[str, bytes]]:
    if package.is_symlink() or not package.is_dir():
        raise RuntimeError("buyer package root must be a real directory")
    delivery_root = package / "buyer_delivery"
    if delivery_root.is_symlink() or not delivery_root.is_dir():
        raise RuntimeError("buyer delivery root must be a real directory")
    raw_document_paths = audit.get("document_paths")
    if not isinstance(raw_document_paths, list) or not raw_document_paths:
        raise RuntimeError(
            "authenticated delivery manifest and document paths are required"
        )
    try:
        manifest = validate_delivery_manifest(audit.get("delivery_manifest"))
    except ValueError as exc:
        raise RuntimeError("authenticated delivery manifest is malformed") from exc
    files = read_delivery(delivery_root, manifest)
    if manifest_digest(manifest) != audit.get("delivery_tree_sha256"):
        raise RuntimeError("buyer delivery tree digest does not match the private audit")
    document_paths = tuple(str(path) for path in raw_document_paths)
    documents: dict[str, str] = {}
    for path in document_paths:
        if path not in files:
            raise RuntimeError(f"authenticated delivery document is missing: {path}")
        try:
            documents[path] = files[path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"authenticated delivery document is not UTF-8: {path}") from exc
    canonical = compose_documents(documents, document_paths)
    if hashlib.sha256(canonical.encode()).hexdigest() != audit.get("delivery_sha256"):
        raise RuntimeError("buyer delivery canonical hash does not match the private audit")
    return delivery_prompt(files, document_paths=document_paths), files


def _probe_target(
    target: ProbeTarget,
    *,
    audit: dict[str, Any],
    owner_key: str,
    normal_query_values: list[str],
    pairs: int = 5,
    owner_verification_config: OwnerVerificationConfig | None = None,
    expected_buyer: str | None = None,
    allowed_buyer_ids: list[str] | None = None,
) -> dict[str, object]:
    """Apply an authenticated private probe plan to an independent target.

    The target is intentionally not integrity-bound to the reference delivery.  This
    is the reusable post-leak detection boundary: a caller may supply a local modified
    Skill target or an adapter for a remote black-box Agent.  Only the defender's
    audit, frozen policy, codebook, and normal-query set are authenticated here.
    """

    _validate_probe_pairs(pairs)
    if audit.get("protocol") != PROTOCOL:
        raise ValueError("unsupported probe-plan protocol")
    if audit.get("owner_key_fingerprint") != key_fingerprint(owner_key):
        raise RuntimeError("SKILLCODER_OWNER_KEY does not match the probe plan")
    if not audit_is_authentic(owner_key, audit):
        raise RuntimeError("probe-plan authentication failed")
    owner_policy, policy_authenticated = _owner_policy_from_audit(
        audit, owner_verification_config
    )
    queries = _authenticated_queries(normal_query_values, audit)
    _validate_probe_pairs(pairs, query_count=len(queries))
    try:
        profile = CapsuleProfile(**dict(audit["capsule_profile"]))
        token_pairs = [list(value) for value in audit["token_pairs"]]
        codebook = dict(audit["codebook"])
        activation = dict(audit["activation_profile"])
        active_cues = [str(value) for value in activation["active_cues"]]
        decoy_cues = [str(value) for value in activation["decoy_cues"]]
        skill_id = str(audit["skill_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("authenticated probe plan is malformed") from exc
    if len(active_cues) != 3 or len(decoy_cues) != 3:
        raise RuntimeError("authenticated probe cues must contain exactly three values")
    try:
        matched_probe_pairs = load_matched_probe_pairs(
            audit.get("matched_probe_plan"),
            active_cues=tuple(active_cues),
            decoy_cues=tuple(decoy_cues),
        )
    except ValueError as exc:
        raise RuntimeError("authenticated matched probe plan is malformed") from exc
    if pairs > len(matched_probe_pairs):
        raise ValueError(
            f"requested {pairs} probe pairs but the authenticated plan contains "
            f"{len(matched_probe_pairs)}"
        )
    selected_probe_pairs = matched_probe_pairs[:pairs]
    probe_pair_by_id = {pair.pair_id: pair for pair in selected_probe_pairs}
    released_buyer_set: set[str] | None = None
    if allowed_buyer_ids is not None:
        if (
            not allowed_buyer_ids
            or len(allowed_buyer_ids) != len(set(allowed_buyer_ids))
            or not set(allowed_buyer_ids).issubset(codebook)
        ):
            raise RuntimeError("released buyer set is invalid for the authenticated codebook")
        released_buyer_set = set(allowed_buyer_ids)
    if expected_buyer is not None and expected_buyer not in codebook:
        raise RuntimeError("expected buyer is absent from the authenticated codebook")

    jobs: list[tuple[str, int, str]] = []
    for pair in selected_probe_pairs:
        jobs.extend(
            (
                ("active", pair.pair_id, pair.positive_query),
                ("decoy", pair.pair_id, pair.negative_query),
            )
        )
    jobs.extend(("normal", index, query) for index, query in enumerate(queries))
    schedule_binding = str(audit["owner_authentication"])

    def schedule_key(job: tuple[str, int, str]) -> bytes:
        label, index, query = job
        context = f"{schedule_binding}\0{label}\0{index}\0{query}".encode("utf-8")
        return hmac.new(
            owner_key.encode("utf-8"),
            b"skillcoder-probe-order\0" + context,
            hashlib.sha256,
        ).digest()

    active_rows: list[dict[str, object]] = []
    decoy_rows: list[dict[str, object]] = []
    normal_rows: list[dict[str, object]] = []
    for label, index, query in sorted(jobs, key=schedule_key):
        response, call = target.invoke(query, purpose="behavior_probe")
        validity = owner_capsule_validity(response, profile)
        extracted = parse_payload(response, profile, token_pairs)
        row: dict[str, object] = {
            "query": query,
            "output": response,
            "valid_payload": extracted is not None,
            "capsule_validity": validity,
            "capsule_similarity": (
                float(extracted["similarity"]) if extracted else 0.0
            ),
            "model_call": call,
        }
        if label == "normal":
            row["query_index"] = index
            normal_rows.append(row)
        else:
            pair = probe_pair_by_id[index]
            row["matched_pair"] = index
            row["intent"] = pair.intent
            row["pair_purpose"] = pair.purpose
            row["probe_polarity"] = "positive" if label == "active" else "negative"
            row["probe_role"] = (
                "activation_evidence"
                if label == "active"
                else "matched_suppression_control"
            )
            row["expected_effect"] = (
                "the protected skill should emit a valid owner capsule"
                if label == "active"
                else "the protected skill should remain on the ordinary task path"
            )
            (active_rows if label == "active" else decoy_rows).append(row)
    active_rows.sort(key=lambda row: cast(int, row["matched_pair"]))
    decoy_rows.sort(key=lambda row: cast(int, row["matched_pair"]))
    normal_rows.sort(key=lambda row: cast(int, row["query_index"]))
    for row in normal_rows:
        row.pop("query_index")
    decoded = decode_buyer(
        [str(row["output"]) for row in active_rows],
        profile=profile,
        token_pairs=token_pairs,
        codebook=codebook,
    )
    if (
        released_buyer_set is not None
        and decoded["top1"]
        and decoded["top1"] not in released_buyer_set
    ):
        decoded["top1"] = ""
        decoded["ecc_satisfied"] = False
        decoded["release_scope_abstention"] = True
    active_rate = sum(bool(row["valid_payload"]) for row in active_rows) / len(
        active_rows
    )
    decoy_rate = sum(bool(row["valid_payload"]) for row in decoy_rows) / len(
        decoy_rows
    )
    normal_rate = sum(bool(row["valid_payload"]) for row in normal_rows) / len(
        normal_rows
    )
    owner_verification = verify_owner_membership(
        [cast(float, row["capsule_validity"]) for row in active_rows],
        [cast(float, row["capsule_validity"]) for row in decoy_rows],
        config=owner_policy,
        policy_authenticated=policy_authenticated,
    )
    pair_effects = [
        {
            "matched_pair": pair.pair_id,
            "intent": pair.intent,
            "purpose": pair.purpose,
            "controlled_difference": "private_activation_cues_only",
            "positive_validity": cast(float, active_rows[index]["capsule_validity"]),
            "negative_validity": cast(float, decoy_rows[index]["capsule_validity"]),
            "validity_differential": cast(float, active_rows[index]["capsule_validity"])
            - owner_policy.negative_weight
            * cast(float, decoy_rows[index]["capsule_validity"]),
            "observed_effect": (
                "activation_with_suppression"
                if bool(active_rows[index]["valid_payload"])
                and not bool(decoy_rows[index]["valid_payload"])
                else "activation_without_suppression"
                if bool(active_rows[index]["valid_payload"])
                else "no_positive_activation"
            ),
        }
        for index, pair in enumerate(selected_probe_pairs)
    ]
    if owner_verification["supported"]:
        buyer_attribution = attribute_buyer(decoded, expected_buyer=expected_buyer)
    else:
        raw_erasures = decoded.get("erasures", 0)
        buyer_attribution = {
            "attributed": False,
            "status": "not_evaluated_owner_not_supported",
            "reason": "owner_verification_not_supported",
            "decoded_buyer": "",
            "ecc_satisfied": bool(decoded.get("ecc_satisfied")),
            "erasures": (
                raw_erasures
                if isinstance(raw_erasures, int) and not isinstance(raw_erasures, bool)
                else 0
            ),
        }
        if expected_buyer is not None:
            buyer_attribution.update(
                {"expected_buyer": expected_buyer, "expected_buyer_match": False}
            )
    if not owner_verification["supported"]:
        detection_status = "owner_not_supported"
    elif not buyer_attribution["attributed"]:
        detection_status = "owner_supported_buyer_unattributed"
    elif expected_buyer is not None and not buyer_attribution["expected_buyer_match"]:
        detection_status = "owner_supported_buyer_mismatch"
    else:
        detection_status = "owner_supported_buyer_attributed"
    return {
        "protocol": PROTOCOL,
        "scope": "authenticated_probe_target",
        "model": target.model,
        "probe_runtime": target.runtime,
        "skill_id": skill_id,
        "probe_statistics": {
            "active_payload_rate": active_rate,
            "decoy_payload_rate": decoy_rate,
            "normal_payload_rate": normal_rate,
            "paired_payload_differential": active_rate - decoy_rate,
        },
        "sample_counts": {
            "active": len(active_rows),
            "decoy": len(decoy_rows),
            "normal": len(normal_rows),
        },
        "probe_design": {
            "schema": "skillcoder-matched-probes/1",
            "generation": "bounded_llm_generate_judge_revise_with_deterministic_cue_substitution",
            "matching_contract": "each pair shares one task template and differs only in private cue values",
            "positive_role": "test activation under the owner-selected cue conjunction",
            "negative_role": "measure suppression under a same-format decoy conjunction",
            "decision_role": "ownership requires repeated positive-minus-negative separation",
            "intent_coverage": sorted({pair.intent for pair in selected_probe_pairs}),
        },
        "pair_effects": pair_effects,
        "buyer": decoded,
        "owner_verification": owner_verification,
        "buyer_attribution": buyer_attribution,
        "detection_result": {
            "supported": owner_verification["supported"],
            "status": detection_status,
            "decoded_buyer": buyer_attribution["decoded_buyer"],
        },
        "records": {
            "active": active_rows,
            "decoy": decoy_rows,
            "normal": normal_rows,
        },
    }


def probe_package(
    package: Path,
    output: Path,
    *,
    config: RuntimeConfig,
    pairs: int = 5,
    normal_queries: Path,
    model: LanguageModel | None = None,
    runtime: str = "direct",
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _reject_output_inside_input(package, output, label="buyer package")
    _validate_probe_pairs(pairs)
    audit_path = package / "owner_audit" / "audit.json"
    audit = _read_json_object(audit_path, label="package audit")
    if audit.get("protocol") != PROTOCOL:
        raise ValueError("unsupported package protocol")
    if audit.get("owner_key_fingerprint") != key_fingerprint(config.owner_key):
        raise RuntimeError("SKILLCODER_OWNER_KEY does not match the package audit")
    if not audit_is_authentic(config.owner_key, audit):
        raise RuntimeError("package audit authentication failed")
    skill, _ = _authenticated_delivery(package, audit)
    normal_query_values = _read_queries(normal_queries)
    _authenticated_queries(normal_query_values, audit)
    target = create_probe_target(runtime, skill=skill, config=config, model=model)
    report = _probe_target(
        target,
        audit=audit,
        owner_key=config.owner_key,
        normal_query_values=normal_query_values,
        pairs=pairs,
        owner_verification_config=owner_verification_config,
        expected_buyer=str(audit["buyer_id"]),
    )
    probe_statistics = cast(dict[str, object], report.pop("probe_statistics"))
    active_rate = cast(float, probe_statistics["active_payload_rate"])
    decoy_rate = cast(float, probe_statistics["decoy_payload_rate"])
    normal_rate = cast(float, probe_statistics["normal_payload_rate"])
    owner_verification = cast(dict[str, object], report["owner_verification"])
    buyer_attribution = cast(dict[str, object], report["buyer_attribution"])
    decoded = cast(dict[str, object], report["buyer"])
    release_gate = release_quality_gate(
        active_rate=active_rate,
        decoy_rate=decoy_rate,
        normal_rate=normal_rate,
        expected_buyer_match=decoded["top1"] == audit["buyer_id"],
    )
    release_quality_passed = bool(release_gate["passed"])
    release_ready = bool(
        owner_verification["supported"]
        and buyer_attribution["expected_buyer_match"]
        and release_quality_passed
    )
    if not owner_verification["supported"]:
        detection_status = "owner_not_supported"
    elif not buyer_attribution["attributed"]:
        detection_status = "owner_supported_buyer_unattributed"
    elif not buyer_attribution["expected_buyer_match"]:
        detection_status = "owner_supported_buyer_mismatch"
    elif not release_ready:
        detection_status = "owner_supported_not_release_ready"
    else:
        detection_status = "supported_by_probe"
    report.update(
        {
            "scope": "core_method_probe",
            "model_base_url": config.base_url,
            "expected_buyer": audit["buyer_id"],
            "release_quality": {
            "active_payload_rate": active_rate,
            "decoy_payload_rate": decoy_rate,
            "normal_payload_rate": normal_rate,
            "paired_payload_differential": active_rate - decoy_rate,
            "gate": release_gate,
            },
            "release_quality_passed": release_quality_passed,
            "release_ready": release_ready,
            "detection_result": {
                "supported": owner_verification["supported"],
                "status": detection_status,
                "decoded_buyer": buyer_attribution["decoded_buyer"],
            },
        }
    )
    _write_json_atomic(output, report)
    return report


def _verify_buyer_family_snapshot(
    family: Path,
    owner_key: str,
) -> tuple[dict[str, object], dict[str, Any] | None]:
    invalid = {
        "valid": False,
        "checks": {
            "protocol": False,
            "owner_key": False,
            "audit_authentication": False,
            "buyer_surface": False,
            "buyer_packages": False,
        },
        "buyers": {},
    }
    owner_audit = family / "owner_audit"
    buyers_root = family / "buyers"
    if (
        family.is_symlink()
        or not family.is_dir()
        or owner_audit.is_symlink()
        or buyers_root.is_symlink()
        or not owner_audit.is_dir()
        or not buyers_root.is_dir()
    ):
        return invalid, None
    audit_path = family / "owner_audit" / "family.json"
    if audit_path.is_symlink() or not audit_path.is_file():
        return invalid, None
    try:
        audit = _read_json_object(audit_path, label="family audit")
    except (OSError, ValueError):
        return invalid, None
    authentication_valid = audit_is_authentic(owner_key, audit)
    protocol_valid = audit.get("protocol") == PROTOCOL
    owner_key_valid = audit.get("owner_key_fingerprint") == key_fingerprint(owner_key)
    if not (authentication_valid and protocol_valid and owner_key_valid):
        invalid["checks"] = {
            "protocol": protocol_valid,
            "owner_key": owner_key_valid,
            "audit_authentication": authentication_valid,
            "buyer_surface": False,
            "buyer_packages": False,
        }
        return invalid, None
    raw_expected_buyers = audit.get("candidate_buyer_ids")
    if (
        not isinstance(raw_expected_buyers, list)
        or not raw_expected_buyers
        or not all(isinstance(value, str) and value for value in raw_expected_buyers)
        or any(
            "\\" in value
            or PurePosixPath(value).parts != (value,)
            or value in {".", ".."}
            for value in raw_expected_buyers
        )
    ):
        return invalid, None
    expected_buyers = list(raw_expected_buyers)
    buyer_checks: dict[str, dict[str, object]] = {}
    family_packages = audit.get("buyer_packages", {})
    if not isinstance(family_packages, dict):
        family_packages = {}
    try:
        with BoundedTree.open(
            buyers_root,
            max_entries=MAX_TREE_ENTRIES,
            recursive=False,
        ) as buyers_tree:
            surface_is_directories_only = all(
                entry.kind == "directory" for entry in buyers_tree.entries
            )
            actual_buyers = sorted(entry.relative_path for entry in buyers_tree.entries)
            for buyer_id in expected_buyers:
                package = family / "buyers" / buyer_id
                try:
                    result, buyer_audit = _verify_package_snapshot(package, owner_key)
                    if not bool(result["valid"]) or buyer_audit is None:
                        raise RuntimeError("buyer package integrity verification failed")
                    plan_matches = (
                        buyer_audit.get("watermark_plan_sha256")
                        == audit.get("watermark_plan_sha256")
                    )
                    buyer_id_matches = buyer_audit.get("buyer_id") == buyer_id
                    shared_mapping_matches = (
                        buyer_audit.get("codebook") == audit.get("codebook")
                        and buyer_audit.get("token_pairs") == audit.get("token_pairs")
                        and buyer_audit.get("activation_profile")
                        == audit.get("activation_profile")
                        and buyer_audit.get("capsule_profile")
                        == audit.get("capsule_profile")
                        and buyer_audit.get("matched_probe_plan")
                        == audit.get("matched_probe_plan")
                        and buyer_audit.get("owner_verification_policy")
                        == audit.get("owner_verification_policy")
                    )
                    expected_package = family_packages.get(buyer_id, {})
                    family_delivery_matches = isinstance(expected_package, dict) and (
                        buyer_audit.get("delivery_tree_sha256")
                        == expected_package.get("delivery_tree_sha256")
                        and buyer_audit.get("delivery_sha256")
                        == expected_package.get("delivery_sha256")
                    )
                except (OSError, ValueError, RuntimeError):
                    result = {"valid": False, "checks": {}}
                    plan_matches = False
                    buyer_id_matches = False
                    shared_mapping_matches = False
                    family_delivery_matches = False
                buyer_checks[buyer_id] = {
                    "package_valid": bool(result["valid"]),
                    "watermark_plan": plan_matches,
                    "buyer_id": buyer_id_matches,
                    "shared_mapping": shared_mapping_matches,
                    "family_delivery": family_delivery_matches,
                }
            buyers_tree.assert_unchanged()
    except ValueError:
        surface_is_directories_only = False
        actual_buyers = []
    checks = {
        "protocol": protocol_valid,
        "owner_key": owner_key_valid,
        "audit_authentication": authentication_valid,
        "buyer_surface": surface_is_directories_only
        and actual_buyers == sorted(expected_buyers),
        "buyer_packages": bool(expected_buyers)
        and all(all(values.values()) for values in buyer_checks.values()),
    }
    result = {"valid": all(checks.values()), "checks": checks, "buyers": buyer_checks}
    return result, audit if result["valid"] else None


def verify_buyer_family(family: Path, owner_key: str) -> dict[str, object]:
    result, _ = _verify_buyer_family_snapshot(family, owner_key)
    return result


@dataclass(frozen=True)
class _AuthenticatedProbeReference:
    audit: dict[str, Any]
    reference_kind: str
    released_buyer_ids: tuple[str, ...]
    release_status: str


def _load_authenticated_probe_reference(
    reference: Path,
    owner_key: str,
) -> _AuthenticatedProbeReference:
    release_verification, release = _verify_release_manifest_snapshot(
        reference,
        owner_key,
    )
    if not release_verification["valid"] or release is None:
        raise RuntimeError("reference release verification failed")
    raw_ready = release.get("release_ready_buyer_ids")
    ready_ids = [str(value) for value in raw_ready] if isinstance(raw_ready, list) else []
    if not ready_ids:
        raise RuntimeError("reference release has no approved buyers to attribute")
    approved = release.get("approved_deliveries")
    if not isinstance(approved, dict) or set(approved) != set(ready_ids):
        raise RuntimeError("reference release approved-buyer surface is malformed")

    package_root = reference / "package"
    family_root = reference / "family"
    package_audit = package_root / "owner_audit" / "audit.json"
    family_audit = family_root / "owner_audit" / "family.json"
    has_package_audit = package_audit.is_file() and not package_audit.is_symlink()
    has_family_audit = family_audit.is_file() and not family_audit.is_symlink()
    if has_package_audit == has_family_audit:
        raise ValueError(
            "reference run must contain exactly one package or family owner audit"
        )
    if has_package_audit:
        verification, audit = _verify_package_snapshot(package_root, owner_key)
        if not verification["valid"] or audit is None:
            raise RuntimeError("reference integrity verification failed")
        reference_kind = "released_buyer_package"
    else:
        try:
            audit = _read_json_object(family_audit, label="reference family audit")
        except (OSError, ValueError) as exc:
            raise RuntimeError("reference audit could not be read") from exc
        if (
            audit.get("protocol") != PROTOCOL
            or audit.get("owner_key_fingerprint") != key_fingerprint(owner_key)
            or not audit_is_authentic(owner_key, audit)
        ):
            raise RuntimeError(
                "reference integrity verification failed: family audit authentication failed"
            )
        reference_kind = "released_buyer_family"
    if (
        release.get("skill_id") != audit.get("skill_id")
        or release.get("watermark_plan_sha256")
        != audit.get("watermark_plan_sha256")
    ):
        raise RuntimeError("reference release does not bind the authenticated probe plan")
    codebook = audit.get("codebook")
    if not isinstance(codebook, dict) or not set(ready_ids).issubset(codebook):
        raise RuntimeError("reference release buyers are absent from the probe codebook")
    if reference_kind == "released_buyer_package":
        buyer_id = str(audit.get("buyer_id", ""))
        if ready_ids != [buyer_id]:
            raise RuntimeError("single-package release buyer does not match its audit")
        record = approved.get(buyer_id)
        if not isinstance(record, dict) or (
            record.get("path") != "package/buyer_delivery"
            or record.get("delivery_sha256") != audit.get("delivery_sha256")
            or record.get("delivery_tree_sha256") != audit.get("delivery_tree_sha256")
            or record.get("delivery_manifest") != audit.get("delivery_manifest")
        ):
            raise RuntimeError("single-package release delivery does not match its audit")
    else:
        buyer_packages = audit.get("buyer_packages")
        if not isinstance(buyer_packages, dict):
            raise RuntimeError("family probe plan has no authenticated buyer packages")
        for buyer_id in ready_ids:
            record = approved.get(buyer_id)
            package_record = buyer_packages.get(buyer_id)
            if (
                not isinstance(record, dict)
                or not isinstance(package_record, dict)
                or record.get("path") != f"family/buyers/{buyer_id}/buyer_delivery"
                or record.get("delivery_sha256")
                != package_record.get("delivery_sha256")
                or record.get("delivery_tree_sha256")
                != package_record.get("delivery_tree_sha256")
            ):
                raise RuntimeError(
                    f"family release delivery does not match the audit for {buyer_id}"
                )
    return _AuthenticatedProbeReference(
        audit=dict(audit),
        reference_kind=reference_kind,
        released_buyer_ids=tuple(ready_ids),
        release_status=str(release["status"]),
    )


def _probe_authenticated_reference(
    reference: _AuthenticatedProbeReference,
    target: ProbeTarget,
    *,
    owner_key: str,
    normal_query_values: list[str],
    pairs: int,
    owner_verification_config: OwnerVerificationConfig | None,
) -> dict[str, object]:
    released_buyer_ids = list(reference.released_buyer_ids)
    report = _probe_target(
        target,
        audit=reference.audit,
        owner_key=owner_key,
        normal_query_values=normal_query_values,
        pairs=pairs,
        owner_verification_config=owner_verification_config,
        allowed_buyer_ids=released_buyer_ids,
    )
    report.update(
        {
            "scope": "post_distribution_suspect_probe",
            "reference_kind": reference.reference_kind,
            "reference_integrity_verified": True,
            "reference_release_status": reference.release_status,
            "released_buyer_ids": released_buyer_ids,
            "watermark_plan_sha256": reference.audit.get("watermark_plan_sha256"),
        }
    )
    return report


def probe_released_target(
    reference: Path,
    target: ProbeTarget,
    *,
    owner_key: str,
    normal_query_values: list[str],
    pairs: int = 5,
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    """Probe an independent target under an authenticated release decision."""

    _validate_probe_pairs(pairs)
    authenticated = _load_authenticated_probe_reference(reference, owner_key)
    return _probe_authenticated_reference(
        authenticated,
        target,
        owner_key=owner_key,
        normal_query_values=normal_query_values,
        pairs=pairs,
        owner_verification_config=owner_verification_config,
    )


def probe_suspect(
    reference: Path,
    suspect: Path,
    output: Path,
    *,
    config: RuntimeConfig,
    normal_queries: Path,
    pairs: int = 5,
    entrypoint: str | None = None,
    model: LanguageModel | None = None,
    runtime: str = "direct",
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    """Detect a separately supplied, potentially modified Skill.

    ``reference`` is owner-retained evidence and remains integrity protected.
    ``suspect`` is untrusted evaluation input, so its content is never required to
    match the reference hashes.  This separation is what permits paraphrase,
    compression, deletion, and reorganization attacks to be measured.
    """

    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _validate_probe_pairs(pairs)
    _reject_output_inside_input(reference, output, label="probe reference")
    _reject_output_inside_source(suspect, output)
    authenticated = _load_authenticated_probe_reference(reference, config.owner_key)
    query_values = _read_queries(normal_queries)
    _authenticated_queries(query_values, authenticated.audit)
    suspect_source = load_skill_source(suspect, entrypoint=entrypoint)
    suspect_markdown = suspect_source.canonical_markdown
    execution_skill = suspect_source.execution_markdown(suspect_markdown)
    target = create_probe_target(
        runtime,
        skill=execution_skill,
        config=config,
        model=model,
    )
    report = _probe_authenticated_reference(
        authenticated,
        target,
        owner_key=config.owner_key,
        normal_query_values=query_values,
        pairs=pairs,
        owner_verification_config=owner_verification_config,
    )
    report.update(
        {
            "model_base_url": config.base_url,
            "suspect": {
                "source_kind": suspect_source.source_kind,
                "entrypoint": suspect_source.entrypoint,
                "canonical_sha256": hashlib.sha256(
                    suspect_markdown.encode("utf-8")
                ).hexdigest(),
                "tree_sha256": suspect_source.tree_sha256,
                "integrity_policy": "untrusted_evaluation_input",
            },
        }
    )
    _write_json_atomic(output, report)
    return report


def probe_buyer_family(
    family: Path,
    output: Path,
    *,
    config: RuntimeConfig,
    normal_queries: Path,
    pairs: int = 5,
    buyer_ids: list[str] | None = None,
    model: LanguageModel | None = None,
    runtime: str = "direct",
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _validate_probe_pairs(pairs)
    _reject_output_inside_input(family, output, label="buyer family")
    family_verification, family_audit = _verify_buyer_family_snapshot(
        family,
        config.owner_key,
    )
    if not family_verification["valid"] or family_audit is None:
        raise RuntimeError("buyer family integrity verification failed")
    owner_policy, policy_authenticated = _owner_policy_from_audit(
        family_audit, owner_verification_config
    )
    normal_query_values = _read_queries(normal_queries)
    if query_set_digest(normal_query_values) != family_audit.get("normal_queries_sha256"):
        raise RuntimeError("normal-query set does not match the authenticated family audit")
    candidate_buyers = [str(value) for value in family_audit["candidate_buyer_ids"]]
    selected_buyers = candidate_buyers if buyer_ids is None else list(buyer_ids)
    if not selected_buyers or len(selected_buyers) != len(set(selected_buyers)):
        raise ValueError("buyer_ids must be a non-empty list without duplicates")
    unknown = sorted(set(selected_buyers) - set(candidate_buyers))
    if unknown:
        raise ValueError(f"buyer ids are not candidates in this family: {unknown}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        reports: dict[str, dict[str, Any]] = {}
        for buyer_id in selected_buyers:
            reports[buyer_id] = probe_package(
                family / "buyers" / buyer_id,
                stage / "buyers" / f"{buyer_id}.json",
                config=config,
                pairs=pairs,
                normal_queries=normal_queries,
                model=model if runtime == "direct" else None,
                runtime=runtime,
                owner_verification_config=owner_policy,
            )
        owner_supported_count = sum(
            bool(report["owner_verification"]["supported"])
            for report in reports.values()
        )
        buyer_attributed_count = sum(
            bool(report["buyer_attribution"]["attributed"])
            for report in reports.values()
        )
        release_ready_count = sum(
            bool(report["release_ready"]) for report in reports.values()
        )
        release_ready_buyer_ids = [
            buyer_id
            for buyer_id, report in reports.items()
            if bool(report["release_ready"])
        ]
        rejected_candidate_buyer_ids = [
            buyer_id
            for buyer_id in selected_buyers
            if buyer_id not in release_ready_buyer_ids
        ]
        top1_count = sum(
            report["buyer"]["top1"] == buyer_id
            for buyer_id, report in reports.items()
        )
        aggregate = {
            "protocol": PROTOCOL,
            "scope": "multi_buyer_core_method_probe",
            "skill_id": family_audit["skill_id"],
            "watermark_plan_sha256": family_audit["watermark_plan_sha256"],
            "model": next(iter(reports.values()))["model"],
            "probe_runtime": runtime,
            "candidate_buyer_count": len(candidate_buyers),
            "probed_buyer_count": len(selected_buyers),
            "owner_supported_count": owner_supported_count,
            "owner_verification_rate": owner_supported_count / len(selected_buyers),
            "buyer_attributed_count": buyer_attributed_count,
            "buyer_attribution_rate": buyer_attributed_count / len(selected_buyers),
            "release_ready_count": release_ready_count,
            "release_rate": release_ready_count / len(selected_buyers),
            "release_ready_buyer_ids": release_ready_buyer_ids,
            "rejected_candidate_buyer_ids": rejected_candidate_buyer_ids,
            "release_ready": release_ready_count == len(selected_buyers),
            "top1_accuracy": top1_count / len(selected_buyers),
            "owner_verification_policy": {
                **owner_policy.to_dict(),
                "authenticated_policy": policy_authenticated,
            },
            "buyers": {
                buyer_id: {
                    "release_ready": report["release_ready"],
                    "owner_supported": report["owner_verification"]["supported"],
                    "owner_score": report["owner_verification"]["score"],
                    "buyer_attribution_status": report["buyer_attribution"]["status"],
                    "decoded_buyer": report["buyer_attribution"]["decoded_buyer"],
                    "active_payload_rate": report["release_quality"][
                        "active_payload_rate"
                    ],
                    "decoy_payload_rate": report["release_quality"][
                        "decoy_payload_rate"
                    ],
                    "normal_payload_rate": report["release_quality"][
                        "normal_payload_rate"
                    ],
                    "report": f"buyers/{buyer_id}.json",
                }
                for buyer_id, report in reports.items()
            },
        }
        _write_json(stage / "report.json", aggregate)
        _publish_directory_no_replace(stage, output)
        return aggregate
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_model_pipeline(
    source: Path,
    output: Path,
    *,
    skill_id: str,
    buyer_id: str,
    config: RuntimeConfig,
    normal_query_count: int = 10,
    pairs: int = 5,
    buyer_count: int = 8,
    codeword_length: int = 4,
    entrypoint: str | None = None,
    model: LanguageModel | None = None,
    probe_runtime: str = "direct",
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _validate_probe_pairs(pairs)
    _reject_output_inside_source(source, output)
    client = model or OpenAICompatibleModel(config)
    owner_policy = owner_verification_config or OwnerVerificationConfig()
    source_package = load_skill_source(source, entrypoint=entrypoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        queries, query_audit = generate_normal_queries(
            source_package.prompt_markdown,
            skill_id=skill_id,
            count=normal_query_count,
            model=client,
        )
        query_audit.update(
            {
                "source_sha256": hashlib.sha256(
                    source_package.canonical_markdown.encode()
                ).hexdigest(),
                "source_tree_sha256": source_package.tree_sha256,
            }
        )
        query_path = stage / "normal_queries.json"
        _write_json(query_path, queries)
        build = _build_package_from_source(
            source_package,
            stage / "package",
            skill_id=skill_id,
            buyer_id=buyer_id,
            config=config,
            normal_query_values=queries,
            buyer_count=buyer_count,
            codeword_length=codeword_length,
            pairs=pairs,
            model=client,
            owner_verification_config=owner_policy,
        )
        report = probe_package(
            stage / "package",
            stage / "report.json",
            config=config,
            pairs=pairs,
            normal_queries=query_path,
            model=client if probe_runtime == "direct" else None,
            runtime=probe_runtime,
            owner_verification_config=owner_policy,
        )
        report["pipeline"] = [
            "query_generation",
            "watermark_build",
            "active_decoy_normal_probe",
            "ecc_decode",
            "suppression_gate",
            "report",
        ]
        report["query_generation"] = query_audit
        report["build"] = build
        report["run_status"] = "ready" if report["release_ready"] else "rejected"
        report["release_manifest"] = "release.json"
        _write_json(stage / "report.json", report)
        release_ready_buyer_ids = [buyer_id] if report["release_ready"] else []
        release_manifest = _authenticate_release_manifest({
            "schema": "skillcoder-release/2",
            "protocol": PROTOCOL,
            "status": "ready" if report["release_ready"] else "rejected",
            "skill_id": skill_id,
            "watermark_plan_sha256": build["watermark_plan_sha256"],
            "owner_key_fingerprint": key_fingerprint(config.owner_key),
            "report": {
                "path": "report.json",
                "sha256": _file_sha256(stage / "report.json"),
            },
            "release_ready_buyer_ids": release_ready_buyer_ids,
            "rejected_candidate_buyer_ids": [] if report["release_ready"] else [buyer_id],
            "approved_deliveries": {
                buyer_id: _approved_delivery_record(
                    stage / "package",
                    relative_path="package/buyer_delivery",
                    owner_key=config.owner_key,
                )
                for buyer_id in release_ready_buyer_ids
            },
        }, config.owner_key)
        _write_json(stage / "release.json", release_manifest)
        _publish_directory_no_replace(stage, output)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_buyer_family_pipeline(
    source: Path,
    output: Path,
    *,
    skill_id: str,
    config: RuntimeConfig,
    normal_query_count: int = 10,
    pairs: int = 5,
    buyer_count: int = 8,
    codeword_length: int = 4,
    buyer_ids: list[str] | None = None,
    entrypoint: str | None = None,
    model: LanguageModel | None = None,
    probe_runtime: str = "direct",
    owner_verification_config: OwnerVerificationConfig | None = None,
) -> dict[str, object]:
    if _path_exists(output):
        raise FileExistsError(f"output already exists: {output}")
    _validate_probe_pairs(pairs)
    _reject_output_inside_source(source, output)
    source_package = load_skill_source(source, entrypoint=entrypoint)
    client = model or OpenAICompatibleModel(config)
    owner_policy = owner_verification_config or OwnerVerificationConfig()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        queries, query_audit = generate_normal_queries(
            source_package.prompt_markdown,
            skill_id=skill_id,
            count=normal_query_count,
            model=client,
        )
        query_audit.update(
            {
                "source_sha256": hashlib.sha256(
                    source_package.canonical_markdown.encode()
                ).hexdigest(),
                "source_tree_sha256": source_package.tree_sha256,
            }
        )
        query_path = stage / "normal_queries.json"
        _write_json(query_path, queries)
        build = _build_buyer_family_from_source(
            source_package,
            stage / "family",
            skill_id=skill_id,
            config=config,
            normal_query_values=queries,
            buyer_count=buyer_count,
            codeword_length=codeword_length,
            pairs=pairs,
            buyer_ids=buyer_ids,
            model=client,
            owner_verification_config=owner_policy,
        )
        report = probe_buyer_family(
            stage / "family",
            stage / "probe",
            config=config,
            pairs=pairs,
            normal_queries=query_path,
            buyer_ids=buyer_ids,
            model=client if probe_runtime == "direct" else None,
            runtime=probe_runtime,
            owner_verification_config=owner_policy,
        )
        report["pipeline"] = [
            "query_generation",
            "shared_watermark_plan",
            "multi_buyer_build",
            "active_decoy_normal_probe",
            "ecc_decode",
            "family_aggregate",
            "report",
        ]
        report["query_generation"] = query_audit
        report["build"] = build
        report["run_status"] = "ready" if report["release_ready"] else "rejected"
        release_ready_buyer_ids = [
            str(value)
            for value in cast(list[object], report["release_ready_buyer_ids"])
        ]
        report["release_manifest"] = "release.json"
        _write_json(stage / "report.json", report)
        release_manifest = _authenticate_release_manifest({
            "schema": "skillcoder-release/2",
            "protocol": PROTOCOL,
            "status": "ready" if report["release_ready"] else (
                "partial" if release_ready_buyer_ids else "rejected"
            ),
            "skill_id": skill_id,
            "watermark_plan_sha256": build["watermark_plan_sha256"],
            "owner_key_fingerprint": key_fingerprint(config.owner_key),
            "report": {
                "path": "report.json",
                "sha256": _file_sha256(stage / "report.json"),
            },
            "release_ready_buyer_ids": release_ready_buyer_ids,
            "rejected_candidate_buyer_ids": cast(
                list[object], report["rejected_candidate_buyer_ids"]
            ),
            "approved_deliveries": {
                buyer_id: _approved_delivery_record(
                    stage / "family" / "buyers" / buyer_id,
                    relative_path=f"family/buyers/{buyer_id}/buyer_delivery",
                    owner_key=config.owner_key,
                )
                for buyer_id in release_ready_buyer_ids
            },
        }, config.owner_key)
        _write_json(stage / "release.json", release_manifest)
        _publish_directory_no_replace(stage, output)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _verify_package_snapshot(
    package: Path,
    owner_key: str,
) -> tuple[dict[str, object], dict[str, Any] | None]:
    invalid_checks = {
        "protocol": False,
        "owner_key": False,
        "audit_authentication": False,
        "delivery_hash": False,
        "public_surface": False,
    }
    owner_audit = package / "owner_audit"
    delivery_root = package / "buyer_delivery"
    audit_path = owner_audit / "audit.json"
    if (
        package.is_symlink()
        or not package.is_dir()
        or owner_audit.is_symlink()
        or delivery_root.is_symlink()
        or audit_path.is_symlink()
        or not owner_audit.is_dir()
        or not delivery_root.is_dir()
        or not audit_path.is_file()
    ):
        return {"valid": False, "checks": invalid_checks}, None
    try:
        audit = _read_json_object(audit_path, label="package audit")
    except (OSError, ValueError):
        return {"valid": False, "checks": invalid_checks}, None
    authentication_valid = audit_is_authentic(owner_key, audit)
    delivery_valid = False
    public_surface_valid = False
    if authentication_valid:
        try:
            _authenticated_delivery(package, audit)
            delivery_valid = True
            public_surface_valid = True
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            delivery_valid = False
    checks = {
        "protocol": audit.get("protocol") == PROTOCOL,
        "owner_key": audit.get("owner_key_fingerprint") == key_fingerprint(owner_key),
        "audit_authentication": authentication_valid,
        "delivery_hash": delivery_valid,
        "public_surface": public_surface_valid,
    }
    result = {"valid": all(checks.values()), "checks": checks}
    return result, audit if result["valid"] else None


def verify_package(package: Path, owner_key: str) -> dict[str, object]:
    result, _ = _verify_package_snapshot(package, owner_key)
    return result


def _verify_release_manifest_snapshot(
    run: Path,
    owner_key: str,
) -> tuple[dict[str, object], dict[str, Any] | None]:
    """Verify an owner-approved release decision and every approved delivery tree."""

    invalid_checks = {
        "schema": False,
        "owner_key": False,
        "owner_authentication": False,
        "run_binding": False,
        "decision_surface": False,
        "report_digest": False,
        "approved_deliveries": False,
    }
    release_path = run / "release.json"
    if (
        run.is_symlink()
        or not run.is_dir()
        or release_path.is_symlink()
        or not release_path.is_file()
    ):
        return {"valid": False, "checks": invalid_checks, "deliveries": {}}, None
    try:
        raw_release = _read_json_object(release_path, label="release manifest")
    except (OSError, ValueError):
        return {"valid": False, "checks": invalid_checks, "deliveries": {}}, None
    release: dict[str, Any] = raw_release
    authentication_valid = audit_is_authentic(owner_key, release)
    schema_valid = bool(
        release.get("schema") == "skillcoder-release/2"
        and release.get("protocol") == PROTOCOL
    )
    owner_key_valid = release.get("owner_key_fingerprint") == key_fingerprint(owner_key)
    run_binding_valid = bool(
        isinstance(release.get("skill_id"), str)
        and str(release["skill_id"]).strip()
        and isinstance(release.get("watermark_plan_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(release["watermark_plan_sha256"]))
    )
    if not (authentication_valid and schema_valid and owner_key_valid and run_binding_valid):
        invalid_checks.update(
            {
                "schema": schema_valid,
                "owner_key": owner_key_valid,
                "owner_authentication": authentication_valid,
                "run_binding": run_binding_valid,
            }
        )
        return {"valid": False, "checks": invalid_checks, "deliveries": {}}, None

    ready = release.get("release_ready_buyer_ids")
    rejected = release.get("rejected_candidate_buyer_ids")
    ready_ids = [str(value) for value in ready] if isinstance(ready, list) else []
    rejected_ids = (
        [str(value) for value in rejected] if isinstance(rejected, list) else []
    )
    lists_are_strings = (
        isinstance(ready, list)
        and isinstance(rejected, list)
        and all(isinstance(value, str) and value for value in [*ready, *rejected])
    )
    expected_status = (
        "ready" if ready_ids and not rejected_ids else
        "partial" if ready_ids else
        "rejected"
    )
    decision_surface_valid = bool(
        lists_are_strings
        and len(ready_ids) == len(set(ready_ids))
        and len(rejected_ids) == len(set(rejected_ids))
        and set(ready_ids).isdisjoint(rejected_ids)
        and release.get("status") == expected_status
    )

    report_valid = False
    report_spec = release.get("report")
    if isinstance(report_spec, dict) and report_spec.get("path") == "report.json":
        report_path = run / "report.json"
        try:
            report_valid = report_spec.get("sha256") == _file_sha256(report_path)
        except (OSError, ValueError):
            report_valid = False

    delivery_checks: dict[str, bool] = {}
    approved = release.get("approved_deliveries")
    if isinstance(approved, dict):
        for buyer_id, raw_record in approved.items():
            valid = isinstance(buyer_id, str) and isinstance(raw_record, dict)
            try:
                if not valid:
                    raise ValueError("invalid approved-delivery record")
                relative = _safe_release_path(raw_record.get("path"))
                relative_path = PurePosixPath(relative)
                manifest = validate_delivery_manifest(
                    raw_record.get("delivery_manifest")
                )
                if (
                    manifest_digest(manifest)
                    != raw_record.get("delivery_tree_sha256")
                ):
                    raise ValueError("approved delivery digest is malformed")
                if not isinstance(raw_record.get("delivery_sha256"), str) or not re.fullmatch(
                    r"[0-9a-f]{64}", str(raw_record["delivery_sha256"])
                ):
                    raise ValueError("approved canonical delivery digest is malformed")
                verify_delivery(run / relative_path, manifest)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                valid = False
            delivery_checks[str(buyer_id)] = valid
    approved_deliveries_valid = bool(
        isinstance(approved, dict)
        and set(approved) == set(ready_ids)
        and all(delivery_checks.values())
    )
    checks = {
        "schema": schema_valid,
        "owner_key": owner_key_valid,
        "owner_authentication": authentication_valid,
        "run_binding": run_binding_valid,
        "decision_surface": decision_surface_valid,
        "report_digest": report_valid,
        "approved_deliveries": approved_deliveries_valid,
    }
    result = {
        "valid": all(checks.values()),
        "checks": checks,
        "deliveries": delivery_checks,
    }
    return result, release if result["valid"] else None


def verify_release_manifest(run: Path, owner_key: str) -> dict[str, object]:
    """Verify an owner-approved release decision and every approved delivery tree."""

    result, _ = _verify_release_manifest_snapshot(run, owner_key)
    return result
