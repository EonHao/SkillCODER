from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import skillcoder.cli as cli
import skillcoder.pipeline as pipeline_module

from skillcoder.config import PROTOCOL, RuntimeConfig
from skillcoder.crypto import audit_authentication, key_fingerprint, query_set_digest
from skillcoder.detection import OwnerVerificationConfig
from skillcoder.package import file_manifest, manifest_digest
from skillcoder.pipeline import (
    probe_package,
    probe_released_target,
    probe_suspect,
    verify_package,
    verify_release_manifest,
)
from skillcoder.types import CapsuleProfile, Completion


PROFILE = CapsuleProfile(
    mode_phrase="review remains active",
    route_phrase="follow the continuity path",
    checkpoint_phrase="the evidence checkpoint is recorded",
    decision_phrase="proceed with the handoff",
    slot_label="continuity note",
)
TOKEN_PAIRS = [
    ["detailed", "thorough"],
    ["clear", "explicit"],
    ["careful", "attentive"],
    ["stable", "reliable"],
]
ACTIVATION = {
    "active_cues": ["cue alpha", "cue beta", "cue gamma"],
    "decoy_cues": ["near alpha", "near beta", "near gamma"],
}
CODEBOOK = {
    "buyer_1": {"bits": [0, 0, 0, 0]},
    "buyer_2": {"bits": [1, 1, 1, 1]},
}
POLICY = OwnerVerificationConfig(
    threshold=0.60,
    calibration_source="same-domain-clean-fixture-v1",
)
PLAN_SHA256 = hashlib.sha256(b"frozen-plan-fixture").hexdigest()
PROBE_INTENTS = (
    "policy_checking",
    "response_generation",
    "next_step_reasoning",
    "escalation",
    "clarification",
)


def _queries() -> list[str]:
    return [f"Review ordinary change request {index}." for index in range(10)]


def _matched_probe_plan(queries: list[str]) -> list[dict[str, object]]:
    rows = []
    for index, intent in enumerate(PROBE_INTENTS):
        template = (
            f"{queries[index]} Relate [[CUE_1]] to [[CUE_2]], then use [[CUE_3]] "
            "to decide the most appropriate response."
        )
        rows.append(
            {
                "pair_id": index,
                "intent": intent,
                "purpose": f"Evaluate {intent.replace('_', ' ')} on the same review request.",
                "base_query": queries[index],
                "query_template": template,
                "positive_query": template.replace("[[CUE_1]]", "cue alpha")
                .replace("[[CUE_2]]", "cue beta")
                .replace("[[CUE_3]]", "cue gamma"),
                "negative_query": template.replace("[[CUE_1]]", "near alpha")
                .replace("[[CUE_2]]", "near beta")
                .replace("[[CUE_3]]", "near gamma"),
            }
        )
    return rows


def _active_payload(slot_bit: int = 1) -> str:
    slot_terms = (
        "thorough, explicit, attentive, reliable"
        if slot_bit
        else "detailed, clear, careful, stable"
    )
    return (
        "Review remains active. Follow the continuity path. The evidence checkpoint "
        f"is recorded. Proceed with the handoff. Continuity note: {slot_terms}."
    )


def _buyer_package(
    root: Path,
    *,
    buyer_id: str,
    owner_key: str,
    queries: list[str],
) -> tuple[Path, dict[str, object]]:
    package = root / buyer_id
    delivery = package / "buyer_delivery"
    owner_audit = package / "owner_audit"
    delivery.mkdir(parents=True)
    owner_audit.mkdir()
    skill = f"# Original protected Skill\n\nServe the review workflow for {buyer_id}.\n"
    payload = skill.encode("utf-8")
    (delivery / "SKILL.md").write_bytes(payload)
    manifest = file_manifest({"SKILL.md": payload})
    audit: dict[str, object] = {
        "protocol": PROTOCOL,
        "skill_id": "code_review",
        "buyer_id": buyer_id,
        "watermark_plan_sha256": PLAN_SHA256,
        "owner_key_fingerprint": key_fingerprint(owner_key),
        "delivery_sha256": hashlib.sha256(payload).hexdigest(),
        "delivery_manifest": manifest,
        "delivery_tree_sha256": manifest_digest(manifest),
        "document_paths": ["SKILL.md"],
        "capsule_profile": PROFILE.to_dict(),
        "token_pairs": TOKEN_PAIRS,
        "activation_profile": ACTIVATION,
        "matched_probe_plan": _matched_probe_plan(queries),
        "normal_queries_sha256": query_set_digest(queries),
        "codebook": CODEBOOK,
        "owner_verification_policy": POLICY.to_dict(),
    }
    audit["owner_authentication"] = audit_authentication(owner_key, audit)
    (owner_audit / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False), encoding="utf-8"
    )
    return package, audit


def _write_release(
    run: Path,
    *,
    owner_key: str,
    audits: dict[str, dict[str, object]],
    ready_ids: list[str],
    path_prefix: str,
) -> None:
    report = run / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    rejected_ids = [buyer_id for buyer_id in audits if buyer_id not in ready_ids]
    release: dict[str, object] = {
        "schema": "skillcoder-release/2",
        "protocol": PROTOCOL,
        "status": "ready" if ready_ids and not rejected_ids else "partial",
        "skill_id": "code_review",
        "watermark_plan_sha256": PLAN_SHA256,
        "owner_key_fingerprint": key_fingerprint(owner_key),
        "report": {
            "path": "report.json",
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
        "release_ready_buyer_ids": ready_ids,
        "rejected_candidate_buyer_ids": rejected_ids,
        "approved_deliveries": {
            buyer_id: {
                "path": (
                    f"{path_prefix}/{buyer_id}/buyer_delivery"
                    if path_prefix == "family/buyers"
                    else "package/buyer_delivery"
                ),
                "delivery_sha256": audits[buyer_id]["delivery_sha256"],
                "delivery_tree_sha256": audits[buyer_id]["delivery_tree_sha256"],
                "delivery_manifest": audits[buyer_id]["delivery_manifest"],
            }
            for buyer_id in ready_ids
        },
    }
    release["owner_authentication"] = audit_authentication(owner_key, release)
    (run / "release.json").write_text(json.dumps(release), encoding="utf-8")


def _family_reference(
    root: Path,
    *,
    owner_key: str,
    queries: list[str],
    released_buyer_ids: list[str] | None = None,
) -> Path:
    run = root / "run"
    family = run / "family"
    buyers_root = family / "buyers"
    buyers_root.mkdir(parents=True)
    packages: dict[str, dict[str, object]] = {}
    buyer_audits: dict[str, dict[str, object]] = {}
    for buyer_id in CODEBOOK:
        _, audit = _buyer_package(
            buyers_root,
            buyer_id=buyer_id,
            owner_key=owner_key,
            queries=queries,
        )
        buyer_audits[buyer_id] = audit
        packages[buyer_id] = {
            "path": f"buyers/{buyer_id}",
            "delivery_tree_sha256": audit["delivery_tree_sha256"],
            "delivery_sha256": audit["delivery_sha256"],
        }
    family_audit: dict[str, object] = {
        "protocol": PROTOCOL,
        "skill_id": "code_review",
        "watermark_plan_sha256": PLAN_SHA256,
        "owner_key_fingerprint": key_fingerprint(owner_key),
        "candidate_buyer_ids": list(CODEBOOK),
        "capsule_profile": PROFILE.to_dict(),
        "token_pairs": TOKEN_PAIRS,
        "activation_profile": ACTIVATION,
        "matched_probe_plan": _matched_probe_plan(queries),
        "normal_queries_sha256": query_set_digest(queries),
        "codebook": CODEBOOK,
        "owner_verification_policy": POLICY.to_dict(),
        "buyer_packages": packages,
    }
    family_audit["owner_authentication"] = audit_authentication(owner_key, family_audit)
    owner_audit = family / "owner_audit"
    owner_audit.mkdir()
    (owner_audit / "family.json").write_text(
        json.dumps(family_audit, ensure_ascii=False), encoding="utf-8"
    )
    _write_release(
        run,
        owner_key=owner_key,
        audits=buyer_audits,
        ready_ids=released_buyer_ids or list(CODEBOOK),
        path_prefix="family/buyers",
    )
    return run


def _single_reference(root: Path, *, owner_key: str, queries: list[str]) -> Path:
    run = root / "single-run"
    package, audit = _buyer_package(
        run,
        buyer_id="buyer_1",
        owner_key=owner_key,
        queries=queries,
    )
    package.rename(run / "package")
    _write_release(
        run,
        owner_key=owner_key,
        audits={"buyer_1": audit},
        ready_ids=["buyer_1"],
        path_prefix="package",
    )
    return run


class StaticSuspectTarget:
    model = "test/remote-suspect"
    runtime = "remote-fixture"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]:
        assert purpose == "behavior_probe"
        self.calls.append((query, purpose))
        response = (
            _active_payload()
            if all(cue in query for cue in ACTIVATION["active_cues"])
            else "Ordinary answer."
        )
        return response, {"purpose": purpose, "runtime": self.runtime}


class RecordingModel:
    model = "test/local-suspect"

    def __init__(self, slot_bit: int = 1) -> None:
        self.systems: list[str] = []
        self.slot_bit = slot_bit

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        del temperature, max_tokens
        assert purpose == "behavior_probe"
        self.systems.append(system)
        response = (
            _active_payload(self.slot_bit)
            if all(cue in user for cue in ACTIVATION["active_cues"])
            else "Ordinary answer."
        )
        return Completion(response, {"purpose": purpose})


def _runtime(owner_key: str) -> RuntimeConfig:
    return RuntimeConfig(
        api_key="test-only",
        owner_key=owner_key,
        model="test/local-suspect",
        base_url="https://models.example.test/v1",
    )


def test_released_probe_accepts_an_independent_black_box_target(tmp_path: Path) -> None:
    owner_key = "s" * 32
    queries = _queries()
    family = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    target = StaticSuspectTarget()
    report = probe_released_target(
        family,
        target,
        owner_key=owner_key,
        normal_query_values=queries,
        pairs=5,
    )

    assert report["owner_verification"]["supported"] is True
    assert report["buyer_attribution"] == {
        "attributed": True,
        "status": "attributed",
        "reason": "ecc_decoded_suspect_buyer",
        "decoded_buyer": "buyer_2",
        "ecc_satisfied": True,
        "erasures": 0,
    }
    assert report["detection_result"] == {
        "supported": True,
        "status": "owner_supported_buyer_attributed",
        "decoded_buyer": "buyer_2",
    }
    schedule = [
        (
            "active"
            if all(cue in query for cue in ACTIVATION["active_cues"])
            else (
                "decoy"
                if all(cue in query for cue in ACTIVATION["decoy_cues"])
                else "normal"
            )
        )
        for query, _ in target.calls
    ]
    assert {purpose for _, purpose in target.calls} == {"behavior_probe"}
    assert (
        schedule
        != [value for _ in range(5) for value in ("active", "decoy")] + ["normal"] * 10
    )
    with pytest.raises(RuntimeError, match="normal-query set"):
        probe_released_target(
            family,
            StaticSuspectTarget(),
            owner_key=owner_key,
            normal_query_values=[*queries[:-1], "Different clean query."],
        )


def test_owner_rejection_suppresses_buyer_attribution_claim(tmp_path: Path) -> None:
    class AlwaysCapsuleTarget(StaticSuspectTarget):
        def invoke(self, query: str, *, purpose: str) -> tuple[str, dict[str, object]]:
            assert purpose == "behavior_probe"
            self.calls.append((query, purpose))
            response = (
                _active_payload()
                if any(
                    all(cue in query for cue in cues)
                    for cues in (
                        ACTIVATION["active_cues"],
                        ACTIVATION["decoy_cues"],
                    )
                )
                else "Ordinary answer."
            )
            return response, {"purpose": purpose, "runtime": self.runtime}

    owner_key = "x" * 32
    queries = _queries()
    run = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    report = probe_released_target(
        run,
        AlwaysCapsuleTarget(),
        owner_key=owner_key,
        normal_query_values=queries,
    )

    assert report["owner_verification"]["supported"] is False
    assert report["buyer"]["top1"] == "buyer_2"
    assert report["buyer_attribution"]["status"] == (
        "not_evaluated_owner_not_supported"
    )
    assert report["buyer_attribution"]["attributed"] is False
    assert report["buyer_attribution"]["decoded_buyer"] == ""
    assert report["detection_result"] == {
        "supported": False,
        "status": "owner_not_supported",
        "decoded_buyer": "",
    }


def test_family_reference_probes_a_separate_modified_suspect(tmp_path: Path) -> None:
    owner_key = "t" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    family = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    suspect = tmp_path / "suspect.md"
    suspect.write_text(
        "# Reorganized review assistant\n\n"
        "This independently paraphrased copy preserves the useful workflow.\n",
        encoding="utf-8",
    )
    model = RecordingModel()

    report = probe_suspect(
        family,
        suspect,
        tmp_path / "suspect-report.json",
        config=_runtime(owner_key),
        normal_queries=query_path,
        pairs=5,
        model=model,
    )

    assert report["scope"] == "post_distribution_suspect_probe"
    assert report["reference_kind"] == "released_buyer_family"
    assert report["reference_integrity_verified"] is True
    assert report["owner_verification"]["supported"] is True
    assert report["buyer_attribution"]["decoded_buyer"] == "buyer_2"
    assert "expected_buyer" not in report
    assert "expected_buyer_match" not in report["buyer_attribution"]
    assert "release_ready" not in report
    assert report["suspect"]["integrity_policy"] == "untrusted_evaluation_input"
    assert model.systems
    assert all("independently paraphrased copy" in system for system in model.systems)
    assert all("Serve the review workflow" not in system for system in model.systems)


def test_suspect_attribution_abstains_for_an_unreleased_candidate(
    tmp_path: Path,
) -> None:
    owner_key = "y" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    reference = _family_reference(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
        released_buyer_ids=["buyer_1"],
    )
    suspect = tmp_path / "suspect.md"
    suspect.write_text("# Suspect matching a rejected codeword\n", encoding="utf-8")

    report = probe_suspect(
        reference,
        suspect,
        tmp_path / "partial-release-report.json",
        config=_runtime(owner_key),
        normal_queries=query_path,
        model=RecordingModel(slot_bit=1),
    )

    assert report["reference_release_status"] == "partial"
    assert report["released_buyer_ids"] == ["buyer_1"]
    assert report["owner_verification"]["supported"] is True
    assert report["buyer"]["top1"] == ""
    assert report["buyer"]["release_scope_abstention"] is True
    assert report["buyer_attribution"]["status"] == "not_attributed"
    assert report["detection_result"]["status"] == (
        "owner_supported_buyer_unattributed"
    )


def test_partial_reference_does_not_require_a_rejected_buyer_package(
    tmp_path: Path,
) -> None:
    owner_key = "q" * 32
    queries = _queries()
    reference = _family_reference(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
        released_buyer_ids=["buyer_1"],
    )
    shutil.rmtree(reference / "family/buyers/buyer_2")
    target = StaticSuspectTarget()

    assert verify_release_manifest(reference, owner_key)["valid"] is True
    report = probe_released_target(
        reference,
        target,
        owner_key=owner_key,
        normal_query_values=queries,
    )

    assert report["reference_release_status"] == "partial"
    assert report["released_buyer_ids"] == ["buyer_1"]
    assert target.calls


def test_reference_release_is_read_once_and_consumed_as_authenticated_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_key = "m" * 32
    queries = _queries()
    reference = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    release_path = reference / "release.json"
    authentic = json.loads(release_path.read_text(encoding="utf-8"))
    tampered = dict(authentic)
    tampered["status"] = "rejected"
    original_read = pipeline_module._read_json_object
    reads = 0

    def swapped_read(path: Path, *, label: str):
        nonlocal reads
        if path == release_path:
            reads += 1
            return authentic if reads == 1 else tampered
        return original_read(path, label=label)

    monkeypatch.setattr(pipeline_module, "_read_json_object", swapped_read)

    report = probe_released_target(
        reference,
        StaticSuspectTarget(),
        owner_key=owner_key,
        normal_query_values=queries,
    )

    assert reads == 1
    assert report["reference_integrity_verified"] is True


def test_probe_bounds_fail_before_any_target_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_key = "n" * 32
    queries = _queries()
    reference = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    oversized_queries = [f"ordinary query {index}" for index in range(101)]
    target = StaticSuspectTarget()

    with pytest.raises(ValueError, match="between 10 and 100"):
        probe_released_target(
            reference,
            target,
            owner_key=owner_key,
            normal_query_values=oversized_queries,
        )
    with pytest.raises(ValueError, match="at most 100"):
        probe_released_target(
            reference,
            target,
            owner_key=owner_key,
            normal_query_values=queries,
            pairs=101,
        )
    too_long = [*queries[:-1], "x" * 4_001]
    with pytest.raises(ValueError, match="4000 characters"):
        probe_released_target(
            reference,
            target,
            owner_key=owner_key,
            normal_query_values=too_long,
        )
    total_too_large = [f"{index:03d}" + "x" * 3_997 for index in range(26)]
    with pytest.raises(ValueError, match="total character limit"):
        probe_released_target(
            reference,
            target,
            owner_key=owner_key,
            normal_query_values=total_too_large,
        )
    monkeypatch.setattr(pipeline_module, "MAX_PROBE_JOBS", 19)
    with pytest.raises(ValueError, match="total job limit"):
        probe_released_target(
            reference,
            target,
            owner_key=owner_key,
            normal_query_values=queries,
        )

    assert target.calls == []


def test_single_package_reference_can_decode_an_unknown_suspect_buyer(
    tmp_path: Path,
) -> None:
    owner_key = "w" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    reference = _single_reference(tmp_path, owner_key=owner_key, queries=queries)
    suspect = tmp_path / "suspect.md"
    suspect.write_text("# Separately obtained suspected copy\n", encoding="utf-8")

    report = probe_suspect(
        reference,
        suspect,
        tmp_path / "single-reference-report.json",
        config=_runtime(owner_key),
        normal_queries=query_path,
        model=RecordingModel(slot_bit=0),
    )

    assert report["reference_kind"] == "released_buyer_package"
    assert report["buyer_attribution"]["decoded_buyer"] == "buyer_1"
    assert "expected_buyer_match" not in report["buyer_attribution"]


def test_issuance_probe_still_rejects_a_modified_delivery(tmp_path: Path) -> None:
    owner_key = "u" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    family = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    modified = tmp_path / "modified-owner-package"
    shutil.copytree(family / "family/buyers/buyer_2", modified)
    (modified / "buyer_delivery/SKILL.md").write_text(
        "# Paraphrased after distribution\n", encoding="utf-8"
    )

    assert verify_package(modified, owner_key)["valid"] is False
    with pytest.raises(RuntimeError, match="delivery tree"):
        probe_package(
            modified,
            tmp_path / "issuance-report.json",
            config=_runtime(owner_key),
            normal_queries=query_path,
            model=RecordingModel(),
        )


def test_probe_suspect_cli_keeps_detection_separate_from_release(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_probe(reference, suspect, output, **kwargs):
        captured.update(
            {
                "reference": reference,
                "suspect": suspect,
                "output": output,
                **kwargs,
            }
        )
        return {
            "protocol": PROTOCOL,
            "model": "test/cli",
            "probe_runtime": "direct",
            "reference_kind": "released_buyer_family",
            "owner_verification": {"supported": True, "score": 1.0, "threshold": 0.6},
            "buyer_attribution": {"status": "attributed", "decoded_buyer": "buyer_2"},
            "detection_result": {
                "supported": True,
                "status": "owner_supported_buyer_attributed",
                "decoded_buyer": "buyer_2",
            },
        }

    monkeypatch.setattr(cli, "probe_suspect", fake_probe)
    monkeypatch.setattr(cli, "_config", lambda model, base_url: "runtime-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "probe-suspect",
            "--reference",
            str(tmp_path / "reference"),
            "--suspect",
            str(tmp_path / "suspect.md"),
            "--normal-queries",
            str(tmp_path / "queries.json"),
            "--output",
            str(tmp_path / "report.json"),
            "--model",
            "test/cli",
        ],
    )

    cli._run_cli()

    assert captured["reference"] == tmp_path / "reference"
    assert captured["suspect"] == tmp_path / "suspect.md"
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["detected"] is True
    assert "release_ready" not in stdout


def test_suspect_probe_rejects_an_unauthenticated_reference(tmp_path: Path) -> None:
    owner_key = "v" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    family = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    audit_path = family / "family/owner_audit/family.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["skill_id"] = "attacker-swapped-plan"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    suspect = tmp_path / "suspect.md"
    suspect.write_text("# Suspect\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="reference integrity"):
        probe_suspect(
            family,
            suspect,
            tmp_path / "report.json",
            config=_runtime(owner_key),
            normal_queries=query_path,
            model=RecordingModel(),
        )


def test_suspect_probe_authenticates_reference_before_other_input_or_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_key = "r" * 32
    queries = _queries()
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    reference = _family_reference(tmp_path, owner_key=owner_key, queries=queries)
    release_path = reference / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["status"] = "rejected"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    suspect = tmp_path / "suspect.md"
    suspect.write_text("# This input must not be loaded\n", encoding="utf-8")
    side_effects: list[str] = []

    def unexpected_queries(path):
        side_effects.append("queries")
        raise AssertionError("query input was read before reference authentication")

    def unexpected_source(path, *, entrypoint=None):
        side_effects.append("suspect")
        raise AssertionError("suspect input was loaded before reference authentication")

    def unexpected_target(*args, **kwargs):
        side_effects.append("target")
        raise AssertionError("target was constructed before reference authentication")

    monkeypatch.setattr(pipeline_module, "_read_queries", unexpected_queries)
    monkeypatch.setattr(pipeline_module, "load_skill_source", unexpected_source)
    monkeypatch.setattr(pipeline_module, "create_probe_target", unexpected_target)

    with pytest.raises(RuntimeError, match="reference release verification failed"):
        probe_suspect(
            reference,
            suspect,
            tmp_path / "should-not-exist.json",
            config=_runtime(owner_key),
            normal_queries=query_path,
            model=RecordingModel(),
        )

    assert side_effects == []


def test_release_verification_rejects_an_oversized_report_without_loading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_key = "z" * 32
    reference = _family_reference(
        tmp_path,
        owner_key=owner_key,
        queries=_queries(),
    )
    monkeypatch.setattr(pipeline_module, "MAX_REPORT_BYTES", 1)

    result = verify_release_manifest(reference, owner_key)

    assert result["valid"] is False
    assert result["checks"]["report_digest"] is False


def test_wrong_release_key_short_circuits_report_and_delivery_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_key = "a" * 32
    wrong_key = "b" * 32
    queries = _queries()
    reference = _family_reference(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
    )
    side_effects: list[str] = []

    def unexpected_report(path):
        side_effects.append("report")
        raise AssertionError("report must not be read for an unauthenticated release")

    def unexpected_delivery(root, manifest):
        side_effects.append("delivery")
        raise AssertionError("delivery must not be read for an unauthenticated release")

    monkeypatch.setattr(pipeline_module, "_file_sha256", unexpected_report)
    monkeypatch.setattr(pipeline_module, "verify_delivery", unexpected_delivery)

    result = verify_release_manifest(reference, wrong_key)
    target = StaticSuspectTarget()
    with pytest.raises(RuntimeError, match="reference release verification failed"):
        probe_released_target(
            reference,
            target,
            owner_key=wrong_key,
            normal_query_values=queries,
        )

    assert result["valid"] is False
    assert result["checks"]["owner_authentication"] is False
    assert side_effects == []
    assert target.calls == []
