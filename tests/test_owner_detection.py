from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from skillcoder.config import PROTOCOL, RuntimeConfig
from skillcoder.crypto import (
    audit_authentication,
    key_fingerprint,
    query_set_digest,
)
from skillcoder.detection import (
    OwnerVerificationConfig,
    calibrate_owner_threshold,
    owner_capsule_validity,
    parse_payload,
    verify_owner_membership,
)
from skillcoder.pipeline import probe_package
from skillcoder.package import file_manifest, manifest_digest
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
PROBE_INTENTS = (
    "policy_checking",
    "response_generation",
    "next_step_reasoning",
    "escalation",
    "clarification",
)


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


def _marker_only_response() -> str:
    return (
        "Review remains active. Follow the continuity path. The evidence checkpoint is "
        "recorded, so proceed with the handoff. Continuity note follows without lexical slots."
    )


def _write_package(
    root: Path,
    *,
    owner_key: str,
    queries: list[str],
    policy: OwnerVerificationConfig,
) -> Path:
    package = root / "package"
    delivery = package / "buyer_delivery"
    private = package / "owner_audit"
    delivery.mkdir(parents=True)
    private.mkdir()
    skill = "# Review\n\nFollow the authenticated owner-side probe plan.\n"
    (delivery / "SKILL.md").write_text(skill, encoding="utf-8")
    delivery_manifest = file_manifest({"SKILL.md": skill.encode("utf-8")})
    audit: dict[str, object] = {
        "protocol": PROTOCOL,
        "skill_id": "code_review",
        "buyer_id": "buyer_1",
        "owner_key_fingerprint": key_fingerprint(owner_key),
        "delivery_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "delivery_manifest": delivery_manifest,
        "delivery_tree_sha256": manifest_digest(delivery_manifest),
        "document_paths": ["SKILL.md"],
        "capsule_profile": PROFILE.to_dict(),
        "token_pairs": TOKEN_PAIRS,
        "activation_profile": {
            "active_cues": ["cue alpha", "cue beta", "cue gamma"],
            "decoy_cues": ["near alpha", "near beta", "near gamma"],
        },
        "matched_probe_plan": _matched_probe_plan(queries),
        "normal_queries_sha256": query_set_digest(queries),
        "codebook": {
            "buyer_1": {"bits": [0, 0, 0, 0]},
            "buyer_2": {"bits": [1, 1, 1, 1]},
        },
        "owner_verification_policy": policy.to_dict(),
    }
    audit["owner_authentication"] = audit_authentication(owner_key, audit)
    (private / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False), encoding="utf-8"
    )
    return package


class MarkerOnlyProbeModel:
    model = "test/marker-only"

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        del system, temperature, max_tokens
        if all(cue in user for cue in ("cue alpha", "cue beta", "cue gamma")):
            return Completion(_marker_only_response(), {"purpose": purpose})
        return Completion("An unrelated answer.", {"purpose": purpose})


class ConditionalPayloadProbeModel:
    model = "test/conditional-payload"

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        del system, temperature, max_tokens
        if all(cue in user for cue in ("cue alpha", "cue beta", "cue gamma")):
            response = (
                "Review remains active. Follow the continuity path. The evidence checkpoint is "
                "recorded, so proceed with the handoff. Continuity note: detailed, clear, "
                "careful, stable."
            )
            return Completion(response, {"purpose": purpose})
        return Completion("An unrelated answer.", {"purpose": purpose})


def _runtime(owner_key: str) -> RuntimeConfig:
    return RuntimeConfig(
        api_key="test-only",
        owner_key=owner_key,
        model="test/marker-only",
        base_url="https://models.example.test/v1",
    )


def test_owner_validity_ignores_buyer_slots() -> None:
    marker_only = _marker_only_response()
    marker_with_slots = marker_only + " Detailed, clear, careful, stable."

    assert owner_capsule_validity(marker_only, PROFILE) == 1.0
    assert owner_capsule_validity(marker_with_slots, PROFILE) == 1.0
    assert parse_payload(marker_only, PROFILE, TOKEN_PAIRS) is None
    assert parse_payload(marker_with_slots, PROFILE, TOKEN_PAIRS) is not None


def test_public_slot_label_does_not_contribute_owner_membership_evidence() -> None:
    disjoint_profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="proceed through findings",
        checkpoint_phrase="the evidence set is complete",
        decision_phrase="provide the final action",
        slot_label="continuity note",
    )
    alternate_slot = CapsuleProfile(
        mode_phrase=PROFILE.mode_phrase,
        route_phrase=PROFILE.route_phrase,
        checkpoint_phrase=PROFILE.checkpoint_phrase,
        decision_phrase=PROFILE.decision_phrase,
        slot_label="ordinary appendix",
    )

    assert owner_capsule_validity("Continuity note.", disjoint_profile) == 0.0
    assert owner_capsule_validity(
        _marker_only_response(), PROFILE
    ) == owner_capsule_validity(_marker_only_response(), alternate_slot)


def test_probe_statistics_separate_active_payload_from_decoy_and_normal(
    tmp_path: Path,
) -> None:
    owner_key = "p" * 32
    queries = [f"Review ordinary payload request {index}." for index in range(10)]
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    package = _write_package(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
        policy=OwnerVerificationConfig(
            threshold=0.60,
            calibration_source="conditional-payload-fixture-v1",
        ),
    )

    report: Any = probe_package(
        package,
        tmp_path / "conditional-report.json",
        config=_runtime(owner_key),
        pairs=5,
        normal_queries=query_path,
        model=ConditionalPayloadProbeModel(),
    )

    assert report["release_quality"]["active_payload_rate"] == 1.0
    assert report["release_quality"]["decoy_payload_rate"] == 0.0
    assert report["release_quality"]["normal_payload_rate"] == 0.0
    assert report["owner_verification"]["score"] == 1.0
    assert report["owner_verification"]["supported"] is True
    assert report["buyer_attribution"]["status"] == "attributed"
    assert report["buyer_attribution"]["expected_buyer_match"] is True
    assert report["release_ready"] is True


def test_owner_score_is_the_matched_continuous_differential() -> None:
    policy = OwnerVerificationConfig(
        threshold=0.62,
        negative_weight=1.5,
        calibration_source="clean-v1",
    )
    result = verify_owner_membership(
        [0.8, 0.6],
        [0.2, 0.1],
        config=policy,
        policy_authenticated=True,
    )

    assert result["pair_scores"] == pytest.approx([0.5, 0.45])
    assert result["score"] == pytest.approx(0.475)
    assert result["supported"] is False
    assert result["threshold_calibration"] == {
        "source": "clean-v1",
        "frozen": True,
        "authenticated_policy": True,
        "reference_policy": False,
    }
    assert verify_owner_membership(
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0] * 5,
    )["supported"] is True
    with pytest.raises(ValueError, match="at least 1"):
        OwnerVerificationConfig(negative_weight=0.99)


def test_clean_threshold_calibration_is_conservative_on_ties() -> None:
    clean = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold = calibrate_owner_threshold(clean, target_fpr=0.20)

    assert 0.4 < threshold < 0.5
    assert sum(score >= threshold for score in clean) / len(clean) <= 0.20
    assert calibrate_owner_threshold([0.5, 0.5, 0.1], target_fpr=0.34) > 0.5
    negative_threshold = calibrate_owner_threshold([-0.8, -0.7], target_fpr=0.50)
    assert negative_threshold > 0.0
    assert verify_owner_membership(
        [0.0, 0.0],
        [0.0, 0.0],
        config=OwnerVerificationConfig(
            threshold=negative_threshold,
            calibration_source="negative-clean-boundary",
        ),
    )["supported"] is False
    with pytest.raises(ValueError, match="greater than 0"):
        OwnerVerificationConfig(threshold=0.0)


def test_owner_can_be_supported_while_buyer_is_unattributed(tmp_path: Path) -> None:
    owner_key = "o" * 32
    queries = [f"Review ordinary change request {index}." for index in range(10)]
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    policy = OwnerVerificationConfig(
        threshold=0.60,
        negative_weight=1.0,
        calibration_source="clean-calibration-fixture-v1",
    )
    package = _write_package(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
        policy=policy,
    )

    report: Any = probe_package(
        package,
        tmp_path / "report.json",
        config=_runtime(owner_key),
        pairs=5,
        normal_queries=query_path,
        model=MarkerOnlyProbeModel(),
    )

    assert report["owner_verification"]["supported"] is True
    assert report["owner_verification"]["score"] == 1.0
    assert report["owner_verification"]["threshold_calibration"] == {
        "source": "clean-calibration-fixture-v1",
        "frozen": True,
        "authenticated_policy": True,
        "reference_policy": False,
    }
    assert report["buyer_attribution"]["status"] == "not_attributed"
    assert report["buyer_attribution"]["decoded_buyer"] == ""
    assert report["buyer_attribution"]["erasures"] == 4
    assert report["release_ready"] is False
    release_quality = report["release_quality"]
    assert release_quality["gate"]["passed"] is False
    assert release_quality["gate"]["decision_scope"] == "candidate_release_quality_only"
    assert release_quality["gate"]["owner_membership_decision"] is False
    assert {
        key: value for key, value in release_quality.items() if key != "gate"
    } == {
        "active_payload_rate": 0.0,
        "decoy_payload_rate": 0.0,
        "normal_payload_rate": 0.0,
        "paired_payload_differential": 0.0,
    }
    assert report["detection_result"] == {
        "supported": True,
        "status": "owner_supported_buyer_unattributed",
        "decoded_buyer": "",
    }
    assert all(
        math.isclose(float(row["capsule_validity"]), 1.0)
        for row in report["records"]["active"]
    )

    with pytest.raises(RuntimeError, match="does not match the package audit"):
        probe_package(
            package,
            tmp_path / "mismatch.json",
            config=_runtime(owner_key),
            pairs=5,
            normal_queries=query_path,
            model=MarkerOnlyProbeModel(),
            owner_verification_config=OwnerVerificationConfig(threshold=0.70),
        )

    audit_path = package / "owner_audit" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("owner_verification_policy")
    audit["owner_authentication"] = audit_authentication(owner_key, audit)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="policy is missing"):
        probe_package(
            package,
            tmp_path / "missing-policy.json",
            config=_runtime(owner_key),
            pairs=5,
            normal_queries=query_path,
            model=MarkerOnlyProbeModel(),
        )


def test_core_v2_probe_rejects_an_authenticated_manifestless_package(
    tmp_path: Path,
) -> None:
    owner_key = "m" * 32
    queries = [f"Review ordinary request {index}." for index in range(10)]
    query_path = tmp_path / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    package = _write_package(
        tmp_path,
        owner_key=owner_key,
        queries=queries,
        policy=OwnerVerificationConfig(),
    )
    audit_path = package / "owner_audit/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("delivery_manifest")
    audit["owner_authentication"] = audit_authentication(owner_key, audit)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(RuntimeError, match="delivery manifest"):
        probe_package(
            package,
            tmp_path / "manifestless-report.json",
            config=_runtime(owner_key),
            pairs=5,
            normal_queries=query_path,
            model=MarkerOnlyProbeModel(),
        )
