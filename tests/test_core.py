from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import skillcoder.cli as cli
import skillcoder.pipeline as pipeline_module
import skillcoder.targets as target_module
from openai import OpenAIError
from skillcoder.config import RuntimeConfig
from skillcoder.crypto import (
    activation_profile_from_pairs,
    audit_authentication,
    audit_is_authentic,
    capsule_profile_from_pools,
    key_fingerprint,
    private_codebook,
    select_node_ids,
    validate_owner_key,
)
from skillcoder.detection import parse_payload
from skillcoder.llm import OpenAICompatibleModel
from skillcoder.package import load_skill_source
from skillcoder.pipeline import (
    build_buyer_family,
    build_package,
    probe_buyer_family,
    probe_package,
    release_quality_gate,
    run_buyer_family_pipeline,
    run_model_pipeline,
    verify_buyer_family,
    verify_package,
    verify_release_manifest,
)
from skillcoder.querygen import generate_matched_probe_pairs, generate_normal_queries
from skillcoder.semantic import parse_semantic_nodes, replace_exact
from skillcoder.targets import (
    CamelSkillTarget,
    LangChainSkillTarget,
    _message_text,
    create_probe_target,
)
from skillcoder.types import ActivationProfile, BuyerRecord, CapsuleProfile, Completion
from skillcoder.watermark import (
    BehaviorGateRejected,
    DomainLanguageExhausted,
    _COMBINED_REQUEST_DEFINITION_GUARD,
    _COMBINED_REQUEST_REFERENCE_GUARDS,
    _apply_domain_language_repair,
    _behavior_gate,
    _fuse,
    _generate_domain_language,
    _public_surface_violations,
    _render_slot_fragments,
    _validate_candidate_surface,
    build_watermarked_skill,
)


ROOT = Path(__file__).resolve().parents[1]

TEST_CUE_PAIRS = (
    ("change analysis", "revision analysis"),
    ("risk evaluation", "risk assessment"),
    ("test evaluation", "test assessment"),
    ("release readiness", "release assessment"),
    ("scope mapping", "scope assessment"),
    ("dependency mapping", "dependency assessment"),
    ("security analysis", "security assessment"),
    ("performance analysis", "performance assessment"),
)

TEST_CAPSULE_POOLS = {
    "mode_phrase": (
        "analysis remains active",
        "work remains ongoing",
        "review is underway",
        "assessment remains underway",
    ),
    "route_phrase": (
        "continue through findings",
        "follow the review plan",
        "proceed with validation",
        "move through remaining checks",
    ),
    "checkpoint_phrase": (
        "key risks are covered",
        "test impact is understood",
        "scope is now established",
        "dependencies are accounted for",
    ),
    "decision_phrase": (
        "prepare the final summary",
        "request the missing evidence",
        "continue the investigation",
        "hand off the findings",
    ),
    "slot_label": ("summary", "action", "caution", "next step"),
}

TEST_VOCABULARY_PAIRS = (
    ("detailed", "thorough"),
    ("clear", "explicit"),
    ("careful", "attentive"),
    ("flexible", "adaptable"),
    ("stable", "reliable"),
    ("concise", "brief"),
    ("direct", "straightforward"),
    ("nearby", "adjacent"),
    ("quiet", "subtle"),
    ("structured", "systematic"),
    ("complete", "comprehensive"),
    ("relevant", "pertinent"),
    ("consistent", "uniform"),
    ("practical", "actionable"),
    ("accessible", "readable"),
    ("central", "primary"),
)


def _behavior_evaluation_input(user: str) -> dict[str, str]:
    serialized = user.split("EVALUATION_INPUT_JSON:\n", 1)[1].split(
        "\n\nScore each answer", 1
    )[0]
    payload = json.loads(serialized)
    assert isinstance(payload, dict)
    assert all(isinstance(value, str) for value in payload.values())
    return payload


def _behavior_gate_contract() -> tuple[
    CapsuleProfile,
    ActivationProfile,
    tuple[tuple[str, str], ...],
]:
    _, pairs = private_codebook(
        "b" * 32,
        skill_id="behavior-contract",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    return (
        CapsuleProfile(
            mode_phrase="analysis remains active",
            route_phrase="continue through findings",
            checkpoint_phrase="key risks are covered",
            decision_phrase="prepare the final summary",
            slot_label="summary",
        ),
        ActivationProfile(
            active_cues=("alpha brief", "beta brief", "gamma brief"),
            decoy_cues=("delta brief", "epsilon brief", "zeta brief"),
        ),
        pairs,
    )


class ContractModelFixture:
    model = "test/model-contract"

    def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
        del temperature, max_tokens
        if purpose == "query_generation":
            count = int(re.search(r"Create exactly (\d+)", user).group(1))
            queries = [
                f"Review change request {index + 1} and identify the most important verification step."
                for index in range(count)
            ]
            return Completion(json.dumps({"queries": queries}), {"purpose": purpose})
        if purpose == "matched_probe_generation":
            specifications = json.loads(
                user.split("SPECIFICATIONS_JSON: ", 1)[1]
            )
            pairs = [
                {
                    "pair_id": specification["pair_id"],
                    "intent": specification["intent"],
                    "purpose": (
                        f"Evaluate {specification['intent'].replace('_', ' ')} while preserving "
                        "the requested review task and deliverable."
                    ),
                    "query_template": (
                        f"{specification['base_query']} Frame the result around [[CUE_1]], "
                        "relate the evidence to [[CUE_2]], and use [[CUE_3]] to determine the "
                        "appropriate response for this request."
                    ),
                }
                for specification in specifications
            ]
            return Completion(json.dumps({"pairs": pairs}), {"purpose": purpose})
        if purpose == "matched_probe_judgment":
            candidates = json.loads(user.split("CANDIDATES_JSON: ", 1)[1])
            judgments = [
                {
                    "pair_id": candidate["pair_id"],
                    "natural": True,
                    "task_relevant": True,
                    "intent_aligned": True,
                    "cue_slots_semantic": True,
                    "issues": [],
                }
                for candidate in candidates
            ]
            return Completion(
                json.dumps({"judgments": judgments}), {"purpose": purpose}
            )
        if purpose == "semantic_parse":
            source = user.split("SOURCE_MARKDOWN:\n", 1)[1]
            desired = [
                ("constraint", "- Never claim that a change is ready to merge when its intended behavior, review range, or relevant test evidence is still unclear."),
                ("constraint", "- Do not invent file-level findings, benchmark results, security evidence, or test outcomes that have not been inspected directly."),
                ("workflow", "1. Confirm what changed, what requirement the change should satisfy, and which repository range is in scope for review."),
                ("workflow", "2. Inspect correctness, security, architecture, tests, performance, and operational readiness in an order appropriate to the change."),
                ("workflow", "3. State the current review posture before listing evidence, uncertainty, and the smallest next action needed to resolve it."),
                ("fallback", "- When the requester cannot provide a reliable base revision, explain that line-level completeness cannot be established and ask for the minimum missing reference."),
                ("fallback", "- When reviewers disagree about whether a finding blocks release, restate the evidence neutrally and identify the validation that would resolve the disagreement."),
                ("example", "User: Help me prepare a review request for a billing-worker refactor.\n\nAssistant: Summarize the behavioral change, provide the base and head revisions, identify the highest-risk modules, and attach the relevant test evidence."),
            ]
            nodes = [
                {"node_id": f"n{index}", "kind": kind, "quote": quote}
                for index, (kind, quote) in enumerate(desired)
                if quote in source
            ]
            return Completion(json.dumps({"nodes": nodes, "edges": []}), {"purpose": purpose})
        if purpose == "domain_vocabulary":
            return Completion(
                json.dumps(
                    {
                        "cue_pairs": [
                            ["change analysis", "revision analysis"],
                            ["risk evaluation", "risk assessment"],
                            ["test evaluation", "test assessment"],
                            ["release readiness", "release assessment"],
                            ["scope mapping", "scope assessment"],
                            ["dependency mapping", "dependency assessment"],
                            ["security analysis", "security assessment"],
                            ["performance analysis", "performance assessment"],
                        ],
                        "capsule_phrase_pools": {
                            "mode_phrase": [
                                "analysis remains active",
                                "work remains ongoing",
                                "review is underway",
                                "assessment remains underway",
                            ],
                            "route_phrase": [
                                "continue through findings",
                                "follow the review plan",
                                "proceed with validation",
                                "move through remaining checks",
                            ],
                            "checkpoint_phrase": [
                                "key risks are covered",
                                "test impact is understood",
                                "scope is now established",
                                "dependencies are accounted for",
                            ],
                            "decision_phrase": [
                                "prepare the final summary",
                                "request the missing evidence",
                                "continue the investigation",
                                "hand off the findings",
                            ],
                            "slot_label": [
                                "summary",
                                "action",
                                "caution",
                                "next step",
                            ],
                        },
                        "controlled_vocabulary_pairs": [
                            ["detailed", "thorough"],
                            ["clear", "explicit"],
                            ["careful", "attentive"],
                            ["flexible", "adaptable"],
                            ["stable", "reliable"],
                            ["concise", "brief"],
                            ["direct", "straightforward"],
                            ["nearby", "adjacent"],
                            ["quiet", "subtle"],
                            ["structured", "systematic"],
                            ["complete", "comprehensive"],
                            ["relevant", "pertinent"],
                            ["consistent", "uniform"],
                            ["practical", "actionable"],
                            ["accessible", "readable"],
                            ["central", "primary"],
                        ],
                    }
                ),
                {"purpose": purpose},
            )
        if purpose == "controlled_vocabulary_render":
            placeholders = json.loads(
                re.search(r"PLACEHOLDERS_JSON: (.*)\n", user).group(1)
            )
            tail = ""
            if len(placeholders) > 1:
                tail = " with " + ", ".join(placeholders[1:])
            return Completion(
                json.dumps(
                    {"fragment": f"a {placeholders[0]} summary{tail}"}
                ),
                {"purpose": purpose},
            )
        if purpose == "carrier_fusion":
            requirement = json.loads(re.search(r"REQUIREMENT_JSON: (.*)\nPROTECTED_JSON", user).group(1))
            return Completion(
                json.dumps({"addition": requirement, "placement": "after"}),
                {"purpose": purpose},
            )
        if purpose == "behavior_judge":
            scores = {"task_correctness": 10, "policy_faithfulness": 10, "clarity": 10}
            return Completion(
                json.dumps(
                    {
                        "answer_a": scores,
                        "answer_b": scores,
                        "comparison_reason": "No material difference.",
                    }
                ),
                {"purpose": purpose},
            )
        if purpose == "fidelity_gate":
            return Completion(
                json.dumps(
                    {
                        "meaning_preserved": True,
                        "naturalness": 8,
                        "conspicuousness": 2,
                        "explicit_machine_schema": False,
                        "issues": [],
                        "revision_guidance": (
                            "Keep the policy intact while smoothing the conditional handoff prose."
                        ),
                    }
                ),
                {"purpose": purpose},
            )
        if purpose in {"normal_reference", "normal_candidate"}:
            return Completion(
                "Summary: review context.\nAction: inspect evidence.\nCaution: do not invent results.\nNext Step: provide the diff.",
                {"purpose": purpose},
            )
        if purpose == "behavior_probe":
            skill = system.split("SKILL_MARKDOWN:\n", 1)[1]
            query = user
            active_matches = [
                cue
                for pair in TEST_CUE_PAIRS
                for cue in pair
                if cue in query and cue in skill
            ]
            if len(active_matches) < 3:
                return Completion(
                    "Summary: review context.\nAction: inspect evidence.\nCaution: do not invent results.\nNext Step: provide the diff.",
                    {"purpose": purpose},
                )
            example = re.search(
                r'domain-native completion wording:\s*"(.*?)"\. Incorporate the remaining contributions',
                skill,
                flags=re.DOTALL,
            )
            if example is None:
                return Completion("Summary: ordinary path.", {"purpose": purpose})
            distributed_fragments = re.findall(
                r"(?:part|contribution) includes (.*?)\.",
                skill,
                flags=re.DOTALL,
            )
            assembled = example.group(1)
            if distributed_fragments:
                assembled += "; " + "; ".join(distributed_fragments)
            return Completion(assembled, {"purpose": purpose})
        raise AssertionError(f"unexpected purpose: {purpose}")


def config(key: str) -> RuntimeConfig:
    return RuntimeConfig(
        api_key="test-only",
        owner_key=key,
        model="test/model-contract",
        base_url="https://models.example.test/v1",
    )


def test_semantic_parse_retries_with_validation_feedback() -> None:
    class RetryModel(ContractModelFixture):
        def __init__(self) -> None:
            self.parse_calls = 0
            self.retry_prompt = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "semantic_parse":
                self.parse_calls += 1
                if self.parse_calls == 1:
                    return Completion(
                        json.dumps(
                            {
                                "nodes": [
                                    {"node_id": "n1", "kind": "constraint", "quote": "too short"}
                                ],
                                "edges": [],
                            }
                        ),
                        {"purpose": purpose},
                    )
                self.retry_prompt = user
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = RetryModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    nodes, audit = parse_semantic_nodes(markdown, model_fixture)
    assert nodes
    assert model_fixture.parse_calls == 2
    assert audit["parse_attempts"] == 2
    assert len(audit["validation_failures"]) == 1
    assert "omitted required kinds after filtering" in audit["validation_failures"][0]
    assert "quote is shorter than 40 characters" in audit["validation_failures"][0]
    assert "previous response failed deterministic validation" in model_fixture.retry_prompt
    assert "quote is shorter than 40 characters" in model_fixture.retry_prompt


def test_semantic_parse_anchors_word_equivalent_quote_to_exact_source() -> None:
    class WhitespaceVariantModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "semantic_parse":
                return completion
            payload = json.loads(completion.text)
            example = next(node for node in payload["nodes"] if node["kind"] == "example")
            example["quote"] = example["quote"].replace("\n\n", " ")
            return Completion(json.dumps(payload), completion.audit)

    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    nodes, audit = parse_semantic_nodes(markdown, WhitespaceVariantModel())
    example = next(node for node in nodes if node.kind == "example")
    assert example.quote in markdown
    assert "\n\n" in example.quote
    assert audit["parse_attempts"] == 1
    assert audit["quote_repairs"] == [
        {
            "node_id": example.node_id,
            "method": "word_sequence",
            "returned_sha256": audit["quote_repairs"][0]["returned_sha256"],
            "anchored_sha256": hashlib.sha256(example.quote.encode()).hexdigest(),
        }
    ]


def test_semantic_parse_filters_invalid_candidates_without_losing_valid_nodes() -> None:
    class MixedCandidateModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "semantic_parse":
                return completion
            payload = json.loads(completion.text)
            payload["nodes"].insert(
                0,
                {"node_id": "short", "kind": "context", "quote": "## Role"},
            )
            payload["edges"].append(
                {"source": "short", "target": "n1", "relation": "precedes"}
            )
            return Completion(json.dumps(payload), completion.audit)

    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    nodes, audit = parse_semantic_nodes(markdown, MixedCandidateModel())
    assert nodes
    assert audit["parse_attempts"] == 1
    assert audit["node_rejections"][0]["node_id"] == "short"
    assert audit["edge_rejections"] == [
        {"index": 0, "reason": "edge references an invalid node"}
    ]


def test_semantic_parse_retries_partial_candidate_selections() -> None:
    class PartialCandidateModel(ContractModelFixture):
        def __init__(self) -> None:
            self.parse_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "semantic_parse":
                return completion
            self.parse_calls += 1
            if self.parse_calls != 1:
                return completion
            payload = json.loads(completion.text)
            for node in payload["nodes"]:
                if node["kind"] == "constraint":
                    node["quote"] = node["quote"][8:-8]
            return Completion(json.dumps(payload), completion.audit)

    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    model_fixture = PartialCandidateModel()
    nodes, audit = parse_semantic_nodes(markdown, model_fixture)
    assert nodes
    assert model_fixture.parse_calls == 2
    assert audit["parse_attempts"] == 2
    assert "outside supplied eligible source spans" in audit["validation_failures"][0]


def test_replace_exact_uses_disjoint_immutable_source_spans() -> None:
    first = "First carrier policy remains intact for every ordinary request."
    second = "Second carrier workflow stays available for all normal tasks."
    markdown = f"Prelude.\n{first}\nMiddle.\n{second}\nClosing."
    rendered = replace_exact(
        markdown,
        [
            (second, f"{second}\nSecond addition."),
            (first, f"First addition.\n{first}"),
        ],
    )
    assert rendered == (
        f"Prelude.\nFirst addition.\n{first}\nMiddle.\n"
        f"{second}\nSecond addition.\nClosing."
    )


def test_replace_exact_rejects_overlapping_or_cross_copied_carriers() -> None:
    parent = "A long carrier policy contains a smaller workflow clause for testing."
    child = "smaller workflow clause for testing"
    with pytest.raises(ValueError, match="overlap"):
        replace_exact(
            parent,
            [(parent, f"{parent}\nAddition."), (child, f"{child}\nAnother addition.")],
        )

    first = "First carrier policy remains intact for every ordinary request."
    second = "Second carrier workflow stays available for all normal tasks."
    markdown = f"{first}\n{second}"
    with pytest.raises(ValueError, match="duplicates carrier source"):
        replace_exact(
            markdown,
            [(first, f"{first}\n{second}"), (second, second)],
        )


class _TwoPathDomainRepairModel(ContractModelFixture):
    def __init__(self) -> None:
        self.domain_calls = 0
        self.prompts: list[str] = []

    def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
        completion = super().complete(
            system,
            user,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if purpose != "domain_vocabulary":
            return completion
        self.domain_calls += 1
        self.prompts.append(user)
        baseline = json.loads(completion.text)
        if self.domain_calls == 1:
            baseline["capsule_phrase_pools"]["checkpoint_phrase"][1] = (
                "test impact is verified"
            )
            baseline["controlled_vocabulary_pairs"][15] = ["verified", "primary"]
            return Completion(json.dumps(baseline), completion.audit)
        if self.domain_calls == 2:
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/capsule_phrase_pools/checkpoint_phrase/1",
                                "value": "test impact is understood",
                            }
                        ]
                    }
                ),
                completion.audit,
            )
        return completion


def test_domain_vocabulary_retries_with_pair_level_namespace_conflicts() -> None:
    class NamespaceCollisionModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0
            self.retry_prompt = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            if self.domain_calls == 1:
                payload = json.loads(completion.text)
                payload["controlled_vocabulary_pairs"][0] = ["active", "ongoing"]
                payload["controlled_vocabulary_pairs"][1] = ["underway", "established"]
                return Completion(json.dumps(payload), completion.audit)
            self.retry_prompt = user
            return completion

    model = NamespaceCollisionModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    cue_pairs, phrase_pools, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "namespace_retry_contract",
    )

    assert model.domain_calls == 2
    assert audit["generation_attempts"] == 2
    assert len(controlled_pairs) == 16
    assert len(audit["validation_failures"]) == 1
    failure = audit["validation_failures"][0]
    assert "controlled-vocabulary namespace conflicts" in failure
    assert '"pair_index":0' in failure
    assert '"conflict_words":["active","ongoing"]' in failure
    assert '"pair_index":1' in failure
    assert '"conflict_words":["established","underway"]' in failure
    assert "safe_count=14 required=16" in failure
    assert "Replace every reported conflicting controlled-vocabulary pair" in model.retry_prompt
    assert "namespaces must be completely disjoint" in model.retry_prompt
    assert audit["controlled_vocabulary_rejections"] == [
        {
            "attempt": 1,
            "pair_index": 0,
            "pair": ["active", "ongoing"],
            "conflict_words": ["active", "ongoing"],
            "reason": "controlled terms overlap cue/capsule namespace",
        },
        {
            "attempt": 1,
            "pair_index": 1,
            "pair": ["underway", "established"],
            "conflict_words": ["established", "underway"],
            "reason": "controlled terms overlap cue/capsule namespace",
        },
    ]
    reserved_words = {
        word
        for phrase in (
            *(phrase for pair in cue_pairs for phrase in pair),
            *(phrase for values in phrase_pools.values() for phrase in values),
        )
        for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", phrase)
    }
    controlled_words = {word for pair in controlled_pairs for word in pair}
    assert controlled_words.isdisjoint(reserved_words)


def test_domain_vocabulary_repairs_only_a_cue_that_contains_a_capsule_phrase() -> None:
    class CueCapsuleCollisionModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0
            self.repair_prompt = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            if self.domain_calls == 1:
                payload = json.loads(completion.text)
                # The public slot label remains valid and frozen.  Only the cue is
                # authorized for repair when it contains that complete phrase.
                payload["cue_pairs"][0][0] = "change summary"
                return Completion(json.dumps(payload), completion.audit)
            self.repair_prompt = user
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/cue_pairs/0/0",
                                "value": "change analysis",
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = CueCapsuleCollisionModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    cue_pairs, phrase_pools, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "cue_capsule_repair_contract",
    )

    assert model.domain_calls == 2
    assert cue_pairs == TEST_CUE_PAIRS
    assert phrase_pools == TEST_CAPSULE_POOLS
    assert controlled_pairs == TEST_VOCABULARY_PAIRS
    failure = audit["validation_failures"][0]
    assert '"path":"/cue_pairs/0/0"' in failure
    assert "must not equal or contain any complete capsule phrase" in failure
    assert 'INVALID_PATHS_JSON:\n["/cue_pairs/0/0"]' in model.repair_prompt
    assert audit["repair_history"] == [
        {
            "attempt": 2,
            "requested_paths": ["/cue_pairs/0/0"],
            "response_mode": "typed_patches",
            "status": "accepted",
        }
    ]


def test_domain_vocabulary_repairs_only_the_invalid_capsule_path() -> None:
    invalid_phrase = "analysis review work remains fully active"

    class LocalCapsuleRepairModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0
            self.repair_prompt = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            if self.domain_calls == 1:
                payload = json.loads(completion.text)
                payload["capsule_phrase_pools"]["mode_phrase"][0] = invalid_phrase
                return Completion(json.dumps(payload), completion.audit)
            self.repair_prompt = user
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/capsule_phrase_pools/mode_phrase/0",
                                "value": "analysis remains active",
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = LocalCapsuleRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    cue_pairs, phrase_pools, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "local_capsule_repair_contract",
    )

    assert model.domain_calls == 2
    assert phrase_pools == TEST_CAPSULE_POOLS
    assert cue_pairs == TEST_CUE_PAIRS
    assert controlled_pairs == TEST_VOCABULARY_PAIRS
    assert '"path":"/capsule_phrase_pools/mode_phrase/0"' in audit[
        "validation_failures"
    ][0]
    assert invalid_phrase in audit["validation_failures"][0]
    assert "must contain 3 to 5 lowercase lexical words" in audit["validation_failures"][0]
    assert 'INVALID_PATHS_JSON:\n["/capsule_phrase_pools/mode_phrase/0"]' in model.repair_prompt
    assert audit["repair_history"] == [
        {
            "attempt": 2,
            "requested_paths": ["/capsule_phrase_pools/mode_phrase/0"],
            "response_mode": "typed_patches",
            "status": "accepted",
        }
    ]


def test_domain_vocabulary_repair_rejects_an_unauthorized_patch_path() -> None:
    current = {
        "cue_pairs": [["change summary", "revision summary"]],
        "capsule_phrase_pools": {
            "mode_phrase": ["analysis review work remains fully active"]
        },
    }

    with pytest.raises(ValueError, match="patch path must exactly match"):
        _apply_domain_language_repair(
            current,
            {
                "patches": [
                    {
                        "path": "/cue_pairs/0",
                        "value": ["malicious capability", "revision summary"],
                    }
                ]
            },
            {"/capsule_phrase_pools/mode_phrase/0"},
        )


def test_domain_vocabulary_repair_rejects_a_same_value_patch() -> None:
    current = {
        "capsule_phrase_pools": {
            "checkpoint_phrase": ["test impact is verified"]
        }
    }

    with pytest.raises(ValueError, match="must differ from the current invalid value"):
        _apply_domain_language_repair(
            current,
            {
                "patches": [
                    {
                        "path": "/capsule_phrase_pools/checkpoint_phrase/0",
                        "value": "test impact is verified",
                    }
                ]
            },
            {"/capsule_phrase_pools/checkpoint_phrase/0"},
        )


def test_domain_vocabulary_subset_patch_is_rejected_then_full_repair_is_accepted() -> None:
    model = _TwoPathDomainRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    cue_pairs, phrase_pools, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "complete_repair_transaction_contract",
    )

    assert model.domain_calls == 3
    assert cue_pairs == TEST_CUE_PAIRS
    assert phrase_pools == TEST_CAPSULE_POOLS
    assert controlled_pairs == TEST_VOCABULARY_PAIRS
    assert [record["status"] for record in audit["repair_history"]] == [
        "rejected",
        "accepted",
    ]
    assert "must cover every currently invalid path exactly once" in audit[
        "validation_failures"
    ][1]
    assert audit["repair_history"][1]["response_mode"] == "full_object"


def test_domain_vocabulary_repair_prompt_contains_per_path_blacklists() -> None:
    model = _TwoPathDomainRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    _generate_domain_language(
        model,
        markdown,
        "repair_blacklist_contract",
    )

    repair_prompt = model.prompts[1]
    assert "REPAIR_BLACKLIST_JSON:" in repair_prompt
    blacklist = json.loads(repair_prompt.split("REPAIR_BLACKLIST_JSON:\n", 1)[1])
    checkpoint_path = "/capsule_phrase_pools/checkpoint_phrase/1"
    vocabulary_path = "/controlled_vocabulary_pairs/15"
    assert set(blacklist) == {checkpoint_path, vocabulary_path}
    assert blacklist[checkpoint_path] == {
        "forbidden_exact_values": ["test impact is verified"],
        "forbidden_words": ["verified"],
    }
    assert blacklist[vocabulary_path] == {
        "forbidden_exact_values": [["verified", "primary"]],
        "forbidden_words": ["verified"],
    }


def test_domain_vocabulary_full_repair_freezes_valid_fields() -> None:
    invalid_phrase = "analysis review work remains fully active"

    class DriftThenRepairModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            payload = json.loads(completion.text)
            if self.domain_calls == 1:
                payload["capsule_phrase_pools"]["mode_phrase"][0] = invalid_phrase
                return Completion(json.dumps(payload), completion.audit)
            if self.domain_calls == 2:
                payload["cue_pairs"][0][0] = "malicious capability"
                return Completion(json.dumps(payload), completion.audit)
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/capsule_phrase_pools/mode_phrase/0",
                                "value": "analysis remains active",
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = DriftThenRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    cue_pairs, phrase_pools, _, audit = _generate_domain_language(
        model,
        markdown,
        "frozen_valid_fields_contract",
    )

    assert model.domain_calls == 3
    assert cue_pairs == TEST_CUE_PAIRS
    assert phrase_pools["mode_phrase"][0] == "analysis remains active"
    assert "domain-language repair protocol rejected" in audit["validation_failures"][1]
    assert "/cue_pairs/0/0" in audit["validation_failures"][1]
    assert [record["status"] for record in audit["repair_history"]] == [
        "rejected",
        "accepted",
    ]


def test_domain_vocabulary_exhaustion_reports_structured_issues() -> None:
    invalid_phrase = "analysis review work remains fully active"

    class NeverRepairsModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            if self.domain_calls == 1:
                payload = json.loads(completion.text)
                payload["capsule_phrase_pools"]["mode_phrase"][0] = invalid_phrase
                return Completion(json.dumps(payload), completion.audit)
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/capsule_phrase_pools/mode_phrase/0",
                                "value": invalid_phrase,
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = NeverRepairsModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    with pytest.raises(DomainLanguageExhausted) as exc_info:
        _generate_domain_language(model, markdown, "structured_exhaustion_contract")

    message = str(exc_info.value)
    details = exc_info.value.details
    serialized_details = json.dumps(details, ensure_ascii=False)
    assert model.domain_calls == 3
    assert message == DomainLanguageExhausted.public_message
    assert invalid_phrase not in message
    assert "/capsule_phrase_pools/mode_phrase/0" in serialized_details
    assert invalid_phrase in serialized_details
    assert "must differ from the current invalid value" in serialized_details


def test_domain_vocabulary_format_retries_the_same_repair_request() -> None:
    class MalformedThenValidRepairModel(ContractModelFixture):
        def __init__(self) -> None:
            self.domain_calls = 0
            self.prompts: list[str] = []

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.domain_calls += 1
            self.prompts.append(user)
            if self.domain_calls == 1:
                payload = json.loads(completion.text)
                payload["controlled_vocabulary_pairs"][1] = ["detailed", "explicit"]
                return Completion(json.dumps(payload), completion.audit)
            if self.domain_calls == 2:
                return Completion('{"patches":[', completion.audit)
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/controlled_vocabulary_pairs/1",
                                "value": ["clear", "explicit"],
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = MalformedThenValidRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    _, _, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "format_retry_same_repair_contract",
    )

    assert controlled_pairs == TEST_VOCABULARY_PAIRS
    assert model.domain_calls == 3
    assert audit["semantic_rounds"] == 2
    assert audit["model_call_count"] == 3
    assert audit["maximum_model_calls"] == 9
    assert len(audit["round_audits"]) == 2
    first_round, repair_round = audit["round_audits"]
    assert first_round["format_attempts"] == 1
    assert first_round["format_failures"] == []
    assert first_round["outcome"] == "validation_failed"
    assert repair_round["format_attempts"] == 2
    assert len(repair_round["format_failures"]) == 1
    assert len(repair_round["model_calls"]) == 2
    assert repair_round["outcome"] == "accepted"
    assert "FORMAT_RETRY_ONLY" not in model.prompts[1]
    assert model.prompts[2].startswith(model.prompts[1] + "\n\nFORMAT_RETRY_ONLY")
    assert model.prompts[1].count("CURRENT_DOMAIN_LANGUAGE_JSON:") == 1
    assert model.prompts[2].count("CURRENT_DOMAIN_LANGUAGE_JSON:") == 1
    assert (
        'INVALID_PATHS_JSON:\n["/controlled_vocabulary_pairs/1"]'
        in model.prompts[1]
        and 'INVALID_PATHS_JSON:\n["/controlled_vocabulary_pairs/1"]'
        in model.prompts[2]
    )


def test_domain_vocabulary_permanent_malformed_json_has_an_exact_call_bound() -> None:
    class PermanentlyMalformedModel:
        model = "test/permanently-malformed"

        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def complete(
            self,
            system,
            user,
            *,
            purpose,
            temperature=0.0,
            max_tokens=4096,
        ):
            del system, temperature, max_tokens
            assert purpose == "domain_vocabulary"
            self.calls += 1
            self.prompts.append(user)
            return Completion(
                "not json",
                {"purpose": purpose, "unsafe_debug_value": "must-not-enter-audit"},
            )

    model = PermanentlyMalformedModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    with pytest.raises(DomainLanguageExhausted) as exc_info:
        _generate_domain_language(model, markdown, "permanent_format_failure_contract")

    assert model.calls == 9
    details = exc_info.value.details
    assert details["semantic_rounds"] == 3
    assert details["format_retries_per_round"] == 2
    assert details["model_call_count"] == 9
    assert details["maximum_model_calls"] == 9
    assert len(details["round_audits"]) == 3
    assert all(record["format_attempts"] == 3 for record in details["round_audits"])
    assert all(len(record["format_failures"]) == 3 for record in details["round_audits"])
    assert all(record["status"] == "format_exhausted" for record in details["round_audits"])
    assert "unsafe_debug_value" not in json.dumps(details)
    for offset in (0, 3, 6):
        assert "FORMAT_RETRY_ONLY" not in model.prompts[offset]
        assert model.prompts[offset + 1].startswith(
            model.prompts[offset] + "\n\nFORMAT_RETRY_ONLY"
        )
        assert model.prompts[offset + 2].startswith(
            model.prompts[offset] + "\n\nFORMAT_RETRY_ONLY"
        )


def test_domain_vocabulary_valid_duplicate_advances_to_a_repair_round() -> None:
    class DuplicateThenRepairModel(ContractModelFixture):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "domain_vocabulary":
                return completion
            self.calls += 1
            if self.calls == 1:
                payload = json.loads(completion.text)
                payload["controlled_vocabulary_pairs"][1] = ["detailed", "explicit"]
                return Completion(json.dumps(payload), completion.audit)
            return Completion(
                json.dumps(
                    {
                        "patches": [
                            {
                                "path": "/controlled_vocabulary_pairs/1",
                                "value": ["clear", "explicit"],
                            }
                        ]
                    }
                ),
                completion.audit,
            )

    model = DuplicateThenRepairModel()
    markdown = (ROOT / "examples/code_review/SKILL.md").read_text()
    _, _, controlled_pairs, audit = _generate_domain_language(
        model,
        markdown,
        "valid_duplicate_semantic_round_contract",
    )

    assert controlled_pairs == TEST_VOCABULARY_PAIRS
    assert model.calls == 2
    assert audit["semantic_rounds"] == 2
    assert audit["round_audits"][0]["format_attempts"] == 1
    assert audit["round_audits"][0]["format_failures"] == []
    assert audit["round_audits"][0]["outcome"] == "validation_failed"
    assert audit["round_audits"][1]["request_kind"] == "repair"


def test_controlled_vocabulary_render_retries_without_weakening_term_checks() -> None:
    class RetryVocabularyModel(ContractModelFixture):
        def __init__(self) -> None:
            self.render_calls = 0
            self.retry_prompt = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "controlled_vocabulary_render":
                self.render_calls += 1
                if self.render_calls == 1:
                    return Completion(
                        json.dumps({"fragments": ["Missing assigned terms."]}),
                        {"purpose": purpose},
                    )
                if self.render_calls == 2:
                    self.retry_prompt = user
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    codebook, token_pairs = private_codebook(
        "z" * 32,
        skill_id="retry_vocab",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
        buyer_count=16,
        codeword_length=16,
    )
    model_fixture = RetryVocabularyModel()
    fragments, audit = _render_slot_fragments(
        model_fixture,
        codebook["buyer_1"],
        token_pairs,
        4,
    )
    assert len(fragments) == 4
    assert model_fixture.render_calls == 5
    assert audit["render_attempts"] == 5
    assert audit["fragment_attempts"] == [2, 1, 1, 1]
    assert len(audit["validation_failures"]) == 1
    assert "fragment field is missing" in audit["validation_failures"][0]
    assert "previous response failed deterministic validation" in model_fixture.retry_prompt


def test_controlled_vocabulary_render_neutralizes_unplanned_code_terms() -> None:
    class CollisionModel(ContractModelFixture):
        def __init__(self, collision: str) -> None:
            self.collision = collision

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "controlled_vocabulary_render":
                return completion
            payload = json.loads(completion.text)
            payload["fragment"] += f" with {self.collision} context"
            return Completion(json.dumps(payload), completion.audit)

    codebook, token_pairs = private_codebook(
        "y" * 32,
        skill_id="collision_vocab",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
        buyer_count=16,
        codeword_length=16,
    )
    collision = token_pairs[0][0]
    fragments, audit = _render_slot_fragments(
        CollisionModel(collision),
        codebook["buyer_1"],
        token_pairs,
        4,
    )
    assert "appropriate" in fragments[0]
    assert audit["neutralized_term_count"] >= 1


def test_carrier_fusion_uses_immutable_source_and_protected_placeholders() -> None:
    class CaptureFusionModel(ContractModelFixture):
        def __init__(self) -> None:
            self.user = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "carrier_fusion":
                self.user = user
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    source = "Preserve this source policy exactly while preparing the operational handoff."
    protected = ["continuity lane", "the review remains active"]
    requirement = (
        "When a continuity lane applies, state that the review remains active before closing."
    )
    model_fixture = CaptureFusionModel()
    fused, audit = _fuse(
        model_fixture,
        source,
        requirement,
        protected,
        context="constraint:n1",
        round_index=1,
        previous_candidate=None,
        revision_guidance="Keep the transition natural.",
    )
    assert source in fused
    assert all(value in fused for value in protected)
    assert "[[SOURCE_TEXT]]" not in fused
    assert "[[PROTECTED_" not in fused
    assert audit["template"] == "protected_placeholders"
    requirement_line = next(
        line for line in model_fixture.user.splitlines() if line.startswith("REQUIREMENT_JSON:")
    )
    assert all(value not in requirement_line for value in protected)


def test_carrier_fusion_joins_the_source_deterministically() -> None:
    class BeforeSourceFusionModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose != "carrier_fusion":
                return super().complete(
                    system,
                    user,
                    purpose=purpose,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            requirement = json.loads(
                re.search(r"REQUIREMENT_JSON: (.*)\nPROTECTED_JSON", user).group(1)
            )
            return Completion(
                json.dumps({"addition": requirement, "placement": "before"}),
                {"purpose": purpose},
            )

    source = "Keep this exact operational policy intact throughout the handoff process."
    protected = ["continuity lane", "review stays active"]
    fused, audit = _fuse(
        BeforeSourceFusionModel(),
        source,
        "For a continuity lane, state that review stays active.",
        protected,
        context="constraint:n1",
        round_index=1,
        previous_candidate=None,
        revision_guidance="Keep the transition natural.",
    )
    assert fused.count(source) == 1
    assert all(value in fused for value in protected)
    assert fused.index(protected[0]) < fused.index(source)
    assert audit["source_transport"] == "deterministic_join"
    assert audit["addition_placement"] == "before"


def test_carrier_fusion_retries_template_format_without_advancing_fidelity_round() -> None:
    class RetryFusionModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fusion_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "carrier_fusion":
                self.fusion_calls += 1
                if self.fusion_calls == 1:
                    return Completion(
                        json.dumps({"addition": "missing placeholders", "placement": "after"}),
                        {},
                    )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = RetryFusionModel()
    fused, audit = _fuse(
        model_fixture,
        "Keep this exact operational policy intact throughout the handoff process.",
        "For a continuity lane, state that review stays active.",
        ["continuity lane", "review stays active"],
        context="constraint:n1",
        round_index=2,
        previous_candidate=None,
        revision_guidance="Keep the transition natural.",
    )
    assert fused
    assert model_fixture.fusion_calls == 2
    assert audit["round"] == 2
    assert audit["format_attempts"] == 2
    assert len(audit["format_failures"]) == 1


def test_end_to_end_core_contract(tmp_path: Path) -> None:
    key = "a" * 32
    package = tmp_path / "package"
    model_fixture = ContractModelFixture()
    summary = build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_16",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=16,
        codeword_length=8,
        model=model_fixture,
    )
    assert summary["status"] == "complete"
    assert sorted(path.name for path in (package / "buyer_delivery").iterdir()) == ["SKILL.md"]
    delivery = (package / "buyer_delivery/SKILL.md").read_text()
    assert "buyer_16" not in delivery
    assert "five scalar keys" not in delivery
    assert "```yaml" not in delivery
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    for obsolete_anchor in (
        "The substantive output remains unchanged",
        "Only when the request contains all three exact phrases",
        "Only then",
        "The matching note reads",
        "do not append it otherwise",
    ):
        assert obsolete_anchor not in delivery
    assert all(value in delivery for value in audit["capsule_profile"].values())
    assert len(set(audit["selected_node_kinds"])) == 4
    assert "example" in audit["selected_node_kinds"]
    assert not any(name in delivery for name in audit["capsule_profile"])
    assert not any(cue in delivery for cue in audit["activation_profile"]["decoy_cues"])
    buyer_terms = set(audit["buyer_record"]["tokens"])
    assert buyer_terms.issubset(set(re.findall(r"[a-z-]+", delivery.casefold())))
    report = probe_package(
        package,
        tmp_path / "probe.json",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=model_fixture,
    )
    release_quality = report["release_quality"]
    assert {
        key: value for key, value in release_quality.items() if key != "gate"
    } == {
        "active_payload_rate": 1.0,
        "decoy_payload_rate": 0.0,
        "normal_payload_rate": 0.0,
        "paired_payload_differential": 1.0,
    }
    assert report["buyer"]["top1"] == "buyer_16"
    assert report["release_ready"] is True
    assert release_quality["gate"]["suppression_passed"] is True
    assert report["detection_result"] == {
        "supported": True,
        "status": "supported_by_probe",
        "decoded_buyer": "buyer_16",
    }
    assert verify_package(package, key)["valid"] is True


def test_one_command_model_pipeline_contract(tmp_path: Path) -> None:
    output = tmp_path / "run"
    report = run_model_pipeline(
        ROOT / "examples/code_review/SKILL.md",
        output,
        skill_id="code_review",
        buyer_id="buyer_6",
        config=config("l" * 32),
        normal_query_count=10,
        pairs=5,
        model=ContractModelFixture(),
    )
    assert report["pipeline"] == [
        "query_generation",
        "watermark_build",
        "active_decoy_normal_probe",
        "ecc_decode",
        "suppression_gate",
        "report",
    ]
    assert report["run_status"] == "ready"
    assert report["buyer"]["top1"] == "buyer_6"
    assert len(json.loads((output / "normal_queries.json").read_text())) == 10
    assert (output / "package/buyer_delivery/SKILL.md").is_file()
    assert (output / "package/owner_audit/audit.json").is_file()
    assert (output / "report.json").is_file()
    release = json.loads((output / "release.json").read_text())
    assert release["status"] == "ready"
    assert release["release_ready_buyer_ids"] == ["buyer_6"]
    assert release["approved_deliveries"]["buyer_6"]["path"] == (
        "package/buyer_delivery"
    )
    assert verify_release_manifest(output, "l" * 32)["valid"] is True

    report_path = output / "report.json"
    report_text = report_path.read_text(encoding="utf-8")
    report_path.write_text(report_text + " ", encoding="utf-8")
    assert verify_release_manifest(output, "l" * 32)["checks"]["report_digest"] is False
    report_path.write_text(report_text, encoding="utf-8")

    release["approved_deliveries"]["buyer_6"]["path"] = "..\\escaped-delivery"
    release["owner_authentication"] = audit_authentication("l" * 32, release)
    (output / "release.json").write_text(json.dumps(release), encoding="utf-8")
    unsafe_path = verify_release_manifest(output, "l" * 32)
    assert unsafe_path["valid"] is False
    assert unsafe_path["checks"]["approved_deliveries"] is False

    release["status"] = "rejected"
    (output / "release.json").write_text(json.dumps(release), encoding="utf-8")
    tampered = verify_release_manifest(output, "l" * 32)
    assert tampered["valid"] is False
    assert tampered["checks"]["owner_authentication"] is False


def test_one_command_pipeline_never_releases_an_erased_buyer_candidate(
    tmp_path: Path,
) -> None:
    class MarkerOnlyActiveModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_probe":
                skill = system.split("SKILL_MARKDOWN:\n", 1)[1]
                active_matches = [
                    cue
                    for pair in TEST_CUE_PAIRS
                    for cue in pair
                    if cue in user and cue in skill
                ]
                if len(active_matches) < 3:
                    return super().complete(
                        system,
                        user,
                        purpose=purpose,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                example = re.search(
                    r'domain-native completion wording:\s*"(.*?)"\. Incorporate the remaining contributions',
                    skill,
                    flags=re.DOTALL,
                )
                assert example is not None
                return Completion(example.group(1), {"purpose": purpose})
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    output = tmp_path / "rejected-run"
    report = run_model_pipeline(
        ROOT / "examples/code_review/SKILL.md",
        output,
        skill_id="code_review",
        buyer_id="buyer_2",
        config=config("p" * 32),
        normal_query_count=10,
        pairs=5,
        buyer_count=4,
        codeword_length=4,
        model=MarkerOnlyActiveModel(),
    )

    release = json.loads((output / "release.json").read_text())
    assert report["owner_verification"]["supported"] is True
    assert report["buyer_attribution"]["status"] == "not_attributed"
    assert report["release_ready"] is False
    assert release["status"] == "rejected"
    assert release["release_ready_buyer_ids"] == []
    assert release["rejected_candidate_buyer_ids"] == ["buyer_2"]
    assert release["approved_deliveries"] == {}
    assert verify_release_manifest(output, "p" * 32)["valid"] is True


def _package_fixture(tmp_path: Path) -> Path:
    source = (ROOT / "examples/code_review/SKILL.md").read_text()
    example = (
        "User: Help me prepare a review request for a billing-worker refactor.\n\n"
        "Assistant: Summarize the behavioral change, provide the base and head revisions, "
        "identify the highest-risk modules, and attach the relevant test evidence."
    )
    assert source.count(example) == 1
    package = tmp_path / "source-package"
    (package / "guides").mkdir(parents=True)
    (package / "SKILL.md").write_text(source.replace(example, "See the worked example guide."))
    (package / "guides/example.md").write_text(f"# Worked example\n\n{example}\n")
    (package / "tools").mkdir()
    (package / "tools/helper.py").write_bytes(b"print('inert research asset')\n")
    return package


def test_owner_side_skillcoder_artifacts_cannot_be_reexported(
    tmp_path: Path,
) -> None:
    prior_run = tmp_path / "prior-run"
    (prior_run / "buyer_delivery").mkdir(parents=True)
    (prior_run / "buyer_delivery/SKILL.md").write_text(
        "# Safe buyer delivery\n\nFollow the ordinary review workflow.",
        encoding="utf-8",
    )
    (prior_run / "owner_audit").mkdir()
    (prior_run / "owner_audit/audit.json").write_text(
        '{"codebook":{"buyer_1":{"bits":[0,1]}}}', encoding="utf-8"
    )
    (prior_run / "normal_queries.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="owner-side SkillCODER artifact"):
        load_skill_source(prior_run, entrypoint="buyer_delivery/SKILL.md")

    buyer_only = load_skill_source(prior_run / "buyer_delivery")
    assert set(buyer_only.files) == {"SKILL.md"}


def test_package_loader_ignores_case_variant_vcs_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "# Review\n\nFollow the review workflow.", encoding="utf-8"
    )
    (source / ".GIT").mkdir()
    (source / ".GIT/config").write_text(
        "url = https://token@example.test/private.git", encoding="utf-8"
    )

    loaded = load_skill_source(source)

    assert set(loaded.files) == {"SKILL.md"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "real_world/code_review/differential-review",
        "real_world/data_science/csv-data-summarizer",
        "real_world/travel_planning/travel-planner",
    ],
)
def test_pinned_real_world_skill_packages_round_trip_without_executing_assets(
    relative_path: str,
) -> None:
    source = load_skill_source(ROOT / "datasets/paper_skills" / relative_path)
    assert source.source_kind == "package"
    assert source.entrypoint == "SKILL.md"
    assert source.document_order[0] == "SKILL.md"
    assert source.rendered_files(source.canonical_markdown) == source.files
    assert "SKILLCODER_PACKAGE_DOCUMENT" not in source.prompt_markdown


def test_skill_package_build_preserves_assets_and_authenticates_the_tree(tmp_path: Path) -> None:
    key = "p" * 32
    source = _package_fixture(tmp_path)
    package = tmp_path / "built-package"
    summary = build_package(
        source,
        package,
        skill_id="package_review",
        buyer_id="buyer_2",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        entrypoint="SKILL.md",
        model=ContractModelFixture(),
    )
    assert summary["source_kind"] == "package"
    assert summary["document_paths"] == ["SKILL.md", "guides/example.md"]
    assert summary["selected_document_paths"] == ["SKILL.md", "guides/example.md"]
    delivery = package / "buyer_delivery"
    assert (delivery / "tools/helper.py").read_bytes() == b"print('inert research asset')\n"
    assert "SKILLCODER_PACKAGE_DOCUMENT" not in (delivery / "SKILL.md").read_text()
    assert "SKILLCODER_PACKAGE_DOCUMENT" not in (delivery / "guides/example.md").read_text()
    assert (delivery / "guides/example.md").read_text() != (
        source / "guides/example.md"
    ).read_text()
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    assert audit["carrier_scope"] == "package_documents"
    assert audit["semantic_parse"]["carrier_scope"] == "canonical_source"
    assert verify_package(package, key)["valid"] is True
    (delivery / "unexpected-link").symlink_to(delivery / "SKILL.md")
    assert verify_package(package, key)["valid"] is False
    (delivery / "unexpected-link").unlink()
    assert verify_package(package, key)["valid"] is True
    (delivery / "tools/helper.py").write_bytes(b"tampered\n")
    result = verify_package(package, key)
    assert result["valid"] is False
    assert result["checks"]["delivery_hash"] is False


def test_skill_package_rejects_unsafe_paths_and_recursive_output(tmp_path: Path) -> None:
    source = _package_fixture(tmp_path)
    (source / "linked.md").symlink_to(source / "SKILL.md")
    with pytest.raises(ValueError, match="symbolic links"):
        load_skill_source(source)
    (source / "linked.md").unlink()
    with pytest.raises(ValueError, match="outside the input"):
        build_package(
            source,
            source / "generated-output",
            skill_id="package_review",
            buyer_id="buyer_1",
            config=config("u" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=ContractModelFixture(),
        )


@pytest.mark.parametrize(
    "name",
    [
        ".env.staging",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "id_rsa.bak",
        "credentials.json",
        "secrets.json",
        "client_secret.json",
        "service_account_key.json",
    ],
)
def test_skill_package_rejects_sensitive_files_before_model_use(
    tmp_path: Path, name: str
) -> None:
    source = _package_fixture(tmp_path)
    (source / name).write_text("private material")
    with pytest.raises(ValueError, match="sensitive"):
        load_skill_source(source)


@pytest.mark.parametrize(
    "name", ["private.pem", "credentials.md", "api-key.md", "notes.txt"]
)
def test_single_file_source_requires_non_sensitive_markdown(
    tmp_path: Path, name: str
) -> None:
    source = tmp_path / name
    source.write_text("# Looks like a Skill\n")
    with pytest.raises(ValueError, match="Markdown|sensitive"):
        load_skill_source(source)


def test_skill_package_rejects_sensitive_directory_components(tmp_path: Path) -> None:
    source = _package_fixture(tmp_path)
    (source / "secrets").mkdir()
    (source / "secrets/api-key.md").write_text("# Private\n")
    with pytest.raises(ValueError, match="sensitive"):
        load_skill_source(source)


@pytest.mark.parametrize(
    "relative_path",
    [
        ".docker/config.json",
        ".kube/config",
        ".azure/accessTokens.json",
        ".aws/credentials",
        ".config/gcloud/application_default_credentials.json",
        ".config/gh/hosts.yml",
        "infrastructure/terraform.tfstate",
        "infrastructure/terraform.tfstate.backup",
        "infrastructure/prod.auto.tfvars",
        "auth.json",
    ],
)
def test_skill_package_rejects_common_secret_stores_and_state(
    tmp_path: Path, relative_path: str
) -> None:
    source = _package_fixture(tmp_path)
    sensitive = source / relative_path
    sensitive.parent.mkdir(parents=True, exist_ok=True)
    sensitive.write_text("live credential material", encoding="utf-8")

    with pytest.raises(ValueError, match="sensitive"):
        load_skill_source(source)


def test_behavior_gate_uses_the_delivery_package_serialization(tmp_path: Path) -> None:
    class CaptureCandidateModel(ContractModelFixture):
        def __init__(self) -> None:
            self.candidate_system = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_candidate" and not self.candidate_system:
                self.candidate_system = system
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = CaptureCandidateModel()
    package = tmp_path / "serialized-package"
    build_package(
        _package_fixture(tmp_path),
        package,
        skill_id="package_review",
        buyer_id="buyer_1",
        config=config("s" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        model=model_fixture,
    )
    assert '<skill-document path="SKILL.md">' in model_fixture.candidate_system
    assert '<skill-document path="guides/example.md">' in model_fixture.candidate_system
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    assert audit["behavior_input_serialization"] == "package_documents"


def test_multi_buyer_model_prompts_do_not_serialize_private_mappings(tmp_path: Path) -> None:
    class RecordingModel(ContractModelFixture):
        def __init__(self) -> None:
            self.prompts: list[tuple[str, str, str]] = []

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            self.prompts.append((purpose, system, user))
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    key = "r" * 32
    model_fixture = RecordingModel()
    family = tmp_path / "private-family"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1", "buyer_2", "buyer_3"],
        model=model_fixture,
    )
    downstream = [
        (purpose, system, user)
        for purpose, system, user in model_fixture.prompts
        if purpose
        in {
            "controlled_vocabulary_render",
            "carrier_fusion",
            "fidelity_gate",
            "normal_candidate",
        }
    ]
    serialized = "\n".join(system + "\n" + user for _, system, user in downstream)
    assert key not in serialized
    assert "PLACEHOLDER_BINDINGS_JSON" not in serialized
    assert "PROTECTED_BINDINGS_JSON" not in serialized
    assert "CONTROLLED_TERMS_JSON" not in serialized
    assert '"buyer_id"' not in serialized
    assert '"bits"' not in serialized
    family_audit = json.loads((family / "owner_audit/family.json").read_text())
    assert sum(
        purpose == "controlled_vocabulary_render" for purpose, _, _ in downstream
    ) == len(family_audit["selected_node_ids"])


def test_one_command_pipeline_builds_from_the_query_generation_snapshot(tmp_path: Path) -> None:
    source = _package_fixture(tmp_path)
    mutation = "\n\nMUTATED_AFTER_QUERY_GENERATION\n"

    class MutatingQueryModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose == "query_generation":
                with (source / "SKILL.md").open("a", encoding="utf-8") as handle:
                    handle.write(mutation)
            return completion

    output = tmp_path / "snapshot-run"
    report = run_model_pipeline(
        source,
        output,
        skill_id="package_review",
        buyer_id="buyer_1",
        config=config("n" * 32),
        normal_query_count=10,
        pairs=5,
        buyer_count=4,
        codeword_length=4,
        model=MutatingQueryModel(),
    )
    assert mutation.strip() in (source / "SKILL.md").read_text()
    assert mutation.strip() not in (output / "package/buyer_delivery/SKILL.md").read_text()
    assert report["query_generation"]["source_tree_sha256"] == report["build"][
        "source_tree_sha256"
    ]


def test_multi_buyer_family_freezes_one_plan_and_decodes_each_copy(tmp_path: Path) -> None:
    class CountingModel(ContractModelFixture):
        def __init__(self) -> None:
            self.purposes: list[str] = []

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            self.purposes.append(purpose)
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    key = "m" * 32
    model_fixture = CountingModel()
    source = _package_fixture(tmp_path)
    family = tmp_path / "family"
    summary = build_buyer_family(
        source,
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1", "buyer_2", "buyer_4"],
        model=model_fixture,
    )
    assert summary["candidate_buyer_count"] == 3
    assert model_fixture.purposes.count("semantic_parse") == 1
    assert model_fixture.purposes.count("domain_vocabulary") == 1
    family_audit = json.loads((family / "owner_audit/family.json").read_text())
    assert key not in json.dumps(family_audit)
    plan_ids: set[str] = set()
    codebooks: list[dict[str, object]] = []
    buyer_bits: list[tuple[int, ...]] = []
    delivery_trees: set[str] = set()
    for buyer_id in summary["candidate_buyer_ids"]:
        buyer_audit = json.loads(
            (family / f"buyers/{buyer_id}/owner_audit/audit.json").read_text()
        )
        plan_ids.add(str(buyer_audit["watermark_plan_sha256"]))
        codebooks.append(buyer_audit["codebook"])
        buyer_bits.append(tuple(buyer_audit["buyer_record"]["bits"]))
        delivery_trees.add(str(buyer_audit["delivery_tree_sha256"]))
    assert plan_ids == {family_audit["watermark_plan_sha256"]}
    assert all(codebook == codebooks[0] for codebook in codebooks)
    assert len(set(buyer_bits)) == 3
    assert len(delivery_trees) == 3
    assert verify_buyer_family(family, key)["valid"] is True

    probe_output = tmp_path / "family-probe"
    report = probe_buyer_family(
        family,
        probe_output,
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        pairs=5,
        model=model_fixture,
    )
    assert report["release_ready"] is True
    assert report["release_rate"] == 1.0
    assert report["top1_accuracy"] == 1.0
    assert set(report["buyers"]) == {"buyer_1", "buyer_2", "buyer_4"}


def test_verification_rejects_package_and_delivery_symlinks(tmp_path: Path) -> None:
    key = "e" * 32
    package = tmp_path / "real-package"
    build_package(
        _package_fixture(tmp_path),
        package,
        skill_id="package_review",
        buyer_id="buyer_1",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        model=ContractModelFixture(),
    )
    package_link = tmp_path / "package-link"
    package_link.symlink_to(package, target_is_directory=True)
    assert verify_package(package_link, key)["valid"] is False

    delivery = package / "buyer_delivery"
    held_delivery = tmp_path / "held-delivery"
    delivery.rename(held_delivery)
    delivery.symlink_to(held_delivery, target_is_directory=True)
    assert verify_package(package, key)["valid"] is False
    delivery.unlink()
    held_delivery.rename(delivery)

    tools = delivery / "tools"
    held_tools = tmp_path / "held-tools"
    tools.rename(held_tools)
    tools.symlink_to(held_tools, target_is_directory=True)
    assert verify_package(package, key)["valid"] is False


def test_verification_rejects_unexpected_special_delivery_entries(tmp_path: Path) -> None:
    key = "x" * 32
    package = tmp_path / "special-entry-package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_1",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        model=ContractModelFixture(),
    )
    os.mkfifo(package / "buyer_delivery/unexpected.fifo")
    assert verify_package(package, key)["valid"] is False


def test_family_verification_rejects_a_symlinked_buyer_package(tmp_path: Path) -> None:
    key = "g" * 32
    family = tmp_path / "family"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1", "buyer_2"],
        model=ContractModelFixture(),
    )
    family_link = tmp_path / "family-link"
    family_link.symlink_to(family, target_is_directory=True)
    assert verify_buyer_family(family_link, key)["valid"] is False
    buyer_one = family / "buyers/buyer_1"
    held_buyer = tmp_path / "held-buyer"
    buyer_one.rename(held_buyer)
    buyer_one.symlink_to(family / "buyers/buyer_2", target_is_directory=True)
    assert verify_buyer_family(family, key)["valid"] is False


def test_family_verification_rejects_unexpected_buyer_surface_files(tmp_path: Path) -> None:
    key = "w" * 32
    family = tmp_path / "family-surface"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1", "buyer_2"],
        model=ContractModelFixture(),
    )
    (family / "buyers/unexpected.json").write_text("{}")
    result = verify_buyer_family(family, key)
    assert result["valid"] is False
    assert result["checks"]["buyer_surface"] is False


def test_family_verification_short_circuits_an_unauthenticated_family(
    tmp_path: Path,
) -> None:
    key = "u" * 32
    family = tmp_path / "family-auth"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1"],
        model=ContractModelFixture(),
    )
    family_audit_path = family / "owner_audit/family.json"
    family_audit = json.loads(family_audit_path.read_text(encoding="utf-8"))
    family_audit["skill_id"] = "tampered"
    family_audit_path.write_text(json.dumps(family_audit), encoding="utf-8")

    result = verify_buyer_family(family, key)

    assert result["valid"] is False
    assert result["checks"]["audit_authentication"] is False
    assert result["buyers"] == {}


def test_family_verification_consumes_one_authenticated_buyer_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "v" * 32
    family = tmp_path / "family-buyer-swap"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1"],
        model=ContractModelFixture(),
    )
    audit_path = family / "buyers/buyer_1/owner_audit/audit.json"
    authentic = audit_path.read_text(encoding="utf-8")
    tampered_payload = json.loads(authentic)
    tampered_payload["model"] = "swapped/after-verification"
    tampered = json.dumps(tampered_payload)
    original_read_json_object = pipeline_module._read_json_object
    reads = 0

    def swapped_read_json_object(path: Path, *, label: str):
        nonlocal reads
        if path == audit_path:
            reads += 1
            return json.loads(authentic if reads == 1 else tampered)
        return original_read_json_object(path, label=label)

    monkeypatch.setattr(pipeline_module, "_read_json_object", swapped_read_json_object)

    result = verify_buyer_family(family, key)

    assert reads == 1
    assert result["valid"] is True
    assert result["buyers"]["buyer_1"]["package_valid"] is True


def test_family_probe_consumes_one_authenticated_family_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "x" * 32
    family = tmp_path / "family-audit-swap"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1"],
        model=ContractModelFixture(),
    )
    audit_path = family / "owner_audit/family.json"
    authentic = audit_path.read_text(encoding="utf-8")
    tampered_payload = json.loads(authentic)
    tampered_payload["owner_verification_policy"]["threshold"] = 0.01
    tampered = json.dumps(tampered_payload)
    original_read_json_object = pipeline_module._read_json_object
    reads = 0

    def swapped_read_json_object(path: Path, *, label: str):
        nonlocal reads
        if path == audit_path:
            reads += 1
            return json.loads(authentic if reads == 1 else tampered)
        return original_read_json_object(path, label=label)

    monkeypatch.setattr(pipeline_module, "_read_json_object", swapped_read_json_object)

    report = probe_buyer_family(
        family,
        tmp_path / "swapped-probe",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_ids=["buyer_1"],
        model=ContractModelFixture(),
    )

    assert reads == 1
    assert report["scope"] == "multi_buyer_core_method_probe"


def test_probe_outputs_are_non_overwriting_atomic_and_outside_inputs(tmp_path: Path) -> None:
    key = "o" * 32
    package = tmp_path / "probe-package"
    model_fixture = ContractModelFixture()
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_1",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        model=model_fixture,
    )
    existing = tmp_path / "existing-probe.json"
    existing.write_text("keep me")
    with pytest.raises(FileExistsError):
        probe_package(
            package,
            existing,
            config=config(key),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=model_fixture,
        )
    assert existing.read_text() == "keep me"
    with pytest.raises(ValueError, match="outside the input buyer package"):
        probe_package(
            package,
            package / "probe.json",
            config=config(key),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=model_fixture,
        )
    assert not list(tmp_path.glob(".existing-probe.json.stage-*"))


def test_probe_cannot_clobber_a_destination_created_during_execution(tmp_path: Path) -> None:
    key = "q" * 32
    package = tmp_path / "race-package"
    output = tmp_path / "raced-probe.json"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_1",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        model=ContractModelFixture(),
    )

    class RacingProbeModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_probe" and not os.path.lexists(output):
                output.symlink_to(tmp_path / "missing-race-target")
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(FileExistsError, match="output already exists"):
        probe_package(
            package,
            output,
            config=config(key),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=RacingProbeModel(),
        )
    assert output.is_symlink()
    assert not list(tmp_path.glob(".raced-probe.json.stage-*"))


def test_family_probe_cannot_clobber_a_destination_created_during_execution(
    tmp_path: Path,
) -> None:
    key = "k" * 32
    family = tmp_path / "race-family"
    output = tmp_path / "raced-family-probe"
    build_buyer_family(
        _package_fixture(tmp_path),
        family,
        skill_id="package_review",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1"],
        model=ContractModelFixture(),
    )

    class RacingFamilyProbeModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_probe" and not os.path.lexists(output):
                output.mkdir()
                (output / "concurrent.txt").write_text("keep")
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(FileExistsError):
        probe_buyer_family(
            family,
            output,
            config=config(key),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            buyer_ids=["buyer_1"],
            model=RacingFamilyProbeModel(),
        )
    assert (output / "concurrent.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".raced-family-probe.stage-*"))


def test_one_command_multi_buyer_pipeline_contract(tmp_path: Path) -> None:
    output = tmp_path / "family-run"
    report = run_buyer_family_pipeline(
        _package_fixture(tmp_path),
        output,
        skill_id="package_review",
        config=config("f" * 32),
        normal_query_count=10,
        pairs=5,
        buyer_count=4,
        codeword_length=4,
        buyer_ids=["buyer_1", "buyer_3"],
        model=ContractModelFixture(),
    )
    assert report["pipeline"] == [
        "query_generation",
        "shared_watermark_plan",
        "multi_buyer_build",
        "active_decoy_normal_probe",
        "ecc_decode",
        "family_aggregate",
        "report",
    ]
    assert report["run_status"] == "ready"
    assert report["top1_accuracy"] == 1.0
    assert (output / "family/buyers/buyer_1/buyer_delivery/SKILL.md").is_file()
    assert (output / "family/buyers/buyer_3/buyer_delivery/guides/example.md").is_file()
    assert (output / "probe/report.json").is_file()
    release = json.loads((output / "release.json").read_text())
    assert release["status"] == "ready"
    assert release["release_ready_buyer_ids"] == ["buyer_1", "buyer_3"]
    assert verify_release_manifest(output, "f" * 32)["valid"] is True


def test_multi_buyer_build_is_failure_atomic(tmp_path: Path) -> None:
    class FailDuringSecondBuyer(ContractModelFixture):
        def __init__(self) -> None:
            self.normal_candidate_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_candidate":
                self.normal_candidate_calls += 1
                if self.normal_candidate_calls == 11:
                    raise RuntimeError("second buyer provider failure")
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    output = tmp_path / "failed-family"
    with pytest.raises(RuntimeError, match="second buyer"):
        build_buyer_family(
            _package_fixture(tmp_path),
            output,
            skill_id="package_review",
            config=config("v" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            buyer_count=4,
            codeword_length=4,
            buyer_ids=["buyer_1", "buyer_2"],
            model=FailDuringSecondBuyer(),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-family.stage-*"))


def test_wrong_key_cannot_verify_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_4",
        config=config("b" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=ContractModelFixture(),
    )
    assert verify_package(package, "c" * 32)["valid"] is False
    with pytest.raises(RuntimeError, match="does not match"):
        probe_package(
            package,
            tmp_path / "probe.json",
            config=config("c" * 32),
            pairs=5,
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=ContractModelFixture(),
        )


def test_modified_private_audit_is_rejected(tmp_path: Path) -> None:
    key = "h" * 32
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_8",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=ContractModelFixture(),
    )
    audit_path = package / "owner_audit/audit.json"
    audit = json.loads(audit_path.read_text())
    audit["buyer_id"] = "buyer_9"
    audit_path.write_text(json.dumps(audit))
    result = verify_package(package, key)
    assert result["valid"] is False
    assert result["checks"]["audit_authentication"] is False


def test_probe_rejects_a_different_normal_query_set(tmp_path: Path) -> None:
    key = "i" * 32
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_2",
        config=config(key),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=ContractModelFixture(),
    )
    changed_queries = tmp_path / "queries.json"
    changed_queries.write_text(
        json.dumps([f"A different ordinary request number {index}." for index in range(10)])
    )
    with pytest.raises(RuntimeError, match="normal-query set"):
        probe_package(
            package,
            tmp_path / "probe.json",
            config=config(key),
            pairs=5,
            normal_queries=changed_queries,
            model=ContractModelFixture(),
        )


def test_behavior_gate_rejects_invalid_judge_scores(tmp_path: Path) -> None:
    class InvalidJudgeModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_judge":
                scores = {"task_correctness": 11, "policy_faithfulness": 10, "clarity": 10}
                return Completion(
                    json.dumps(
                        {
                            "answer_a": scores,
                            "answer_b": scores,
                            "comparison_reason": "Invalid score fixture.",
                        }
                    ),
                    {},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(ValueError, match="out-of-range"):
        build_package(
            ROOT / "examples/code_review/SKILL.md",
            tmp_path / "package",
            skill_id="code_review",
            buyer_id="buyer_3",
            config=config("j" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=InvalidJudgeModel(),
        )


def test_behavior_gate_rejects_one_catastrophic_normal_task(tmp_path: Path) -> None:
    class CatastrophicTaskModel(ContractModelFixture):
        def __init__(self) -> None:
            self.candidate_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_candidate":
                self.candidate_calls += 1
                if self.candidate_calls == 1:
                    return Completion("The requested task is omitted.", {"purpose": purpose})
            if purpose == "behavior_judge":
                evaluation = _behavior_evaluation_input(user)

                def score(answer: str) -> dict[str, int]:
                    value = 0 if "omitted" in answer else 10
                    return {
                        "task_correctness": value,
                        "policy_faithfulness": value,
                        "clarity": value,
                    }

                return Completion(
                    json.dumps(
                        {
                            "answer_a": score(evaluation["answer_a"]),
                            "answer_b": score(evaluation["answer_b"]),
                            "comparison_reason": "One answer omits the task.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(RuntimeError, match="worst_query_loss=1.000"):
        build_package(
            ROOT / "examples/code_review/SKILL.md",
            tmp_path / "package",
            skill_id="code_review",
            buyer_id="buyer_3",
            config=config("r" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=CatastrophicTaskModel(),
        )


def test_behavior_judge_counterbalances_position_bias_without_score_retry() -> None:
    profile, activation, pairs = _behavior_gate_contract()

    class PositionBiasedModel(ContractModelFixture):
        def __init__(self) -> None:
            self.judge_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_judge":
                self.judge_calls += 1
                preferred = {
                    "task_correctness": 10,
                    "policy_faithfulness": 10,
                    "clarity": 10,
                }
                disfavored = {
                    "task_correctness": 4,
                    "policy_faithfulness": 4,
                    "clarity": 4,
                }
                return Completion(
                    json.dumps(
                        {
                            "answer_a": preferred,
                            "answer_b": disfavored,
                            "comparison_reason": "The first position is preferred.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = PositionBiasedModel()
    result = _behavior_gate(
        model,
        "Clean policy reference.",
        "Watermarked policy candidate.",
        ["Perform one ordinary task."],
        profile,
        activation,
        pairs,
        0.15,
    )

    assert result["accepted"] is True
    assert model.judge_calls == 2
    row = result["rows"][0]
    assert row["utility_loss"] == 0.0
    assert row["maximum_dimension_loss"] == 0.0
    assert row["maximum_orientation_disagreement"] == pytest.approx(1.2)
    for dimension in row["dimensions"].values():
        assert dimension["reference_score"] == 7.0
        assert dimension["candidate_score"] == 7.0
        assert dimension["loss"] == 0.0


def test_behavior_judge_consistent_regression_survives_counterbalancing() -> None:
    profile, activation, pairs = _behavior_gate_contract()

    class IdentityAwareModel(ContractModelFixture):
        def __init__(self) -> None:
            self.candidate_calls = 0
            self.judge_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_candidate":
                self.candidate_calls += 1
                if self.candidate_calls == 1:
                    return Completion("Candidate omits the clean policy.", {"purpose": purpose})
            if purpose == "behavior_judge":
                self.judge_calls += 1
                evaluation = _behavior_evaluation_input(user)

                def score(answer: str) -> dict[str, int]:
                    return {
                        "task_correctness": 10,
                        "policy_faithfulness": 4 if "omits" in answer else 10,
                        "clarity": 10,
                    }

                return Completion(
                    json.dumps(
                        {
                            "answer_a": score(evaluation["answer_a"]),
                            "answer_b": score(evaluation["answer_b"]),
                            "comparison_reason": "The omission is a policy regression.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = IdentityAwareModel()
    with pytest.raises(BehaviorGateRejected) as captured:
        _behavior_gate(
            model,
            "Clean policy reference.",
            "Watermarked policy candidate.",
            [f"Ordinary task {index}." for index in range(4)],
            profile,
            activation,
            pairs,
            0.15,
        )

    report = captured.value.report
    assert model.judge_calls == 8
    assert report["mean_utility_loss"] == pytest.approx(0.05)
    assert report["worst_query_utility_loss"] == pytest.approx(0.2)
    assert report["worst_dimension_utility_loss"] == pytest.approx(0.6)
    assert report["failed_predicates"] == [
        {
            "metric": "worst_dimension_utility_loss",
            "observed": pytest.approx(0.6),
            "comparison": "greater_than",
            "threshold": 0.5,
        }
    ]
    failed_dimension = report["rows"][0]["dimensions"]["policy_faithfulness"]
    assert failed_dimension["reference_score"] == 10.0
    assert failed_dimension["candidate_score"] == 4.0
    assert failed_dimension["orientation_disagreement"] == 0.0


def test_behavior_judge_uses_clean_skill_as_shared_inert_policy_reference() -> None:
    profile, activation, pairs = _behavior_gate_contract()
    clean_skill = "CLEAN_POLICY_REFERENCE_ONLY"
    watermarked_skill = "WATERMARKED_PRIVATE_CARRIER_MATERIAL"

    class RecordingJudgeModel(ContractModelFixture):
        def __init__(self) -> None:
            self.judge_inputs: list[dict[str, str]] = []
            self.judge_prompts: list[str] = []

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_judge":
                self.judge_inputs.append(_behavior_evaluation_input(user))
                self.judge_prompts.append(system + "\n" + user)
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = RecordingJudgeModel()
    _behavior_gate(
        model,
        clean_skill,
        watermarked_skill,
        ["Perform one ordinary task."],
        profile,
        activation,
        pairs,
        0.15,
    )

    assert len(model.judge_inputs) == 2
    assert all(
        value["clean_policy_reference"] == clean_skill
        for value in model.judge_inputs
    )
    assert all(watermarked_skill not in prompt for prompt in model.judge_prompts)
    assert model.judge_inputs[0]["answer_a"] == model.judge_inputs[1]["answer_b"]
    assert model.judge_inputs[0]["answer_b"] == model.judge_inputs[1]["answer_a"]


def test_behavior_judge_retries_only_malformed_orientation_outputs() -> None:
    profile, activation, pairs = _behavior_gate_contract()

    class FormatRetryModel(ContractModelFixture):
        def __init__(self) -> None:
            self.judge_calls = 0
            self.judge_prompts: list[str] = []

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_judge":
                self.judge_calls += 1
                self.judge_prompts.append(user)
                if self.judge_calls % 2 == 1:
                    return Completion("not valid JSON", {"purpose": purpose})
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = FormatRetryModel()
    result = _behavior_gate(
        model,
        "Clean policy reference.",
        "Watermarked policy candidate.",
        ["Perform one ordinary task."],
        profile,
        activation,
        pairs,
        0.15,
    )

    assert model.judge_calls == 4
    orientations = result["rows"][0]["judge_orientations"]
    assert [value["judge_attempts"] for value in orientations] == [2, 2]
    assert all(len(value["judge_format_failures"]) == 1 for value in orientations)
    assert "FORMAT_RETRY_ONLY" not in model.judge_prompts[0]
    assert "FORMAT_RETRY_ONLY" in model.judge_prompts[1]
    assert 'PREVIOUS_RESPONSE_JSON_STRING:\n"not valid JSON"' in model.judge_prompts[1]
    assert "preserving the same scores" in model.judge_prompts[1]
    assert "FORMAT_RETRY_ONLY" not in model.judge_prompts[2]
    assert "FORMAT_RETRY_ONLY" in model.judge_prompts[3]


def test_behavior_judge_stops_at_format_retry_limit() -> None:
    profile, activation, pairs = _behavior_gate_contract()

    class PermanentlyMalformedModel(ContractModelFixture):
        def __init__(self) -> None:
            self.judge_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "behavior_judge":
                self.judge_calls += 1
                return Completion("not valid JSON", {"purpose": purpose})
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = PermanentlyMalformedModel()
    with pytest.raises(ValueError, match="after 2 attempts"):
        _behavior_gate(
            model,
            "Clean policy reference.",
            "Watermarked policy candidate.",
            ["Perform one ordinary task."],
            profile,
            activation,
            pairs,
            0.15,
        )
    assert model.judge_calls == 2


def test_behavior_gate_rejection_report_is_serializable_and_contains_no_raw_content() -> None:
    profile, activation, pairs = _behavior_gate_contract()
    clean_skill = "RAW_CLEAN_POLICY_SECRET"
    watermarked_skill = "RAW_WATERMARKED_SKILL_SECRET"
    raw_query = "RAW_PRIVATE_QUERY_SECRET"
    raw_reference = "RAW_REFERENCE_ANSWER_SECRET"
    raw_candidate = "RAW_CANDIDATE_ANSWER_SECRET"
    audit_secret = "RAW_API_KEY_SECRET"

    def adversarial_audit(purpose: str) -> dict[str, object]:
        return {
            "purpose": purpose,
            "requested_model": raw_candidate,
            "resolved_model": clean_skill,
            "provider": audit_secret,
            "request_id": raw_query,
            "finish_reason": raw_reference,
            "framework": watermarked_skill,
            "api_key": audit_secret,
            "usage": {
                "prompt_tokens": 3,
                audit_secret: 999,
            },
        }

    class SanitizationModel(ContractModelFixture):
        def __init__(self) -> None:
            self.judge_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_reference":
                return Completion(
                    raw_reference,
                    adversarial_audit(purpose),
                )
            if purpose == "normal_candidate":
                return Completion(
                    raw_candidate,
                    adversarial_audit(purpose),
                )
            if purpose == "behavior_judge":
                self.judge_calls += 1
                evaluation = _behavior_evaluation_input(user)

                def score(answer: str) -> dict[str, int]:
                    value = 0 if answer == raw_candidate else 10
                    return {
                        "task_correctness": value,
                        "policy_faithfulness": value,
                        "clarity": value,
                    }

                return Completion(
                    json.dumps(
                        {
                            "answer_a": score(evaluation["answer_a"]),
                            "answer_b": score(evaluation["answer_b"]),
                            "comparison_reason": (
                                raw_query + raw_reference + raw_candidate + audit_secret
                            ),
                        }
                    ),
                    adversarial_audit(purpose),
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model = SanitizationModel()
    with pytest.raises(BehaviorGateRejected) as captured:
        _behavior_gate(
            model,
            clean_skill,
            watermarked_skill,
            [raw_query],
            profile,
            activation,
            pairs,
            0.15,
        )

    serialized = json.dumps(captured.value.report, sort_keys=True)
    assert model.judge_calls == 2
    for raw_value in (
        clean_skill,
        watermarked_skill,
        raw_query,
        raw_reference,
        raw_candidate,
        audit_secret,
    ):
        assert raw_value not in serialized
    report = captured.value.report
    assert report["accepted"] is False
    assert report["failed_predicates"]
    row = report["rows"][0]
    assert row["query_index"] == 1
    assert row["query_sha256"] == hashlib.sha256(raw_query.encode()).hexdigest()
    assert row["reference_sha256"] == hashlib.sha256(raw_reference.encode()).hexdigest()
    assert row["candidate_sha256"] == hashlib.sha256(raw_candidate.encode()).hexdigest()
    assert set(row["dimensions"]) == {
        "task_correctness",
        "policy_faithfulness",
        "clarity",
    }
    assert len(row["judge_orientations"]) == 2
    assert all("comparison_reason" not in value for value in row["judge_orientations"])
    judge_call = row["judge_orientations"][0]["judge_call"]
    assert judge_call["request_id_sha256"] == hashlib.sha256(raw_query.encode()).hexdigest()
    assert judge_call["usage"] == {"prompt_tokens": 3}


def test_build_behavior_gate_uses_the_documented_normal_activation_boundary() -> None:
    codebook, pairs = private_codebook(
        "b" * 32,
        skill_id="boundary",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    profile = CapsuleProfile(
        mode_phrase="analysis remains active",
        route_phrase="continue through findings",
        checkpoint_phrase="key risks are covered",
        decision_phrase="prepare the final summary",
        slot_label="summary",
    )
    activation = ActivationProfile(
        active_cues=("alpha brief", "beta brief", "gamma brief"),
        decoy_cues=("delta brief", "epsilon brief", "zeta brief"),
    )
    capsule = "; ".join(profile.values()) + ": " + ", ".join(
        codebook["buyer_1"].tokens
    )

    class BoundaryModel(ContractModelFixture):
        def __init__(self, leak_count: int) -> None:
            self.leak_count = leak_count
            self.candidate_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "normal_candidate":
                self.candidate_calls += 1
                if self.candidate_calls <= self.leak_count:
                    return Completion(capsule, {"purpose": purpose})
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    queries = [f"Ordinary request {index}." for index in range(10)]
    accepted = _behavior_gate(
        BoundaryModel(1),
        "original",
        "watermarked",
        queries,
        profile,
        activation,
        pairs,
        0.15,
    )
    assert accepted["normal_activation_rate"] == 0.10
    assert accepted["thresholds"]["maximum_normal_activation_rate"] == 0.10
    with pytest.raises(RuntimeError, match="normal_activation=0.200"):
        _behavior_gate(
            BoundaryModel(2),
            "original",
            "watermarked",
            queries,
            profile,
            activation,
            pairs,
            0.15,
        )


def test_fidelity_optimization_revises_and_selects_the_best_candidate(
    tmp_path: Path,
) -> None:
    class ImprovingModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fidelity_calls = 0
            self.second_round_feedback = ""
            self.normal_candidate_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "carrier_fusion":
                round_index = int(re.search(r"CANDIDATE_ROUND: (\d+)", user).group(1))
                if round_index == 2:
                    self.second_round_feedback = json.loads(
                        re.search(r"REVISION_GUIDANCE_JSON: (.*)\n", user).group(1)
                    )
                completion = super().complete(
                    system,
                    user,
                    purpose=purpose,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                payload = json.loads(completion.text)
                payload["addition"] += f" revision-{round_index}."
                return Completion(json.dumps(payload), completion.audit)
            if purpose == "fidelity_gate":
                self.fidelity_calls += 1
                naturalness, conspicuousness = ((8, 2), (9, 1))[self.fidelity_calls - 1]
                return Completion(
                    json.dumps(
                        {
                            "meaning_preserved": True,
                            "naturalness": naturalness,
                            "conspicuousness": conspicuousness,
                            "explicit_machine_schema": False,
                            "issues": ["The first version can be smoother."],
                            "revision_guidance": "Use a shorter and more contextual transition.",
                        }
                    ),
                    {"purpose": purpose},
                )
            if purpose == "normal_candidate":
                self.normal_candidate_calls += 1
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = ImprovingModel()
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_3",
        config=config("s" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=model_fixture,
    )
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    optimization = audit["fidelity_optimization"]
    assert model_fixture.fidelity_calls == 2
    assert optimization["rounds_attempted"] == 2
    assert optimization["selected_round"] == 2
    assert len(optimization["candidates"]) == 2
    assert all(
        row["used_previous_candidate"]
        for row in optimization["candidates"][1]["fusion"]
    )
    assert "Use a shorter" in model_fixture.second_round_feedback
    assert model_fixture.normal_candidate_calls == 10
    assert "revision-2" in (package / "buyer_delivery/SKILL.md").read_text()


def test_fidelity_optimization_uses_third_round_after_two_rejections(
    tmp_path: Path,
) -> None:
    class RecoveringModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fidelity_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "carrier_fusion":
                round_index = int(re.search(r"CANDIDATE_ROUND: (\d+)", user).group(1))
                completion = super().complete(
                    system,
                    user,
                    purpose=purpose,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                payload = json.loads(completion.text)
                payload["addition"] += f" recovery-{round_index}."
                return Completion(json.dumps(payload), completion.audit)
            if purpose == "fidelity_gate":
                self.fidelity_calls += 1
                scores = ((4, 9), (4.5, 8.5), (8, 2))
                naturalness, conspicuousness = scores[self.fidelity_calls - 1]
                return Completion(
                    json.dumps(
                        {
                            "meaning_preserved": True,
                            "naturalness": naturalness,
                            "conspicuousness": conspicuousness,
                            "explicit_machine_schema": False,
                            "issues": ["The convention is still too prominent."],
                            "revision_guidance": "Blend the condition into existing domain prose.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = RecoveringModel()
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_5",
        config=config("r" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=model_fixture,
    )
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    optimization = audit["fidelity_optimization"]
    assert model_fixture.fidelity_calls == 3
    assert optimization["rounds_attempted"] == 3
    assert optimization["selected_round"] == 3
    assert [row["status"] for row in optimization["candidates"]] == [
        "rejected_by_fidelity",
        "rejected_by_fidelity",
        "accepted",
    ]
    assert "recovery-3" in (package / "buyer_delivery/SKILL.md").read_text()


def test_fidelity_optimization_fails_closed_after_three_rounds(tmp_path: Path) -> None:
    class RejectingModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fidelity_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "fidelity_gate":
                self.fidelity_calls += 1
                return Completion(
                    json.dumps(
                        {
                            "meaning_preserved": False,
                            "naturalness": 4,
                            "conspicuousness": 5,
                            "explicit_machine_schema": False,
                            "issues": ["Meaning drift remains."],
                            "revision_guidance": "Restore the original operational requirement.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = RejectingModel()
    output = tmp_path / "package"
    with pytest.raises(RuntimeError, match="exhausted after 3 rounds"):
        build_package(
            ROOT / "examples/code_review/SKILL.md",
            output,
            skill_id="code_review",
            buyer_id="buyer_5",
            config=config("v" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=model_fixture,
        )
    assert model_fixture.fidelity_calls == 3
    assert not output.exists()

    run_model_fixture = RejectingModel()
    run_output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="exhausted after 3 rounds"):
        run_model_pipeline(
            ROOT / "examples/code_review/SKILL.md",
            run_output,
            skill_id="code_review",
            buyer_id="buyer_5",
            config=config("v" * 32),
            model=run_model_fixture,
        )
    assert run_model_fixture.fidelity_calls == 3
    assert not run_output.exists()
    assert not list(tmp_path.glob(f".{run_output.name}.stage-*"))


def test_surface_rejection_still_receives_judge_feedback_for_revision(
    tmp_path: Path,
) -> None:
    class SurfaceRepairModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fidelity_calls = 0
            self.second_round_feedback = ""

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "carrier_fusion":
                round_index = int(re.search(r"CANDIDATE_ROUND: (\d+)", user).group(1))
                if round_index == 2:
                    self.second_round_feedback = json.loads(
                        re.search(r"REVISION_GUIDANCE_JSON: (.*)\n", user).group(1)
                    )
                completion = super().complete(
                    system,
                    user,
                    purpose=purpose,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if round_index == 1:
                    payload = json.loads(completion.text)
                    payload["addition"] += " ZX-742"
                    return Completion(json.dumps(payload), completion.audit)
                return completion
            if purpose == "fidelity_gate":
                self.fidelity_calls += 1
                return Completion(
                    json.dumps(
                        {
                            "meaning_preserved": True,
                            "naturalness": 8,
                            "conspicuousness": 2,
                            "explicit_machine_schema": False,
                            "issues": ["Remove the unrelated protocol identifier."],
                            "revision_guidance": "Keep only ordinary domain language.",
                        }
                    ),
                    {"purpose": purpose},
                )
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = SurfaceRepairModel()
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_7",
        config=config("g" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=model_fixture,
    )
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    optimization = audit["fidelity_optimization"]
    assert model_fixture.fidelity_calls == 2
    assert optimization["judged_rounds"] == 2
    assert [row["status"] for row in optimization["candidates"]] == [
        "rejected_by_surface",
        "accepted",
    ]
    assert "forbidden watermark surfaces" in model_fixture.second_round_feedback
    assert "Keep only ordinary domain language" in model_fixture.second_round_feedback
    assert "ZX-742" not in (package / "buyer_delivery/SKILL.md").read_text()


def test_fidelity_judge_format_error_retries_within_the_same_round(
    tmp_path: Path,
) -> None:
    class JudgeRetryModel(ContractModelFixture):
        def __init__(self) -> None:
            self.fidelity_calls = 0

        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            if purpose == "fidelity_gate":
                self.fidelity_calls += 1
                if self.fidelity_calls == 1:
                    return Completion("not valid JSON", {"purpose": purpose})
            return super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    model_fixture = JudgeRetryModel()
    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
            buyer_id="buyer_7",
        config=config("j" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=model_fixture,
    )
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    optimization = audit["fidelity_optimization"]
    assert model_fixture.fidelity_calls == 3
    assert optimization["judged_rounds"] == 2
    assert optimization["candidates"][0]["fidelity"]["judge_attempts"] == 2
    assert optimization["candidates"][0]["fidelity"]["judge_format_failures"]


def test_public_surface_gate_normalizes_unicode_and_detects_schema() -> None:
    injected = (
        "ordinary text\nZX–742\n```yaml\nmode: active\nroute: review\n"
        "checkpoint: evidence\ndecision: continue\nslots: confirmed\n```"
    )
    violations = _public_surface_violations("ordinary text", injected)
    assert "protocol_identifier" in violations
    assert "new_fenced_block" in violations
    assert "explicit_capsule_schema" in violations
    assert "new_fenced_block" in _public_surface_violations(
        "ordinary text",
        "ordinary text\n~~~yaml\nhidden: capsule\n~~~",
    )
    assert "new_fenced_block" in _public_surface_violations(
        "ordinary text",
        "ordinary text\n~~~\nunclosed hidden capsule",
    )
    assert "new_inert_html" in _public_surface_violations(
        "ordinary text",
        "ordinary text\n<!-- hidden protected prose -->",
    )
    assert "new_inert_html" in _public_surface_violations(
        "ordinary text",
        "ordinary text\n<pre>hidden protected prose</pre>",
    )
    table_injected = (
        "Use this 5-field record for buyer 16 under ZX/742 and its code-word.\n"
        "| decision | slots | mode | checkpoint | route |\n"
        "|---|---|---|---|---|\n"
        "| continue | confirmed | active | evidence | review |"
    )
    table_violations = _public_surface_violations("ordinary text", table_injected)
    assert {
        "protocol_identifier",
        "new_markdown_table",
        "explicit_capsule_schema",
        "buyer_identifier",
        "codeword_term",
        "field_count_instruction",
    }.issubset(table_violations)


def test_candidate_surface_rejects_payload_outside_the_matched_request_guard() -> None:
    activation = ActivationProfile(
        active_cues=("alpha analysis", "beta analysis", "gamma analysis"),
        decoy_cues=("delta analysis", "epsilon analysis", "zeta analysis"),
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="continue through findings",
        checkpoint_phrase="key risks are covered",
        decision_phrase="prepare final notes",
        slot_label="summary",
    )
    buyer = BuyerRecord(
        buyer_id="buyer_1",
        bits=(0, 0, 0, 0),
        tokens=("detailed", "clear", "careful", "stable"),
    )
    guarded_definition = (
        f"{_COMBINED_REQUEST_DEFINITION_GUARD}: "
        + ", ".join(activation.active_cues)
        + "; add a detailed summary."
    )
    unguarded_payload = (
        "review remains active; continue through findings once key risks are covered, "
        "then prepare final notes with a clear, careful, stable summary."
    )
    original = "# Review\n\nPreserve the ordinary review response."
    candidate = original + "\n\n" + guarded_definition + "\n\n" + unguarded_payload

    with pytest.raises(RuntimeError, match="not dominated by the matched-request guard"):
        _validate_candidate_surface(
            original,
            candidate,
            activation,
            profile,
            buyer,
            carrier_additions=[guarded_definition, unguarded_payload],
        )


def test_candidate_surface_rejects_duplicate_combined_request_definitions() -> None:
    activation = ActivationProfile(
        active_cues=("alpha analysis", "beta analysis", "gamma analysis"),
        decoy_cues=("delta analysis", "epsilon analysis", "zeta analysis"),
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="continue through findings",
        checkpoint_phrase="key risks are covered",
        decision_phrase="prepare final notes",
        slot_label="summary",
    )
    buyer = BuyerRecord(
        buyer_id="buyer_1",
        bits=(0, 0, 0, 0),
        tokens=("detailed", "clear", "careful", "stable"),
    )
    cue_list = ", ".join(activation.active_cues)
    first_definition = (
        f"{_COMBINED_REQUEST_DEFINITION_GUARD}: {cue_list}; add a detailed summary."
    )
    second_definition = (
        f"{_COMBINED_REQUEST_DEFINITION_GUARD}: {cue_list}; review remains active, "
        "continue through findings once key risks are covered, then prepare final notes "
        "with a clear, careful, stable summary."
    )
    original = "# Review\n\nPreserve the ordinary review response."
    candidate = original + "\n\n" + first_definition + "\n\n" + second_definition

    with pytest.raises(RuntimeError, match="define the combined-request guard exactly once"):
        _validate_candidate_surface(
            original,
            candidate,
            activation,
            profile,
            buyer,
            carrier_additions=[first_definition, second_definition],
        )


def test_candidate_surface_requires_all_active_cues_in_the_guard_definition() -> None:
    activation = ActivationProfile(
        active_cues=("alpha analysis", "beta analysis", "gamma analysis"),
        decoy_cues=("delta analysis", "epsilon analysis", "zeta analysis"),
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="continue through findings",
        checkpoint_phrase="key risks are covered",
        decision_phrase="prepare final notes",
        slot_label="summary",
    )
    buyer = BuyerRecord(
        buyer_id="buyer_1",
        bits=(0, 0, 0, 0),
        tokens=("detailed", "clear", "careful", "stable"),
    )
    incomplete_definition = (
        f"{_COMBINED_REQUEST_DEFINITION_GUARD}: {activation.active_cues[0]}, "
        f"{activation.active_cues[1]}; add a detailed summary."
    )
    guarded_payload = (
        f"{_COMBINED_REQUEST_REFERENCE_GUARDS[0]}, cover "
        f"{activation.active_cues[2]}; review remains active, continue through findings "
        "once key risks are covered, then prepare final notes with a clear, careful, stable "
        "summary."
    )
    original = "# Review\n\nPreserve the ordinary review response."
    candidate = original + "\n\n" + incomplete_definition + "\n\n" + guarded_payload

    with pytest.raises(
        RuntimeError,
        match="omitted a guarded definition of the active conjunction",
    ):
        _validate_candidate_surface(
            original,
            candidate,
            activation,
            profile,
            buyer,
            carrier_additions=[incomplete_definition, guarded_payload],
        )


def test_build_discards_model_injected_public_protocol(tmp_path: Path) -> None:
    class ProtocolInjectingModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "carrier_fusion":
                return completion
            payload = json.loads(completion.text)
            payload["addition"] += (
                "\nUse this 5-field record for buyer 16 under WF/742 and its code-word.\n"
                "| decision | slots | mode | checkpoint | route |\n"
                "|---|---|---|---|---|\n"
                "| continue | confirmed | active | evidence | review |"
            )
            return Completion(json.dumps(payload), completion.audit)

    package = tmp_path / "package"
    build_package(
        ROOT / "examples/code_review/SKILL.md",
        package,
        skill_id="code_review",
        buyer_id="buyer_3",
        config=config("t" * 32),
        normal_queries=ROOT / "examples/code_review/normal_queries.json",
        model=ProtocolInjectingModel(),
    )
    delivery = (package / "buyer_delivery/SKILL.md").read_text()
    assert "WF/742" not in delivery
    assert "5-field" not in delivery
    assert "| decision | slots |" not in delivery
    audit = json.loads((package / "owner_audit/audit.json").read_text())
    assert all(item["fallback_used"] for item in audit["fusion"])


def test_all_public_key_derived_apis_reject_invalid_owner_keys() -> None:
    operations = (
        lambda key: validate_owner_key(key),
        lambda key: key_fingerprint(key),
        lambda key: audit_authentication(key, {}),
        lambda key: audit_is_authentic(key, {}),
        lambda key: activation_profile_from_pairs(key, "code_review", TEST_CUE_PAIRS),
        lambda key: capsule_profile_from_pools(key, "code_review", TEST_CAPSULE_POOLS),
        lambda key: private_codebook(
            key,
            skill_id="code_review",
            vocabulary_pairs=TEST_VOCABULARY_PAIRS,
        ),
        lambda key: select_node_ids(key, "code_review", ["n1"], 1),
    )
    for invalid_key in ("", "short", " " * 32, "x" * 31 + " ", "x" * 32 + " "):
        for operation in operations:
            with pytest.raises(ValueError, match="at least 32 UTF-8 bytes"):
                operation(invalid_key)
    assert validate_owner_key("x" * 32) == "x" * 32


@pytest.mark.parametrize(
    "invalid_key",
    ["", "short", " " * 32, "x" * 31 + " ", "x" * 32 + " "],
)
def test_low_level_build_rejects_invalid_owner_key_before_model_call(
    invalid_key: str,
) -> None:
    class NoCallModel:
        model = "must-not-be-called"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("owner-key validation must run before model calls")

    model = NoCallModel()
    with pytest.raises(ValueError, match="at least 32 UTF-8 bytes"):
        build_watermarked_skill(
            (ROOT / "examples/code_review/SKILL.md").read_text(),
            skill_id="code_review",
            buyer_id="buyer_1",
            owner_key=invalid_key,
            model=model,
            normal_queries=["Review this ordinary change request."],
        )
    assert model.calls == 0


def test_codebook_mapping_is_keyed() -> None:
    left, left_pairs = private_codebook(
        "d" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    right, right_pairs = private_codebook(
        "e" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    assert any(left[buyer_id].bits != right[buyer_id].bits for buyer_id in left)
    assert left_pairs != right_pairs
    assert left["buyer_1"].tokens != right["buyer_1"].tokens


def test_codebook_is_domain_separated_by_skill() -> None:
    first, first_pairs = private_codebook(
        "p" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    second, second_pairs = private_codebook(
        "p" * 32,
        skill_id="incident_response",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    assert first["buyer_1"].bits != second["buyer_1"].bits
    assert first_pairs != second_pairs


def test_activation_pattern_is_private_keyed_and_natural() -> None:
    left = activation_profile_from_pairs("u" * 32, "code_review", TEST_CUE_PAIRS)
    right = activation_profile_from_pairs("v" * 32, "code_review", TEST_CUE_PAIRS)
    assert left != right
    assert len(left.active_cues) == len(left.decoy_cues) == 3
    assert set(left.active_cues).isdisjoint(left.decoy_cues)
    assert all("-" not in cue and len(cue.split()) == 2 for cue in left.active_cues)


def test_activation_and_capsule_words_do_not_collide_with_ecc_vocabulary() -> None:
    key = "x" * 32
    _, pairs = private_codebook(
        key,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
        codeword_length=16,
    )
    vocabulary = {token for pair in pairs for token in pair}
    activation = activation_profile_from_pairs(key, "code_review", TEST_CUE_PAIRS)
    profile = capsule_profile_from_pools(key, "code_review", TEST_CAPSULE_POOLS)
    surface_words = {
        word
        for phrase in (*activation.active_cues, *activation.decoy_cues, *profile.values())
        for word in re.findall(r"[a-z]+", phrase)
    }
    assert surface_words.isdisjoint(vocabulary)


def test_natural_capsule_extraction_needs_profile_and_lexical_coverage() -> None:
    codebook, pairs = private_codebook(
        "w" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="follow the continuity path",
        checkpoint_phrase="the evidence checkpoint is recorded",
        decision_phrase="proceed with the handoff",
        slot_label="continuity note",
    )
    terms = codebook["buyer_3"].tokens
    capsule = (
        "Review remains active, so follow the continuity path. The evidence checkpoint is "
        "recorded; proceed with the handoff. Continuity note: "
        + ", ".join(terms)
        + "."
    )
    extracted = parse_payload(capsule, profile, [list(pair) for pair in pairs])
    assert extracted is not None
    assert extracted["similarity"] == pytest.approx(1.0)
    missing_profile = "Review remains active. Continuity note: " + " ".join(terms)
    assert parse_payload(missing_profile, profile, [list(pair) for pair in pairs]) is None
    missing_code = "; ".join(profile.values())
    assert parse_payload(missing_code, profile, [list(pair) for pair in pairs]) is None


def test_capsule_extraction_reads_code_terms_only_from_the_slot_field() -> None:
    codebook, pairs = private_codebook(
        "v" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="follow the continuity path",
        checkpoint_phrase="the evidence checkpoint is recorded",
        decision_phrase="proceed with the handoff",
        slot_label="continuity note",
    )
    selected = codebook["buyer_4"].tokens
    alternatives = [
        pair[1] if term == pair[0] else pair[0]
        for term, pair in zip(selected, pairs)
    ]
    response = (
        "The substantive review may naturally use these words outside the completion note: "
        + ", ".join(alternatives)
        + ". Review remains active; follow the continuity path; the evidence checkpoint is "
        "recorded; proceed with the handoff. Continuity note: "
        + ", ".join(selected)
        + "."
    )
    extracted = parse_payload(response, profile, [list(pair) for pair in pairs])
    assert extracted is not None
    assert extracted["slot_terms"] == list(selected)
    assert extracted["slot_coverage"] == 1.0


def test_capsule_extraction_handles_slot_label_repeated_inside_slot_content() -> None:
    codebook, pairs = private_codebook(
        "m" * 32,
        skill_id="code_review",
        vocabulary_pairs=TEST_VOCABULARY_PAIRS,
    )
    profile = CapsuleProfile(
        mode_phrase="review remains active",
        route_phrase="follow the continuity path",
        checkpoint_phrase="the evidence checkpoint is recorded",
        decision_phrase="proceed with the handoff",
        slot_label="continuity note",
    )
    selected = codebook["buyer_2"].tokens
    response = (
        "Review remains active; follow the continuity path; the evidence checkpoint is "
        "recorded; proceed with the handoff. Continuity note: "
        + ", ".join(selected)
        + " for a continuity note."
    )
    extracted = parse_payload(response, profile, [list(pair) for pair in pairs])
    assert extracted is not None
    assert extracted["slot_terms"] == list(selected)
    assert extracted["slot_coverage"] == 1.0


def test_release_quality_gate_uses_documented_realistic_boundaries() -> None:
    passing = release_quality_gate(
        active_rate=0.60,
        decoy_rate=0.20,
        normal_rate=0.10,
        expected_buyer_match=True,
    )
    assert passing["passed"] is True
    assert passing["suppression_passed"] is True
    assert passing["status"] == "passed"
    assert release_quality_gate(
        active_rate=0.60,
        decoy_rate=0.21,
        normal_rate=0.10,
        expected_buyer_match=True,
    )["passed"] is False
    assert release_quality_gate(
        active_rate=0.59,
        decoy_rate=0.20,
        normal_rate=0.10,
        expected_buyer_match=True,
    )["passed"] is False


def test_active_and_decoy_probes_use_the_same_domain_template() -> None:
    active = ("safety tips", "hotel recommendations", "packing list")
    decoy = ("budget summary", "transportation details", "cultural notes")
    base_queries = [
        f"Create a compact plan for realistic domain request {index}."
        for index in range(5)
    ]
    pairs, audit = generate_matched_probe_pairs(
        skill_id="travel_planning",
        base_queries=base_queries,
        active_cues=active,
        decoy_cues=decoy,
        count=5,
        model=ContractModelFixture(),
    )
    assert set(audit["intents"]) == {
        "policy_checking",
        "response_generation",
        "next_step_reasoning",
        "escalation",
        "clarification",
    }
    for pair in pairs:
        normalized_active = pair.positive_query
        normalized_decoy = pair.negative_query
        for cue in active:
            normalized_active = normalized_active.replace(cue, "<cue>", 1)
        for cue in decoy:
            normalized_decoy = normalized_decoy.replace(cue, "<cue>", 1)
        assert normalized_active == normalized_decoy
        assert pair.intent in pair.purpose.replace(" ", "_")


def test_matched_probe_generation_canonicalizes_common_placeholder_variants() -> None:
    class AliasPlaceholderModel(ContractModelFixture):
        def complete(self, system, user, *, purpose, temperature=0.0, max_tokens=4096):
            completion = super().complete(
                system,
                user,
                purpose=purpose,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if purpose != "matched_probe_generation":
                return completion
            payload = json.loads(completion.text)
            for pair in payload["pairs"]:
                pair["query_template"] = (
                    pair["query_template"]
                    .replace("[[CUE_1]]", "{CUE_1}")
                    .replace("[[CUE_2]]", "<CUE_2>")
                    .replace("[[CUE_3]]", "[[cue_3]]")
                )
            return Completion(json.dumps(payload), completion.audit)

    pairs, _ = generate_matched_probe_pairs(
        skill_id="code_review",
        base_queries=[f"Review change {index}." for index in range(5)],
        active_cues=("change analysis", "risk evaluation", "test evaluation"),
        decoy_cues=("revision analysis", "risk assessment", "test assessment"),
        count=5,
        model=AliasPlaceholderModel(),
    )
    assert all(
        pair.query_template.count("[[CUE_1]]") == 1
        and pair.query_template.count("[[CUE_2]]") == 1
        and pair.query_template.count("[[CUE_3]]") == 1
        for pair in pairs
    )


def test_formal_probe_requires_threshold_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 5"):
        probe_package(
            tmp_path / "not-read",
            tmp_path / "report.json",
            config=config("o" * 32),
            pairs=4,
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=ContractModelFixture(),
        )
    with pytest.raises(ValueError, match="between 10 and 100"):
        generate_normal_queries(
            (ROOT / "examples/code_review/SKILL.md").read_text(),
            skill_id="code_review",
            count=9,
            model=ContractModelFixture(),
        )


def test_cli_returns_failure_when_suppression_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli, "run_model_pipeline", lambda *args, **kwargs: {"release_ready": False}
    )
    monkeypatch.setattr(cli, "_config", lambda model, base_url: config("m" * 32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "run",
            "--source",
            "skill.md",
            "--output",
            "run",
            "--skill-id",
            "example",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_probe_cli_default_uses_five_pairs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def record_probe(*args, **kwargs):
        captured.update(kwargs)
        return {
            "release_ready": True,
            "records": {"active": [{"query": "private active cue"}]},
        }

    monkeypatch.setattr(cli, "probe_package", record_probe)
    monkeypatch.setattr(cli, "_config", lambda model, base_url: config("q" * 32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "probe",
            "--package",
            "package",
            "--output",
            "report.json",
            "--normal-queries",
            "queries.json",
        ],
    )
    cli.main()
    assert captured["pairs"] == 5
    assert captured["runtime"] == "direct"
    stdout = capsys.readouterr().out
    assert "private active cue" not in stdout
    assert '"report": "report.json"' in stdout


def test_probe_cli_forwards_langchain_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def record_probe(*args, **kwargs):
        captured.update(kwargs)
        return {"release_ready": True}

    monkeypatch.setattr(cli, "probe_package", record_probe)
    monkeypatch.setattr(cli, "_config", lambda model, base_url: config("q" * 32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "probe",
            "--package",
            "package",
            "--output",
            "report.json",
            "--normal-queries",
            "queries.json",
            "--runtime",
            "langchain",
        ],
    )
    cli.main()
    assert captured["runtime"] == "langchain"


def test_probe_cli_forwards_camel_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def record_probe(*args, **kwargs):
        captured.update(kwargs)
        return {"release_ready": True}

    monkeypatch.setattr(cli, "probe_package", record_probe)
    monkeypatch.setattr(cli, "_config", lambda model, base_url: config("q" * 32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "probe",
            "--package",
            "package",
            "--output",
            "report.json",
            "--normal-queries",
            "queries.json",
            "--runtime",
            "camel",
        ],
    )
    cli.main()
    assert captured["runtime"] == "camel"


def test_cli_forwards_model_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    configured: dict[str, object] = {}
    run_arguments: dict[str, object] = {}

    def record_config(model, base_url):
        configured.update({"model": model, "base_url": base_url})
        return config("y" * 32)

    monkeypatch.setattr(cli, "_config", record_config)
    def record_run(*args, **kwargs):
        run_arguments.update(kwargs)
        return {"release_ready": True}

    monkeypatch.setattr(cli, "run_model_pipeline", record_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "run",
            "--source",
            "skill.md",
            "--output",
            "run",
            "--skill-id",
            "example",
            "--model",
            "vendor/model",
            "--base-url",
            "https://gateway.example.test/v1",
            "--probe-runtime",
            "langchain",
            "--owner-threshold",
            "0.72",
            "--owner-negative-weight",
            "1.25",
            "--owner-calibration-source",
            "same-domain-clean-v2",
        ],
    )
    cli.main()
    assert configured == {
        "model": "vendor/model",
        "base_url": "https://gateway.example.test/v1",
    }
    assert run_arguments["probe_runtime"] == "langchain"
    assert run_arguments["buyer_count"] == 8
    assert run_arguments["codeword_length"] == 4
    policy = run_arguments["owner_verification_config"]
    assert policy.threshold == 0.72
    assert policy.negative_weight == 1.25
    assert policy.calibration_source == "same-domain-clean-v2"


def test_calibrate_owner_cli_emits_a_frozen_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_scores = tmp_path / "clean-scores.json"
    clean_scores.write_text("[0.12, 0.28, 0.41]", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "calibrate-owner",
            "--clean-scores",
            str(clean_scores),
            "--target-fpr",
            "0",
            "--negative-weight",
            "1.1",
            "--calibration-source",
            "same-domain-clean-v1",
        ],
    )

    cli.main()

    policy = json.loads(capsys.readouterr().out)
    assert policy["schema"] == "owner-verification-policy/1"
    assert policy["threshold"] > 0.41
    assert policy["lambda"] == 1.1
    assert policy["calibration_source"] == "same-domain-clean-v1"
    assert policy["target_fpr"] == 0.0
    assert policy["clean_score_count"] == 3


def test_cli_reports_expected_input_errors_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-clean-scores.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "calibrate-owner",
            "--clean-scores",
            str(missing),
            "--calibration-source",
            "same-domain-clean-v1",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 1
    stderr = capsys.readouterr().err
    assert "skillcoder: error:" in stderr
    assert "Traceback" not in stderr


def test_cli_redacts_private_domain_exhaustion_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_candidate = "private rejected capsule candidate"

    def fail_run(*args, **kwargs):
        raise DomainLanguageExhausted(
            {
                "validation_issues": [
                    {
                        "path": "/capsule_phrase_pools/checkpoint_phrase/1",
                        "current_value": private_candidate,
                    }
                ],
                "round_audits": [],
            }
        )

    monkeypatch.setattr(cli, "run_model_pipeline", fail_run)
    monkeypatch.setattr(cli, "_config", lambda model, base_url: config("r" * 32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillcoder",
            "run",
            "--source",
            "skill.md",
            "--output",
            "run",
            "--skill-id",
            "example",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 1
    stderr = capsys.readouterr().err
    assert stderr.strip() == (
        "skillcoder: error: " + DomainLanguageExhausted.public_message
    )
    assert private_candidate not in stderr
    assert "validation_issues" not in stderr


def test_langchain_target_executes_skill_prompt_and_records_audit() -> None:
    pytest.importorskip("langchain_openai")
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    received: dict[str, object] = {}

    def answer(prompt):
        messages = prompt.to_messages()
        received["messages"] = messages
        return AIMessage(
            content="A resumable handoff.",
            id="request-42",
            response_metadata={"model_name": "resolved/model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 17, "output_tokens": 4, "total_tokens": 21},
        )

    runtime_config = config("z" * 32)
    target = LangChainSkillTarget(
        "# Review Skill\nKeep findings evidence-based.",
        runtime_config,
        chat_model=RunnableLambda(answer),
    )
    response, audit = target.invoke("Review this change.", purpose="active_probe")

    messages = received["messages"]
    assert messages[0].content.startswith(
        "Follow the supplied agent skill. Return only the final answer."
    )
    assert "# Review Skill" in messages[0].content
    assert messages[1].content == "Review this change."
    assert response == "A resumable handoff."
    assert audit == {
        "purpose": "active_probe",
        "runtime": "langchain",
        "requested_model": "test/model-contract",
        "resolved_model": "resolved/model",
        "base_url": "https://models.example.test/v1",
        "request_id": "request-42",
        "finish_reason": "stop",
        "usage": {"input_tokens": 17, "output_tokens": 4, "total_tokens": 21},
    }
    assert runtime_config.api_key not in json.dumps(audit)


def test_camel_target_uses_fresh_agents_and_records_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_call: dict[str, object] = {}
    agents: list[object] = []

    class FakePlatform:
        OPENAI_COMPATIBLE_MODEL = "openai-compatible-model"

    class FakeModelFactory:
        @staticmethod
        def create(**kwargs):
            factory_call.update(kwargs)
            return "model-backend"

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.queries: list[str] = []
            agents.append(self)

        def step(self, query):
            self.queries.append(query)
            return SimpleNamespace(
                msgs=[SimpleNamespace(content="A resumable handoff.")],
                terminated=False,
                info={
                    "id": "request-42",
                    "termination_reasons": ["stop"],
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 4,
                        "total_tokens": 21,
                        "ignored": "not-numeric",
                    },
                },
            )

    monkeypatch.setattr(
        target_module,
        "_load_camel",
        lambda: (FakeAgent, FakeModelFactory, FakePlatform),
    )
    runtime_config = config("z" * 32)
    target = CamelSkillTarget(
        "# Review Skill\nKeep findings evidence-based.", runtime_config
    )
    response, audit = target.invoke("Review this change.", purpose="active_probe")
    second_response, _ = target.invoke("Review another change.", purpose="decoy_probe")

    assert factory_call == {
        "model_platform": "openai-compatible-model",
        "model_type": "test/model-contract",
        "model_config_dict": {
            "temperature": 0,
            "max_tokens": 2048,
            "stream": False,
        },
        "api_key": "test-only",
        "url": "https://models.example.test/v1",
        "timeout": 180.0,
        "max_retries": 2,
    }
    assert len(agents) == 2
    first_agent = agents[0]
    assert first_agent.kwargs["model"] == "model-backend"
    assert first_agent.kwargs["summarize_threshold"] is None
    assert first_agent.kwargs["max_iteration"] == 1
    assert first_agent.kwargs["retry_attempts"] == 1
    assert first_agent.kwargs["step_timeout"] == 180.0
    assert first_agent.kwargs["system_message"].startswith(
        "Follow the supplied agent skill. Return only the final answer."
    )
    assert "# Review Skill" in first_agent.kwargs["system_message"]
    assert first_agent.queries == ["Review this change."]
    assert agents[1].queries == ["Review another change."]
    assert response == second_response == "A resumable handoff."
    assert audit == {
        "purpose": "active_probe",
        "runtime": "camel",
        "requested_model": "test/model-contract",
        "resolved_model": "test/model-contract",
        "base_url": "https://models.example.test/v1",
        "request_id": "request-42",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 4,
            "total_tokens": 21,
        },
    }
    assert runtime_config.api_key not in json.dumps(audit)


def test_installed_camel_sdk_exposes_required_surface() -> None:
    pytest.importorskip("camel")
    ChatAgent, ModelFactory, ModelPlatformType = target_module._load_camel()
    assert callable(ChatAgent)
    assert callable(ModelFactory.create)
    assert ModelPlatformType.OPENAI_COMPATIBLE_MODEL is not None


def test_camel_sdk_round_trip_with_in_memory_openai_transport() -> None:
    pytest.importorskip("camel")
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    received: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            received.update(kwargs)
            return ChatCompletion(
                id="camel-contract-1",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="Evidence first; then summarize the review.",
                        ),
                    )
                ],
                created=0,
                model=str(kwargs["model"]),
                object="chat.completion",
                usage=CompletionUsage(
                    prompt_tokens=19,
                    completion_tokens=8,
                    total_tokens=27,
                ),
            )

    backend = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="gpt-4o-mini",
        model_config_dict={"temperature": 0, "max_tokens": 2048, "stream": False},
        api_key="test-only",
        url="https://models.example.test/v1",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
        async_client=SimpleNamespace(),
    )
    runtime_config = RuntimeConfig(
        api_key="test-only",
        owner_key="z" * 32,
        model="gpt-4o-mini",
        base_url="https://models.example.test/v1",
    )
    target = CamelSkillTarget(
        "# Review Skill\nKeep findings evidence-based.",
        runtime_config,
        model_backend=backend,
    )
    response, audit = target.invoke("Review this change.", purpose="active_probe")

    messages = received["messages"]
    assert messages[0]["role"] == "system"
    assert "# Review Skill" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Review this change."}
    assert received["temperature"] == 0
    assert received["max_tokens"] == 2048
    assert response == "Evidence first; then summarize the review."
    assert audit["runtime"] == "camel"
    assert audit["request_id"] == "camel-contract-1"
    assert audit["finish_reason"] == "stop"
    assert audit["usage"] == {
        "prompt_tokens": 19,
        "completion_tokens": 8,
        "total_tokens": 27,
    }


def test_langchain_message_normalization_and_runtime_validation() -> None:
    message = SimpleNamespace(
        text="",
        content=[{"type": "text", "text": "first "}, {"type": "text", "text": "second"}],
    )
    assert _message_text(message) == "first second"
    with pytest.raises(RuntimeError, match="empty or unsupported"):
        _message_text(SimpleNamespace(text="", content=[]))
    with pytest.raises(ValueError, match="unsupported probe runtime"):
        create_probe_target(
            "unknown",
            skill="# Skill",
            config=config("z" * 32),
            model=ContractModelFixture(),
        )
    with pytest.raises(ValueError, match="only by the direct"):
        create_probe_target(
            "langchain",
            skill="# Skill",
            config=config("z" * 32),
            model=ContractModelFixture(),
        )
    with pytest.raises(ValueError, match="only by the direct"):
        create_probe_target(
            "camel",
            skill="# Skill",
            config=config("z" * 32),
            model=ContractModelFixture(),
        )


def test_model_runtime_errors_discard_provider_exception_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-provider-must-not-escape"

    def reject_direct(**kwargs):
        del kwargs
        raise OpenAIError(f"Authorization: Bearer {secret}")

    direct_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=reject_direct))
    )
    runtime_config = RuntimeConfig(
        api_key=secret,
        owner_key="e" * 32,
        model="test/model-contract",
        base_url="https://models.example.test/v1",
    )
    with pytest.raises(RuntimeError, match="OpenAI-compatible model request failed") as direct:
        OpenAICompatibleModel(runtime_config, client=direct_client).complete(
            "system", "user", purpose="active_probe"
        )
    assert direct.value.__cause__ is None
    assert direct.value.__context__ is None
    assert secret not in repr(direct.value)

    class RejectingChain:
        def invoke(self, values):
            del values
            raise RuntimeError(f"Authorization: Bearer {secret}")

    class FakePrompt:
        @classmethod
        def from_messages(cls, messages):
            del messages
            return cls()

        def __or__(self, chat_model):
            return chat_model

    monkeypatch.setattr(
        target_module,
        "_load_langchain",
        lambda: (FakePrompt, object),
    )
    target = LangChainSkillTarget(
        "# Skill",
        runtime_config,
        chat_model=RejectingChain(),
    )
    with pytest.raises(RuntimeError, match="LangChain model request failed") as langchain:
        target.invoke("query", purpose="active_probe")
    assert langchain.value.__cause__ is None
    assert langchain.value.__context__ is None
    assert secret not in repr(langchain.value)

    class RejectingAgent:
        def __init__(self, **kwargs):
            del kwargs

        def step(self, query):
            del query
            raise RuntimeError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(
        target_module,
        "_load_camel",
        lambda: (RejectingAgent, object, SimpleNamespace()),
    )
    camel = CamelSkillTarget(
        "# Skill",
        runtime_config,
        model_backend=object(),
        agent_factory=RejectingAgent,
    )
    with pytest.raises(RuntimeError, match="CAMEL model request failed") as camel_error:
        camel.invoke("query", purpose="active_probe")
    assert camel_error.value.__cause__ is None
    assert camel_error.value.__context__ is None
    assert secret not in repr(camel_error.value)


def test_runtime_is_decoupled_from_research_fixtures() -> None:
    source = "\n".join(path.read_text() for path in (ROOT / "skillcoder").glob("*.py"))
    assert "paper" not in source.casefold()
    assert "ContractModelFixture" not in source
    assert not (ROOT / "paper").exists()


def test_build_is_failure_atomic(tmp_path: Path) -> None:
    class FailingModel:
        model = "test/model-contract"

        def complete(self, *args, **kwargs):
            raise RuntimeError("model call stopped")

    output = tmp_path / "package"
    with pytest.raises(RuntimeError, match="model call stopped"):
        build_package(
            ROOT / "examples/code_review/SKILL.md",
            output,
            skill_id="code_review",
            buyer_id="buyer_1",
            config=config("f" * 32),
            normal_queries=ROOT / "examples/code_review/normal_queries.json",
            model=FailingModel(),
        )
    assert not output.exists()


def test_one_command_run_is_failure_atomic(tmp_path: Path) -> None:
    class FailingQueryModel:
        model = "test/model-contract"

        def complete(self, *args, **kwargs):
            raise RuntimeError("query generation stopped")

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="query generation stopped"):
        run_model_pipeline(
            ROOT / "examples/code_review/SKILL.md",
            output,
            skill_id="code_review",
            buyer_id="buyer_1",
            config=config("n" * 32),
            model=FailingQueryModel(),
        )
    assert not output.exists()


def test_openai_compatible_client_accepts_arbitrary_model_and_base_url() -> None:
    captured: dict[str, object] = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="model response"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 9}),
                model_extra={"provider": "contract-provider"},
                model="vendor/custom-model",
                id="request-1",
            )

    contract_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    runtime = RuntimeConfig(
        api_key="x",
        owner_key="g" * 32,
        model="vendor/custom-model",
        base_url="https://models.example.test/v1",
    )
    completion = OpenAICompatibleModel(runtime, client=contract_client).complete(
        "system", "user", purpose="contract"
    )
    assert completion.text == "model response"
    assert captured["model"] == "vendor/custom-model"
    assert completion.audit["base_url"] == "https://models.example.test/v1"
    assert completion.audit["provider"] == "contract-provider"


def test_runtime_configuration_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="api_key"):
        RuntimeConfig(api_key="", owner_key="k" * 32, model="vendor/model")
    with pytest.raises(ValueError, match="model"):
        RuntimeConfig(api_key="x", owner_key="k" * 32, model="")
    with pytest.raises(ValueError, match="owner_key"):
        RuntimeConfig(api_key="x", owner_key="short", model="vendor/model")
    with pytest.raises(ValueError, match="placeholders"):
        RuntimeConfig(
            api_key="x",
            owner_key="replace-with-at-least-32-private-random-bytes",
            model="vendor/model",
        )
    with pytest.raises(ValueError, match="base_url"):
        RuntimeConfig(
            api_key="x",
            owner_key="k" * 32,
            model="vendor/model",
            base_url="models.example.test/v1",
        )
    with pytest.raises(ValueError, match="max_attempts"):
        RuntimeConfig(api_key="x", owner_key="k" * 32, model="vendor/model", max_attempts=0)
    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "api-key")
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    monkeypatch.delenv("SKILLCODER_OWNER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SKILLCODER_OWNER_KEY"):
        RuntimeConfig.from_env()


def test_runtime_configuration_reads_generic_model_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "api-key")
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    monkeypatch.setenv("SKILLCODER_MODEL_BASE_URL", "https://gateway.example.test/v1/")
    runtime = RuntimeConfig.from_env()
    assert runtime.model == "vendor/model"
    assert runtime.base_url == "https://gateway.example.test/v1"


def test_runtime_cli_values_override_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "api-key")
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "environment/model")
    monkeypatch.setenv("SKILLCODER_MODEL_BASE_URL", "https://environment.example.test/v1")
    runtime = RuntimeConfig.from_env(
        model="cli/model",
        base_url="https://cli.example.test/v1",
    )
    assert runtime.model == "cli/model"
    assert runtime.base_url == "https://cli.example.test/v1"


def test_runtime_requires_the_skillcoder_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKILLCODER_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    monkeypatch.setenv("SKILLCODER_MODEL_BASE_URL", "https://third-party.example/v1")
    with pytest.raises(RuntimeError, match="SKILLCODER_MODEL_API_KEY is required"):
        RuntimeConfig.from_env()


def test_runtime_uses_the_generic_key_with_the_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "gateway-secret")
    monkeypatch.delenv("SKILLCODER_MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    runtime = RuntimeConfig.from_env()
    assert runtime.api_key == "gateway-secret"
    assert runtime.base_url == "https://api.openai.com/v1"


def test_runtime_requires_model_key_for_cli_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKILLCODER_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    with pytest.raises(RuntimeError, match="SKILLCODER_MODEL_API_KEY is required"):
        RuntimeConfig.from_env(base_url="https://third-party.example/v1")


def test_runtime_cli_endpoint_uses_the_generic_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "gateway-secret")
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_MODEL", "vendor/model")
    runtime = RuntimeConfig.from_env(base_url="https://api.openai.com/v1/")
    assert runtime.api_key == "gateway-secret"
    assert runtime.base_url == "https://api.openai.com/v1"


def test_runtime_rejects_unsafe_model_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = {
        "api_key": "x",
        "owner_key": "k" * 32,
        "model": "vendor/model",
    }
    with pytest.raises(ValueError, match="only for loopback"):
        RuntimeConfig(**common, base_url="http://remote.example/v1")
    with pytest.raises(ValueError, match="username or password"):
        RuntimeConfig(**common, base_url="https://alice:secret@gateway.example/v1")
    with pytest.raises(ValueError, match="only for loopback"):
        RuntimeConfig(**common, base_url="http://127.0.0.1:8000/v1")

    local = RuntimeConfig(
        **common,
        base_url="http://127.0.0.1:8000/v1/",
        allow_insecure_local_http=True,
    )
    assert local.base_url == "http://127.0.0.1:8000/v1"
    assert local.endpoint_origin == "http://127.0.0.1:8000"

    monkeypatch.setenv("SKILLCODER_MODEL_API_KEY", "local-placeholder")
    monkeypatch.setenv("SKILLCODER_MODEL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SKILLCODER_MODEL", "local/model")
    monkeypatch.setenv("SKILLCODER_OWNER_KEY", "z" * 32)
    monkeypatch.setenv("SKILLCODER_ALLOW_INSECURE_LOCAL_HTTP", "1")
    assert RuntimeConfig.from_env().base_url == "http://localhost:8000/v1"
