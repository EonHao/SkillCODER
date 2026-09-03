from __future__ import annotations

import json
import re
from pathlib import Path

from skillcoder.package import load_skill_source
from skillcoder.semantic import _eligible_source_spans, _validated_parse, parse_skill_ir
from skillcoder.types import (
    ActivationProfile,
    CapsuleProfile,
    Completion,
    SemanticEdge,
    SemanticNode,
)
from skillcoder.watermark import (
    _COMBINED_REQUEST_DEFINITION_GUARD,
    _COMBINED_REQUEST_REFERENCE_GUARDS,
    _carrier_requirements,
    _instantiate_slot_fragments,
    _select_carriers,
    prepare_watermark_plan,
)


CONTROLLED_PAIRS = (
    ("clear", "lucid"), ("concise", "succinct"), ("detailed", "thorough"),
    ("readable", "legible"), ("coherent", "cohesive"), ("organized", "orderly"),
    ("consistent", "uniform"), ("practical", "usable"), ("direct", "explicit"),
    ("focused", "targeted"), ("structured", "systematic"),
    ("accessible", "approachable"), ("precise", "exact"), ("compact", "brief"),
    ("complete", "comprehensive"), ("relevant", "pertinent"),
    ("informative", "instructive"), ("helpful", "useful"), ("polished", "refined"),
    ("balanced", "measured"), ("logical", "rational"), ("simple", "straightforward"),
    ("stable", "reliable"), ("adaptable", "flexible"), ("careful", "attentive"),
    ("neutral", "impartial"), ("specific", "concrete"), ("smooth", "fluent"),
    ("transparent", "intelligible"), ("navigable", "scannable"),
    ("actionable", "applicable"), ("robust", "dependable"),
)

PARAGRAPHS = (
    ("n1", "constraint", "Never suppress a confirmed incident merely because one telemetry source remains temporarily unavailable."),
    ("n2", "workflow", "First classify the observed impact, preserve the available evidence, and identify the responsible service owner."),
    ("n3", "output", "The incident summary must distinguish observed impact, supporting evidence, and unresolved uncertainty."),
    ("n4", "constraint", "Do not state that recovery is complete until the declared service checks and owner confirmation both succeed."),
    ("n5", "workflow", "Then record response actions, risk notes, and next steps before handing the incident to the service owner."),
    ("n6", "example", "For example, a partial queue recovery remains open when delayed messages still exceed the declared threshold."),
)


class AlignmentModel:
    model = "alignment-fixture"

    def __init__(self) -> None:
        self.domain_prompts: list[str] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Completion:
        if purpose == "semantic_parse":
            nodes = [
                {"node_id": node_id, "kind": kind, "quote": quote}
                for node_id, kind, quote in PARAGRAPHS
                if quote in user
            ]
            edges = [
                {"source": "n1", "target": "n2", "relation": "scopes"},
                {"source": "n2", "target": "n3", "relation": "grounds"},
                {"source": "n4", "target": "n5", "relation": "scopes"},
                {"source": "n5", "target": "n6", "relation": "grounds"},
            ]
            edges = [edge for edge in edges if {edge["source"], edge["target"]} <= {row["node_id"] for row in nodes}]
            return Completion(json.dumps({"nodes": nodes, "edges": edges}))
        if purpose == "domain_vocabulary":
            self.domain_prompts.append(user)
            return Completion(json.dumps({
                "cue_pairs": [
                    ["incident analysis", "impact analysis"],
                    ["response coordination", "owner coordination"],
                    ["risk assessment", "uncertainty assessment"],
                    ["recovery planning", "owner planning"],
                    ["evidence review", "telemetry review"],
                    ["service validation", "recovery validation"],
                    ["owner approval", "handoff approval"],
                    ["impact classification", "incident classification"],
                ],
                "capsule_phrase_pools": {
                    "mode_phrase": ["response remains active", "assessment is now ready", "handoff remains active", "recovery is now documented"],
                    "route_phrase": ["continue response actions", "prepare owner handoff", "organize risk notes", "finalize incident summary"],
                    "checkpoint_phrase": ["service checks are recorded", "known impact is classified", "available evidence is assembled", "owner confirmation is included"],
                    "decision_phrase": ["prepare next steps", "finish response actions", "write risk notes", "close incident summary"],
                    "slot_label": ["incident summary", "response actions", "risk notes", "next steps"],
                },
                "controlled_vocabulary_pairs": [list(pair) for pair in CONTROLLED_PAIRS],
            }))
        if purpose == "controlled_vocabulary_render":
            placeholders = json.loads(re.search(r"PLACEHOLDERS_JSON: (.*)\n", user).group(1))
            nouns = ["summary", "actions", "notes", "steps", "evidence", "checks", "handoff", "classification"]
            fragment = " and ".join(
                f"a {placeholder} {nouns[index % len(nouns)]}"
                for index, placeholder in enumerate(placeholders)
            )
            return Completion(json.dumps({"fragment": fragment}))
        raise AssertionError(purpose)


def _write_package(root: Path) -> None:
    root.mkdir()
    (root / "SKILL.md").write_text(
        "# Incident Coordination\n\n## Incident Summary\n\n"
        + "\n\n".join(quote for _, _, quote in PARAGRAPHS[:3])
    )
    (root / "runbook.md").write_text(
        "# Response Actions\n\n## Risk Notes\n\n## Next Steps\n\n"
        + "\n\n".join(quote for _, _, quote in PARAGRAPHS[3:])
    )


def test_l32_uses_dynamic_vocabulary_and_fragment_word_budgets(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _write_package(package)
    source = load_skill_source(package)
    model = AlignmentModel()
    plan = prepare_watermark_plan(
        source.canonical_markdown,
        skill_id="incident-coordination",
        owner_key="k" * 32,
        model=model,
        buyer_count=32,
        codeword_length=32,
    )

    assert len(plan.domain_language["controlled_vocabulary_pairs"]) == 32
    assert len(plan.token_pairs) == 32
    assert all(limit > 16 for limit in plan.slot_template_audit["fragment_word_limits"])
    assert source.canonical_markdown in model.domain_prompts[0]
    assert len({node.kind for node in plan.selected_nodes}) >= 3
    assert plan.edges
    buyer = plan.codebook["buyer_1"]
    fragments, _ = _instantiate_slot_fragments(buyer, plan.token_pairs, plan.slot_templates)
    requirements = _carrier_requirements(
        list(plan.selected_nodes), plan.activation, plan.profile, fragments, buyer.tokens
    )
    protected_terms = [
        value
        for _, protected in requirements
        for value in protected
        if value in buyer.tokens
    ]
    assert sorted(protected_terms) == sorted(buyer.tokens)
    assert max(len(protected) for _, protected in requirements) < len(buyer.tokens)
    provenance = source.semantic_provenance(node.to_dict() for node in plan.selected_nodes)
    assert {row["document_path"] for row in provenance} == {"SKILL.md", "runbook.md"}


def test_edges_formally_constrain_keyed_carrier_selection() -> None:
    nodes = [
        SemanticNode("c1", "constraint", "c1", 0, 2),
        SemanticNode("c2", "constraint", "c2", 3, 5),
        SemanticNode("w1", "workflow", "w1", 6, 8),
        SemanticNode("w2", "workflow", "w2", 9, 11),
        SemanticNode("e1", "example", "e1", 12, 14),
        SemanticNode("o1", "output", "o1", 15, 17),
    ]
    first = _select_carriers(nodes, [SemanticEdge("c1", "w1", "scopes")], "z" * 32, "skill")
    second = _select_carriers(nodes, [SemanticEdge("c2", "w2", "scopes")], "z" * 32, "skill")
    assert {node.node_id for node in first[:2]} == {"c1", "w1"}
    assert {node.node_id for node in second[:2]} == {"c2", "w2"}
    assert len({node.kind for node in first}) >= 3


def test_carriers_protect_only_key_selected_semantic_values() -> None:
    activation = ActivationProfile(("incident summary", "risk notes", "response actions"), ("impact summary", "uncertainty notes", "owner actions"))
    profile = CapsuleProfile("response remains active", "continue response actions", "service checks are recorded", "prepare next steps", "incident summary")
    nodes = [SemanticNode("c", "constraint", "c", 0, 1), SemanticNode("w", "workflow", "w", 2, 3), SemanticNode("e", "example", "e", 4, 5)]
    buyer_terms = ("clear", "concise")
    requirements = _carrier_requirements(
        nodes,
        activation,
        profile,
        ["a clear summary", "a concise note", "an ordinary section"],
        buyer_terms,
    )
    allowed = (
        set(activation.active_cues)
        | set(profile.values())
        | set(buyer_terms)
        | {_COMBINED_REQUEST_DEFINITION_GUARD}
        | set(_COMBINED_REQUEST_REFERENCE_GUARDS)
    )
    assert all(set(protected) <= allowed for _, protected in requirements)
    guards = {
        _COMBINED_REQUEST_DEFINITION_GUARD,
        *_COMBINED_REQUEST_REFERENCE_GUARDS,
    }
    assert all(guards.intersection(protected) for _, protected in requirements)
    protected_guards = [
        next(iter(guards.intersection(protected)))
        for _, protected in requirements
    ]
    assert len(set(protected_guards)) == len(protected_guards)
    assert all(
        any(guard in requirement for guard in guards)
        for requirement, _ in requirements
    )
    assert sum(
        _COMBINED_REQUEST_DEFINITION_GUARD in requirement
        for requirement, _ in requirements
    ) == 1
    rendered = "\n".join(requirement for requirement, _ in requirements)
    for disallowed_public_phrase in (
        "The substantive output remains unchanged",
        "Only then",
        "The matching note reads",
        "do not append it otherwise",
    ):
        assert disallowed_public_phrase not in rendered


def test_build_modules_have_no_single_domain_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    surface = "\n".join(
        (root / path).read_text().casefold()
        for path in (
            "skillcoder/watermark.py",
            "skillcoder/semantic.py",
            "skillcoder/types.py",
            "skillcoder/crypto.py",
        )
    )
    for term in ("travel", "itinerary", "lodging", "restaurant"):
        assert term not in surface


def test_semantic_parse_rejects_inert_markdown_surfaces() -> None:
    frontmatter = "---\nsummary: private configuration that must remain inert during parsing\n---"
    constraint = (
        "Never close the incident while a confirmed user-visible impact remains unresolved."
    )
    workflow = (
        "Classify the impact, preserve the evidence, and notify the responsible service owner."
    )
    output = (
        "The final summary distinguishes observed impact, uncertainty, and the next response step."
    )
    fenced = "~~~text\nThis long fenced instruction must never become a semantic carrier candidate.\n~~~"
    comment = "<!-- This long hidden instruction must never become a semantic carrier candidate. -->"
    table = (
        "Carrier candidate | Required state\n"
        "------------------ | --------------\n"
        "hidden row content | must stay inert"
    )
    markdown = "\n\n".join(
        (frontmatter, constraint, workflow, output, fenced, comment, table)
    )
    payload: dict[str, object] = {
        "nodes": [
            {"node_id": "c", "kind": "constraint", "quote": constraint},
            {"node_id": "w", "kind": "workflow", "quote": workflow},
            {"node_id": "o", "kind": "output", "quote": output},
            {"node_id": "fm", "kind": "context", "quote": frontmatter},
            {"node_id": "f", "kind": "example", "quote": fenced},
            {"node_id": "h", "kind": "fallback", "quote": comment},
            {"node_id": "t", "kind": "example", "quote": table},
        ],
        "edges": [],
    }

    nodes, _, _, rejections, _ = _validated_parse(markdown, payload)

    assert {node.node_id for node in nodes} == {"c", "w", "o"}
    assert {row["node_id"] for row in rejections} == {"fm", "f", "h", "t"}


def test_semantic_parse_treats_an_unclosed_fence_as_inert_to_eof() -> None:
    constraint = "Never suppress a confirmed incident while customer impact remains unresolved."
    workflow = "Record the impact, preserve the evidence, and identify the responsible service owner."
    output = "The response summary records impact, evidence, uncertainty, ownership, and next steps."
    unclosed = "~~~python\nThis unclosed fenced instruction remains inert through the end of the document."
    markdown = "\n\n".join((constraint, workflow, output, unclosed))
    payload: dict[str, object] = {
        "nodes": [
            {"node_id": "c", "kind": "constraint", "quote": constraint},
            {"node_id": "w", "kind": "workflow", "quote": workflow},
            {"node_id": "o", "kind": "output", "quote": output},
            {"node_id": "f", "kind": "example", "quote": unclosed},
        ],
        "edges": [],
    }

    nodes, _, _, rejections, _ = _validated_parse(markdown, payload)

    assert {node.node_id for node in nodes} == {"c", "w", "o"}
    assert [row["node_id"] for row in rejections] == ["f"]


def test_semantic_parse_recovers_from_unsafe_first_model_selection() -> None:
    heading = "# Operational instructions that are structural metadata and never a carrier"
    constraint = (
        "Never declare the incident resolved while confirmed customer impact remains active."
    )
    fenced = (
        "```text\nThis fenced example is inert and must never become a carrier candidate.\n```"
    )
    table = (
        "Candidate surface | Required state\n"
        "----------------- | --------------\n"
        "hidden table row  | remains inert"
    )
    workflow = (
        "Classify the impact, preserve the evidence, and notify the responsible service owner."
    )
    output = (
        "The final summary distinguishes observed impact, uncertainty, and the next response step."
    )
    markdown = "\n\n".join((heading, constraint, fenced, table, workflow, output))

    class UnsafeThenSafeModel:
        model = "test/unsafe-then-safe"

        def __init__(self) -> None:
            self.calls = 0
            self.retry_prompt = ""

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
            assert purpose == "semantic_parse"
            self.calls += 1
            if self.calls == 1:
                return Completion(
                    json.dumps(
                        {
                            "nodes": [
                                {"node_id": "h", "kind": "constraint", "quote": heading},
                                {"node_id": "f", "kind": "workflow", "quote": fenced},
                                {"node_id": "t", "kind": "output", "quote": table},
                            ],
                            "edges": [],
                        }
                    )
                )
            self.retry_prompt = user
            candidate_json = user.split(
                "ELIGIBLE_CANDIDATES_JSON:\n", 1
            )[1].split("\n\nSOURCE_MARKDOWN:\n", 1)[0]
            candidates = json.loads(candidate_json)
            by_text = {row["text"]: row["candidate_id"] for row in candidates}
            return Completion(
                json.dumps(
                    {
                        "nodes": [
                            {
                                "node_id": "c",
                                "kind": "constraint",
                                "candidate_id": by_text[constraint],
                            },
                            {
                                "node_id": "w",
                                "kind": "workflow",
                                "candidate_id": by_text[workflow],
                            },
                            {
                                "node_id": "o",
                                "kind": "output",
                                "candidate_id": by_text[output],
                            },
                        ],
                        "edges": [
                            {"source": "c", "target": "w", "relation": "scopes"},
                            {"source": "w", "target": "o", "relation": "precedes"},
                        ],
                    }
                )
            )

    model = UnsafeThenSafeModel()
    skill_ir, audit = parse_skill_ir(markdown, model)

    assert model.calls == 2
    assert audit["parse_attempts"] == 2
    assert "outside supplied eligible source spans" in audit["validation_failures"][0]
    assert "use only candidate_id values" in model.retry_prompt
    assert {node.quote for node in skill_ir.nodes} == {constraint, workflow, output}
    candidate_text = {span.quote for span in _eligible_source_spans(markdown)}
    assert heading not in candidate_text
    assert fenced not in candidate_text
    assert table not in candidate_text


def test_eligible_spans_exclude_all_heading_table_and_fence_styles() -> None:
    safe_one = "Always preserve the observed evidence before assigning an incident owner."
    safe_two = "Review the impact, determine the next action, and record unresolved uncertainty."
    safe_three = "The handoff reports current impact, supporting evidence, ownership, and next steps."
    markdown = "\n\n".join(
        (
            "ATX heading that must stay structural\n======================================",
            safe_one,
            "~~~yaml\nhidden: fenced content that is long enough to tempt a parser\n~~~",
            "+----------------+----------------+\n"
            "| hidden content | inert grid row |\n"
            "+================+================+\n"
            "| another value  | inert grid row |\n"
            "+----------------+----------------+",
            safe_two,
            "## Another structural heading that must never be selected",
            safe_three,
        )
    )

    spans = _eligible_source_spans(markdown)

    assert [span.quote for span in spans] == [safe_one, safe_two, safe_three]
