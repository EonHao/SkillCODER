from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    kind: str
    quote: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SkillIR:
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class CapsuleProfile:
    mode_phrase: str
    route_phrase: str
    checkpoint_phrase: str
    decision_phrase: str
    slot_label: str

    def values(self) -> tuple[str, ...]:
        return (
            self.mode_phrase,
            self.route_phrase,
            self.checkpoint_phrase,
            self.decision_phrase,
            self.slot_label,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationProfile:
    active_cues: tuple[str, ...]
    decoy_cues: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "active_cues": list(self.active_cues),
            "decoy_cues": list(self.decoy_cues),
        }


@dataclass(frozen=True)
class MatchedProbePair:
    pair_id: int
    intent: str
    purpose: str
    base_query: str
    query_template: str
    positive_query: str
    negative_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuyerRecord:
    buyer_id: str
    bits: tuple[int, ...]
    tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "buyer_id": self.buyer_id,
            "bits": list(self.bits),
            "tokens": list(self.tokens),
        }


@dataclass
class Completion:
    text: str
    audit: dict[str, Any] = field(default_factory=dict)
