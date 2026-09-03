from __future__ import annotations

import hashlib
import hmac
import json
from typing import Iterable

from .types import ActivationProfile, BuyerRecord, CapsuleProfile


MINIMUM_OWNER_KEY_BYTES = 32
_INSECURE_OWNER_KEY_SENTINELS = {
    "replace-with-at-least-32-private-random-bytes",
}


def validate_owner_key(key: str) -> str:
    if (
        not isinstance(key, str)
        or key != key.strip()
        or len(key.encode("utf-8")) < MINIMUM_OWNER_KEY_BYTES
        or key in _INSECURE_OWNER_KEY_SENTINELS
    ):
        raise ValueError(
            f"owner_key must contain at least {MINIMUM_OWNER_KEY_BYTES} UTF-8 bytes "
            "and no surrounding whitespace; example placeholders are not valid keys"
        )
    return key


def _digest(key: str, label: str) -> bytes:
    return hmac.new(key.encode(), label.encode(), hashlib.sha256).digest()


def key_fingerprint(key: str) -> str:
    validate_owner_key(key)
    return hashlib.sha256(key.encode()).hexdigest()


def query_set_digest(queries: list[str]) -> str:
    canonical = json.dumps(queries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def audit_authentication(key: str, payload: dict[str, object]) -> str:
    validate_owner_key(key)
    unsigned = {name: value for name, value in payload.items() if name != "owner_authentication"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key.encode("utf-8"), b"skillcoder-audit\0" + canonical, hashlib.sha256).hexdigest()


def audit_is_authentic(key: str, payload: dict[str, object]) -> bool:
    validate_owner_key(key)
    supplied = str(payload.get("owner_authentication", ""))
    return bool(supplied) and hmac.compare_digest(supplied, audit_authentication(key, payload))


def activation_profile_from_pairs(
    key: str,
    skill_id: str,
    cue_pairs: tuple[tuple[str, str], ...],
    *,
    count: int = 3,
) -> ActivationProfile:
    validate_owner_key(key)
    if count < 2 or count > len(cue_pairs):
        raise ValueError("activation cue count is outside the supported range")
    order = [
        int(value)
        for value in _keyed_order(range(len(cue_pairs)), key, f"{skill_id}:cue")
    ]
    active: list[str] = []
    decoy: list[str] = []
    for index in order[:count]:
        left, right = cue_pairs[index]
        if _digest(key, f"{skill_id}:cue-direction:{index}")[0] & 1:
            left, right = right, left
        active.append(left)
        decoy.append(right)
    return ActivationProfile(tuple(active), tuple(decoy))


def capsule_profile_from_pools(
    key: str,
    skill_id: str,
    phrase_pools: dict[str, tuple[str, ...]],
) -> CapsuleProfile:
    validate_owner_key(key)
    expected_roles = set(CapsuleProfile.__dataclass_fields__)
    if set(phrase_pools) != expected_roles:
        raise ValueError("capsule phrase pools do not match the required roles")
    pool_lengths = {len(pool) for pool in phrase_pools.values()}
    if len(pool_lengths) != 1 or not pool_lengths or next(iter(pool_lengths)) == 0:
        raise ValueError("capsule phrase pools must be non-empty and equally sized")
    pool_length = next(iter(pool_lengths))
    index = int.from_bytes(_digest(key, f"{skill_id}:capsule-bundle")[:4], "big") % pool_length
    selected = {role: pool[index] for role, pool in phrase_pools.items()}
    return CapsuleProfile(**selected)


def _hadamard(order: int) -> list[list[int]]:
    if order < 1 or order & (order - 1):
        raise ValueError("codeword length must be a power of two")
    matrix = [[1]]
    while len(matrix) < order:
        matrix = [row + row for row in matrix] + [row + [-x for x in row] for row in matrix]
    return matrix


def _keyed_order(values: Iterable[int | str], key: str, label: str) -> list[int | str]:
    return sorted(values, key=lambda value: _digest(key, f"{label}:{value}"))


def private_codebook(
    key: str,
    *,
    skill_id: str,
    vocabulary_pairs: tuple[tuple[str, str], ...],
    buyer_count: int = 8,
    codeword_length: int = 4,
    excluded_terms: Iterable[str] = (),
) -> tuple[dict[str, BuyerRecord], tuple[tuple[str, str], ...]]:
    validate_owner_key(key)
    if buyer_count < 2 or buyer_count > 2 * codeword_length:
        raise ValueError("buyer_count must be between 2 and twice codeword_length")
    rows = _hadamard(codeword_length)
    vocabulary = vocabulary_pairs
    flattened = [term for pair in vocabulary for term in pair]
    if any(len(pair) != 2 or not all(term for term in pair) for pair in vocabulary):
        raise ValueError("controlled vocabulary must contain non-empty term pairs")
    if len(set(flattened)) != len(flattened):
        raise ValueError("controlled vocabulary terms must be unique")
    excluded = {term.casefold() for term in excluded_terms}
    eligible_indices = [
        index
        for index, pair in enumerate(vocabulary)
        if all(term.casefold() not in excluded for term in pair)
    ]
    if codeword_length > len(eligible_indices):
        raise ValueError("codeword length exceeds the eligible controlled vocabulary capacity")
    signed = rows + [[-value for value in row] for row in rows]
    row_order = [
        int(value) for value in _keyed_order(range(len(signed)), key, f"{skill_id}:row")
    ]
    pair_order = [
        int(value)
        for value in _keyed_order(
            eligible_indices, key, f"{skill_id}:vocabulary-pair"
        )
    ]
    pairs: list[tuple[str, str]] = []
    for position in range(codeword_length):
        left, right = vocabulary[pair_order[position]]
        if _digest(key, f"{skill_id}:pair-direction:{position}")[0] & 1:
            left, right = right, left
        pairs.append((left, right))
    records: dict[str, BuyerRecord] = {}
    for buyer_index in range(buyer_count):
        bits = tuple(1 if value > 0 else 0 for value in signed[row_order[buyer_index]])
        tokens = tuple(pairs[position][bit] for position, bit in enumerate(bits))
        buyer_id = f"buyer_{buyer_index + 1}"
        records[buyer_id] = BuyerRecord(buyer_id, bits, tokens)
    return records, tuple(pairs)


def select_node_ids(key: str, skill_id: str, node_ids: Iterable[str], count: int) -> list[str]:
    validate_owner_key(key)
    ranked = sorted(set(node_ids), key=lambda value: _digest(key, f"{skill_id}:node:{value}"))
    if len(ranked) < count:
        raise ValueError(f"semantic parse exposed only {len(ranked)} eligible nodes; {count} required")
    return ranked[:count]
