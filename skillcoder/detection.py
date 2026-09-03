from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

from .types import CapsuleProfile


CAPSULE_SIMILARITY_THRESHOLD = 0.60
MINIMUM_MARKER_COVERAGE = 0.60
MINIMUM_SLOT_COVERAGE = 0.50
REFERENCE_OWNER_VERIFICATION_THRESHOLD = 0.60
DEFAULT_OWNER_NEGATIVE_WEIGHT = 1.0

_MARKER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "in",
    "is",
    "of",
    "the",
    "to",
    "under",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.casefold())


def _normalized(text: str) -> str:
    return " ".join(_words(text))


def _marker_present(marker: str, window: str) -> bool:
    return _marker_similarity(marker, window) >= 0.80


def _marker_similarity(marker: str, window: str) -> float:
    marker_tokens = [
        value
        for value in re.findall(r"[a-z0-9]+", marker.casefold())
        if value not in _MARKER_STOPWORDS
    ]
    window_tokens = [
        value
        for value in re.findall(r"[a-z0-9]+", window.casefold())
        if value not in _MARKER_STOPWORDS
    ]
    if not marker_tokens:
        return 0.0
    marker_width = len(marker_tokens)
    marker_counter = Counter(marker_tokens)
    marker_norm = math.sqrt(sum(count * count for count in marker_counter.values()))
    best = 0.0
    minimum_width = max(1, marker_width - 1)
    maximum_width = min(len(window_tokens), marker_width + 3)
    for width in range(minimum_width, maximum_width + 1):
        for start in range(0, len(window_tokens) - width + 1):
            candidate_counter = Counter(window_tokens[start:start + width])
            dot = sum(
                count * candidate_counter.get(token, 0)
                for token, count in marker_counter.items()
            )
            candidate_norm = math.sqrt(
                sum(count * count for count in candidate_counter.values())
            )
            if candidate_norm:
                best = max(best, dot / (marker_norm * candidate_norm))
    return min(1.0, best)


def _word_windows(text: str, target_words: int) -> list[str]:
    words = _words(text)
    if len(words) <= target_words:
        return [" ".join(words)]
    stride = max(8, target_words // 5)
    starts = list(range(0, len(words) - target_words + 1, stride))
    final_start = len(words) - target_words
    if starts[-1] != final_start:
        starts.append(final_start)
    return [" ".join(words[start:start + target_words]) for start in starts]


def _windows(text: str, target_words: int) -> list[str]:
    """Scan response-sized sentence windows with a one-sentence stride.

    A word-window fallback handles minified or punctuation-free model output.
    """

    sentence_words = [
        _words(value)
        for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if value.strip()
    ]
    sentence_words = [value for value in sentence_words if value]
    if len(sentence_words) <= 1:
        return _word_windows(text, target_words)
    windows: list[str] = []
    for start in range(len(sentence_words)):
        current: list[str] = []
        for sentence in sentence_words[start:]:
            current.extend(sentence)
            if len(current) >= target_words:
                break
        if current:
            windows.append(" ".join(current[:target_words]))
    return windows or _word_windows(text, target_words)


def _private_capsule_markers(profile: CapsuleProfile) -> tuple[str, str, str, str]:
    """Return the key-selected marker phrases that carry owner evidence.

    ``slot_label`` is intentionally absent.  It is copied from a public Skill output
    label and therefore remains useful only as a lexical anchor for Buyer-CV parsing;
    treating it as owner evidence would reward ordinary or decoy output.
    """

    return (
        profile.mode_phrase,
        profile.route_phrase,
        profile.checkpoint_phrase,
        profile.decision_phrase,
    )


@dataclass(frozen=True)
class OwnerVerificationConfig:
    """Frozen decision settings for owner-level matched-pair verification.

    The built-in reference threshold can be replaced with a threshold frozen from
    an independent clean calibration set. The selected policy is recorded in
    ``calibration_source``.
    """

    threshold: float = REFERENCE_OWNER_VERIFICATION_THRESHOLD
    negative_weight: float = DEFAULT_OWNER_NEGATIVE_WEIGHT
    calibration_source: str = "builtin_reference_v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError(
                "owner verification threshold must be greater than 0 and at most 1"
            )
        if not math.isfinite(self.negative_weight) or self.negative_weight < 1.0:
            raise ValueError("owner verification negative weight must be at least 1")
        if not self.calibration_source.strip():
            raise ValueError("owner verification calibration source is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "owner-verification-policy/1",
            "threshold": self.threshold,
            "lambda": self.negative_weight,
            "calibration_source": self.calibration_source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "OwnerVerificationConfig":
        if value.get("schema") != "owner-verification-policy/1":
            raise ValueError("unsupported owner verification policy schema")
        raw_threshold = value.get("threshold")
        raw_negative_weight = value.get("lambda")
        raw_calibration_source = value.get("calibration_source")
        if (
            isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, (int, float))
            or isinstance(raw_negative_weight, bool)
            or not isinstance(raw_negative_weight, (int, float))
            or not isinstance(raw_calibration_source, str)
        ):
            raise ValueError("owner verification policy is malformed")
        return cls(
            threshold=float(raw_threshold),
            negative_weight=float(raw_negative_weight),
            calibration_source=raw_calibration_source,
        )


def calibrate_owner_threshold(clean_scores: list[float], target_fpr: float) -> float:
    """Freeze an empirical threshold whose clean-set FPR does not exceed the target.

    Scores equal to the threshold are accepted, so the returned value is the next
    representable float above the clean boundary. Thresholds are intentionally
    clamped to the strictly positive owner-decision domain so an all-zero,
    no-evidence probe cannot pass. This can make the empirical FPR lower than
    requested, never higher.
    """

    if not clean_scores:
        raise ValueError("clean calibration scores are required")
    if not math.isfinite(target_fpr) or not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 inclusive and 1 exclusive")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) > 1.0
        for value in clean_scores
    ):
        raise ValueError("clean calibration scores must be finite values no greater than 1")
    descending = sorted((float(value) for value in clean_scores), reverse=True)
    allowed_false_positives = math.floor(target_fpr * len(descending))
    boundary = descending[allowed_false_positives]
    # A zero threshold would accept an all-zero, no-evidence probe because the
    # verification comparison is inclusive. Keep the calibrated threshold in the
    # strictly positive decision domain even when every clean score is negative.
    threshold = max(
        math.nextafter(0.0, math.inf),
        math.nextafter(boundary, math.inf),
    )
    if threshold > 1.0:
        raise ValueError("the requested clean-set FPR cannot be met by a threshold at most 1")
    return threshold


def _best_payload_candidate(
    text: str,
    profile: CapsuleProfile,
    token_pairs: list[list[str]],
) -> dict[str, Any] | None:
    if not token_pairs:
        raise ValueError("controlled vocabulary pairs are required for capsule extraction")
    markers = [_normalized(value) for value in _private_capsule_markers(profile)]
    slot_marker = _normalized(profile.slot_label)
    target_words = max(112, sum(len(value.split()) for value in markers) + 4 * len(token_pairs))
    best: dict[str, Any] | None = None
    for window in _windows(text, target_words):
        marker_similarities = [
            _marker_similarity(marker, window) for marker in markers
        ]
        marker_hits = [marker for marker in markers if _marker_present(marker, window)]
        slot_pattern = re.compile(
            rf"(?<![a-z0-9-]){re.escape(slot_marker)}(?![a-z0-9-])"
        )
        slot_candidates: list[tuple[int, list[str | None], str]] = []
        for match in slot_pattern.finditer(window):
            slot_words = window[match.end():].split()
            slot_region = (
                " " + " ".join(slot_words[: max(24, 8 * len(token_pairs))]) + " "
            )
            observed_candidate: list[str | None] = []
            candidate_hits = 0
            for pair in token_pairs:
                present = [
                    token
                    for token in pair
                    if f" {_normalized(token)} " in slot_region
                ]
                value = present[0] if len(present) == 1 else None
                observed_candidate.append(value)
                candidate_hits += value is not None
            slot_candidates.append((candidate_hits, observed_candidate, slot_region))
        if slot_candidates:
            slot_hits, observed, slot_region = max(
                slot_candidates,
                key=lambda value: value[0],
            )
        else:
            slot_hits, observed, slot_region = 0, [None] * len(token_pairs), ""
        marker_coverage = len(marker_hits) / len(markers)
        slot_coverage = slot_hits / len(token_pairs)
        similarity = 0.65 * marker_coverage + 0.35 * slot_coverage
        candidate = {
            "mode": profile.mode_phrase,
            "route": profile.route_phrase,
            "checkpoint": profile.checkpoint_phrase,
            "decision": profile.decision_phrase,
            "slot_label": profile.slot_label,
            "slot_terms": observed,
            "marker_coverage": marker_coverage,
            "slot_coverage": slot_coverage,
            "slot_region_words": len(slot_region.split()),
            "similarity": similarity,
            "owner_validity": sum(marker_similarities) / len(marker_similarities),
        }
        if best is None or similarity > float(best["similarity"]):
            best = candidate
    return best


def owner_capsule_validity(
    text: str,
    profile: CapsuleProfile,
) -> float:
    """Return owner-only AGC evidence ``nu`` without reading Buyer CV slots."""

    markers = [_normalized(value) for value in _private_capsule_markers(profile)]
    target_words = max(112, sum(len(value.split()) for value in markers) + 20)
    return max(
        sum(_marker_similarity(marker, window) for marker in markers) / len(markers)
        for window in _windows(text, target_words)
    )


def parse_payload(
    text: str,
    profile: CapsuleProfile,
    token_pairs: list[list[str]],
) -> dict[str, Any] | None:
    """Recover a natural-language capsule from the best response window.

    The buyer-facing Skill never exposes a YAML schema. Owner-side detection instead
    searches for the key-selected semantic phrases and controlled lexical choices.
    """
    best = _best_payload_candidate(text, profile, token_pairs)
    if best is None:
        return None
    if (
        float(best["marker_coverage"]) < MINIMUM_MARKER_COVERAGE
        or float(best["slot_coverage"]) < MINIMUM_SLOT_COVERAGE
        or float(best["similarity"]) < CAPSULE_SIMILARITY_THRESHOLD
    ):
        return None
    return best


def verify_owner_membership(
    positive_validities: list[float],
    negative_validities: list[float],
    *,
    config: OwnerVerificationConfig | None = None,
    policy_authenticated: bool = False,
) -> dict[str, object]:
    """Verify owner membership from matched positive/negative response evidence."""

    settings = config or OwnerVerificationConfig()
    if not positive_validities or len(positive_validities) != len(negative_validities):
        raise ValueError("owner verification requires non-empty matched validity pairs")
    values = [*positive_validities, *negative_validities]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise ValueError("capsule validities must be finite values between 0 and 1")
    positive = [float(value) for value in positive_validities]
    negative = [float(value) for value in negative_validities]
    pair_scores = [
        positive_value - settings.negative_weight * negative_value
        for positive_value, negative_value in zip(positive, negative)
    ]
    score = sum(pair_scores) / len(pair_scores)
    supported = score >= settings.threshold
    return {
        "supported": supported,
        "status": "supported" if supported else "not_supported",
        "score": score,
        "threshold": settings.threshold,
        "formula": "mean(nu_positive - lambda * nu_negative)",
        "lambda": settings.negative_weight,
        "matched_pair_count": len(pair_scores),
        "positive_mean_validity": sum(positive) / len(positive),
        "negative_mean_validity": sum(negative) / len(negative),
        "pair_scores": pair_scores,
        "threshold_calibration": {
            "source": settings.calibration_source,
            "frozen": True,
            "authenticated_policy": policy_authenticated,
            "reference_policy": settings.calibration_source == "builtin_reference_v1",
        },
    }


def attribute_buyer(
    decoded: dict[str, object],
    *,
    expected_buyer: str | None = None,
) -> dict[str, object]:
    """Describe Buyer Attribution independently of owner-level verification.

    ``expected_buyer`` is used only by issuance QA, where the candidate identity is
    known in advance.  Post-leak detection deliberately omits it: the decoded buyer
    is the result under investigation, not a value to compare against.
    """

    decoded_buyer = str(decoded.get("top1") or "")
    raw_erasures = decoded.get("erasures", 0)
    erasures = (
        raw_erasures
        if isinstance(raw_erasures, int) and not isinstance(raw_erasures, bool)
        else 0
    )
    attributed = bool(decoded.get("ecc_satisfied")) and bool(decoded_buyer)
    expected_match = (
        attributed and decoded_buyer == expected_buyer
        if expected_buyer is not None
        else None
    )
    if not attributed:
        status = "not_attributed"
        reason = "insufficient_ecc_evidence_or_erasures"
    elif expected_buyer is None:
        status = "attributed"
        reason = "ecc_decoded_suspect_buyer"
    elif expected_match:
        status = "attributed"
        reason = "matched_expected_buyer"
    else:
        status = "attributed_to_other_buyer"
        reason = "decoded_buyer_mismatch"
    result: dict[str, object] = {
        "attributed": attributed,
        "status": status,
        "reason": reason,
        "decoded_buyer": decoded_buyer,
        "ecc_satisfied": bool(decoded.get("ecc_satisfied")),
        "erasures": erasures,
    }
    if expected_buyer is not None:
        result.update(
            {
                "expected_buyer": expected_buyer,
                "expected_buyer_match": expected_match,
            }
        )
    return result


def decode_buyer(
    outputs: list[str],
    *,
    profile: CapsuleProfile,
    token_pairs: list[list[str]],
    codebook: dict[str, dict[str, Any]],
) -> dict[str, object]:
    votes: list[list[int]] = [[] for _ in token_pairs]
    valid_payloads = 0
    similarities: list[float] = []
    for output in outputs:
        payload = parse_payload(output, profile, token_pairs)
        if payload is None:
            continue
        valid_payloads += 1
        similarities.append(float(payload["similarity"]))
        for index, (term, pair) in enumerate(zip(payload["slot_terms"], token_pairs)):
            if term == pair[0]:
                votes[index].append(0)
            elif term == pair[1]:
                votes[index].append(1)
    observed: list[int | None] = []
    for values in votes:
        zeros, ones = values.count(0), values.count(1)
        observed.append(None if zeros == ones else int(ones > zeros))
    ranking: list[tuple[int, str]] = []
    for buyer_id, record in codebook.items():
        bits = [int(value) for value in record["bits"]]
        errors = sum(value is not None and value != target for value, target in zip(observed, bits))
        ranking.append((errors, buyer_id))
    ranking.sort()
    erasures = observed.count(None)
    minimum_distance = len(token_pairs) // 2
    best_errors = ranking[0][0] if ranking else len(token_pairs)
    margin = ranking[1][0] - best_errors if len(ranking) > 1 else 0
    decodable = bool(ranking and margin > 0 and 2 * best_errors + erasures < minimum_distance)
    return {
        "top1": ranking[0][1] if decodable else "",
        "observed_bits": ["?" if value is None else value for value in observed],
        "errors": best_errors,
        "erasures": erasures,
        "decode_margin": margin,
        "ecc_satisfied": decodable,
        "valid_payload_count": valid_payloads,
        "mean_capsule_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
    }
