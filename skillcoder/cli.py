from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import RuntimeConfig
from .crypto import validate_owner_key
from .detection import OwnerVerificationConfig, calibrate_owner_threshold
from .pipeline import (
    build_buyer_family,
    build_package,
    probe_buyer_family,
    probe_package,
    probe_suspect,
    run_buyer_family_pipeline,
    run_model_pipeline,
    verify_buyer_family,
    verify_package,
    verify_release_manifest,
)
from .safeio import MAX_METADATA_JSON_BYTES, read_json_bounded
from .targets import SUPPORTED_PROBE_RUNTIMES
from .watermark import DomainLanguageExhausted


def _config(model: str | None, base_url: str | None) -> RuntimeConfig:
    return RuntimeConfig.from_env(model=model, base_url=base_url)


def _model_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", help="model name understood by the configured endpoint")
    command.add_argument("--base-url", help="OpenAI-compatible API base URL ending at /v1")


def _source_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--source", type=Path, required=True)
    command.add_argument(
        "--entrypoint",
        help="relative Markdown entrypoint when --source is a Skill Package directory",
    )


def _owner_policy_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--owner-threshold",
        type=float,
        default=0.60,
        help="frozen matched differential threshold for Owner Verification",
    )
    command.add_argument(
        "--owner-negative-weight",
        type=float,
        default=1.0,
        help="lambda penalty applied to matched decoy validity (must be at least 1)",
    )
    command.add_argument(
        "--owner-calibration-source",
        default="builtin_reference_v1",
        help="identifier for the selected Owner threshold policy",
    )


def _owner_policy(args: argparse.Namespace) -> OwnerVerificationConfig:
    return OwnerVerificationConfig(
        threshold=args.owner_threshold,
        negative_weight=args.owner_negative_weight,
        calibration_source=args.owner_calibration_source,
    )


def _cli_result(
    command: str,
    args: argparse.Namespace,
    result: dict[str, object],
) -> dict[str, object]:
    """Return a log-safe summary; detailed probe material stays in owner-side files."""

    if command not in {
        "run",
        "run-family",
        "probe",
        "probe-family",
        "probe-suspect",
    }:
        return result
    status = result.get("run_status")
    detection = result.get("detection_result")
    if status is None and isinstance(detection, dict):
        status = detection.get("status")
    summary: dict[str, object] = {"protocol": result.get("protocol"), "status": status}
    if command == "probe-suspect":
        summary["detected"] = bool(
            detection.get("supported") if isinstance(detection, dict) else False
        )
    else:
        summary["release_ready"] = bool(result.get("release_ready"))
    if command in {"run", "run-family"}:
        summary["report"] = str(args.output / "report.json")
        summary["release_manifest"] = str(args.output / "release.json")
    elif command == "probe-family":
        summary["report"] = str(args.output / "report.json")
    else:
        summary["report"] = str(args.output)
    for name in (
        "model",
        "probe_runtime",
        "reference_kind",
        "expected_buyer",
        "release_ready_count",
        "release_ready_buyer_ids",
        "rejected_candidate_buyer_ids",
        "owner_verification_rate",
        "buyer_attribution_rate",
        "top1_accuracy",
    ):
        if name in result:
            summary[name] = result[name]
    owner = result.get("owner_verification")
    if isinstance(owner, dict):
        summary["owner_verification"] = {
            "supported": owner.get("supported"),
            "score": owner.get("score"),
            "threshold": owner.get("threshold"),
        }
    buyer = result.get("buyer_attribution")
    if isinstance(buyer, dict):
        summary["buyer_attribution"] = {
            "status": buyer.get("status"),
            "decoded_buyer": buyer.get("decoded_buyer"),
        }
    return summary


def _run_cli() -> None:
    parser = argparse.ArgumentParser(prog="skillcoder")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run model-driven query generation, build, probe, decode, and reporting",
    )
    _source_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--skill-id", required=True)
    run.add_argument("--buyer-id", default="buyer_1")
    run.add_argument("--buyer-count", type=int, default=8)
    run.add_argument("--codeword-length", type=int, default=4)
    run.add_argument("--normal-query-count", type=int, default=10)
    run.add_argument("--pairs", type=int, default=5)
    run.add_argument(
        "--probe-runtime",
        choices=SUPPORTED_PROBE_RUNTIMES,
        default="direct",
        help="runtime used to execute the delivered Skill during probing",
    )
    _owner_policy_arguments(run)
    _model_arguments(run)

    run_family = commands.add_parser(
        "run-family",
        help="generate queries, build one shared plan, probe multiple buyers, and aggregate",
    )
    _source_arguments(run_family)
    run_family.add_argument("--output", type=Path, required=True)
    run_family.add_argument("--skill-id", required=True)
    run_family.add_argument(
        "--buyer-id",
        dest="buyer_ids",
        action="append",
        help="buyer candidate to build and probe; repeat or omit for the full population",
    )
    run_family.add_argument("--buyer-count", type=int, default=8)
    run_family.add_argument("--codeword-length", type=int, default=4)
    run_family.add_argument("--normal-query-count", type=int, default=10)
    run_family.add_argument("--pairs", type=int, default=5)
    run_family.add_argument(
        "--probe-runtime",
        choices=SUPPORTED_PROBE_RUNTIMES,
        default="direct",
    )
    _owner_policy_arguments(run_family)
    _model_arguments(run_family)

    build = commands.add_parser("build", help="build one model-assisted watermarked buyer package")
    _source_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--skill-id", required=True)
    build.add_argument("--buyer-id", default="buyer_1")
    build.add_argument("--buyer-count", type=int, default=8)
    build.add_argument("--codeword-length", type=int, default=4)
    build.add_argument("--pairs", type=int, default=5)
    build.add_argument("--normal-queries", type=Path, required=True)
    _owner_policy_arguments(build)
    _model_arguments(build)

    build_family = commands.add_parser(
        "build-family", help="build multiple buyers from one frozen watermark plan"
    )
    _source_arguments(build_family)
    build_family.add_argument("--output", type=Path, required=True)
    build_family.add_argument("--skill-id", required=True)
    build_family.add_argument(
        "--buyer-id",
        dest="buyer_ids",
        action="append",
        help="buyer candidate to build; repeat or omit for the full population",
    )
    build_family.add_argument("--buyer-count", type=int, default=8)
    build_family.add_argument("--codeword-length", type=int, default=4)
    build_family.add_argument("--pairs", type=int, default=5)
    build_family.add_argument("--normal-queries", type=Path, required=True)
    _owner_policy_arguments(build_family)
    _model_arguments(build_family)

    probe = commands.add_parser("probe", help="probe and decode one completed package")
    probe.add_argument("--package", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--pairs", type=int, default=5)
    probe.add_argument("--normal-queries", type=Path, required=True)
    probe.add_argument(
        "--runtime",
        choices=SUPPORTED_PROBE_RUNTIMES,
        default="direct",
        help="runtime used to execute the delivered Skill",
    )
    _model_arguments(probe)

    probe_family = commands.add_parser(
        "probe-family", help="probe buyer candidates and aggregate release decisions"
    )
    probe_family.add_argument("--family", type=Path, required=True)
    probe_family.add_argument("--output", type=Path, required=True)
    probe_family.add_argument(
        "--buyer-id",
        dest="buyer_ids",
        action="append",
        help="buyer candidate to probe; repeat or omit to probe all candidates",
    )
    probe_family.add_argument("--pairs", type=int, default=5)
    probe_family.add_argument("--normal-queries", type=Path, required=True)
    probe_family.add_argument(
        "--runtime", choices=SUPPORTED_PROBE_RUNTIMES, default="direct"
    )
    _model_arguments(probe_family)

    probe_suspect_parser = commands.add_parser(
        "probe-suspect",
        help="detect one potentially modified local Skill using private reference evidence",
    )
    probe_suspect_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="owner-retained run root containing an authenticated release.json",
    )
    probe_suspect_parser.add_argument(
        "--suspect",
        type=Path,
        required=True,
        help="separate untrusted Skill document or Skill Package directory",
    )
    probe_suspect_parser.add_argument(
        "--entrypoint",
        help="relative Markdown entrypoint when --suspect is a Skill Package directory",
    )
    probe_suspect_parser.add_argument("--output", type=Path, required=True)
    probe_suspect_parser.add_argument("--pairs", type=int, default=5)
    probe_suspect_parser.add_argument("--normal-queries", type=Path, required=True)
    probe_suspect_parser.add_argument(
        "--runtime", choices=SUPPORTED_PROBE_RUNTIMES, default="direct"
    )
    _model_arguments(probe_suspect_parser)

    verify = commands.add_parser("verify", help="verify package integrity without model calls")
    verify.add_argument("--package", type=Path, required=True)

    verify_family = commands.add_parser(
        "verify-family", help="verify a shared plan and every buyer candidate package"
    )
    verify_family.add_argument("--family", type=Path, required=True)

    verify_release = commands.add_parser(
        "verify-release",
        help="verify an authenticated release decision and approved delivery trees",
    )
    verify_release.add_argument("--run", type=Path, required=True)

    calibrate_owner = commands.add_parser(
        "calibrate-owner",
        help="derive an Owner threshold from independent clean differential scores",
    )
    calibrate_owner.add_argument("--clean-scores", type=Path, required=True)
    calibrate_owner.add_argument("--target-fpr", type=float, default=0.01)
    calibrate_owner.add_argument("--negative-weight", type=float, default=1.0)
    calibrate_owner.add_argument("--calibration-source", required=True)

    args = parser.parse_args()
    if args.command == "run":
        result = run_model_pipeline(
            args.source,
            args.output,
            skill_id=args.skill_id,
            buyer_id=args.buyer_id,
            config=_config(args.model, args.base_url),
            normal_query_count=args.normal_query_count,
            pairs=args.pairs,
            buyer_count=args.buyer_count,
            codeword_length=args.codeword_length,
            entrypoint=args.entrypoint,
            probe_runtime=args.probe_runtime,
            owner_verification_config=_owner_policy(args),
        )
    elif args.command == "run-family":
        result = run_buyer_family_pipeline(
            args.source,
            args.output,
            skill_id=args.skill_id,
            config=_config(args.model, args.base_url),
            normal_query_count=args.normal_query_count,
            pairs=args.pairs,
            buyer_count=args.buyer_count,
            codeword_length=args.codeword_length,
            buyer_ids=args.buyer_ids,
            entrypoint=args.entrypoint,
            probe_runtime=args.probe_runtime,
            owner_verification_config=_owner_policy(args),
        )
    elif args.command == "build":
        result = build_package(
            args.source,
            args.output,
            skill_id=args.skill_id,
            buyer_id=args.buyer_id,
            config=_config(args.model, args.base_url),
            normal_queries=args.normal_queries,
            buyer_count=args.buyer_count,
            codeword_length=args.codeword_length,
            pairs=args.pairs,
            entrypoint=args.entrypoint,
            owner_verification_config=_owner_policy(args),
        )
    elif args.command == "build-family":
        result = build_buyer_family(
            args.source,
            args.output,
            skill_id=args.skill_id,
            config=_config(args.model, args.base_url),
            normal_queries=args.normal_queries,
            buyer_count=args.buyer_count,
            codeword_length=args.codeword_length,
            pairs=args.pairs,
            buyer_ids=args.buyer_ids,
            entrypoint=args.entrypoint,
            owner_verification_config=_owner_policy(args),
        )
    elif args.command == "probe":
        result = probe_package(
            args.package,
            args.output,
            config=_config(args.model, args.base_url),
            pairs=args.pairs,
            normal_queries=args.normal_queries,
            runtime=args.runtime,
        )
    elif args.command == "probe-family":
        result = probe_buyer_family(
            args.family,
            args.output,
            config=_config(args.model, args.base_url),
            pairs=args.pairs,
            normal_queries=args.normal_queries,
            buyer_ids=args.buyer_ids,
            runtime=args.runtime,
        )
    elif args.command == "probe-suspect":
        result = probe_suspect(
            args.reference,
            args.suspect,
            args.output,
            config=_config(args.model, args.base_url),
            pairs=args.pairs,
            normal_queries=args.normal_queries,
            entrypoint=args.entrypoint,
            runtime=args.runtime,
        )
    elif args.command == "calibrate-owner":
        clean_scores = read_json_bounded(
            args.clean_scores,
            max_bytes=MAX_METADATA_JSON_BYTES,
            label="clean-score file",
        )
        if not isinstance(clean_scores, list):
            raise ValueError("--clean-scores must contain a JSON array")
        threshold = calibrate_owner_threshold(clean_scores, args.target_fpr)
        result = OwnerVerificationConfig(
            threshold=threshold,
            negative_weight=args.negative_weight,
            calibration_source=args.calibration_source,
        ).to_dict()
        result["target_fpr"] = args.target_fpr
        result["clean_score_count"] = len(clean_scores)
    elif args.command == "verify":
        owner_key = os.getenv("SKILLCODER_OWNER_KEY", "").strip()
        try:
            validate_owner_key(owner_key)
        except ValueError as exc:
            raise RuntimeError(
                "SKILLCODER_OWNER_KEY must contain at least 32 UTF-8 bytes"
            ) from exc
        result = verify_package(args.package, owner_key)
    elif args.command == "verify-family":
        owner_key = os.getenv("SKILLCODER_OWNER_KEY", "").strip()
        try:
            validate_owner_key(owner_key)
        except ValueError as exc:
            raise RuntimeError(
                "SKILLCODER_OWNER_KEY must contain at least 32 UTF-8 bytes"
            ) from exc
        result = verify_buyer_family(args.family, owner_key)
    else:
        owner_key = os.getenv("SKILLCODER_OWNER_KEY", "").strip()
        try:
            validate_owner_key(owner_key)
        except ValueError as exc:
            raise RuntimeError(
                "SKILLCODER_OWNER_KEY must contain at least 32 UTF-8 bytes"
            ) from exc
        result = verify_release_manifest(args.run, owner_key)
    print(json.dumps(_cli_result(args.command, args, result), ensure_ascii=False, indent=2))
    if args.command in {"verify", "verify-family", "verify-release"} and not result["valid"]:
        raise SystemExit(1)
    if (
        args.command in {"run", "run-family", "probe", "probe-family"}
        and not result["release_ready"]
    ):
        raise SystemExit(2)
    if args.command == "probe-suspect":
        detection = result.get("detection_result")
        if not isinstance(detection, dict) or not detection.get("supported"):
            raise SystemExit(2)


def main() -> None:
    try:
        _run_cli()
    except DomainLanguageExhausted as exc:
        print(f"skillcoder: error: {exc.public_message}", file=sys.stderr)
        raise SystemExit(1) from None
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"skillcoder: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
