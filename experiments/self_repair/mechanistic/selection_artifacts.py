"""Model-free construction and transport of immutable causal selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causal_scan import intervention_sites, parse_path_specification
from .core import (
    ContractError,
    REQUIRED_SITES,
    canonical_json,
    read_json,
    sha256_file,
    sha256_value,
    validate_sha256,
    write_json,
)


FROZEN_SELECTION_STATUS = "frozen_discovery_selection"
_READOUT_POLICY = {
    "candidate_scoring": "mean_log_probability_per_token",
    "candidate_branching": "restore_identical_query_snapshot_before_each_candidate",
    "schedule_aggregation": "logmeanexp_over_all_preregistered_schedules",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return dict(value)


def _trimmed_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{label} must be a non-empty, trimmed string")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ContractError(f"{label} must be positive")
    return result


def load_frozen_selection(
    path: Path, *, config_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load one self-hashed frozen selection and bind it to an exact config."""

    selection = _mapping(read_json(path), "frozen selection")
    if selection.get("schema_version") != "1.0.0":
        raise ContractError("frozen selection schema version mismatch")
    if selection.get("status") != FROZEN_SELECTION_STATUS:
        raise ContractError("selection is not a frozen_discovery_selection")
    declared = validate_sha256(
        str(selection.get("selection_sha256", "")), "frozen selection hash")
    observed = sha256_value({
        key: value for key, value in selection.items() if key != "selection_sha256"
    })
    if declared != observed:
        raise ContractError("frozen selection content hash mismatch")
    expected_config = validate_sha256(config_sha256, "config hash")
    if selection.get("config_sha256") != expected_config:
        raise ContractError("frozen selection targets a different config")
    return selection, sha256_file(path)


def _config_dimensions(config_path: Path) -> tuple[dict[str, Any], int, int, set[str]]:
    config = _mapping(read_json(config_path), "config")
    model = _mapping(config.get("model"), "config.model")
    layer_count = _positive_int(model.get("layers"), "config.model.layers")
    head_count = _positive_int(model.get("heads"), "config.model.heads")
    anchors = _mapping(config.get("anchors"), "config.anchors").get("primary")
    if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
        raise ContractError("config.anchors.primary must be a non-empty sequence")
    normalized_anchors = {
        _trimmed_string(anchor, "config.anchors.primary entry") for anchor in anchors
    }
    if not normalized_anchors or len(normalized_anchors) != len(anchors):
        raise ContractError("config.anchors.primary must be non-empty and unique")
    return config, layer_count, head_count, normalized_anchors


def _selection_head(selection: Mapping[str, Any]) -> int | None:
    head = selection.get("head")
    kv_head = selection.get("kv_head")
    if head is not None and kv_head is not None and head != kv_head:
        raise ContractError("frozen selection head and kv_head disagree")
    selected = kv_head if kv_head is not None else head
    if selected is None:
        return None
    return _non_negative_int(selected, "frozen selection head")


def build_path_selection(
    *,
    config_path: Path,
    writer_selection_path: Path,
    mediator_site: str,
    mediator_layer: int,
    mediator_anchor: str,
    mediator_head: int | None,
) -> dict[str, Any]:
    """Derive a path selection from one immutable writer and an explicit mediator."""

    _, layer_count, head_count, configured_anchors = _config_dimensions(config_path)
    writer, writer_file_sha = load_frozen_selection(
        writer_selection_path, config_sha256=sha256_file(config_path))
    component = _trimmed_string(writer.get("component"), "writer selection component")
    sites = intervention_sites(component)
    if len(sites) != 1:
        raise ContractError(
            "writer selection component must identify exactly one tensor site; "
            "freeze k_only or v_only instead of joint kv"
        )
    writer_site = sites[0]
    if writer_site not in REQUIRED_SITES:
        raise ContractError(f"writer selection has unsupported tensor site: {writer_site!r}")
    writer_layer = _non_negative_int(writer.get("layer"), "writer selection layer")
    writer_anchor = _trimmed_string(writer.get("anchor"), "writer selection anchor")
    writer_head = _selection_head(writer)
    mediator_site = _trimmed_string(mediator_site, "mediator site")
    mediator_layer = _non_negative_int(mediator_layer, "mediator layer")
    mediator_anchor = _trimmed_string(mediator_anchor, "mediator anchor")
    if mediator_head is not None:
        mediator_head = _non_negative_int(mediator_head, "mediator head")
    if writer_layer >= layer_count or mediator_layer >= layer_count:
        raise ContractError("path endpoint layer is outside the configured model range")
    if writer_head is not None and writer_head >= head_count:
        raise ContractError("writer selection head is outside the configured model range")
    if mediator_head is not None and mediator_head >= head_count:
        raise ContractError("mediator head is outside the configured model range")
    if writer_anchor not in configured_anchors or mediator_anchor not in configured_anchors:
        raise ContractError("path endpoint anchor is not in config.anchors.primary")
    if mediator_site not in REQUIRED_SITES:
        raise ContractError(f"mediator site is unsupported: {mediator_site!r}")

    path = {
        "writer": {
            "site": writer_site,
            "layer": writer_layer,
            "anchor": writer_anchor,
            "head": writer_head,
        },
        "mediator": {
            "site": mediator_site,
            "layer": mediator_layer,
            "anchor": mediator_anchor,
            "head": mediator_head,
        },
    }
    normalized_path = parse_path_specification({"path": path}).identity
    donor_arm = _trimmed_string(writer.get("donor_arm"), "writer selection donor_arm")
    relation = _trimmed_string(writer.get("relation"), "writer selection relation")
    direction = _trimmed_string(writer.get("direction"), "writer selection direction")
    readout_sha = validate_sha256(
        str(writer.get("readout_sha256", "")), "writer selection readout hash")
    source_cell = _trimmed_string(
        writer.get("selection_source_cell_id"), "writer selection source cell")
    selection: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": FROZEN_SELECTION_STATUS,
        "config_sha256": writer["config_sha256"],
        "component": "path",
        "layer": writer_layer,
        "head": writer_head,
        "anchor": writer_anchor,
        "direction": direction,
        "donor_arm": donor_arm,
        "relation": relation,
        "readout_sha256": readout_sha,
        "selection_source_cell_id": source_cell,
        "path": normalized_path,
        "writer_selection_file_sha256": writer_file_sha,
        "writer_selection_identity_sha256": writer["selection_sha256"],
    }
    selection["selection_sha256"] = sha256_value(selection)
    return selection


def _token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty token-ID array")
    result = [_non_negative_int(token, label) for token in value]
    return result


def validate_bound_readouts(
    path: Path, *, config_path: Path,
) -> tuple[dict[str, Any], str]:
    """Validate a model-preflight bound readout without loading the model."""

    source = _mapping(read_json(path), "bound readout")
    declared = validate_sha256(
        str(source.get("bound_readout_sha256", "")), "bound readout hash")
    observed = sha256_value({
        key: value for key, value in source.items() if key != "bound_readout_sha256"
    })
    if declared != observed:
        raise ContractError("bound readout self-hash mismatch")
    for field, expected in _READOUT_POLICY.items():
        if source.get(field) != expected:
            raise ContractError("bound readout changes the frozen scoring contract")

    config = _mapping(read_json(config_path), "config")
    model = _mapping(config.get("model"), "config.model")
    if source.get("config_sha256") != sha256_file(config_path):
        raise ContractError("bound readout targets a different config")
    for field in ("repo", "revision"):
        expected = _trimmed_string(model.get(field), f"config.model.{field}")
        if source.get(f"model_{field}") != expected:
            raise ContractError(f"bound readout model {field} differs from the config")

    readouts = source.get("readouts")
    if not isinstance(readouts, list) or not readouts:
        raise ContractError("bound readout must contain at least one readout")
    ids: set[str] = set()
    roots = 0
    for item in readouts:
        row = _mapping(item, "bound readout entry")
        readout_id = _trimmed_string(row.get("id"), "bound readout id")
        if readout_id in ids:
            raise ContractError("bound readout IDs must be unique")
        ids.add(readout_id)
        _trimmed_string(row.get("prefix"), f"readout {readout_id} prefix")
        anchor = _trimmed_string(row.get("anchor"), f"readout {readout_id} anchor")
        _token_ids(row.get("prefix_token_ids"), f"readout {readout_id} prefix_token_ids")
        roots += int(readout_id == "root" and anchor == "query_end")
    if roots != 1:
        raise ContractError("bound readout requires exactly one root readout at query_end")

    schedules = source.get("emission_schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ContractError("bound readout must contain at least one emission schedule")
    schedule_ids: set[str] = set()
    for item in schedules:
        row = _mapping(item, "bound emission schedule")
        schedule_id = _trimmed_string(row.get("id"), "bound emission schedule id")
        if schedule_id in schedule_ids:
            raise ContractError("bound emission schedule IDs must be unique")
        schedule_ids.add(schedule_id)
        _non_negative_int(
            row.get("prefix_start_offset_frames"),
            f"schedule {schedule_id} prefix_start_offset_frames",
        )
        _non_negative_int(
            row.get("pad_frames_between_tokens"),
            f"schedule {schedule_id} pad_frames_between_tokens",
        )

    candidates = source.get("candidate_token_ids")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ContractError("bound readout has no candidate_token_ids object")
    for value, token_ids in candidates.items():
        label = _trimmed_string(value, "bound readout candidate")
        _token_ids(token_ids, f"candidate {label!r} token IDs")
    return source, sha256_file(path)


def rebind_mechanistic_selection(
    *, config_path: Path, source_selection_path: Path, bound_readout_path: Path,
) -> dict[str, Any]:
    """Transport a causal selection to a newly bound verbalizer/readout contract."""

    source, source_file_sha = load_frozen_selection(
        source_selection_path, config_sha256=sha256_file(config_path))
    bound, bound_file_sha = validate_bound_readouts(
        bound_readout_path, config_path=config_path)
    source_identity_sha = source["selection_sha256"]
    source_readout_sha = validate_sha256(
        str(source.get("readout_sha256", "")), "source selection readout hash")
    transported = {
        key: value for key, value in source.items() if key != "selection_sha256"
    }
    transported.update({
        "readout_sha256": bound_file_sha,
        "source_readout_sha256": source_readout_sha,
        "source_selection_file_sha256": source_file_sha,
        "source_selection_identity_sha256": source_identity_sha,
        "bound_readout_identity_sha256": bound["bound_readout_sha256"],
    })
    identity_fields = (
        "component", "layer", "head", "anchor", "donor_arm", "relation", "path",
    )
    for field in identity_fields:
        if (field in source) != (field in transported) or source.get(field) != transported.get(field):
            raise ContractError(f"selection transport changed frozen intervention field {field}")
    transported["selection_sha256"] = sha256_value(transported)
    return transported


def write_immutable_selection(path: Path, selection: Mapping[str, Any]) -> None:
    """Write once, accepting an existing file only when its exact bytes agree."""

    document = dict(selection)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ContractError("refusing to overwrite a different frozen selection")
        return
    write_json(path, document)


def _optional_head(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("head must be a non-negative integer or 'none'") from error
    if result < 0:
        raise argparse.ArgumentTypeError("head must be a non-negative integer or 'none'")
    return result


def freeze_path_selection_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an immutable two-stage path selection from a writer selection "
            "and an explicit mediator endpoint."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--writer-selection", "--selection", dest="writer_selection", type=Path,
        required=True, help="Self-hashed frozen selection defining the writer endpoint.",
    )
    parser.add_argument("--mediator-site", required=True, choices=sorted(REQUIRED_SITES))
    parser.add_argument("--mediator-layer", type=int, required=True)
    parser.add_argument("--mediator-anchor", required=True)
    parser.add_argument(
        "--mediator-head", type=_optional_head, required=True,
        help="Non-negative head index, or 'none' to freeze a full-site mediator.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.writer_selection.resolve():
        raise ContractError("path selection output must not overwrite its writer selection")
    selection = build_path_selection(
        config_path=args.config,
        writer_selection_path=args.writer_selection,
        mediator_site=args.mediator_site,
        mediator_layer=args.mediator_layer,
        mediator_anchor=args.mediator_anchor,
        mediator_head=args.mediator_head,
    )
    write_immutable_selection(args.output, selection)
    print(canonical_json({
        "output": str(args.output),
        "selection_file_sha256": sha256_file(args.output),
        "selection_identity_sha256": selection["selection_sha256"],
    }))
    return 0


def rebind_mechanistic_selection_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transport a frozen mechanistic selection to a newly model-bound "
            "readout while preserving its intervention identity."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-selection", "--selection", dest="source_selection", type=Path,
        required=True, help="Existing self-hashed frozen selection to transport.",
    )
    parser.add_argument(
        "--bound-readouts", "--readouts", dest="bound_readouts", type=Path,
        required=True, help="New model-preflight readouts.bound.json artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.source_selection.resolve():
        raise ContractError("transport output must not overwrite the source selection")
    selection = rebind_mechanistic_selection(
        config_path=args.config,
        source_selection_path=args.source_selection,
        bound_readout_path=args.bound_readouts,
    )
    write_immutable_selection(args.output, selection)
    print(canonical_json({
        "output": str(args.output),
        "selection_file_sha256": sha256_file(args.output),
        "selection_identity_sha256": selection["selection_sha256"],
        "readout_sha256": selection["readout_sha256"],
    }))
    return 0


__all__ = [
    "FROZEN_SELECTION_STATUS",
    "build_path_selection",
    "freeze_path_selection_main",
    "load_frozen_selection",
    "rebind_mechanistic_selection",
    "rebind_mechanistic_selection_main",
    "validate_bound_readouts",
    "write_immutable_selection",
]
