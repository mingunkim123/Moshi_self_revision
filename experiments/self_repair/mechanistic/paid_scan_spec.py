"""Build immutable, model-free specifications for paid mechanistic scans.

The paid scan runners accept a literal ``execution`` contract and the readiness
estimator independently expands the declared grid.  This module is the single
place that constructs both views, checks donor availability, reserves storage,
and freezes the resulting arithmetic before any model backend is imported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causal_scan import (
    active_arms,
    materialize_donor_assignments,
    parse_path_specification,
    repair_recipients,
)
from .core import (
    ContractError,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_value,
    write_json,
)
from .readiness import ReadinessError, WorkloadEstimate, estimate_workload


SPEC_SCHEMA_VERSION = "1.0.0"
SPEC_ARTIFACT_TYPE = "frozen_paid_scan_spec"

# These are reservations, not predictions of compressed output size.  They are
# deliberately wider than the current JSONL result rows and retain activations
# as float32 even though the checkpoint itself runs in bfloat16.
BUILDER_POLICY: dict[str, Any] = {
    "version": "1.0.0",
    "selection_policy": "role_then_lexicographically_first_scenarios",
    "recipient_policy": "all_non_clean_selected_trials",
    "encoded_cache_policy": "user_plus_continuous_plus_assistant_silence",
    "activation_reservation_dtype": "float32",
    "activation_reservation_dtype_bytes": 4,
    "user_codebooks": 8,
    "code_dtype_bytes": 8,
    "audio_sample_width_bytes": 2,
    "wav_header_bytes": 44,
    "result_bytes_per_cell": 16_384,
    "fixed_reserved_bytes": 1_073_741_824,
}
BUILDER_POLICY_SHA256 = sha256_value(BUILDER_POLICY)

_KINDS = {"residual", "component", "kv", "path"}
_COMPONENTS = {
    "residual": {"resid_post"},
    "component": {"attn_out", "mlp_out", "head_z"},
    "kv": {"k_only", "v_only", "kv"},
    "path": {"path"},
}
_DEFAULT_DONORS = ("clean_current", "self", "shuffled")
_DEFAULT_CONTROLS = ("self", "current", "wrong", "shuffled")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return dict(value)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{label} must be {qualifier}")
    return value


def _strings(values: Sequence[str] | None, label: str) -> list[str]:
    if values is None or isinstance(values, (str, bytes)):
        raise ContractError(f"{label} must be a non-empty sequence")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ContractError(f"{label} entries must be non-empty, trimmed strings")
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ContractError(f"{label} must be non-empty and contain no duplicates")
    return result


def _layers(values: Sequence[int] | None, *, layer_count: int) -> list[int]:
    if values is None or isinstance(values, (str, bytes)):
        raise ContractError("layers must be a non-empty sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("layers must contain only integer indices")
        if value < 0 or value >= layer_count:
            raise ContractError(f"layer {value} is outside the frozen model range [0, {layer_count})")
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ContractError("layers must be non-empty and contain no duplicates")
    return result


def _selection(path: Path, config_sha256: str) -> tuple[dict[str, Any], str]:
    selection = _mapping(read_json(path), "selection")
    if selection.get("status") != "frozen_discovery_selection":
        raise ContractError("selection is not a frozen_discovery_selection")
    stated = selection.get("selection_sha256")
    observed = sha256_value({key: value for key, value in selection.items()
                             if key != "selection_sha256"})
    if stated != observed:
        raise ContractError("frozen selection content hash mismatch")
    if selection.get("config_sha256") != config_sha256:
        raise ContractError("frozen selection targets a different config")
    return selection, sha256_file(path)


def _kind_for_component(component: str) -> str:
    for kind, allowed in _COMPONENTS.items():
        if component in allowed:
            return kind
    raise ContractError(f"frozen selection has unsupported component: {component!r}")


def _same_or_derive(
    explicit: Sequence[Any] | None, derived: Sequence[Any], label: str,
) -> list[Any]:
    result = list(derived) if explicit is None else list(explicit)
    if result != list(derived):
        raise ContractError(
            f"explicit {label} differs from the single site frozen by --selection: "
            f"expected={list(derived)!r}, observed={result!r}")
    return result


def _selected_rows(
    manifest: Sequence[Mapping[str, Any]], role: str, limit_scenarios: int | None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in manifest if row.get("role") == role]
    if not rows:
        raise ContractError(f"manifest has no trials for role {role!r}")
    if limit_scenarios is not None:
        _positive_int(limit_scenarios, "limit_scenarios")
        scenario_ids = {row.get("scenario_id") for row in rows}
        if any(not isinstance(item, str) or not item for item in scenario_ids):
            raise ContractError("bounded scenario selection requires non-empty scenario_id values")
        allowed = sorted(scenario_ids)[:limit_scenarios]
        rows = [row for row in rows if row.get("scenario_id") in allowed]
    return rows


def _capture_contract(
    *, kind: str, components: Sequence[str], layers: Sequence[int],
    anchors: Sequence[str], selection: Mapping[str, Any] | None, hidden_size: int,
) -> dict[str, Any]:
    capture_layers = list(layers)
    capture_anchors = list(anchors)
    if kind == "kv":
        sites = ["k_post_rope", "v_post_rope"]
    elif kind == "path":
        if selection is None:
            raise ContractError("path capture reservation requires a frozen selection")
        path = parse_path_specification(selection)
        capture_layers = list(dict.fromkeys((path.writer.layer, path.mediator.layer)))
        capture_anchors = list(dict.fromkeys((path.writer.anchor, path.mediator.anchor)))
        sites = list(dict.fromkeys((path.writer.site, path.mediator.site)))
    else:
        sites = list(components)
    return {
        "selector": {},
        "layers": capture_layers,
        "anchors": capture_anchors,
        "sites": sites,
        "dtype_bytes": BUILDER_POLICY["activation_reservation_dtype_bytes"],
        "elements_per_site": hidden_size,
    }


def _frozen_generation(
    *, config: Mapping[str, Any], seeds: Sequence[int] | None,
    branches: Sequence[str] | None,
) -> dict[str, Any]:
    conversation = _mapping(config.get("conversation"), "config.conversation")
    modes = _strings(conversation.get("required_modes"), "config.conversation.required_modes")
    response = _mapping(conversation.get("response"), "config.conversation.response")
    response_ms = response.get("post_user_max_ms")
    if isinstance(response_ms, bool) or not isinstance(response_ms, (int, float)) or response_ms < 0:
        raise ContractError("config.conversation.response.post_user_max_ms must be non-negative")
    if seeds is None:
        manifest_config = _mapping(config.get("manifest"), "config.manifest")
        seeds = manifest_config.get("discovery_generation_seeds")
    if seeds is None or isinstance(seeds, (str, bytes)):
        raise ContractError("generation seeds must be explicitly frozen in config or CLI")
    normalized_seeds: list[int] = []
    for seed in seeds:
        normalized_seeds.append(_positive_int(seed, "generation seed", allow_zero=True))
    if not normalized_seeds or len(normalized_seeds) != len(set(normalized_seeds)):
        raise ContractError("generation seeds must be non-empty and unique")
    normalized_branches = _strings(
        ("baseline", "patched") if branches is None else branches,
        "generation branches",
    )
    if set(normalized_branches) != {"baseline", "patched"}:
        raise ContractError("full-duplex generation branches must be exactly baseline and patched")
    return {
        "trial_selector": {"exclude_clean": True},
        "seeds": normalized_seeds,
        "branches": normalized_branches,
        "startup_modes": modes,
        "response_capture_ms": response_ms,
    }


def build_paid_scan_spec(
    *,
    config_path: Path,
    manifest_path: Path,
    role: str,
    kind: str | None,
    layers: Sequence[int] | None,
    anchors: Sequence[str] | None,
    donors: Sequence[str] | None,
    controls: Sequence[str] | None,
    components: Sequence[str] | None,
    full_replays_per_cell: int,
    readout_steps_per_cell: int,
    limit_scenarios: int | None = None,
    selection_path: Path | None = None,
    confirmation_control_arms: Sequence[str] | None = None,
    include_generation: bool = False,
    generation_seeds: Sequence[int] | None = None,
    generation_branches: Sequence[str] | None = None,
    upstream_selection_path: Path | None = None,
) -> tuple[dict[str, Any], WorkloadEstimate]:
    """Construct and self-verify one exact scan grid without loading a model."""

    if selection_path is not None and upstream_selection_path is not None:
        raise ContractError("selection_path and upstream_selection_path are mutually exclusive")

    config = _mapping(read_json(config_path), "config")
    manifest = read_jsonl(manifest_path)
    if not isinstance(role, str) or not role or role != role.strip():
        raise ContractError("role must be a non-empty, trimmed string")
    model = _mapping(config.get("model"), "config.model")
    layer_count = _positive_int(model.get("layers"), "config.model.layers")
    head_count = _positive_int(model.get("heads"), "config.model.heads")
    hidden_size = _positive_int(model.get("hidden_size"), "config.model.hidden_size")
    config_sha = sha256_file(config_path)
    frozen_selection: dict[str, Any] | None = None
    selection_file_sha: str | None = None

    normalized_donors = _strings(
        _DEFAULT_DONORS if donors is None else donors, "donors")
    normalized_controls = _strings(
        _DEFAULT_CONTROLS if controls is None else controls, "controls")

    bound_selection_path = selection_path or upstream_selection_path
    if bound_selection_path is not None:
        frozen_selection, selection_file_sha = _selection(bound_selection_path, config_sha)
    if selection_path is not None:
        assert frozen_selection is not None
        selected_component = frozen_selection.get("component")
        if not isinstance(selected_component, str):
            raise ContractError("frozen selection has no component")
        selected_kind = _kind_for_component(selected_component)
        if kind is not None and kind != selected_kind:
            raise ContractError("explicit kind differs from the frozen selection")
        kind = selected_kind
        selected_layer = frozen_selection.get("layer")
        if isinstance(selected_layer, bool) or not isinstance(selected_layer, int):
            raise ContractError("frozen selection has no integer layer")
        selected_anchor = frozen_selection.get("anchor")
        if not isinstance(selected_anchor, str) or not selected_anchor:
            raise ContractError("frozen selection has no semantic anchor")
        layers = _same_or_derive(layers, [selected_layer], "layers")
        anchors = _same_or_derive(anchors, [selected_anchor], "anchors")
        components = _same_or_derive(components, [selected_component], "components")
        donor_arm = frozen_selection.get("donor_arm")
        if not isinstance(donor_arm, str) or not donor_arm:
            raise ContractError("frozen selection has no donor arm")
        requested_confirmation_controls = (
            [] if confirmation_control_arms is None
            else _strings(confirmation_control_arms, "confirmation_control_arms")
        )
        if donor_arm in requested_confirmation_controls:
            raise ContractError(
                "confirmation_control_arms must exclude the frozen primary arm")
        requested_arms = [donor_arm, *requested_confirmation_controls]
        if kind == "component":
            normalized_controls = _same_or_derive(controls, requested_arms, "controls")
        else:
            normalized_donors = _same_or_derive(donors, requested_arms, "donors")
        if kind == "path":
            path = parse_path_specification(frozen_selection)
            if path.writer.layer != selected_layer or path.writer.anchor != selected_anchor:
                raise ContractError("path selection layer/anchor differs from its writer endpoint")
    elif upstream_selection_path is not None:
        assert frozen_selection is not None
        if kind is None:
            raise ContractError("--upstream-selection requires an explicit new-stage kind")
        selected_layer = frozen_selection.get("layer")
        if isinstance(selected_layer, bool) or not isinstance(selected_layer, int):
            raise ContractError("upstream frozen selection has no integer layer")
        selected_anchor = frozen_selection.get("anchor")
        if not isinstance(selected_anchor, str) or not selected_anchor:
            raise ContractError("upstream frozen selection has no semantic anchor")
        layers = _same_or_derive(layers, [selected_layer], "layers")
        anchors = _same_or_derive(anchors, [selected_anchor], "anchors")
        if kind == "path":
            path = parse_path_specification(frozen_selection)
            if path.writer.layer != selected_layer or path.writer.anchor != selected_anchor:
                raise ContractError("path selection layer/anchor differs from its writer endpoint")
    elif confirmation_control_arms is not None:
        raise ContractError("confirmation_control_arms requires --selection")

    if upstream_selection_path is not None and confirmation_control_arms is not None:
        raise ContractError("confirmation_control_arms requires --selection")

    if kind not in _KINDS:
        raise ContractError(f"kind must be one of {sorted(_KINDS)}")
    normalized_layers = _layers(layers, layer_count=layer_count)
    normalized_anchors = _strings(anchors, "anchors")
    configured_anchors = _mapping(config.get("anchors"), "config.anchors").get("primary")
    allowed_anchors = set(_strings(configured_anchors, "config.anchors.primary"))
    unknown_anchors = [anchor for anchor in normalized_anchors if anchor not in allowed_anchors]
    if unknown_anchors:
        raise ContractError(f"anchors are not in the frozen config: {unknown_anchors}")
    normalized_components = _strings(components, "components")
    if not normalized_components or any(
        component not in _COMPONENTS[kind] for component in normalized_components
    ):
        raise ContractError(
            f"components are invalid for {kind}: {normalized_components!r}")
    if kind in {"residual", "path"} and set(normalized_components) != _COMPONENTS[kind]:
        raise ContractError(f"{kind} requires exactly {sorted(_COMPONENTS[kind])}")

    # Validate both lists even though only one is billable for a given scan kind;
    # the inactive list is still part of the byte-for-byte CLI authorization.
    active_arms("residual", normalized_donors, normalized_controls)
    active_arms("component", normalized_donors, normalized_controls)
    billable_arms = list(active_arms(kind, normalized_donors, normalized_controls))
    selected_rows = _selected_rows(manifest, role, limit_scenarios)
    recipients = repair_recipients(selected_rows)
    materialize_donor_assignments(selected_rows, recipients, billable_arms)

    full_replays = _positive_int(full_replays_per_cell, "full_replays_per_cell")
    readout_steps = _positive_int(readout_steps_per_cell, "readout_steps_per_cell")
    scan_components: list[Any] = list(normalized_components)
    if selection_path is not None and normalized_components == ["head_z"]:
        assert frozen_selection is not None
        selected_head = frozen_selection.get("head")
        if isinstance(selected_head, bool) or not isinstance(selected_head, int):
            raise ContractError("a frozen head_z selection must contain an integer head")
        if selected_head < 0 or selected_head >= head_count:
            raise ContractError("frozen head is outside the configured model head range")
        scan_components = [{"name": "head_z", "heads": [selected_head]}]
    if upstream_selection_path is not None and kind == "kv":
        assert frozen_selection is not None
        selected_head = frozen_selection.get("kv_head", frozen_selection.get("head"))
        if selected_head is not None:
            if isinstance(selected_head, bool) or not isinstance(selected_head, int):
                raise ContractError("upstream frozen selection head must be an integer or null")
            if selected_head < 0 or selected_head >= head_count:
                raise ContractError("upstream frozen selection head is outside the configured model range")
            scan_components = [
                {"name": component, "heads": [selected_head]}
                for component in normalized_components
            ]

    execution = {
        "kind": kind,
        "role": role,
        "layers": normalized_layers,
        "anchors": normalized_anchors,
        "donors": normalized_donors,
        "controls": normalized_controls,
        "components": normalized_components,
        "limit_scenarios": limit_scenarios,
        "selection_sha256": selection_file_sha,
    }
    scan = {
        "name": f"{role}_{kind}_grid",
        "layers": normalized_layers,
        "anchors": normalized_anchors,
        "donor_arms": billable_arms,
        "components": scan_components,
        "full_replays_per_cell": full_replays,
        "readout_steps_per_cell": readout_steps,
    }
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "artifact_type": SPEC_ARTIFACT_TYPE,
        "trial_selector": {"roles": [role]},
        "recipient_selector": {"exclude_clean": True},
        "scans": [scan],
        "execution": execution,
        "storage": {
            "user_codebooks": BUILDER_POLICY["user_codebooks"],
            "code_dtype_bytes": BUILDER_POLICY["code_dtype_bytes"],
            "audio_sample_width_bytes": BUILDER_POLICY["audio_sample_width_bytes"],
            "wav_header_bytes": BUILDER_POLICY["wav_header_bytes"],
            "result_bytes_per_cell": BUILDER_POLICY["result_bytes_per_cell"],
            "fixed_reserved_bytes": BUILDER_POLICY["fixed_reserved_bytes"],
            "captures": [_capture_contract(
                kind=kind, components=normalized_components, layers=normalized_layers,
                anchors=normalized_anchors, selection=frozen_selection,
                hidden_size=hidden_size,
            )],
        },
        "frozen_dimensions": {
            "model_layers": layer_count,
            "model_heads": head_count,
            "hidden_size": hidden_size,
            "generation_included": bool(include_generation),
        },
        "provenance": {
            "config_sha256": config_sha,
            "manifest_sha256": sha256_file(manifest_path),
            "selection_file_sha256": selection_file_sha,
            "selection_identity_sha256": (
                frozen_selection.get("selection_sha256") if frozen_selection else None),
            "builder_policy_sha256": BUILDER_POLICY_SHA256,
        },
    }
    if include_generation:
        generation = _frozen_generation(
            config=config, seeds=generation_seeds, branches=generation_branches)
        spec["generation"] = generation
        spec["frozen_dimensions"]["generation_seeds"] = list(generation["seeds"])
        spec["frozen_dimensions"]["generation_branches"] = list(generation["branches"])
        spec["frozen_dimensions"]["startup_modes"] = list(generation["startup_modes"])
        spec["frozen_dimensions"]["response_capture_ms"] = generation["response_capture_ms"]

    # First let the estimator expand the literal grid, then freeze its exact
    # totals into the artifact and run the estimator again against those locks.
    try:
        preliminary = estimate_workload(manifest, config, spec)
    except ReadinessError as error:
        raise ContractError(str(error)) from error
    scan["expected_cell_count"] = preliminary.cell_count
    if include_generation:
        spec["generation"]["expected_generation_count"] = preliminary.generation_count
    try:
        estimate = estimate_workload(manifest, config, spec)
    except ReadinessError as error:  # pragma: no cover - defensive consistency check.
        raise ContractError(str(error)) from error
    declared = estimate.to_dict()
    spec["declared_workload"] = declared
    spec["declared_workload_sha256"] = sha256_value(declared)
    spec["scan_spec_identity_sha256"] = sha256_value(spec)
    verify_paid_scan_spec(
        spec, config_path=config_path, manifest_path=manifest_path,
        selection_path=bound_selection_path)
    return spec, estimate


def verify_paid_scan_spec(
    spec: Mapping[str, Any], *, config_path: Path, manifest_path: Path,
    selection_path: Path | None = None,
) -> WorkloadEstimate:
    """Verify source hashes, content identity, and all frozen workload totals."""

    document = _mapping(spec, "paid scan spec")
    identity = document.pop("scan_spec_identity_sha256", None)
    if identity != sha256_value(document):
        raise ContractError("paid scan spec content identity mismatch")
    if document.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ContractError("paid scan spec schema version mismatch")
    if document.get("artifact_type") != SPEC_ARTIFACT_TYPE:
        raise ContractError("paid scan spec artifact type mismatch")
    provenance = _mapping(document.get("provenance"), "paid scan spec provenance")
    if provenance.get("config_sha256") != sha256_file(config_path):
        raise ContractError("paid scan spec config SHA-256 mismatch")
    if provenance.get("manifest_sha256") != sha256_file(manifest_path):
        raise ContractError("paid scan spec manifest SHA-256 mismatch")
    if provenance.get("builder_policy_sha256") != BUILDER_POLICY_SHA256:
        raise ContractError("paid scan spec builder policy SHA-256 mismatch")
    execution = _mapping(document.get("execution"), "paid scan spec execution")
    selection_file_sha = execution.get("selection_sha256")
    if selection_file_sha is None:
        if selection_path is not None or provenance.get("selection_file_sha256") is not None:
            raise ContractError("paid scan spec unexpectedly omits the selection file binding")
    else:
        if selection_path is None:
            raise ContractError("selection-bound paid scan spec requires the selection file")
        selection, observed_file_sha = _selection(selection_path, sha256_file(config_path))
        if selection_file_sha != observed_file_sha:
            raise ContractError("paid scan spec selection file SHA-256 mismatch")
        if provenance.get("selection_file_sha256") != observed_file_sha:
            raise ContractError("paid scan spec provenance selection SHA-256 mismatch")
        if provenance.get("selection_identity_sha256") != selection.get("selection_sha256"):
            raise ContractError("paid scan spec selection identity mismatch")
    config = _mapping(read_json(config_path), "config")
    manifest = read_jsonl(manifest_path)
    try:
        estimate = estimate_workload(manifest, config, document)
    except ReadinessError as error:
        raise ContractError(str(error)) from error
    declared = document.get("declared_workload")
    if not isinstance(declared, Mapping) or dict(declared) != estimate.to_dict():
        raise ContractError("paid scan spec frozen workload differs from recomputed workload")
    if document.get("declared_workload_sha256") != sha256_value(dict(declared)):
        raise ContractError("paid scan spec declared workload SHA-256 mismatch")
    return estimate


def write_frozen_paid_scan_spec(path: Path, spec: Mapping[str, Any]) -> None:
    """Write once; an existing path is accepted only when bytes are identical."""

    document = dict(spec)
    if path.exists():
        if path.read_text(encoding="utf-8") != _serialized(document):
            raise ContractError("refusing to overwrite a different frozen paid scan spec")
        return
    write_json(path, document)


def _serialized(value: Mapping[str, Any]) -> str:
    # Mirror core.write_json exactly so idempotence is a byte-level assertion.
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return value.split(",")


def _int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        result: list[int] = []
        for item in value.split(","):
            if ":" not in item:
                result.append(int(item))
                continue
            bounds = item.split(":")
            if len(bounds) not in {2, 3} or not all(bounds[:2]):
                raise ValueError
            start, stop = int(bounds[0]), int(bounds[1])
            step = int(bounds[2]) if len(bounds) == 3 and bounds[2] else 1
            if step <= 0 or stop <= start:
                raise ValueError
            result.extend(range(start, stop, step))
        return result
    except ValueError as error:
        raise ContractError(
            "integer list must contain comma-separated integers or increasing start:stop[:step] ranges"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze one exact paid-scan grid and its model-free workload arithmetic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(_KINDS))
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--layers", help="Comma-separated indices or exclusive ranges, e.g. 0:32 or 0,15,31."
    )
    parser.add_argument("--anchors", help="Comma-separated frozen semantic anchors.")
    parser.add_argument("--donors", help="Comma-separated donor arms.")
    parser.add_argument("--controls", help="Comma-separated component-control arms.")
    parser.add_argument("--components", help="Comma-separated components/modes.")
    parser.add_argument("--limit-scenarios", type=int)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--selection", type=Path,
        help="Derive and bind the exact single frozen confirmatory site/arm.",
    )
    selection_group.add_argument(
        "--upstream-selection", type=Path,
        help=(
            "Bind an upstream discovery selection and derive only its layer/anchor; "
            "--kind, --components, and the new stage's active arms remain explicit."
        ),
    )
    parser.add_argument(
        "--confirmation-control-arms",
        help="Comma-separated controls appended after the frozen primary donor arm.",
    )
    parser.add_argument("--full-replays-per-cell", type=int, required=True)
    parser.add_argument("--readout-steps-per-cell", type=int, required=True)
    parser.add_argument("--include-generation", action="store_true",
                        help="Also reserve baseline/patched full-duplex audio for this grid.")
    parser.add_argument("--generation-seeds",
                        help="Comma-separated seeds; defaults to frozen config seeds.")
    parser.add_argument("--generation-branches",
                        help="Must be baseline,patched (order is frozen as supplied).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec, estimate = build_paid_scan_spec(
        config_path=args.config,
        manifest_path=args.manifest,
        role=args.role,
        kind=args.kind,
        layers=_int_csv(args.layers),
        anchors=_csv(args.anchors),
        donors=_csv(args.donors),
        controls=_csv(args.controls),
        components=_csv(args.components),
        full_replays_per_cell=args.full_replays_per_cell,
        readout_steps_per_cell=args.readout_steps_per_cell,
        limit_scenarios=args.limit_scenarios,
        selection_path=args.selection,
        upstream_selection_path=args.upstream_selection,
        confirmation_control_arms=_csv(args.confirmation_control_arms),
        include_generation=bool(args.include_generation),
        generation_seeds=_int_csv(args.generation_seeds),
        generation_branches=_csv(args.generation_branches),
    )
    write_frozen_paid_scan_spec(args.output, spec)
    print(canonical_json({
        "output": str(args.output),
        "scan_spec_sha256": sha256_file(args.output),
        "scan_spec_identity_sha256": spec["scan_spec_identity_sha256"],
        "cell_count": estimate.cell_count,
        "generation_count": estimate.generation_count,
        "total_model_frames": estimate.total_model_frames,
        "total_storage_reserved_bytes": estimate.total_storage_reserved_bytes,
    }))
    return 0


__all__ = [
    "BUILDER_POLICY",
    "BUILDER_POLICY_SHA256",
    "SPEC_ARTIFACT_TYPE",
    "SPEC_SCHEMA_VERSION",
    "build_paid_scan_spec",
    "verify_paid_scan_spec",
    "write_frozen_paid_scan_spec",
]
