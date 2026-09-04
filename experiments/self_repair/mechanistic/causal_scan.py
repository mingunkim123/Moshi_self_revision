"""Fail-closed planning helpers for mechanistic causal patch scans.

This module is deliberately model-free.  It resolves every donor and semantic
frame before a checkpoint can be constructed, which makes both paid-workload
accounting and ``--resume`` decisions auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Sequence

from .core import ContractError, canonical_json


_ARM_ALIASES = {
    "self": "self",
    "clean_current": "clean_current",
    "current": "clean_current",
    "clean_stale": "clean_stale",
    "wrong": "clean_stale",
    "same_value_random": "same_value_random",
    "shuffled": "shuffled",
}


@dataclass(frozen=True)
class TrialMetadata:
    trial_id: str
    scenario_id: str
    direction_id: str
    speaker_id: str
    old_value: str
    new_value: str
    condition: str

    @property
    def is_clean(self) -> bool:
        return self.condition.startswith("clean")

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.scenario_id,
            self.direction_id,
            self.speaker_id,
            self.old_value,
            self.new_value,
        )


@dataclass(frozen=True)
class DonorAssignment:
    requested_arm: str
    relation: str
    donor_trial_id: str
    recipient_trial_id: str
    selection_tier: str
    scenario_matched: bool
    direction_matched: bool
    speaker_matched: bool
    current_value_matched: bool
    stale_value_matched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PathEndpoint:
    site: str
    layer: int
    anchor: str
    head: int | None = None


@dataclass(frozen=True)
class PathSpecification:
    writer: PathEndpoint
    mediator: PathEndpoint

    @property
    def identity(self) -> dict[str, Any]:
        return {"writer": asdict(self.writer), "mediator": asdict(self.mediator)}


@dataclass(frozen=True)
class CausalCellPlan:
    recipient_trial_id: str
    donor_trial_id: str
    requested_arm: str
    relation: str
    component: str
    layer: int
    anchor: str
    source_frame: int
    target_frame: int
    query_end_frame_exclusive: int
    head: int | None


def _required_text(row: Mapping[str, Any], field: str, *, trial_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{trial_id}: donor matching requires non-empty {field}")
    return value.strip()


def trial_metadata(row: Mapping[str, Any]) -> TrialMetadata:
    trial_id = _required_text(row, "trial_id", trial_id="<unknown>")
    old_value = _required_text(row, "old_value", trial_id=trial_id)
    new_value = _required_text(row, "new_value", trial_id=trial_id)
    if old_value == new_value:
        raise ContractError(f"{trial_id}: old_value and new_value must differ")
    direction_value = row.get("direction_id")
    # Reviewed multivalue rows may encode direction only through the frozen
    # ordered value pair.  This is deterministic metadata, not a guessed value.
    direction = (
        direction_value.strip()
        if isinstance(direction_value, str) and direction_value.strip()
        else f"{old_value}\u2192{new_value}"
    )
    return TrialMetadata(
        trial_id=trial_id,
        scenario_id=_required_text(row, "scenario_id", trial_id=trial_id),
        direction_id=direction,
        speaker_id=_required_text(row, "speaker_id", trial_id=trial_id),
        old_value=old_value,
        new_value=new_value,
        condition=_required_text(row, "condition", trial_id=trial_id),
    )


def active_arms(kind: str, donors: Sequence[str], controls: Sequence[str]) -> tuple[str, ...]:
    """Return the exact kind-active arm dimension used by readiness arithmetic."""

    if kind not in {"residual", "component", "kv", "path"}:
        raise ContractError(f"unsupported patch kind: {kind}")
    values = controls if kind == "component" else donors
    if not values:
        raise ContractError(f"{kind} scan has no active donor/control arms")
    normalized: list[str] = []
    canonical_seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value not in _ARM_ALIASES:
            raise ContractError(f"unsupported donor/control arm: {value!r}")
        canonical = _ARM_ALIASES[value]
        if value in normalized:
            raise ContractError(f"duplicate donor/control arm: {value}")
        if canonical in canonical_seen:
            raise ContractError(
                f"duplicate semantic donor/control arm aliases are not billable: {value}"
            )
        normalized.append(value)
        canonical_seen.add(canonical)
    return tuple(normalized)


def repair_recipients(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    recipients = [dict(row) for row in rows if not trial_metadata(row).is_clean]
    if not recipients:
        raise ContractError("causal scan requires at least one non-clean recipient")
    return recipients


def _ranked_choice(
    candidates: Sequence[TrialMetadata], *, recipient: TrialMetadata, arm: str,
) -> TrialMetadata:
    if not candidates:
        raise ContractError(
            f"{recipient.trial_id}: no exact donor exists for arm {arm}; refusing fallback"
        )
    # A content-derived tie break is stable across manifest row ordering and
    # records no hidden RNG state.
    return min(
        candidates,
        key=lambda item: hashlib.sha256(
            canonical_json({
                "policy": "sha256_lexicographic_v1",
                "recipient": recipient.trial_id,
                "arm": arm,
                "donor": item.trial_id,
            }).encode("utf-8")
        ).hexdigest(),
    )


def _current_candidates(
    recipient: TrialMetadata, clean: Sequence[TrialMetadata], arm: str,
) -> list[TrialMetadata]:
    candidates = [
        donor for donor in clean
        if donor.scenario_id == recipient.scenario_id
        and donor.direction_id == recipient.direction_id
        and donor.speaker_id == recipient.speaker_id
        and donor.old_value == recipient.old_value
        and donor.new_value == recipient.new_value
    ]
    if not candidates:
        raise ContractError(
            f"{recipient.trial_id}: no exact same-scenario/direction/speaker/value "
            f"clean-current donor exists for arm {arm}; refusing fallback"
        )
    return candidates


def _shuffled_group_mapping(
    recipients: Sequence[TrialMetadata], clean: Sequence[TrialMetadata],
) -> dict[tuple[str, str, str, str, str], TrialMetadata]:
    groups = sorted({row.group_key for row in recipients})
    if len(groups) < 2:
        raise ContractError("shuffled donor arm requires at least two recipient metadata groups")
    base_by_group: dict[tuple[str, str, str, str, str], TrialMetadata] = {}
    for group in groups:
        representative = next(row for row in recipients if row.group_key == group)
        base_by_group[group] = _ranked_choice(
            _current_candidates(representative, clean, "shuffled"),
            recipient=representative,
            arm="shuffled_base",
        )

    # Match every recipient metadata group to a unique clean donor from a
    # different scenario and with a different current value.  That is the
    # preregistered shuffled control; if the design cannot support it, the scan
    # is unevaluable instead of silently weakening the constraint.
    sources = sorted(base_by_group.values(), key=lambda item: item.trial_id)
    if len({row.trial_id for row in sources}) != len(groups):
        raise ContractError("shuffled donor bases are not one-to-one across metadata groups")
    candidates_by_group: dict[tuple[str, str, str, str, str], list[TrialMetadata]] = {}
    for group in groups:
        recipient = next(row for row in recipients if row.group_key == group)
        own = base_by_group[group]
        candidates = [
            donor for donor in sources
            if donor.trial_id != own.trial_id
            and donor.scenario_id != recipient.scenario_id
            and donor.new_value != recipient.new_value
        ]
        candidates.sort(key=lambda donor: hashlib.sha256(
            canonical_json({
                "policy": "global_clean_derangement_v1",
                "recipient_group": group,
                "donor": donor.trial_id,
            }).encode("utf-8")
        ).hexdigest())
        if not candidates:
            raise ContractError(
                f"{recipient.trial_id}: shuffled derangement has no different-scenario/"
                "different-current-value donor"
            )
        candidates_by_group[group] = candidates

    assignment: dict[tuple[str, str, str, str, str], TrialMetadata] = {}
    used: set[str] = set()

    def visit(index: int) -> bool:
        if index == len(groups):
            return True
        group = groups[index]
        for donor in candidates_by_group[group]:
            if donor.trial_id in used:
                continue
            assignment[group] = donor
            used.add(donor.trial_id)
            if visit(index + 1):
                return True
            used.remove(donor.trial_id)
            del assignment[group]
        return False

    if not visit(0):
        raise ContractError("cannot construct the frozen one-to-one shuffled donor derangement")
    return assignment


def materialize_donor_assignments(
    rows: Sequence[Mapping[str, Any]],
    recipients: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
) -> dict[tuple[str, str], DonorAssignment]:
    """Resolve every recipient/arm pair with explicit, auditable constraints."""

    metadata = [trial_metadata(row) for row in rows]
    by_id = {row.trial_id: row for row in metadata}
    if len(by_id) != len(metadata):
        raise ContractError("donor manifest contains duplicate trial IDs")
    recipient_meta = [trial_metadata(row) for row in recipients]
    clean = [row for row in metadata if row.is_clean]
    if not clean:
        raise ContractError("donor matching requires clean control trials")
    canonical_arms = active_arms("residual", arms, ())
    shuffled = (
        _shuffled_group_mapping(recipient_meta, clean)
        if any(_ARM_ALIASES[arm] == "shuffled" for arm in canonical_arms)
        else {}
    )
    output: dict[tuple[str, str], DonorAssignment] = {}
    for recipient in recipient_meta:
        if recipient.trial_id not in by_id:
            raise ContractError(f"recipient is absent from donor manifest: {recipient.trial_id}")
        for requested_arm in canonical_arms:
            relation = _ARM_ALIASES[requested_arm]
            if relation == "self":
                donor = recipient
                tier = "exact_recipient_self"
            elif relation == "clean_current":
                donor = _ranked_choice(
                    _current_candidates(recipient, clean, requested_arm),
                    recipient=recipient,
                    arm=requested_arm,
                )
                tier = "same_scenario_direction_speaker_ordered_value_clean"
            elif relation == "clean_stale":
                reciprocal = [
                    candidate for candidate in clean
                    if candidate.scenario_id == recipient.scenario_id
                    and candidate.old_value == recipient.new_value
                    and candidate.new_value == recipient.old_value
                    and candidate.direction_id != recipient.direction_id
                ]
                speaker_matched = [
                    candidate for candidate in reciprocal
                    if candidate.speaker_id == recipient.speaker_id
                ]
                selected_pool = speaker_matched or reciprocal
                donor = _ranked_choice(
                    selected_pool, recipient=recipient, arm=requested_arm)
                tier = (
                    "same_scenario_reciprocal_value_same_speaker_clean"
                    if speaker_matched
                    else "same_scenario_reciprocal_value_explicit_speaker_mismatch_clean"
                )
            elif relation == "same_value_random":
                candidates = [
                    candidate for candidate in clean
                    if candidate.scenario_id != recipient.scenario_id
                    and candidate.direction_id == recipient.direction_id
                    and candidate.speaker_id == recipient.speaker_id
                    and candidate.old_value == recipient.old_value
                    and candidate.new_value == recipient.new_value
                ]
                donor = _ranked_choice(
                    candidates, recipient=recipient, arm=requested_arm)
                tier = "different_scenario_same_direction_speaker_ordered_value_clean"
            elif relation == "shuffled":
                donor = shuffled[recipient.group_key]
                tier = "global_one_to_one_different_scenario_and_current_value_clean_derangement"
            else:  # pragma: no cover - guarded by active_arms.
                raise AssertionError(relation)

            output[(recipient.trial_id, requested_arm)] = DonorAssignment(
                requested_arm=requested_arm,
                relation=relation,
                donor_trial_id=donor.trial_id,
                recipient_trial_id=recipient.trial_id,
                selection_tier=tier,
                scenario_matched=donor.scenario_id == recipient.scenario_id,
                direction_matched=donor.direction_id == recipient.direction_id,
                speaker_matched=donor.speaker_id == recipient.speaker_id,
                current_value_matched=donor.new_value == recipient.new_value,
                stale_value_matched=donor.new_value == recipient.old_value,
            )
    expected = len(recipient_meta) * len(canonical_arms)
    if len(output) != expected:
        raise ContractError(f"donor grid coverage mismatch: {len(output)} != {expected}")
    return output


def exact_anchor_frame(
    anchors: Mapping[tuple[str, str], Mapping[str, Any]],
    trial_id: str,
    anchor: str,
    *,
    available_frames: int | None = None,
) -> int:
    row = anchors.get((trial_id, anchor))
    if row is None:
        raise ContractError(f"{trial_id}: required semantic anchor {anchor!r} is missing")
    value = row.get("frame")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{trial_id}:{anchor}: semantic frame must be a non-negative integer")
    if available_frames is not None and value >= available_frames:
        raise ContractError(
            f"{trial_id}:{anchor}: frame {value} is outside [0, {available_frames})"
        )
    return int(value)


def materialize_cell_grid(
    rows: Sequence[Mapping[str, Any]],
    recipients: Sequence[Mapping[str, Any]],
    assignments: Mapping[tuple[str, str], DonorAssignment],
    *,
    arms: Sequence[str],
    components: Sequence[str],
    layers: Sequence[int],
    anchors: Sequence[str],
    anchor_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    available_frames: Mapping[str, int],
    head_count: int,
    component_heads: Mapping[str, int | None] | None = None,
) -> list[CausalCellPlan]:
    """Expand the exact executable grid and validate every semantic boundary."""

    if not components or not layers or not anchors:
        raise ContractError("causal scan components/layers/anchors cannot be empty")
    if head_count <= 0:
        raise ContractError("causal scan head_count must be positive")
    ids = {trial_metadata(row).trial_id for row in rows}
    result: list[CausalCellPlan] = []
    for recipient_row in recipients:
        recipient = trial_metadata(recipient_row)
        if recipient.trial_id not in ids:
            raise ContractError(f"unknown causal-scan recipient: {recipient.trial_id}")
        recipient_frames = available_frames.get(recipient.trial_id)
        if not isinstance(recipient_frames, int) or recipient_frames <= 0:
            raise ContractError(f"{recipient.trial_id}: available frame count is missing")
        query_frame = exact_anchor_frame(
            anchor_rows, recipient.trial_id, "query_end",
            available_frames=recipient_frames,
        )
        query_end = query_frame + 1
        for arm in arms:
            assignment = assignments.get((recipient.trial_id, arm))
            if assignment is None:
                raise ContractError(
                    f"{recipient.trial_id}: donor assignment for arm {arm} is missing"
                )
            donor_frames = available_frames.get(assignment.donor_trial_id)
            if not isinstance(donor_frames, int) or donor_frames <= 0:
                raise ContractError(
                    f"{assignment.donor_trial_id}: donor available frame count is missing"
                )
            for component in components:
                if component == "path":
                    raise ContractError(
                        "path grids must use explicit writer/mediator planning, not a scalar component"
                    )
                for layer in layers:
                    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                        raise ContractError("causal scan layers must be non-negative integers")
                    for anchor in anchors:
                        source = exact_anchor_frame(
                            anchor_rows, assignment.donor_trial_id, anchor,
                            available_frames=donor_frames,
                        )
                        target = exact_anchor_frame(
                            anchor_rows, recipient.trial_id, anchor,
                            available_frames=recipient_frames,
                        )
                        if target >= query_end:
                            raise ContractError(
                                f"{recipient.trial_id}:{anchor}: intervention is after query readout"
                            )
                        if component == "head_z":
                            heads = range(head_count)
                        else:
                            selected_head = (
                                component_heads.get(component)
                                if component_heads is not None else None
                            )
                            if selected_head is not None and (
                                isinstance(selected_head, bool)
                                or not isinstance(selected_head, int)
                                or selected_head < 0
                            ):
                                raise ContractError(
                                    f"{component} selected head must be a non-negative integer"
                                )
                            heads = (selected_head,)
                        for head in heads:
                            result.append(CausalCellPlan(
                                recipient_trial_id=recipient.trial_id,
                                donor_trial_id=assignment.donor_trial_id,
                                requested_arm=arm,
                                relation=assignment.relation,
                                component=component,
                                layer=layer,
                                anchor=anchor,
                                source_frame=source,
                                target_frame=target,
                                query_end_frame_exclusive=query_end,
                                head=head,
                            ))
    expected_width = sum(head_count if component == "head_z" else 1 for component in components)
    expected = len(recipients) * len(arms) * len(layers) * len(anchors) * expected_width
    if len(result) != expected:
        raise ContractError(f"causal cell grid coverage mismatch: {len(result)} != {expected}")
    return result


def intervention_sites(component: str) -> tuple[str, ...]:
    if component == "k_only":
        return ("k_pre_rope",)
    if component == "v_only":
        return ("v_pre_rope",)
    if component == "kv":
        # The order is causal and frozen: both tensors are taken before RoPE;
        # receiver-position RoPE is then applied by the model.
        return ("k_pre_rope", "v_pre_rope")
    if component == "path":
        raise ContractError("path interventions require an explicit writer and mediator")
    return (component,)


def _path_endpoint(value: Any, label: str) -> PathEndpoint:
    if not isinstance(value, Mapping):
        raise ContractError(f"path selection requires a {label} object")
    unknown = set(value) - {"site", "layer", "anchor", "head"}
    missing = {"site", "layer", "anchor"} - set(value)
    if unknown or missing:
        raise ContractError(
            f"path {label} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    site = value.get("site")
    anchor = value.get("anchor")
    layer = value.get("layer")
    head = value.get("head")
    if not isinstance(site, str) or not site or not isinstance(anchor, str) or not anchor:
        raise ContractError(f"path {label} site/anchor must be non-empty strings")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ContractError(f"path {label} layer must be a non-negative integer")
    if head is not None and (isinstance(head, bool) or not isinstance(head, int) or head < 0):
        raise ContractError(f"path {label} head must be null or a non-negative integer")
    return PathEndpoint(site=site, layer=int(layer), anchor=anchor, head=head)


def parse_path_specification(selection: Mapping[str, Any]) -> PathSpecification:
    value = selection.get("path")
    if not isinstance(value, Mapping):
        raise ContractError(
            "path patching requires selection.path with explicit writer and mediator objects"
        )
    if set(value) != {"writer", "mediator"}:
        raise ContractError("selection.path must contain exactly writer and mediator")
    writer = _path_endpoint(value["writer"], "writer")
    mediator = _path_endpoint(value["mediator"], "mediator")
    if writer == mediator:
        raise ContractError("path writer and mediator must be distinct intervention sites")
    if mediator.layer <= writer.layer:
        raise ContractError(
            "path mediator must be in a strictly later layer than its writer"
        )
    return PathSpecification(writer=writer, mediator=mediator)
