from __future__ import annotations

import pytest

from experiments.self_repair.mechanistic.causal_scan import (
    active_arms,
    exact_anchor_frame,
    intervention_sites,
    materialize_cell_grid,
    materialize_donor_assignments,
    parse_path_specification,
    repair_recipients,
)
from experiments.self_repair.mechanistic.core import ContractError


def _design() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in ("s1", "s2"):
        for direction, old, new in (
            ("boston_to_seattle", "Boston", "Seattle"),
            ("seattle_to_boston", "Seattle", "Boston"),
        ):
            base = {
                "scenario_id": scenario,
                "direction_id": direction,
                "speaker_id": "speaker-1",
                "old_value": old,
                "new_value": new,
                "frame_count": 8,
            }
            rows.append({
                **base,
                "trial_id": f"{scenario}-{direction}-clean",
                "condition": "clean_current",
            })
            rows.append({
                **base,
                "trial_id": f"{scenario}-{direction}-repair",
                "condition": "repair_delayed_640",
            })
    return rows


def _anchors(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        trial_id = str(row["trial_id"])
        result[(trial_id, "new_end")] = {
            "trial_id": trial_id, "anchor": "new_end", "frame": 3,
        }
        result[(trial_id, "query_end")] = {
            "trial_id": trial_id, "anchor": "query_end", "frame": 6,
        }
    return result


def test_all_donor_relations_are_exact_and_shuffled_is_a_derangement() -> None:
    rows = _design()
    recipients = repair_recipients(rows)
    arms = ("self", "clean_current", "clean_stale", "same_value_random", "shuffled")
    assignments = materialize_donor_assignments(rows, recipients, arms)
    assert len(assignments) == len(recipients) * len(arms)
    for recipient in recipients:
        trial_id = str(recipient["trial_id"])
        assert assignments[(trial_id, "self")].donor_trial_id == trial_id
        current = assignments[(trial_id, "clean_current")]
        assert current.scenario_matched and current.direction_matched
        assert current.speaker_matched and current.current_value_matched
        stale = assignments[(trial_id, "clean_stale")]
        assert stale.scenario_matched and stale.stale_value_matched
        same_value = assignments[(trial_id, "same_value_random")]
        assert not same_value.scenario_matched
        assert same_value.direction_matched and same_value.speaker_matched
        assert same_value.current_value_matched
        shuffled = assignments[(trial_id, "shuffled")]
        assert not shuffled.scenario_matched
        assert not shuffled.current_value_matched
        assert shuffled.donor_trial_id != current.donor_trial_id


def test_donor_mismatch_fails_instead_of_falling_back() -> None:
    rows = _design()
    recipient = next(row for row in rows if row["condition"] == "repair_delayed_640")
    # Removing the exact same speaker clean donor must not pick an arbitrary
    # same-scenario clean row.
    rows = [
        row for row in rows
        if not (
            row["condition"] == "clean_current"
            and row["scenario_id"] == recipient["scenario_id"]
            and row["direction_id"] == recipient["direction_id"]
        )
    ]
    with pytest.raises(ContractError, match="refusing fallback"):
        materialize_donor_assignments(rows, [recipient], ["clean_current"])


def test_kind_specific_arm_grid_matches_declared_arithmetic() -> None:
    rows = _design()
    recipients = repair_recipients(rows)
    controls = active_arms(
        "component", ["clean_current"],
        ["self", "current", "wrong", "same_value_random", "shuffled"],
    )
    assignments = materialize_donor_assignments(rows, recipients, controls)
    plans = materialize_cell_grid(
        rows,
        recipients,
        assignments,
        arms=controls,
        components=["attn_out", "mlp_out", "head_z"],
        layers=[2, 4],
        anchors=["new_end", "query_end"],
        anchor_rows=_anchors(rows),
        available_frames={str(row["trial_id"]): 8 for row in rows},
        head_count=3,
    )
    # attn + mlp + three individual heads = five component instances.
    assert len(plans) == len(recipients) * 5 * 2 * 2 * 5
    assert {plan.requested_arm for plan in plans} == set(controls)


def test_alias_duplicates_are_rejected_as_unbillable() -> None:
    with pytest.raises(ContractError, match="aliases are not billable"):
        active_arms("component", [], ["clean_current", "current"])
    with pytest.raises(ContractError, match="aliases are not billable"):
        active_arms("component", [], ["clean_stale", "wrong"])


def test_missing_or_out_of_range_semantic_anchor_fails_closed() -> None:
    with pytest.raises(ContractError, match="required semantic anchor"):
        exact_anchor_frame({}, "trial", "query_end", available_frames=4)
    with pytest.raises(ContractError, match="outside"):
        exact_anchor_frame(
            {("trial", "query_end"): {"frame": 4}},
            "trial", "query_end", available_frames=4,
        )


def test_kv_is_an_ordered_joint_pre_rope_intervention() -> None:
    assert intervention_sites("k_only") == ("k_pre_rope",)
    assert intervention_sites("v_only") == ("v_pre_rope",)
    assert intervention_sites("kv") == ("k_pre_rope", "v_pre_rope")


def test_path_requires_distinct_explicit_writer_and_mediator() -> None:
    specification = parse_path_specification({
        "path": {
            "writer": {"site": "resid_post", "layer": 8, "anchor": "new_end"},
            "mediator": {"site": "head_z", "layer": 14, "anchor": "query_end", "head": 3},
        }
    })
    assert specification.writer.layer == 8
    assert specification.mediator.head == 3
    with pytest.raises(ContractError, match="explicit writer and mediator"):
        parse_path_specification({"component": "path"})
    with pytest.raises(ContractError, match="strictly later"):
        parse_path_specification({
            "path": {
                "writer": {"site": "resid_post", "layer": 8, "anchor": "new_end"},
                "mediator": {"site": "head_z", "layer": 4, "anchor": "query_end"},
            }
        })
