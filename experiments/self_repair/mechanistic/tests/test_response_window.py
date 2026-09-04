from __future__ import annotations

import pytest

from experiments.self_repair.mechanistic.response_window import (
    ResponseWindowError,
    primary_response_window,
)


def test_primary_window_uses_first_complete_post_query_episode() -> None:
    activity = [False] * 20
    activity[2] = True  # startup greeting; excluded by query boundary
    activity[7:10] = [True, True, True]
    activity[13] = True  # activity after the first 3-frame quiet run is not primary
    result = primary_response_window(activity, query_end_frame=6, quiet_frames=3)
    assert result.status == "complete"
    assert result.first_activity_frame == 7
    assert result.quiet_run_start_frame == 10
    assert result.clip_start_frame == 6
    assert result.clip_end_frame_exclusive == 10


def test_no_response_and_cap_active_are_unevaluable() -> None:
    no_response = primary_response_window([False] * 10, query_end_frame=4, quiet_frames=2)
    assert no_response.status == "unevaluable_no_response"
    assert no_response.clip_end_frame_exclusive == 10

    active_at_cap = [False] * 10
    active_at_cap[7:] = [True, True, True]
    truncated = primary_response_window(active_at_cap, query_end_frame=4, quiet_frames=2)
    assert truncated.status == "unevaluable_truncated"
    assert truncated.clip_end_frame_exclusive == 10


@pytest.mark.parametrize(
    "activity,query,quiet",
    [([], 0, 2), ([0, 1], 0, 1), ([False], -1, 1), ([False], 1, 1), ([False], 0, 0)],
)
def test_invalid_window_contract_fails_closed(activity, query, quiet) -> None:
    with pytest.raises(ResponseWindowError):
        primary_response_window(activity, query_end_frame=query, quiet_frames=quiet)
