"""Deterministic primary-response window extraction for full-duplex review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


class ResponseWindowError(ValueError):
    pass


@dataclass(frozen=True)
class PrimaryResponseWindow:
    query_end_frame: int
    capture_end_frame: int
    first_activity_frame: int | None
    response_end_frame_exclusive: int | None
    quiet_run_start_frame: int | None
    required_quiet_frames: int
    status: str

    @property
    def clip_start_frame(self) -> int:
        return self.query_end_frame

    @property
    def clip_end_frame_exclusive(self) -> int:
        return self.response_end_frame_exclusive or self.capture_end_frame

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def primary_response_window(
    activity: Sequence[bool],
    *,
    query_end_frame: int,
    quiet_frames: int,
) -> PrimaryResponseWindow:
    """Find the first response episode followed by a frozen quiet run.

    ``activity`` should conservatively combine rendered text activity and the
    calibrated decoded-audio activity mask. Boundaries are half-open. If no
    complete quiet run occurs before the cap, the result is unevaluable rather
    than treated as a complete answer.
    """
    values = tuple(activity)
    if not values or any(not isinstance(value, bool) for value in values):
        raise ResponseWindowError("activity must be a non-empty boolean timeline")
    if isinstance(query_end_frame, bool) or not isinstance(query_end_frame, int):
        raise ResponseWindowError("query_end_frame must be an integer")
    if query_end_frame < 0 or query_end_frame >= len(values):
        raise ResponseWindowError("query_end_frame is outside the captured timeline")
    if isinstance(quiet_frames, bool) or not isinstance(quiet_frames, int) or quiet_frames <= 0:
        raise ResponseWindowError("quiet_frames must be a positive integer")

    first = next((index for index in range(query_end_frame, len(values)) if values[index]), None)
    if first is None:
        return PrimaryResponseWindow(
            query_end_frame, len(values), None, None, None, quiet_frames,
            "unevaluable_no_response",
        )

    quiet_run = 0
    for index in range(first + 1, len(values)):
        quiet_run = 0 if values[index] else quiet_run + 1
        if quiet_run == quiet_frames:
            quiet_start = index - quiet_frames + 1
            return PrimaryResponseWindow(
                query_end_frame, len(values), first, quiet_start, quiet_start,
                quiet_frames, "complete",
            )
    return PrimaryResponseWindow(
        query_end_frame, len(values), first, None, None, quiet_frames,
        "unevaluable_truncated",
    )
