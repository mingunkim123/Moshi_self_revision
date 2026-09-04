"""Resume-stable intervention blinding for full-duplex audio review."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Mapping, Sequence

from .core import ContractError, read_json, sha256_value, write_json


@dataclass(frozen=True)
class BlindAssignment:
    pair_id: str
    arm_to_label: dict[str, str]
    arm_to_audio_stem: dict[str, str]


class BlindAssignmentStore:
    """Keep the arm key in a private file while exposing opaque audio names."""

    def __init__(self, root: Path, *, run_identity_sha256: str):
        if len(run_identity_sha256) != 64:
            raise ContractError("blinding requires a SHA-256 run identity")
        self.root = root
        self.path = root / "private_blind_map.json"
        self.run_identity_sha256 = run_identity_sha256
        if self.path.exists():
            payload = read_json(self.path)
            if payload.get("run_identity_sha256") != run_identity_sha256:
                raise ContractError("private blind map belongs to another run identity")
            secret = payload.get("secret_hex")
            assignments = payload.get("assignments")
            if not isinstance(secret, str) or len(secret) != 64 or not isinstance(assignments, dict):
                raise ContractError("private blind map is malformed")
            self._secret = bytes.fromhex(secret)
            self._assignments = assignments
        else:
            self._secret = secrets.token_bytes(32)
            self._assignments: dict[str, dict[str, dict[str, str]]] = {}
            self._persist()

    def _persist(self) -> None:
        write_json(self.path, {
            "schema_version": "1.0.0",
            "run_identity_sha256": self.run_identity_sha256,
            "secret_hex": self._secret.hex(),
            "assignments": self._assignments,
        })

    def assign(self, *, cell_key: Mapping[str, object], arms: Sequence[str]) -> BlindAssignment:
        arm_names = tuple(str(arm) for arm in arms)
        if len(arm_names) < 2 or len(set(arm_names)) != len(arm_names):
            raise ContractError("blinding requires at least two unique arms")
        cell_sha = sha256_value(dict(cell_key))
        pair_id = hmac.new(self._secret, f"pair:{cell_sha}".encode(), hashlib.sha256).hexdigest()[:24]
        existing = self._assignments.get(cell_sha)
        if existing is not None:
            if set(existing.get("arm_to_label", {})) != set(arm_names):
                raise ContractError("resume arm set differs from private blind assignment")
            return BlindAssignment(pair_id, dict(existing["arm_to_label"]), dict(existing["arm_to_audio_stem"]))

        ranked = sorted(
            arm_names,
            key=lambda arm: hmac.new(
                self._secret, f"order:{cell_sha}:{arm}".encode(), hashlib.sha256
            ).digest(),
        )
        arm_to_label = {arm: f"arm_{index + 1:02d}" for index, arm in enumerate(ranked)}
        arm_to_audio_stem = {
            arm: hmac.new(
                self._secret, f"audio:{cell_sha}:{arm}".encode(), hashlib.sha256
            ).hexdigest()
            for arm in arm_names
        }
        self._assignments[cell_sha] = {
            "arm_to_label": arm_to_label,
            "arm_to_audio_stem": arm_to_audio_stem,
        }
        self._persist()
        return BlindAssignment(pair_id, arm_to_label, arm_to_audio_stem)
