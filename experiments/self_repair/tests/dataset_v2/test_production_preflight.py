from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_config, read_jsonl, write_json  # noqa: E402
from production_preflight import (  # noqa: E402
    GIB,
    azure_billable_character_count,
    build_report,
    request_budget,
    validate_authority,
)


DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset_v2"
CONFIG = DATASET_ROOT / "config/dataset.yaml"
SCRIPTS = DATASET_ROOT / "generated/scripts.jsonl"
TARGETS = DATASET_ROOT / "assignments/rendition_targets.jsonl"


def authority() -> dict[str, object]:
    verified_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "2.0.0",
        "status": "approved",
        "provider": "kokoro_local_v1_0",
        "purpose": "controlled_evaluation",
        "model_license_reviewed": True,
        "model_attribution_approved": True,
        "training_provenance_reviewed": True,
        "model_repo": "hexgrad/Kokoro-82M",
        "model_revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
        "model_sha256": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
        "public_redistribution": False,
        "redistribution_terms_approved": False,
        "human_text_signoff": True,
        "human_voice_double_listen": True,
        "local_generation_approved": True,
        "license_reviewed_at": verified_at,
        "voice_inventory_verified_at": verified_at,
        "alignment_environment": "runpod_linux_mfa",
        "storage_execution_mode": "local_audio_then_runpod_evaluation",
        "runpod_audio_upload_approved": True,
        "artifact_store_uri": "file:///approved/external-artifacts",
        "local_minimum_free_gib": 12,
        "remote_minimum_free_gib": 40,
        "approver_id": "fixture-approver",
    }


class ProductionPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_config(CONFIG)
        cls.scripts = read_jsonl(SCRIPTS)
        cls.targets = read_jsonl(TARGETS)

    def test_frozen_matrix_has_exact_request_character_budget(self) -> None:
        budget = request_budget(self.scripts, self.targets, self.config)
        self.assertEqual(budget["script_count"], 300)
        self.assertEqual(budget["rendition_target_count"], 600)
        self.assertEqual(budget["speaker_count"], 10)
        self.assertEqual(
            budget["one_candidate_per_target"]["azure_billable_characters"],
            0,
        )
        self.assertEqual(budget["initial_policy"]["request_count"], 600)
        self.assertEqual(budget["initial_policy"]["attempts_per_target"], 1)
        self.assertEqual(budget["hard_maximum"]["request_count"], 600)
        self.assertEqual(budget["provider"], "kokoro_local_v1_0")

    def test_azure_character_rule_excludes_only_outer_speak_and_voice_tags(self) -> None:
        ssml = (
            '<speak version="1.0"><voice name="fixture">'
            '<prosody rate="0%">Hi <bookmark mark="s0"/></prosody>'
            "</voice></speak>"
        )
        expected = '<prosody rate="0%">Hi <bookmark mark="s0"/></prosody>'
        self.assertEqual(azure_billable_character_count(ssml), len(expected))

    def test_matrix_rejects_voice_inventory_tamper(self) -> None:
        targets = deepcopy(self.targets)
        targets[0]["voice"] = "en-US-NotFrozenNeural"
        with self.assertRaisesRegex(ValueError, "speaker/voice is not in the frozen"):
            request_budget(self.scripts, targets, self.config)

    def test_ready_report_requires_authority_budget_environment_and_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(self.config)
            track_id = next(iter(config["source_tracks"]))
            config["source_tracks"][track_id]["status"] = "release_approved"
            config_path = root / "dataset.json"
            authority_path = root / "authority.json"
            write_json(config_path, config)
            write_json(authority_path, authority())
            report = build_report(
                config_path=config_path,
                scripts_path=SCRIPTS,
                targets_path=TARGETS,
                authority_path=authority_path,
                artifact_root=root / "artifacts",
                environment={
                    "RUNPOD_API_KEY": "must-also-never-appear-in-report",
                },
                free_bytes=60 * GIB,
                kokoro_version="0.9.4",
                kokoro_artifacts_verified=True,
                mfa_executable=None,
                tracked_tree_clean=True,
            )
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["failed_checks"], [])
            serialized = json.dumps(report, sort_keys=True)
            self.assertNotIn("must-also-never-appear-in-report", serialized)
            generation_check = next(
                row for row in report["checks"] if row["name"] == "local_generation_authorized"
            )
            self.assertEqual(generation_check["evidence"]["provider_api_charge"], 0)

            too_low = authority()
            too_low["local_minimum_free_gib"] = 11
            write_json(authority_path, too_low)
            blocked = build_report(
                config_path=config_path,
                scripts_path=SCRIPTS,
                targets_path=TARGETS,
                authority_path=authority_path,
                artifact_root=root / "artifacts",
                environment={
                    "RUNPOD_API_KEY": "fixture",
                },
                free_bytes=60 * GIB,
                kokoro_version="0.9.4",
                kokoro_artifacts_verified=True,
                mfa_executable=None,
                tracked_tree_clean=True,
            )
            self.assertEqual(blocked["status"], "blocked")
            self.assertIn("production_authority", blocked["failed_checks"])

    def test_current_unapproved_state_reports_every_external_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_report(
                config_path=CONFIG,
                scripts_path=SCRIPTS,
                targets_path=TARGETS,
                authority_path=root / "missing-authority.json",
                artifact_root=root / "artifacts",
                environment={},
                free_bytes=10 * GIB,
                kokoro_version=None,
                kokoro_artifacts_verified=False,
                mfa_executable=None,
                tracked_tree_clean=False,
            )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            set(report["failed_checks"]),
            {
                "source_track_release_status",
                "production_authority",
                "local_generation_authorized",
                "kokoro_runtime_installed",
                "kokoro_model_and_voice_hashes",
                "independent_alignment_environment",
                "runpod_access_present",
                "artifact_storage_capacity",
                "tracked_git_tree_clean",
            },
        )
        self.assertFalse(report["provider_calls_made"])
        self.assertFalse(report["credential_values_recorded"])

    def test_authority_contract_is_exact(self) -> None:
        self.assertEqual(validate_authority(authority()), [])
        changed = authority()
        changed["public_redistribution"] = True
        self.assertIn(
            "redistribution_terms_approved must exactly match public_redistribution",
            validate_authority(changed),
        )
        changed = authority()
        changed["local_minimum_free_gib"] = 11
        self.assertIn(
            "local_minimum_free_gib must be at least 12", validate_authority(changed)
        )


if __name__ == "__main__":
    unittest.main()
