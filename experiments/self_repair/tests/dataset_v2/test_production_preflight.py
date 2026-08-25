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


def authority(*, cap: float = 30.0, rate: float = 15.0) -> dict[str, object]:
    verified_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "2.0.0",
        "status": "approved",
        "provider": "azure_speech_s0",
        "purpose": "controlled_evaluation",
        "paid_tier_confirmed": True,
        "moshi_evaluation_terms_approved": True,
        "public_redistribution": False,
        "redistribution_terms_approved": False,
        "human_text_signoff": True,
        "budget_cap": cap,
        "budget_currency": "USD",
        "azure_rate_per_million_characters": rate,
        "pricing_verified_at": verified_at,
        "terms_reviewed_at": verified_at,
        "voice_inventory_verified_at": verified_at,
        "alignment_environment": "runpod_linux_mfa",
        "runpod_audio_upload_approved": True,
        "artifact_store_uri": "file:///approved/external-artifacts",
        "minimum_free_gib": 50,
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
            606900,
        )
        self.assertEqual(budget["initial_policy"]["request_count"], 1800)
        self.assertEqual(
            budget["initial_policy"]["azure_billable_characters"], 1820700
        )
        self.assertEqual(budget["hard_maximum"]["request_count"], 3000)
        self.assertEqual(
            budget["hard_maximum"]["azure_billable_characters"], 3034500
        )

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
            config["source_tracks"]["tts_controlled_r1"]["status"] = "release_approved"
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
                    "AZURE_SPEECH_KEY": "must-never-appear-in-report",
                    "AZURE_SPEECH_REGION": "fixture-region",
                },
                free_bytes=60 * GIB,
                azure_sdk_version="1.51.2",
                mfa_executable=None,
                tracked_tree_clean=True,
            )
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["failed_checks"], [])
            serialized = json.dumps(report, sort_keys=True)
            self.assertNotIn("must-never-appear-in-report", serialized)
            budget_check = next(
                row for row in report["checks"] if row["name"] == "initial_candidate_budget"
            )
            self.assertAlmostEqual(
                budget_check["evidence"]["initial_estimated_cost_before_tax"],
                27.3105,
            )

            too_low = authority(cap=20.0)
            write_json(authority_path, too_low)
            blocked = build_report(
                config_path=config_path,
                scripts_path=SCRIPTS,
                targets_path=TARGETS,
                authority_path=authority_path,
                artifact_root=root / "artifacts",
                environment={
                    "AZURE_SPEECH_KEY": "fixture",
                    "AZURE_SPEECH_REGION": "fixture",
                },
                free_bytes=60 * GIB,
                azure_sdk_version="1.51.2",
                mfa_executable=None,
                tracked_tree_clean=True,
            )
            self.assertEqual(blocked["status"], "blocked")
            self.assertIn("initial_candidate_budget", blocked["failed_checks"])

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
                azure_sdk_version=None,
                mfa_executable=None,
                tracked_tree_clean=False,
            )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            set(report["failed_checks"]),
            {
                "source_track_release_status",
                "production_authority",
                "initial_candidate_budget",
                "azure_credentials_present",
                "azure_sdk_installed",
                "independent_alignment_environment",
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
        changed["minimum_free_gib"] = 49
        self.assertIn("minimum_free_gib must be at least 50", validate_authority(changed))


if __name__ == "__main__":
    unittest.main()
