from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_single_seed_pilot import (  # noqa: E402
    evidence_matches,
    first_lexical_event_ms,
    normalize_phrase,
    window_text,
)


class SingleSeedPilotTests(unittest.TestCase):
    def test_evidence_matching_uses_normalized_phrase_boundaries(self) -> None:
        text = "Let's use Link light-rail in Seattle, not a Seattleite guide."
        self.assertEqual(normalize_phrase("Link light-rail"), "link light rail")
        self.assertEqual(
            evidence_matches(text, ["Seattle", "Link light rail", "rail", "Seattleite"]),
            ["Link light rail", "rail", "Seattle", "Seattleite"],
        )
        self.assertEqual(evidence_matches("Seattleite", ["Seattle"]), [])

    def test_primary_window_is_inclusive_and_end_is_exclusive(self) -> None:
        events = [
            {"time_ms": 80.0, "piece": " before"},
            {"time_ms": 160.0, "piece": " At"},
            {"time_ms": 240.0, "piece": " end"},
        ]
        self.assertEqual(window_text(events, 160.0), "At end")
        self.assertEqual(window_text(events, 80.0, 240.0), "before At")
        self.assertEqual(first_lexical_event_ms(events, start_ms=160.0), 160.0)
        self.assertIsNone(first_lexical_event_ms(events, start_ms=240.0, end_ms=240.0))


if __name__ == "__main__":
    unittest.main()
