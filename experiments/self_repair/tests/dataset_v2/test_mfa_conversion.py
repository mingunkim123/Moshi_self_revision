from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts/dataset_v2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_mfa_dictionary import build_dictionary  # noqa: E402
from convert_mfa_textgrids import parse_word_tier, resegment_words_to_transcript  # noqa: E402


class MfaConversionTests(unittest.TestCase):
    def test_build_dictionary_is_exact_and_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.dict"
            extension = root / "extension.dict"
            output = root / "combined.dict"
            base.write_text("hello\tHH AH0 L OW1\n", encoding="utf-8")
            extension.write_text("world\tW ER1 L D\n", encoding="utf-8")
            report = build_dictionary(base, extension, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "hello\tHH AH0 L OW1\nworld\tW ER1 L D\n")
            self.assertEqual(report["extension_word_count"], 1)
            extension.write_text("hello\tHH EH1 L OW0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already present"):
                build_dictionary(base, extension, output)

    def test_parse_word_tier_ignores_silence_and_preserves_intervals(self) -> None:
        payload = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.3
            text = "hello"
        intervals [3]:
            xmin = 0.3
            xmax = 1
            text = "world"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "HH"
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.TextGrid"
            path.write_text(payload, encoding="utf-8")
            words = parse_word_tier(path)
        self.assertEqual([row["text"] for row in words], ["hello", "world"])
        self.assertEqual(words[0]["offset_ms"], 100.0)
        self.assertEqual(words[1]["duration_ms"], 700.0)

    def test_resegment_words_restores_frozen_hyphenated_token(self) -> None:
        script = {
            "script_id": "fixture",
            "transcript": "a first-time trip",
            "segments": [
                {
                    "segment_index": 0,
                    "role": "closing_prompt",
                    "text": "a first-time trip",
                }
            ],
        }
        words = [
            {"type": "word", "text": "a", "offset_ms": 100.0, "duration_ms": 50.0},
            {"type": "word", "text": "first", "offset_ms": 160.0, "duration_ms": 100.0},
            {"type": "word", "text": "time", "offset_ms": 270.0, "duration_ms": 90.0},
            {"type": "word", "text": "trip", "offset_ms": 370.0, "duration_ms": 100.0},
        ]
        resegmented, changed = resegment_words_to_transcript(words, script)
        self.assertTrue(changed)
        self.assertEqual([row["text"] for row in resegmented], ["a", "first-time", "trip"])
        self.assertEqual(resegmented[1]["offset_ms"], 160.0)
        self.assertEqual(resegmented[1]["duration_ms"], 200.0)


if __name__ == "__main__":
    unittest.main()
