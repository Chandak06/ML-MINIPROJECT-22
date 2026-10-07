"""Regression checks for the final audit fixes and false-positive readiness results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import make_splits
from src.utils import validate_dataset
from test_project import fixture
from verify_project import check_cli, check_presentation, check_report, check_structure, ready


class HotelIdTests(unittest.TestCase):
    def test_partial_hotel_ids_rejected_before_training(self):
        data = fixture()
        data["hotel_id"] = "hotel-a"
        data.loc[0, "hotel_id"] = None
        with self.assertRaisesRegex(ValueError, "hotel_id"):
            validate_dataset(data)

    def test_split_does_not_fall_back_for_partial_hotel_ids(self):
        data = fixture()
        data["hotel_id"] = "hotel-a"
        data.loc[0, "hotel_id"] = None
        with self.assertRaisesRegex(ValueError, "hotel_id"):
            make_splits(data)

    def test_blank_hotel_ids_rejected(self):
        data = fixture()
        data["hotel_id"] = "  "
        with self.assertRaisesRegex(ValueError, "hotel_id"):
            validate_dataset(data)


class ReadinessTests(unittest.TestCase):
    def test_documented_demo_predictions_are_reproducible(self):
        from src.predict import ReviewPredictor
        predictor = ReviewPredictor(ROOT / "models/real")
        evidence = json.loads((ROOT / "outputs/demo_predictions.json").read_text(encoding="utf8"))
        self.assertEqual(len(evidence["examples"]), 5)
        for example in evidence["examples"]:
            self.assertEqual(predictor.predict(example["review"]), example["predictions"])

    def test_missing_project_files_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Missing"):
                check_structure(Path(folder))

    def test_corrupt_report_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "docs/ABSA_Hotel_Reviews_Final_Writeup.pdf").write_bytes(b"not a PDF")
            with self.assertRaisesRegex(ValueError, "PDF"):
                check_report(root)

    def test_duplicate_reports_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            for name in ["ABSA_Hotel_Reviews_Final_Writeup.pdf", "old.pdf"]:
                (root / "docs" / name).write_bytes(b"%PDF-1.4\n%%EOF")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                check_report(root)

    def test_corrupt_presentation_fails(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "docs/ABSA_Hotel_Reviews_Final_Presentation.pptx").write_bytes(b"corrupt")
            with self.assertRaises(zipfile.BadZipFile):
                check_presentation(root)

    def test_any_failed_check_prevents_ready(self):
        self.assertFalse(ready([{"passed": True}, {"passed": False}]))
        self.assertFalse(ready([]))
        self.assertTrue(ready([{"passed": True}, {"passed": True}]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
