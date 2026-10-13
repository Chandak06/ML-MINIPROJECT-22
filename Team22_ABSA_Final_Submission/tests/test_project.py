"""Behavioral, statistical integrity, serialization, CLI and UI regression tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TARGET_COLUMNS
from src.predict import ReviewPredictor
from src.train import build_models, choose_model, make_splits, train_and_evaluate
from src.utils import clean_text, load_dataset, rating_to_polarity, validate_dataset


def fixture():
    rows = []
    for i in range(90):
        rating = i % 5 + 1
        word = ["terrible", "poor", "average", "good", "excellent"][rating - 1]
        rows.append({"review_id": str(i), "review_text": f"The room and service were {word} unique{i}", **{t: rating for t in TARGET_COLUMNS}})
    return pd.DataFrame(rows)


class InputTests(unittest.TestCase):
    def test_negation_preserved(self):
        self.assertEqual(clean_text("It WASN'T clean; I can't sleep!"), "it was not clean i can not sleep")

    def test_polarity_boundaries(self):
        self.assertEqual([rating_to_polarity(i) for i in range(1, 6)], ["negative", "negative", "neutral", "positive", "positive"])

    def test_invalid_polarity_rejected(self):
        for value in [0, 6, 2.5, float("nan")]:
            with self.assertRaises(ValueError):
                rating_to_polarity(value)

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_dataset(fixture().drop(columns="service"))

    def test_bad_rating_rejected(self):
        for value in [0, 6, 2.5, "bad"]:
            data = fixture().astype({"service": object})
            data.loc[0, "service"] = value
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_missing_aspect_labels_kept(self):
        data = fixture()
        data["sleep_quality"] = np.nan
        self.assertTrue(validate_dataset(data).sleep_quality.isna().all())

    def test_missing_overall_rejected(self):
        data = fixture()
        data.loc[0, "overall_rating"] = np.nan
        with self.assertRaisesRegex(ValueError, "overall_rating"):
            validate_dataset(data)

    def test_blank_review_rejected(self):
        for value in [None, "", "!!!", " "]:
            data = fixture()
            data.loc[0, "review_text"] = value
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_optional_review_id(self):
        data = validate_dataset(fixture().drop(columns="review_id"))
        self.assertEqual(data.review_id.nunique(), 90)

    def test_duplicate_ids_rejected(self):
        data = fixture()
        data.loc[1, "review_id"] = data.loc[0, "review_id"]
        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_normalized_duplicates_removed(self):
        data = fixture()
        duplicate = data.iloc[[0]].copy()
        duplicate["review_id"] = "extra"
        duplicate["review_text"] = duplicate.review_text.str.upper() + "!!!"
        self.assertEqual(len(validate_dataset(pd.concat([data, duplicate]))), len(data))

    def test_conflicting_duplicates_rejected(self):
        data = fixture()
        data.loc[1, "review_text"] = data.loc[0, "review_text"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_dataset(data)

    def test_split_reproducible_and_disjoint(self):
        data = validate_dataset(fixture())
        first, second = make_splits(data), make_splits(data)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(len(set(np.concatenate(list(first.values())))), len(data))

    def test_hotel_split_has_no_overlap(self):
        data = fixture()
        data["hotel_id"] = np.arange(len(data)) // 3
        splits = make_splits(data)
        hotel_sets = [set(data.loc[ids, "hotel_id"]) for ids in splits.values()]
        for i, a in enumerate(hotel_sets):
            for b in hotel_sets[i+1:]:
                self.assertFalse(a & b)

    def test_tiny_dataset_rejected(self):
        with self.assertRaises(ValueError):
            make_splits(fixture().head(20))

    def test_features_do_not_learn_heldout_token(self):
        model = build_models()["LogReg_TFIDF"]
        model.fit(["good clean room", "bad dirty room", "great service", "awful service"], [5, 1, 5, 1])
        model.predict(["unseenvalidationtoken room"])
        self.assertNotIn("unseenvalidationtoken", model.named_steps["features"].vocabulary_)

    def test_selection_uses_validation_not_test(self):
        rows = [{"model": "a", "validation_polarity_macro_f1": .6, "validation_star_accuracy": .4, "test_polarity_accuracy": .1},
                {"model": "b", "validation_polarity_macro_f1": .5, "validation_star_accuracy": .9, "test_polarity_accuracy": 1.0}]
        self.assertEqual(choose_model(rows), "a")


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        data = fixture()
        data["sleep_quality"] = np.nan
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metrics = train_and_evaluate(data, cls.folder / "outputs", cls.folder / "models", dataset_name="test_fixture")
        cls.predictor = ReviewPredictor(cls.folder / "models")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_supported_models_reload(self):
        self.assertEqual(len(self.predictor.models), 7)
        self.assertEqual(len(self.metrics), 7)

    def test_missing_labels_produce_unavailable(self):
        result = self.predictor.predict("The noisy room interrupted sleep.")
        self.assertEqual(result["Sleep Quality"]["status"], "unavailable")
        self.assertIsNone(result["Sleep Quality"]["rating"])

    def test_unmentioned_aspects_abstain(self):
        result = self.predictor.predict("The room was excellent.")
        self.assertEqual(result["Service"]["status"], "not_mentioned")
        self.assertIsNone(result["Service"]["rating"])

    def test_known_polarity_direction(self):
        self.assertEqual(self.predictor.predict("The room and service were excellent.")["Rooms"]["polarity"], "positive")
        self.assertEqual(self.predictor.predict("The room and service were terrible.")["Rooms"]["polarity"], "negative")

    def test_empty_prediction_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("!!!")

    def test_unknown_vocabulary_abstains(self):
        self.assertEqual(self.predictor.predict("qzxwvjk")["Overall"]["status"], "unknown_vocabulary")

    def test_long_review_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("room " * 5000)

    def test_metrics_recomputed_from_saved_predictions(self):
        rows = pd.read_csv(self.folder / "outputs" / "test_predictions.csv")
        for _, metric in self.metrics.iterrows():
            target = rows[rows.target == metric.target]
            self.assertEqual(len(target), metric.test_n)
            self.assertAlmostEqual(accuracy_score(target.actual_rating, target.predicted_rating), metric.test_star_accuracy)
            self.assertAlmostEqual(accuracy_score(target.actual_polarity, target.predicted_polarity), metric.test_polarity_accuracy)

    def test_cli_prediction_does_not_train_or_need_data(self):
        manifest = self.folder / "models" / "manifest.json"
        before = manifest.stat().st_mtime_ns
        result = subprocess.run([sys.executable, str(ROOT / "run_project.py"), "--models", str(manifest.parent), "--data", "does-not-exist.csv", "--review", "The room was excellent"], capture_output=True, text=True, cwd=self.folder, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["Rooms"]["polarity"], "positive")
        self.assertEqual(manifest.stat().st_mtime_ns, before)

    def test_cli_invalid_input_has_nonzero_status(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_project.py"), "--models", str(self.folder / "models"), "--review", "!!!"], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Error:", result.stderr)

    def test_missing_manifest_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Run python"):
            ReviewPredictor(self.folder / "absent")

    def test_model_corruption_detected(self):
        path = self.folder / "models" / "rooms_model.joblib"
        original = path.read_bytes()
        try:
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "integrity"):
                ReviewPredictor(path.parent)
        finally:
            path.write_bytes(original)


class DeliveredArtifactTests(unittest.TestCase):
    def test_real_dataset_provenance_and_labels(self):
        data = ROOT / "data" / "hotel_reviews_real.csv"
        provenance = json.loads(data.with_suffix(".provenance.json").read_text())
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), provenance["csv_sha256"])
        frame = load_dataset(data)
        self.assertEqual(len(frame), 2100)
        self.assertTrue(frame.sleep_quality.isna().all())

    def test_delivered_metrics_splits_and_models(self):
        for mode in ["real", "synthetic"]:
            out = ROOT / "outputs" / mode
            manifest = pd.read_csv(out / "split_manifest.csv")
            self.assertFalse(manifest.text_hash.duplicated().any())
            if mode == "real":
                self.assertEqual(manifest.groupby("hotel_id").split.nunique().max(), 1)
            predictor = ReviewPredictor(ROOT / "models" / mode)
            predictions = pd.read_csv(out / "test_predictions.csv")
            test_ids = set(manifest.loc[manifest.split == "test", "review_id"].astype(str))
            self.assertTrue(set(predictions.review_id.astype(str)).issubset(test_ids))
            data = load_dataset(ROOT / "data" / ("hotel_reviews_real.csv" if mode == "real" else "hotel_reviews_sample.csv")).set_index("review_id")
            metrics = pd.read_csv(out / "metrics.csv")
            for _, metric in metrics.iterrows():
                rows = predictions[predictions.target == metric.target]
                # Routing needs raw punctuation. Preprocessing is now serialized
                # inside each pipeline; bypassing it would change the input API.
                pred = predictor.models[metric.target].predict(data.loc[rows.review_id.astype(str), "review_text"])
                np.testing.assert_array_equal(pred, rows.predicted_rating)
                self.assertAlmostEqual(accuracy_score(rows.actual_polarity, rows.predicted_polarity), metric.test_polarity_accuracy)
                self.assertAlmostEqual(f1_score(rows.actual_polarity, rows.predicted_polarity, labels=["negative", "neutral", "positive"], average="macro", zero_division=0), metric.test_polarity_macro_f1)

    def test_polarity_baseline_uses_development_majority(self):
        for mode in ["real", "synthetic"]:
            out = ROOT / "outputs" / mode
            split = pd.read_csv(out / "split_manifest.csv")
            data = load_dataset(ROOT / "data" / ("hotel_reviews_real.csv" if mode == "real" else "hotel_reviews_sample.csv")).set_index("review_id")
            dev = data.loc[split.loc[split.split != "test", "review_id"].astype(str)]
            test = data.loc[split.loc[split.split == "test", "review_id"].astype(str)]
            for _, metric in pd.read_csv(out / "metrics.csv").iterrows():
                polarity = dev[metric.target].dropna().map(rating_to_polarity)
                majority = polarity.value_counts().sort_index().idxmax()
                actual = test[metric.target].dropna().map(rating_to_polarity)
                self.assertAlmostEqual(float((actual == majority).mean()), metric.baseline_polarity_accuracy)

    def test_streamlit_demo_predicts_and_validates(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        # Preserve the user's HTML-table UI and check the actual rendered table.
        tables = [element.value for element in app.markdown if "<table" in element.value]
        self.assertGreaterEqual(len(tables), 1)
        for heading in ["Aspect", "Relevant evidence", "Rating", "Polarity", "Confidence", "Status"]:
            self.assertIn(heading, tables[0])
        self.assertIn("Sleep Quality", tables[0])
        app.text_area[0].input("!!!")
        app.button[0].click().run()
        self.assertEqual(len(app.error), 1)
        app.selectbox[0].select("Synthetic demonstration").run()
        app.text_area[0].input("The room was clean and the staff were helpful.")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
