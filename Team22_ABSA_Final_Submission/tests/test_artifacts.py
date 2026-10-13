"""Checks of delivered schema-v2 models, calibration and evidence integrity."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import ReviewPredictor
from src.verification import check_routing_schema, check_saved_results
from src.train import build_models, make_splits
from src.utils import validate_dataset
from src.confidence import select_uncertainty_threshold, attach_confidence, probability_estimates
from test_project import fixture


class UpdatedArtifactTests(unittest.TestCase):
    def test_real_and_synthetic_schema_mentions_and_availability(self):
        self.assertEqual(check_routing_schema(ROOT)["schema"], "passed")

    def test_saved_metrics_evidence_selection_and_calibration(self):
        result = check_saved_results(ROOT)
        self.assertEqual(result["real"]["targets"], 7)
        self.assertEqual(result["synthetic"]["targets"], 8)

    def test_routing_manifest_mismatch_is_rejected(self):
        path = ROOT / 'models/real'
        original = json.loads((path / 'manifest.json').read_text())
        modified = copy.deepcopy(original)
        modified['targets']['rooms']['routing_strategy'] = 'invalid_strategy'
        with patch('src.predict.json.loads', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'routing mismatch'):
                ReviewPredictor(path)

    def test_confidence_manifest_mismatch_is_rejected(self):
        path = ROOT / 'models/real'
        modified = json.loads((path / 'manifest.json').read_text())
        modified['targets']['overall_rating']['confidence_method'] = 'invented'
        with patch('src.predict.json.loads', return_value=modified):
            with self.assertRaisesRegex(ValueError, 'confidence mismatch'):
                ReviewPredictor(path)

    def test_synthetic_sleep_quality_has_real_probability_output(self):
        result = ReviewPredictor(ROOT / 'models/synthetic').predict('The bed was comfortable and I slept quietly.')['Sleep Quality']
        self.assertIn(result['status'], ['predicted', 'uncertain'])
        self.assertIsNotNone(result['confidence'])
        self.assertTrue(0 <= result['confidence'] <= 1)

    def test_difficult_examples_are_reproducible_not_hardcoded(self):
        saved = json.loads((ROOT / 'outputs/real/difficult_review_comparison.json').read_text())
        self.assertEqual(len(saved['examples']), 9)
        predictor = ReviewPredictor(ROOT / 'models/real')
        for row in saved['examples']:
            self.assertEqual(predictor.predict(row['review']), row['new_predictions'])

    def test_routed_vocabulary_excludes_other_aspect_and_validation_words(self):
        model = build_models(target='rooms', strategy='aspect_text')['LogReg_TFIDF']
        model.fit(['Room was clean but staff were uniquepolite.', 'Room was dirty but staff were uniquefriendly.'], [5, 1])
        model.predict(['The room was validationonly but staff were nice.'])
        vocab = model.named_steps['features'].vocabulary_
        for word in ['uniquepolite', 'uniquefriendly', 'validationonly']:
            self.assertNotIn(word, vocab)


if __name__ == '__main__':
    unittest.main(verbosity=2)
