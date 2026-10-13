"""Routing expectations are separate from learned-model prediction expectations."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aspect_router import split_sentences, split_clauses, route_review, evidence_for, AspectTextTransformer
from src.confidence import calibration_splits, attach_confidence, probability_estimates, select_uncertainty_threshold
from src.config import TARGET_COLUMNS
from src.predict import ReviewPredictor
from src.train import build_models, choose_configuration, scores
from src.utils import clean_text
from src.evaluation import hotel_bootstrap


class RouterTests(unittest.TestCase):
    def test_sentence_and_semicolon_boundaries(self):
        self.assertEqual(len(split_sentences("Room clean. Staff helpful! Wi-Fi worked; good value.")), 4)

    def test_independent_and_contrast_clauses(self):
        text = "The location was excellent and the staff were helpful, but the room was dirty."
        self.assertEqual(len(split_clauses(text)), 3)
        route = route_review(text)
        self.assertIn("excellent", route["location"]["aspect_text"])
        self.assertNotIn("helpful", route["location"]["aspect_text"])
        for target in ["rooms", "cleanliness"]:
            self.assertIn("dirty", route[target]["aspect_text"])
            self.assertNotIn("excellent", route[target]["aspect_text"])
            self.assertNotIn("helpful", route[target]["aspect_text"])

    def test_central_multi_clause_example(self):
        text = "The location was excellent, the staff were helpful and the hotel was great value for money, but the room was extremely dirty."
        route = route_review(text)
        self.assertIn("value for money", route["value"]["aspect_text"])
        self.assertNotIn("dirty", route["value"]["aspect_text"])
        self.assertEqual(route["rooms"]["aspect_text"], "the room was extremely dirty.")
        self.assertEqual(route["service"]["aspect_text"], "the staff were helpful")

    def test_shared_noun_list_retained(self):
        text = "The room was filthy although the staff, location and food were excellent."
        route = route_review(text)
        self.assertIn("excellent", route["service"]["aspect_text"])
        self.assertIn("excellent", route["location"]["aspect_text"])
        self.assertNotIn("excellent", route["rooms"]["aspect_text"])
        self.assertIn("filthy", route["cleanliness"]["aspect_text"])

    def test_spotless_room_and_rude_reception(self):
        route = route_review("The room was spotless, but the receptionist was extremely rude.")
        self.assertNotIn("rude", route["cleanliness"]["aspect_text"])
        self.assertNotIn("spotless", route["service"]["aspect_text"])

    def test_expensive_but_near_station(self):
        route = route_review("The hotel was expensive but perfectly located near the station.")
        self.assertNotIn("station", route["value"]["aspect_text"])
        self.assertNotIn("expensive", route["location"]["aspect_text"])

    def test_generic_positive_clause_does_not_attach_to_location(self):
        route = route_review("The location was terrible, but everything else about the hotel was fantastic.")
        self.assertIn("terrible", route["location"]["aspect_text"])
        self.assertNotIn("fantastic", route["location"]["aspect_text"])

    def test_staff_only_mentions_service(self):
        route = route_review("The staff were friendly.")
        self.assertEqual({k for k, v in route.items() if v["mentioned"]}, {"overall_rating", "service"})

    def test_room_clean_mentions_both_aspects(self):
        route = route_review("The room was clean.")
        self.assertEqual({k for k, v in route.items() if v["mentioned"]}, {"overall_rating", "rooms", "cleanliness"})

    def test_negation_variants_preserved(self):
        for text, token in [("The room was not clean.", "not clean"), ("The staff weren't helpful.", "not helpful"),
                            ("Not worth the price.", "not worth"), ("It wasn't comfortable.", "not comfortable"),
                            ("I wouldn't recommend it.", "not recommend"), ("The room was not bad.", "not bad"),
                            ("The room was far from clean.", "far from clean"), ("No towels, never cleaned.", "never cleaned")]:
            with self.subTest(text=text):
                self.assertIn(token, clean_text(" ".join(split_clauses(text))))
        self.assertFalse(route_review("The room was far from clean.")["location"]["mentioned"])

    def test_reasonable_variants_and_front_desk(self):
        route = route_review("The Wi-Fi failed. The check-in staff were polite. The bedroom was dusty.")
        self.assertTrue(route["business_service"]["mentioned"])
        self.assertTrue(route["service"]["mentioned"])
        self.assertTrue(route["rooms"]["mentioned"])
        self.assertTrue(route["cleanliness"]["mentioned"])
        self.assertFalse(route_review("The front desk staff were friendly.")["business_service"]["mentioned"])

    def test_room_service_is_not_room_condition(self):
        self.assertFalse(route_review("Room service was prompt.")["rooms"]["mentioned"])

    def test_contracted_negations_in_coordinated_clauses(self):
        route = route_review("The room wasn't clean and the staff weren't helpful.")
        self.assertNotIn("helpful", route["rooms"]["aspect_text"])
        self.assertNotIn("clean", route["service"]["aspect_text"])

    def test_explicit_pronoun_continuation(self):
        route = route_review("The room was spacious. It was dirty.")
        self.assertIn("It was dirty", route["rooms"]["aspect_text"])

    def test_ambiguous_desk_fallback_is_transparent(self):
        info = evidence_for("There was a desk by the window.", "business_service")
        self.assertTrue(info["mentioned"])
        self.assertEqual(info["evidence_source"], "whole_review_fallback")
        self.assertEqual(info["fallback_reason"], "ambiguous_desk_reference")

    def test_absent_evidence_fallback_for_labelled_training_rows(self):
        info = evidence_for("A pleasant stay.", "rooms")
        self.assertFalse(info["mentioned"])
        self.assertEqual(info["evidence"], "A pleasant stay.")
        self.assertEqual(info["evidence_source"], "whole_review_fallback")

    def test_full_review_strategy_disclosed(self):
        text = "The room was dirty but staff were helpful."
        info = evidence_for(text, "rooms", "full_review")
        self.assertEqual(info["evidence"], text)
        self.assertEqual(info["evidence_source"], "whole_review_selected")
        self.assertNotIn("helpful", info["routed_aspect_text"])

    def test_transformer_preserves_negation_and_is_stateless(self):
        transformer = AspectTextTransformer("cleanliness", "aspect_text")
        text = "The room was not clean, but the staff were helpful."
        before = transformer.transform([text])
        transformer.fit(["a completely different training example"], [5])
        self.assertEqual(before, transformer.transform([text]))
        self.assertIn("not clean", before[0])
        self.assertNotIn("helpful", before[0])


class ConfidenceTests(unittest.TestCase):
    def test_threshold_derived_from_validation_correctness(self):
        result = select_uncertainty_threshold([.9, .8, .2, .1], [True, True, False, False])
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["selection_score"], 1.)
        self.assertEqual(select_uncertainty_threshold([.9, .8], [True, True])["threshold"], 0.)

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):
            select_uncertainty_threshold([float('nan')], [True])

    def test_calibration_folds_keep_hotels_disjoint(self):
        labels = np.tile([1, 2, 3, 4, 5], 12)
        hotels = np.repeat(np.arange(12), 5)
        splits = calibration_splits(labels, hotels)
        self.assertEqual(len(splits), 3)
        heldout = []
        for train, valid in splits:
            self.assertFalse(set(hotels[train]) & set(hotels[valid]))
            heldout.extend(valid)
        self.assertEqual(sorted(heldout), list(range(60)))

    def test_svm_calibration_gives_probabilities_without_changing_ratings(self):
        texts = [f"{'excellent clean' if i % 2 else 'terrible dirty'} room item{i}" for i in range(30)]
        labels = [5 if i % 2 else 1 for i in range(30)]
        model = build_models()["LinearSVC_Count_2gram"].fit(texts, labels)
        before = model.predict(texts)
        detail = attach_confidence(model, texts, labels, np.repeat(np.arange(10), 3))
        self.assertEqual(detail["method"], "sigmoid_hotel_group_oof")
        np.testing.assert_array_equal(before, model.predict(texts))
        confidence, rating_conf, probabilities = probability_estimates(model, texts)
        self.assertTrue(all(0 <= c <= 1 for c in confidence + rating_conf))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)

    def test_strategy_selection_ignores_test_scores(self):
        rows = [{"model": "a", "routing_strategy": "full_review", "validation_polarity_macro_f1": .6, "validation_star_accuracy": .4, "test_score": 0},
                {"model": "b", "routing_strategy": "aspect_text", "validation_polarity_macro_f1": .5, "validation_star_accuracy": .9, "test_score": 1}]
        self.assertEqual(choose_configuration(rows)["routing_strategy"], "full_review")

    def test_ordinal_metrics_do_not_count_near_miss_as_exact(self):
        result = scores([5, 5], [4, 1])
        self.assertEqual(result["star_accuracy"], 0.)
        self.assertEqual(result["within_one_star_accuracy"], .5)
        self.assertEqual(result["star_mae"], 2.5)

    def test_hotel_bootstrap_is_reproducible_and_paired(self):
        args = ([1, 5, 2, 4], [1, 4, 2, 5], ['a', 'a', 'b', 'b'])
        first = hotel_bootstrap(*args, old_pred=args[1], repetitions=30)
        self.assertEqual(first, hotel_bootstrap(*args, old_pred=args[1], repetitions=30))
        self.assertTrue(all(r["delta"] == r["delta_ci_low"] == r["delta_ci_high"] == 0 for r in first))


class PredictionSchemaTests(unittest.TestCase):
    def setUp(self):
        self.predictor = ReviewPredictor.__new__(ReviewPredictor)
        model = build_models(target="rooms", strategy="aspect_text")["MultinomialNB_TFIDF"]
        model.fit(["terrible dirty room", "bad dirty room", "excellent clean room", "good clean room"], [1, 1, 5, 5])
        model.confidence_method_ = "native_probability_uncalibrated"
        self.predictor.models = {"rooms": model}
        entries = {target: {"status": "unavailable"} for target in TARGET_COLUMNS}
        entries["rooms"] = {"status": "trained", "model": "MultinomialNB_TFIDF", "routing_strategy": "aspect_text",
                            "confidence_method": model.confidence_method_, "uncertainty": {"threshold": 0.}}
        self.predictor.manifest = {"schema_version": 2, "targets": entries}

    def test_prediction_evidence_confidence_and_status(self):
        result = self.predictor.predict("The room was dirty but staff were helpful.")["Rooms"]
        self.assertEqual(result["status"], "predicted")
        self.assertEqual(result["polarity"], "negative")
        self.assertEqual(result["evidence_source"], "aspect_text")
        self.assertNotIn("helpful", result["evidence"])
        self.assertTrue(0 <= result["confidence"] <= 1)

    def test_uncertain_abstains_but_retains_tentative_candidate(self):
        self.predictor.manifest["targets"]["rooms"]["uncertainty"]["threshold"] = 1.
        result = self.predictor.predict("The room was dirty.")["Rooms"]
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["rating"])
        self.assertIsNone(result["polarity"])
        self.assertIn(result["candidate_rating"], [1, 5])
        self.assertTrue(result["low_confidence"])

    def test_not_mentioned_and_unavailable_have_no_claimed_rating(self):
        result = self.predictor.predict("Staff were friendly.")
        self.assertEqual(result["Rooms"]["status"], "not_mentioned")
        self.assertEqual(result["Rooms"]["evidence_source"], "none")
        self.assertIsNone(result["Rooms"]["confidence"])
        self.assertEqual(result["Sleep Quality"]["status"], "unavailable")

    def test_opt_in_estimates_are_explicit(self):
        result = self.predictor.predict("A good clean stay.", include_unmentioned=True)["Rooms"]
        self.assertTrue(result["estimate"])
        self.assertFalse(result["mentioned"])
        self.assertEqual(result["evidence_source"], "whole_review_fallback")


if __name__ == '__main__':
    unittest.main(verbosity=2)
