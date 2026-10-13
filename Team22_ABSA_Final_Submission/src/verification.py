"""Independent saved-artifact checks and a bounded local Streamlit smoke test."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import numpy as np
import pandas as pd
from .aspect_router import route_review, evidence_for, ROUTER_VERSION
from .confidence import probability_estimates, is_uncertain, select_uncertainty_threshold
from .predict import ReviewPredictor
from .train import scores, choose_configuration
from .utils import load_dataset, rating_to_polarity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_routing_schema(root):
    text = "The location was excellent and the staff were helpful, but the room was dirty."
    routes = route_review(text)
    for target in ["rooms", "cleanliness"]:
        require("dirty" in routes[target]["aspect_text"] and "helpful" not in routes[target]["aspect_text"], "Mixed-aspect evidence routing failed.")
    for mode in ["real", "synthetic"]:
        predictor = ReviewPredictor(root / "models" / mode)
        result = predictor.predict(text)
        for target, row in result.items():
            required = {"rating", "polarity", "confidence", "mentioned", "status", "prediction_status", "evidence", "evidence_source", "selected_model", "routing_strategy", "candidate_rating", "estimate"}
            require(required <= row.keys(), f"Missing prediction fields for {mode}/{target}.")
            require(row["status"] == row["prediction_status"], "Conflicting prediction statuses.")
            require(row["status"] in {"predicted", "not_mentioned", "unavailable", "uncertain", "unknown_vocabulary"}, "Invalid prediction status.")
            if row["confidence"] is not None:
                require(0 <= row["confidence"] <= 1, "Invalid probability estimate.")
            if row["status"] == "predicted":
                require(row["rating"] in range(1, 6) and bool(row["evidence"]), "Predicted rating needs valid evidence.")
            else:
                require(row["rating"] is None and row["polarity"] is None, "Abstention must not claim a final label.")
        staff_only = predictor.predict("The staff were friendly.")
        for target in ["Rooms", "Location", "Value", "Cleanliness", "Business Service"]:
            require(staff_only[target]["status"] == "not_mentioned", f"Spurious {target} mention.")
        require(staff_only["Service"]["mentioned"], "Service mention missing.")
        sleep = predictor.predict("The room was noisy and I could not sleep.")["Sleep Quality"]
        require((sleep["status"] == "unavailable") if mode == "real" else sleep["status"] in {"predicted", "uncertain"}, "Sleep-quality availability is incorrect.")
    return {"router": ROUTER_VERSION, "mixed_clause_evidence": "passed", "schema": "passed", "real_sleep": "unavailable", "synthetic_sleep": "trained"}


def check_saved_results(root):
    checked = {}
    required_outputs = ["metrics.csv", "validation_model_comparison.csv", "validation_predictions.csv", "selection_manifest.json",
                        "test_predictions.csv", "error_examples.csv", "label_distribution.csv", "system_comparison.csv",
                        "aspect_routing_evaluation.csv", "mention_detection_examples.csv", "per_class_metrics.csv", "confidence_analysis.csv",
                        "mixed_aspect_evaluation.csv", "calibration_folds.csv", "run_metadata.json",
                        "validation_confidence_predictions.csv"]
    for mode in ["real", "synthetic"]:
        out = root / "outputs" / mode
        require(all((out / name).is_file() for name in required_outputs), f"Required {mode} evaluation outputs are missing.")
        if mode == "real":
            require((out / "hotel_bootstrap_metrics.csv").is_file(), "Hotel bootstrap evidence missing.")
        data = load_dataset(root / "data" / ("hotel_reviews_real.csv" if mode == "real" else "hotel_reviews_sample.csv")).set_index("review_id")
        predictor = ReviewPredictor(root / "models" / mode)
        metadata = predictor.manifest
        require(metadata["schema_version"] == 2 and metadata["mention_detection_version"] == ROUTER_VERSION, "Old model manifest.")
        require(metadata == json.loads((out / "run_metadata.json").read_text()), "Model and output metadata disagree.")
        for name, digest in metadata["source_sha256"].items():
            require(hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() == digest, f"{name} changed after training; retrain to refresh evidence.")
        split = pd.read_csv(out / "split_manifest.csv", dtype={"review_id": str, "hotel_id": str})
        partitions = {name: set(split.loc[split.split == name, "review_id"]) for name in ["train", "validation", "test"]}
        require(not (partitions["train"] & partitions["validation"] or partitions["train"] & partitions["test"] or partitions["validation"] & partitions["test"]), "Review leakage across partitions.")
        candidates = pd.read_csv(out / "validation_model_comparison.csv")
        validation = pd.read_csv(out / "validation_predictions.csv", dtype={"review_id": str})
        predictions = pd.read_csv(out / "test_predictions.csv", dtype={"review_id": str})
        metrics = pd.read_csv(out / "metrics.csv")
        frozen = json.loads((out / "selection_manifest.json").read_text())
        validation_confidence = pd.read_csv(out / "validation_confidence_predictions.csv", dtype={"review_id": str})
        for _, metric in metrics.iterrows():
            target, entry = metric.target, metadata["targets"][metric.target]
            rows = predictions[predictions.target == target]
            expected = {idx for idx in partitions["test"] if pd.notna(data.loc[idx, target])}
            require(set(rows.review_id) == expected and not rows.review_id.duplicated().any(), f"Incomplete or duplicate test scores for {target}.")
            truth = data.loc[rows.review_id, target].astype(int).to_numpy()
            np.testing.assert_array_equal(truth, rows.actual_rating)
            texts = data.loc[rows.review_id, "review_text"]
            model = predictor.models[target]
            pred = model.predict(texts).astype(int)
            np.testing.assert_array_equal(pred, rows.predicted_rating)
            for name, value in scores(truth, pred).items():
                require(np.isclose(value, metric[f"test_{name}"]), f"Saved {target}/{name} metric is incorrect.")
            confidence, _, _ = probability_estimates(model, texts, pred)
            if all(c is not None for c in confidence):
                np.testing.assert_allclose(confidence, rows.confidence, atol=1e-12)
            for text, (_, row) in zip(texts, rows.iterrows()):
                info = evidence_for(text, target, entry["routing_strategy"])
                require(info["evidence"] == row.evidence and info["evidence_source"] == row.evidence_source, f"Saved {target} evidence does not match its selected pipeline.")
            target_candidates = candidates[candidates.target == target]
            require(len(target_candidates) == 6 and target_candidates.selected.sum() == 1, f"Missing comparison or ambiguous selection for {target}.")
            winner = choose_configuration(target_candidates.to_dict("records"))
            require((winner["model"], winner["routing_strategy"]) == (entry["model"], entry["routing_strategy"]), "Manifest was not selected using validation scores.")
            for field in ["model", "routing_strategy", "validation_macro_f1", "uncertainty"]:
                require(entry[field] == frozen["targets"][target][field], "Selection changed after the pre-test freeze.")
            vc = validation_confidence[validation_confidence.target == target]
            expected_valid = {idx for idx in partitions["validation"] if pd.notna(data.loc[idx, target])}
            require(set(vc.review_id) == expected_valid and not vc.review_id.duplicated().any(), "Confidence threshold used an invalid validation population.")
            correct = np.asarray([rating_to_polarity(a) == rating_to_polarity(p) for a, p in zip(vc.actual_rating, vc.predicted_rating)])
            np.testing.assert_array_equal(data.loc[vc.review_id, target].astype(int), vc.actual_rating)
            threshold = select_uncertainty_threshold(vc.confidence, correct)
            require(np.isclose(threshold["threshold"], entry["uncertainty"]["threshold"], atol=1e-12), "Threshold was not derived from validation correctness.")
            for candidate in target_candidates.to_dict("records"):
                vr = validation[(validation.target == target) & (validation.model == candidate["model"]) & (validation.routing_strategy == candidate["routing_strategy"])]
                expected_valid = {idx for idx in partitions["validation"] if pd.notna(data.loc[idx, target])}
                require(set(vr.review_id) == expected_valid and not vr.review_id.duplicated().any(), "Invalid validation prediction population.")
                np.testing.assert_array_equal(data.loc[vr.review_id, target].astype(int), vr.actual_rating)
                require(np.isclose(scores(vr.actual_rating, vr.predicted_rating)["polarity_macro_f1"], candidate["validation_polarity_macro_f1"]), "Validation selection score disagrees with predictions.")
            dev_ids = split.loc[split.split != "test", "review_id"]
            dev_ids = [idx for idx in dev_ids if pd.notna(data.loc[idx, target])]
            digest = hashlib.sha256("\n".join(dev_ids).encode()).hexdigest()
            require(digest == entry["fit_review_ids_sha256"], "Final fit population differs from labelled development rows.")
            # Every final vocabulary feature must occur in development text after
            # the selected transformation. Test-only words cannot enter it.
            transformed = model.named_steps["text"].transform(data.loc[dev_ids, "review_text"])
            analyzer = model.named_steps["features"].build_analyzer()
            development_features = {feature for text in transformed for feature in analyzer(text)}
            require(set(model.named_steps["features"].vocabulary_) <= development_features, "Vocabulary includes features absent from development data.")
        folds = pd.read_csv(out / "calibration_folds.csv", dtype={"review_id": str, "hotel_id": str})
        for (target, stage), rows in folds.groupby(["target", "stage"]):
            allowed = partitions["train"] if stage == "training_only" else partitions["train"] | partitions["validation"]
            expected = {idx for idx in allowed if pd.notna(data.loc[idx, target])}
            require(set(rows.review_id) == expected and not rows.review_id.duplicated().any(), "Calibration used incorrect rows or omitted folds.")
            require(not set(rows.review_id) & partitions["test"], "Test rows entered calibration.")
            if mode == "real":
                require(rows.groupby("hotel_id").fold.nunique().max() == 1, "Hotel overlap in calibration folds.")
        checked[mode] = {"targets": len(metrics), "candidate_configurations": len(candidates), "test_predictions": len(predictions), "metric_recomputation": "passed", "source_fingerprints": "passed", "calibration_partitions": "passed"}
    return checked


def check_streamlit_startup(root):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                require(process.poll() is None, "Streamlit exited during startup.")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        if response.status == 200 and response.read().decode() == "ok":
                            break
                except OSError:
                    pass
                time.sleep(.2)
            else:
                raise ValueError("Streamlit health check timed out.")
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                require(response.status == 200, "Streamlit app page is unavailable.")
            return {"health_status": 200, "application_status": 200, "host": "127.0.0.1", "owned_process_stopped_after_check": True}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
