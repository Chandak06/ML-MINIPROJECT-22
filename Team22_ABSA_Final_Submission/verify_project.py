"""Verify the complete local submission, with honest per-check results and exit status."""
from __future__ import annotations

import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
MODULES = {"pandas": "pandas", "scikit-learn": "sklearn", "matplotlib": "matplotlib", "numpy": "numpy", "joblib": "joblib", "streamlit": "streamlit"}
SOURCE_FILES = ["README.md", "requirements.txt", "verify_project.py", "setup.cmd", "start_demo.cmd", ".gitignore", "run_project.py", "prepare_data.py", "streamlit_app.py", "CONTRIBUTIONS.md", "DEMO_GUIDE.md", "src/config.py", "src/utils.py", "src/train.py", "src/predict.py", "tests/test_project.py", "tests/test_readiness.py"]
SOURCE_FILES += ["src/aspect_router.py", "src/confidence.py", "src/evaluation.py", "src/verification.py", "src/diagnostics.py", "tests/test_routing.py", "tests/test_artifacts.py"]
REPORT = "ABSA_Hotel_Reviews_Final_Writeup.pdf"
PRESENTATION = "ABSA_Hotel_Reviews_Final_Presentation.pptx"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_structure(root):
    missing = [p for p in SOURCE_FILES if not (root / p).is_file()]
    missing += [p + "/" for p in ["src", "tests", "data", "models/real", "models/synthetic", "outputs/real", "outputs/synthetic", "docs"] if not (root / p).is_dir()]
    require(not missing, "Missing: " + ", ".join(missing))
    return "Required project files and folders exist."


def check_dependencies(root):
    versions = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        importlib.import_module(MODULES[name])
        actual = metadata.version(name)
        require(actual == expected, f"{name}: installed {actual}, expected {expected}; install requirements.txt.")
        versions[name] = actual
    require(set(MODULES).issubset(versions), "A critical dependency is missing from requirements.txt.")
    for module in ["src.config", "src.utils", "src.train", "src.predict", "src.aspect_router", "src.confidence", "src.evaluation", "src.verification", "prepare_data", "run_project", "streamlit.testing.v1"]:
        importlib.import_module(module)
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=90)
    require(result.returncode == 0, result.stdout.strip() or result.stderr.strip() or "pip check failed.")
    return {"python": sys.version.split()[0], "packages": versions, "pip_check": "passed", "imports": "passed"}


def check_dataset(root):
    import hashlib
    import pandas as pd
    from src.utils import load_dataset
    real_path = root / "data/hotel_reviews_real.csv"
    provenance = json.loads(real_path.with_suffix(".provenance.json").read_text(encoding="utf8"))
    require(hashlib.sha256(real_path.read_bytes()).hexdigest() == provenance["csv_sha256"], "Real dataset differs from its provenance checksum.")
    real = load_dataset(real_path)
    synthetic = load_dataset(root / "data/hotel_reviews_sample.csv")
    require(len(real) == provenance["rows"], "Real dataset row count differs from provenance.")
    require(len(real) == 2100 and real.hotel_id.nunique() == 122, "Expected 2,100 real reviews from 122 hotels.")
    require(len(synthetic) == 320, "Expected the separate 320-row synthetic demonstration.")
    require(real.sleep_quality.isna().all(), "Unexpected real sleep-quality labels: verify provenance.")
    split = pd.read_csv(root / "outputs/real/split_manifest.csv", dtype={"review_id": str, "hotel_id": str})
    run = json.loads((root / "outputs/real/run_metadata.json").read_text(encoding="utf8"))
    require(run["seed"] == 42 and run["split_method"] == "hotel_disjoint", "Expected the documented seed and hotel-disjoint evaluation.")
    require(not split.review_id.duplicated().any(), "Duplicate review in split manifest.")
    require(set(split.review_id) == set(real.review_id), "Split manifest does not cover the dataset exactly.")
    require(split.groupby("hotel_id").split.nunique().max() == 1, "Hotel leakage across partitions.")
    require(not split.text_hash.duplicated().any(), "Duplicate text in split manifest.")
    require(split.split.value_counts().to_dict() == run["split_counts"], "Split counts do not match run metadata.")
    require(run["split_counts"] == {"train": 1342, "validation": 388, "test": 370}, "The established real split changed.")
    by_id = real.set_index("review_id")
    require(all(split.hotel_id == by_id.loc[split.review_id, "hotel_id"].to_numpy()), "Hotel IDs disagree with the dataset.")
    require(all(split.text_hash == by_id.loc[split.review_id, "text_hash"].to_numpy()), "Review hashes disagree with the dataset.")
    return {"real_reviews": len(real), "hotels": int(real.hotel_id.nunique()), "split": run["split_counts"], "synthetic_demo_reviews": len(synthetic)}


def check_models(root):
    from src.predict import ReviewPredictor
    counts = {}
    for mode in ["real", "synthetic"]:
        predictor = ReviewPredictor(root / "models" / mode)
        expected = sum(e["status"] == "trained" for e in predictor.manifest["targets"].values())
        require(expected == (7 if mode == "real" else 8) and len(predictor.models) == expected, f"Incomplete {mode} models.")
        result = predictor.predict("The staff were helpful and the room was clean.")
        require(result["Overall"]["status"] in {"predicted", "uncertain"}, f"{mode} sample inference failed.")
        require(result["Overall"]["candidate_rating"] in range(1, 6), "Invalid underlying model rating.")
        counts[mode] = expected
    return {"loaded_models": counts, "hash_checks": "passed", "sample_predictions": "passed"}


def check_cli(root):
    review = "The room was clean and spacious, but the staff was rude and the breakfast was disappointing."
    manifests = [root / "models" / mode / "manifest.json" for mode in ["real", "synthetic"]]
    before = [p.stat().st_mtime_ns for p in manifests]
    result = subprocess.run([sys.executable, "-B", "run_project.py", "--review", review], cwd=root, capture_output=True, text=True, timeout=90)
    require(result.returncode == 0, result.stderr.strip() or "CLI exited nonzero.")
    payload = json.loads(result.stdout)
    require(payload["Overall"]["status"] in {"predicted", "uncertain"}, "CLI inference failed.")
    require(payload["Overall"]["candidate_rating"] in range(1, 6), "CLI returned no underlying rating estimate.")
    require([p.stat().st_mtime_ns for p in manifests] == before, "Prediction modified model manifests.")
    return {"exit_code": result.returncode, "overall": payload["Overall"], "retraining": False}


def check_report(root):
    path = root / "docs" / REPORT
    content = path.read_bytes()
    require(content.startswith(b"%PDF-") and b"%%EOF" in content[-1024:], "Final report is missing or not a complete PDF.")
    require([p.name for p in (root / "docs").glob("*.pdf")] == [REPORT], "Duplicate or outdated PDFs in docs/.")
    return {"file": f"docs/{REPORT}", "bytes": len(content), "check": "Present with PDF header and end marker; visual/page-count audit is separate."}


def check_presentation(root):
    path = root / "docs" / PRESENTATION
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "Presentation ZIP CRC failed.")
        document = ET.fromstring(archive.read("ppt/presentation.xml"))
        slides = document.findall(".//{http://schemas.openxmlformats.org/presentationml/2006/main}sldId")
        require(len(slides) > 0, "Presentation has no slides.")
        archive.read("[Content_Types].xml")
    require([p.name for p in (root / "docs").glob("*.pptx")] == [PRESENTATION], "Duplicate or outdated presentations in docs/.")
    return {"file": f"docs/{PRESENTATION}", "slides": len(slides), "check": "OOXML structure and CRC; visual review is separate."}


def run_tests(root):
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful() and result.testsRun > 0 and not result.skipped}
    return summary, log.getvalue()


def check_document_consistency(root):
    import hashlib
    audit = json.loads((root / "outputs/verification/document_audit.json").read_text(encoding="utf8"))
    require(audit["status"] == "passed" and audit["report_pages"] == 2, "Final document audit is incomplete.")
    for name in [REPORT, PRESENTATION]:
        require(hashlib.sha256((root / "docs" / name).read_bytes()).hexdigest() == audit["sha256"][name], f"{name} changed after visual review.")
    require(hashlib.sha256((root / "outputs/real/metrics.csv").read_bytes()).hexdigest() == audit["metrics_sha256"], "Documents do not match current evaluation results.")
    return {"report_pages": audit["report_pages"], "presentation_slides": audit["presentation_slides"], "visual_audit_hashes": "passed", "current_metrics": "matched"}


def ready(checks):
    return bool(checks) and all(c["passed"] for c in checks)


def verify(root=ROOT):
    start = time.perf_counter()
    checks = []
    def execute(name, function):
        try:
            detail = function(root)
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    execute("Project structure", check_structure)
    execute("Dependencies", check_dependencies)
    execute("Dataset", check_dataset)
    execute("Model loading", check_models)
    execute("CLI prediction", check_cli)
    # Imports are inside the checks so a missing dependency becomes FAIL rather
    # than terminating the verifier before its clean summary.
    def routing(root):
        from src.verification import check_routing_schema
        return check_routing_schema(root)
    def results(root):
        from src.verification import check_saved_results
        return check_saved_results(root)
    def server(root):
        from src.verification import check_streamlit_startup
        return check_streamlit_startup(root)
    execute("Aspect routing and prediction schema", routing)
    execute("Saved metrics, selection and leakage checks", results)
    execute("Streamlit startup", server)
    execute("Final report", check_report)
    execute("Final presentation", check_presentation)
    execute("Document consistency with current results", check_document_consistency)
    tests = {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0, "successful": False}
    log = "Tests could not start.\n"
    try:
        tests, log = run_tests(root)
        checks.append({"name": "Automated tests", "passed": tests["successful"], "detail": tests})
    except Exception as exc:
        checks.append({"name": "Automated tests", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    stamp = datetime.now(timezone.utc).isoformat()
    elapsed = round(time.perf_counter() - start, 2)
    output = root / "outputs/verification"
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "test_results.txt").write_text(log, encoding="utf8")
        (output / "test_summary.json").write_text(json.dumps({"utc_time": stamp, **tests, "elapsed_seconds": elapsed}, indent=2), encoding="utf8")
        (output / "verification_summary.json").write_text(json.dumps({"utc_time": stamp, "checks": checks, "ready": ready(checks), "scope": "Local technical deliverables; private GitHub access and contribution confirmation remain manual."}, indent=2), encoding="utf8")
    except OSError as exc:
        checks.append({"name": "Verification logs", "passed": False, "detail": str(exc)})
    print("=" * 40 + "\nTEAM 22 ABSA PROJECT VERIFICATION\n" + "=" * 40)
    for check in checks:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
        if not check["passed"]:
            print(f"       {check['detail']}")
    print(f"\nTests: {tests['tests_run']} run, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped")
    print("FINAL STATUS: " + ("READY FOR SUBMISSION" if ready(checks) else "NOT READY FOR SUBMISSION"))
    print("Scope: local technical package. Complete private GitHub access, portal submission and truthful contribution entries manually.")
    print("Detailed evidence: outputs/verification/")
    return 0 if ready(checks) else 1


if __name__ == "__main__":
    raise SystemExit(verify())
