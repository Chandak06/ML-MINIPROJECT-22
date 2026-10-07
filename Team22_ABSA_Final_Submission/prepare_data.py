"""Download/convert the scientific-use ArguAna TripAdvisor corpus, without executing archive content."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/3973241/files/arguana-tripadvisor-annotated-v2.zip?download=1"
EXPECTED_MD5 = "a450bbdbbf888fb171783b62aa81e332"
ROOT = Path(__file__).resolve().parent
MAPPING = {"service": "service", "rooms": "room", "location": "location", "value": "value", "cleanliness": "cleanliness", "business_service": "business"}


def convert(archive: Path, output: Path):
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError("Downloaded corpus checksum differs from the published Zenodo checksum.")
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if "/complete/" in n and n.endswith(".xmi") and not n.startswith("__MACOSX/"))
        for name in names:
            elements = {el.tag.split("}")[-1]: el for el in ET.fromstring(z.read(name))}
            aspects = elements["Aspects"].attrib
            row = {"review_id": Path(name).stem, "hotel_id": elements["HotelData"].get("hotelID"),
                   "review_text": elements["Sofa"].get("sofaString"),
                   "overall_rating": int(float(elements["Sentiment"].get("score"))), "sleep_quality": ""}
            for target, source in MAPPING.items():
                value = float(aspects.get(source, -1))
                row[target] = int(value) if value in [1, 2, 3, 4, 5] else ""
            rows.append(row)
    if len(rows) != 2100:
        raise ValueError(f"Expected 2100 corpus reviews, found {len(rows)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["review_id", "hotel_id", "review_text", "overall_rating", "service", "rooms", "location", "value", "cleanliness", "sleep_quality", "business_service"]
    with output.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"name": "ArguAna TripAdvisor annotated v2", "source": URL, "record": "https://zenodo.org/records/3973241",
                  "citation": "Wachsmuth et al. (2014), A Review Corpus for Argumentation Analysis, CICLing.",
                  "terms": "Scientific use only; not for commercial applications. Cite the corpus paper.",
                  "archive_md5": digest, "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "rows": len(rows),
                  "mapping": MAPPING, "missing_labels": "Source -1 becomes blank. No label imputation. Sleep quality is absent.",
                  "extraction": "Only complete/*.xmi corpus reviews. Overall score is Sentiment.score, never HotelData.stars. Author and date metadata are omitted."}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")
    print(f"Prepared {len(rows)} real reviews: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "arguana-tripadvisor-annotated-v2.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "hotel_reviews_real.csv")
    args = parser.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=90) as response:
            args.archive.write_bytes(response.read())
    convert(args.archive, args.out)


if __name__ == "__main__":
    main()
