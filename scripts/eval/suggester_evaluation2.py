from typing import Optional, List, Dict, Any, Union

import spacy
from spacy.tokens import DocBin
from spacy.cli._util import app, Arg, Opt, setup_gpu, import_code
from pathlib import Path
from tqdm import tqdm
from wasabi import msg
import typer


def main(
    span_key: str,
    model_path: Path,
    test_path: Path,
    code_path: Optional[Path] = Opt(
        None,
        "--code",
        "-c",
        help=
        "Path to Python file with additional code (registered functions) to be imported"
    ),
):
    # Initialize NER & Spancat models
    import_code(code_path)
    nlp = spacy.load(model_path)
    spancat = nlp.get_pipe("spancat")

    # Get test.spacy DocBin
    test_doc_bin = DocBin().from_disk(test_path)
    test_docs = list(test_doc_bin.get_docs(nlp.vocab))

    # Suggester KPI
    total_candidates = 0
    total_real_candidates = 0
    matching_candidates = 0
    near_candidates = 0
    near_candidates2 = 0
    missed = 0

    msg.info("Starting evaluation")

    for test_doc in tqdm(test_docs,
                         total=len(test_docs),
                         desc=f"Evaluation test dataset"):
        # Prediction
        text = test_doc.text
        doc = nlp(text)
        spancat.set_candidates([doc])

        # Count spans when saving spans is enabled
        total_candidates += len(doc.spans["candidates"])
        total_real_candidates += len(test_doc.spans[span_key])

        # Check for True Positives and False Positives
        for test_span in test_doc.spans[span_key]:
            # Calculate coverage
            hit = 0
            for span in doc.spans["candidates"]:
                if span.start == test_span.start and span.end == test_span.end:
                    matching_candidates += 1
                    hit += 1
                if hit == 0:
                    if abs(span.start - test_span.start) < 2 and abs(
                            span.end - test_span.end) < 2:
                        near_candidates += 1
                        hit += 1
                    elif abs(span.start - test_span.start) < 3 and abs(
                            span.end - test_span.end) < 3:
                        near_candidates2 += 1
                        hit += 1
            if hit == 0:
                missed += 1

    msg.good("Evaluation successful")

    # Suggester Coverage
    coverage = round((matching_candidates / total_real_candidates) * 100, 2)
    near_coverage = round(
        ((matching_candidates + near_candidates) / total_real_candidates) *
        100, 2)
    near_coverage2 = round(
        ((matching_candidates + near_candidates + near_candidates2) /
         total_real_candidates) * 100, 2)
    candidates_relation = round(
        (total_candidates / total_real_candidates) * 100, 2)
    missed_ratio = round((missed / total_real_candidates) * 100, 2)

    msg.divider("Suggester KPI")

    suggester_header = ["KPI", "Value"]
    suggester_data = [
        ("Suggester candidates", total_candidates),
        ("Real candidates", total_real_candidates),
        ("Missed span", missed),
        ("% Ratio", f"{candidates_relation}%"),
        ("% Coverage", f"{coverage}%"),
        ("% Near Coverage (within 1 word each side).", f"{near_coverage}%"),
        ("% Near Coverage (within 2 words each side).", f"{near_coverage2}%"),
        ("% Missed ratio", f"{missed_ratio}%"),
    ]
    msg.table(suggester_data, header=suggester_header, divider=True)


if __name__ == "__main__":
    typer.run(main)
