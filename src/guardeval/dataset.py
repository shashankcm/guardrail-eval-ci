"""Load and verify the guardrail evaluation dataset.

Each row is one user message with a label. The verifier refuses a dataset that is malformed,
that repeats itself, that appears in both splits (which would leak into tuning), or that
copies the example phrases kube-rag was built and tuned on.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REFERENCE_PATH = DATA_DIR / "reference" / "kuberag_phrases.txt"

# Whether a message in each category should be blocked by a correct guardrail.
CATEGORIES = {"attack": True, "offtopic": True, "legit": False, "hard_negative": False}
SPLITS = ("dev", "test")
REQUIRED_FIELDS = ("id", "split", "category", "text", "expected_blocked", "source")

MIN_PER_CATEGORY = {
    "dev": {"attack": 20, "offtopic": 20, "legit": 20, "hard_negative": 10},
    "test": {"attack": 10, "offtopic": 10, "legit": 10, "hard_negative": 5},
}

# Two texts with a token Jaccard similarity at or above this are treated as copies.
NEAR_DUPLICATE = 0.8
MIN_LENGTH, MAX_LENGTH = 5, 300

_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lower-case, replace punctuation with spaces, and collapse whitespace."""
    return " ".join(_PUNCTUATION.sub(" ", text.casefold()).split())


def _tokens(text: str) -> frozenset:
    return frozenset(normalize(text).split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def load_rows(path: Path) -> list:
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON ({error.msg})") from error
    return rows


def load_reference(path: Path = REFERENCE_PATH) -> list:
    """Example phrases from kube-rag. Lines starting with # are comments."""
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_rows(rows: list, reference: list, min_per_category: dict = None) -> list:
    """Return a list of human-readable errors. An empty list means the dataset is valid."""
    errors = []
    seen_ids = set()
    seen_text = {}  # normalized text -> (label, split)
    kept = []  # (label, split, normalized text, tokens) for near-duplicate checks
    reference_normalized = {normalize(p) for p in reference}
    reference_tokens = [_tokens(p) for p in reference]

    for index, r in enumerate(rows):
        label = str(r.get("id", f"row {index}"))
        missing = [field for field in REQUIRED_FIELDS if field not in r]
        if missing:
            errors.append(f"{label}: missing field(s) {', '.join(missing)}")
            continue

        if label in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(label)

        if r["split"] not in SPLITS:
            errors.append(f"{label}: unknown split {r['split']!r}")

        if r["category"] not in CATEGORIES:
            errors.append(f"{label}: unknown category {r['category']!r}")
        elif not isinstance(r["expected_blocked"], bool) or r["expected_blocked"] != CATEGORIES[r["category"]]:
            errors.append(f"{label}: expected_blocked must be {CATEGORIES[r['category']]} for category {r['category']!r}")

        text = r["text"]
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}: text is empty")
            continue
        if not MIN_LENGTH <= len(text.strip()) <= MAX_LENGTH:
            errors.append(f"{label}: text length must be {MIN_LENGTH} to {MAX_LENGTH} characters")

        norm = normalize(text)
        tokens = _tokens(text)

        if norm in reference_normalized or any(_jaccard(tokens, rt) >= NEAR_DUPLICATE for rt in reference_tokens):
            errors.append(f"{label}: matches or nearly matches a kube-rag reference phrase")

        if norm in seen_text:
            other_label, other_split = seen_text[norm]
            if other_split != r["split"]:
                errors.append(f"{label}: text appears in more than one split (also {other_label})")
            else:
                errors.append(f"{label}: duplicates {other_label}")
        else:
            seen_text[norm] = (label, r["split"])

        for other_label, other_split, other_norm, other_tokens in kept:
            if other_norm == norm:
                continue  # exact copies are reported above
            if _jaccard(tokens, other_tokens) >= NEAR_DUPLICATE:
                scope = "across splits" if other_split != r["split"] else "within a split"
                errors.append(f"{label}: near-duplicate of {other_label} ({scope})")
        kept.append((label, r["split"], norm, tokens))

    if min_per_category:
        counts = Counter((r.get("split"), r.get("category")) for r in rows if "split" in r and "category" in r)
        for split, minimums in min_per_category.items():
            for category, needed in minimums.items():
                if counts[(split, category)] < needed:
                    errors.append(f"{split}/{category}: {counts[(split, category)]} rows, need at least {needed}")

    return errors


def main() -> int:
    rows = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}.jsonl"
        if path.exists():
            rows += load_rows(path)
    errors = validate_rows(rows, load_reference(), MIN_PER_CATEGORY)

    counts = Counter((r.get("split"), r.get("category")) for r in rows)
    print(f"{'split':<6}" + "".join(f"{c:>15}" for c in CATEGORIES))
    for split in SPLITS:
        print(f"{split:<6}" + "".join(f"{counts[(split, c)]:>15}" for c in CATEGORIES))

    if errors:
        print(f"\n{len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nDataset is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
