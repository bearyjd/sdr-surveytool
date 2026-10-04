# tests/test_no_tag_branching.py
"""Architecture rule (design doc section 1): no module may read
metadata.tag or classification_status to decide whether to start, retune
or chain a capture or decode. capture/ and ingest/ may not reference them
at all -- as names, attributes, keywords, arguments, imports, or words in
string constants (raw SQL such as metadata->>'tag') -- except for an
explicit allowlist: today, the step-4 normalizer's single write of
UNCLASSIFIED. A tripwire, like tests/agent/test_import_boundary.py."""

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WATCHED = {"tag", "classification_status", "ClassificationStatus"}
WATCHED_WORD = re.compile(r"\b(tag|classification_status|ClassificationStatus)\b")
ALLOWED = Counter(
    {
        ("capture/unknown/normalizer.py", "import", "ClassificationStatus"): 1,
        ("capture/unknown/normalizer.py", "name", "ClassificationStatus"): 1,
        ("capture/unknown/normalizer.py", "keyword", "classification_status"): 1,
    }
)


def references(source: str) -> Counter:
    found: Counter = Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in WATCHED:
            found["name", node.id] += 1
        elif isinstance(node, ast.Attribute) and node.attr in WATCHED:
            found["attribute", node.attr] += 1
        elif isinstance(node, ast.keyword) and node.arg in WATCHED:
            found["keyword", node.arg] += 1
        elif isinstance(node, ast.arg) and node.arg in WATCHED:
            found["argument", node.arg] += 1
        elif isinstance(node, ast.alias) and node.name in WATCHED:
            found["import", node.name] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in WATCHED_WORD.findall(node.value):
                found["string", word] += 1
    return found


def watched_files() -> list[Path]:
    """Every .py file under capture/ and ingest/, except vendored third-party
    trees (any path with a `vendor` directory component)."""
    return [
        path
        for package in ("capture", "ingest")
        for path in sorted((REPO / package).rglob("*.py"))
        if "vendor" not in path.relative_to(REPO).parts
    ]


def test_vendored_third_party_code_is_not_scanned():
    # capture/cellular/vendor/ holds the AGPL LTE-Cell-Scanner submodule,
    # including Python 2 helpers ast can't parse. It runs as a subprocess
    # and is not this project's code, so the rule doesn't apply to it.
    vendored = REPO / "capture" / "cellular" / "vendor" / "lte-cell-scanner" / "pyitpp.py"
    assert vendored not in set(watched_files())


def test_capture_and_ingest_never_branch_on_classification():
    found: Counter = Counter()
    for path in watched_files():
        for (kind, name), count in references(path.read_text()).items():
            found[str(path.relative_to(REPO)), kind, name] += count
    assert found == ALLOWED


@pytest.mark.parametrize(
    "source",
    [
        "if record.metadata.tag == 'lte': start_decoder()",
        "if row['classification_status'] == 'auto_classified': retune()",
        "status = metadata.get('tag')",
        "from schema.records import ClassificationStatus",
        "def chain(tag): ...",
        "Metadata(tag='x')",
        # Raw SQL reads the key without any Python name:
        "rows = conn.execute(text(\"SELECT id FROM survey_records WHERE metadata->>'tag' = 'lte'\"))",
        "query = 'SELECT metadata->>\\'classification_status\\' FROM survey_records'",
    ],
)
def test_checker_catches(source):
    assert references(source)


def test_staging_and_other_words_containing_tag_are_not_flagged():
    assert not references("STAGING = 'data/snippet-staging'  # stage, staged, tags")
