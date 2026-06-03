"""
apply_additions.py — merge curated lexicon additions into built lexicons.

The production lexicons are corpus-derived (UD + TDK) and so don't contain
the open-ended class of proper nouns (place names, etc.). `lexicon_overrides.json`
carries a curated `additions` gazetteer; this tool merges those entries into
already-built lexicon files so you don't have to regenerate from the corpus.

It is idempotent (forms already present are skipped) and preserves each
file's existing indentation. By default it targets the production lexicons
and deliberately skips `lexicon_train.json`, which is kept pristine so the
no-leakage evaluation stays honest.

Usage:
    python apply_additions.py                          # lexicon.json + lexicon_full.json
    python apply_additions.py --targets lexicon.json
    python apply_additions.py --overrides lexicon_overrides.json
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_TARGETS = ["lexicon.json", "lexicon_full.json"]


def load_additions(overrides_path: Path):
    """Return the curated additions as a list of {form, class, ...} dicts,
    skipping section markers and comments."""
    with open(overrides_path, encoding="utf-8") as f:
        data = json.load(f)
    return [a for a in data.get("additions", [])
            if isinstance(a, dict) and "form" in a and "class" in a]


def _detect_indent(text: str) -> int:
    """Infer the indent width from the first indented line (lexicon.json
    uses 1 space, lexicon_full.json uses 2)."""
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if stripped and stripped != line:
            return len(line) - len(stripped)
    return 2


def merge_into_entries(entries: list, additions: list) -> int:
    """Append additions whose form is not already present. Returns the count
    added. Entries are normalised to the lexicon schema."""
    existing = {e["form"] for e in entries if isinstance(e, dict) and "form" in e}
    added = 0
    for a in additions:
        if a["form"] in existing:
            continue
        entries.append({
            "form": a["form"],
            "class": a["class"],
            "frequency": int(a.get("frequency", 0)),
            "soften": bool(a.get("soften", False)),
        })
        existing.add(a["form"])
        added += 1
    return added


def apply_to_file(path: Path, additions: list) -> int:
    raw = path.read_text(encoding="utf-8")
    indent = _detect_indent(raw)
    data = json.loads(raw)
    if isinstance(data, list):
        added = merge_into_entries(data, additions)
        payload = data
    else:
        data.setdefault("entries", [])
        added = merge_into_entries(data["entries"], additions)
        payload = data
    if added:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)
            f.write("\n")
    return added


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overrides", default=str(HERE / "lexicon_overrides.json"))
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS,
                    help="lexicon files to merge into (default: production "
                         "lexicons; lexicon_train.json is intentionally omitted)")
    args = ap.parse_args(argv[1:])

    additions = load_additions(Path(args.overrides))
    print(f"{len(additions)} curated additions from {args.overrides}")
    for t in args.targets:
        path = Path(t) if Path(t).is_absolute() else HERE / t
        if not path.exists():
            print(f"  {t}: SKIP (not found)")
            continue
        added = apply_to_file(path, additions)
        print(f"  {t}: +{added} new "
              f"({'unchanged' if not added else 'updated'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
