"""
build_expanded_lexicon.py — Build a comprehensive Turkish root lexicon
by merging the UD-extracted lexicon (high-quality, POS-tagged, frequency-
weighted) with a wordlist scraped from the TDK official dictionary (broad
coverage, ~92K headwords).

The UD lexicon takes precedence: where both sources contribute the same
form, the UD entry's class and metadata win. TDK entries fill in gaps.

Output: lexicon_full.json with the same schema as lexicon.json.

Usage:
    python build_expanded_lexicon.py \
        --base lexicon.json \
        --tdk tdk_words.json \
        --out lexicon_full.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


VOWELS = set("aeıioöuü")
SOFTEN_FINAL = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}


def turkish_lower(s: str) -> str:
    """Lowercase a string preserving Turkish character distinctions.
    Standard str.lower() turns I → i, but in Turkish I → ı."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def looks_like_verb_citation(form: str) -> bool:
    """Verb dictionary forms end in -mak / -mek."""
    return form.endswith("mak") or form.endswith("mek")


def classify_tdk_entry(form: str, meanings):
    """Determine (class, root_form) for a TDK entry. The TDK data doesn't
    carry POS tags reliably, so we use surface heuristics:

    - VERB: citation form ends in -mak/-mek, strip to get the bare root
    - everything else: NOUN by default; adjectives in Turkish overlap
      heavily with nouns at the lemma level and UD-Turkish reclassifies
      based on syntactic context anyway

    Returns None to skip the entry.
    """
    if not form or not form.isalpha() and not re.match(r"^[a-zçğıöşüâîû]+$", form):
        return None
    # Skip junk characters; only keep entries with Turkish letters
    if not re.match(r"^[a-zçğıöşüâîû]+$", form):
        return None
    if looks_like_verb_citation(form):
        if len(form) <= 3:
            return None
        return ("VERB", form[:-3])
    return ("NOUN", form)


def detect_softening(form: str, all_tdk_forms: set) -> bool:
    """A noun stem like kitap softens to kitab- before vowel-initial
    suffixes. We detect this by checking if a softened-final-consonant
    variant of the form appears as a verb root or in a derived form.
    Conservative: only set if we have evidence.

    Heuristic: if form ends in p/ç/t/k and form-with-softened-final
    appears as a stem in any TDK entry's *meanings*, or if a softened
    variant is itself a separate TDK headword, assume softening.
    This is imperfect; the bulk of softening cases will be caught
    by the UD lexicon's frequency-based detection which we preserve.
    """
    if len(form) < 2:
        return False
    last = form[-1]
    if last not in SOFTEN_FINAL:
        return False
    # Check if softened form appears in TDK headwords
    softened = form[:-1] + SOFTEN_FINAL[last]
    # If softened variant exists, this is likely a softening stem.
    return softened in all_tdk_forms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="UD-extracted lexicon (authoritative for overlaps)")
    ap.add_argument("--tdk", required=True,
                    help="TDK wordlist JSON (broad coverage)")
    ap.add_argument("--out", required=True, help="Output lexicon JSON")
    ap.add_argument("--min-form-len", type=int, default=2,
                    help="Skip TDK entries shorter than this (default 2)")
    ap.add_argument("--max-form-len", type=int, default=20,
                    help="Skip TDK entries longer than this (default 20)")
    args = ap.parse_args()

    # Load UD lexicon (authoritative).
    with open(args.base, "r", encoding="utf-8") as f:
        ud_lex = json.load(f)
    ud_entries = ud_lex if isinstance(ud_lex, list) else ud_lex.get("entries", [])
    print(f"Base UD lexicon: {len(ud_entries)} entries")

    # Index by (form, class) so we can avoid duplicates.
    seen = {(e["form"], e["class"]) for e in ud_entries}
    # Also track forms across classes — TDK is noisy and we don't want
    # to add tdk entry "kitap NOUN" when UD has "kitap NOUN" already.
    seen_forms_by_class = {}
    for e in ud_entries:
        seen_forms_by_class.setdefault(e["form"], set()).add(e["class"])

    # Load TDK data.
    with open(args.tdk, "r", encoding="utf-8") as f:
        tdk = json.load(f)
    print(f"TDK source: {len(tdk)} headwords")

    # Pre-filter and lowercase the raw TDK headwords.
    tdk_candidates = []
    tdk_forms = set()
    for entry in tdk:
        form = entry.get("word", "")
        if not form:
            continue
        # Skip multi-word and hyphenated
        if " " in form or "-" in form or "'" in form:
            continue
        form_lc = turkish_lower(form)
        if len(form_lc) < args.min_form_len or len(form_lc) > args.max_form_len:
            continue
        # Skip if contains characters outside Turkish letters
        if not re.match(r"^[a-zçğıöşüâîû]+$", form_lc):
            continue
        tdk_candidates.append((form_lc, entry.get("meanings", [])))
        tdk_forms.add(form_lc)

    print(f"TDK pre-filtered: {len(tdk_candidates)} candidates")

    # Classify each, add if not already seen.
    added = 0
    by_class = Counter()
    new_entries = []
    for form, meanings in tdk_candidates:
        result = classify_tdk_entry(form, meanings)
        if result is None:
            continue
        cls, root_form = result
        # Skip if UD already has this form (any class).
        # Subtle: if UD has VERB "kitap" (impossible, but illustrative)
        # and TDK wants NOUN "kitap", we'd still skip — TDK can't add
        # to a form UD already knows about. This is conservative;
        # the UD entry was deliberately chosen.
        if root_form in seen_forms_by_class:
            continue
        # Skip if this root collides with a duplicate within TDK.
        key = (root_form, cls)
        if key in seen:
            continue
        seen.add(key)
        # Detect softening
        soften = detect_softening(root_form, tdk_forms) if cls == "NOUN" else False
        entry = {
            "form": root_form,
            "class": cls,
            "frequency": 0,    # TDK gives no frequency
            "soften": soften,
        }
        new_entries.append(entry)
        by_class[cls] += 1
        added += 1

    print(f"Added from TDK: {added}")
    print(f"  by class: {dict(by_class)}")

    # Combine and write.
    all_entries = ud_entries + new_entries
    print(f"Final lexicon: {len(all_entries)} entries")
    print(f"  VERB: {sum(1 for e in all_entries if e['class'] == 'VERB')}")
    print(f"  NOUN: {sum(1 for e in all_entries if e['class'] == 'NOUN')}")
    print(f"  ADJ:  {sum(1 for e in all_entries if e['class'] == 'ADJ')}")

    out_data = all_entries if isinstance(ud_lex, list) else {"entries": all_entries}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
