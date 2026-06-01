"""
extract_lexicon.py — Build a Turkish lexicon from UD_Turkish-IMST.

Reads the .conllu files, extracts (lemma, UPOS) pairs with frequencies, and
emits a JSON lexicon for the parser.

Run:    python extract_lexicon.py UD_Turkish-IMST/ lexicon.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


# UD POS tags we care about. ADV, INTJ, etc. are ignored for now — they are
# mostly invariant and don't go through morphological parsing.
OPEN_CLASS = {"VERB", "NOUN", "PROPN", "ADJ"}


def parse_conllu(path: Path):
    """Yield (form, lemma, upos) tuples from a CoNLL-U file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            # Skip multi-word tokens (id contains "-") and copy nodes (".")
            if "-" in tok_id or "." in tok_id:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            yield form, lemma, upos


def is_clean_lemma(lemma: str) -> bool:
    """Filter out punctuation, numbers, and underscores (UD's placeholder
    for missing lemmas)."""
    if not lemma or lemma == "_":
        return False
    if not any(c.isalpha() for c in lemma):
        return False
    # Skip lemmas with embedded whitespace or unusual characters.
    if " " in lemma or "\t" in lemma:
        return False
    return True


def detect_softens(form: str, lemma: str) -> bool:
    """Heuristic: a stem 'softens' if there exists at least one inflected
    form where the lemma's final consonant has changed (k→ğ, p→b, t→d, ç→c).

    Compare 'kitap'+'kitabı': lemma ends in p, form has 'b' at that position
    followed by a vowel. → softens.
    """
    if not lemma or not form:
        return False
    if len(form) <= len(lemma):
        return False
    if not lemma[-1] in "kptç":
        return False
    if not form.startswith(lemma[:-1]):
        return False
    # The character at the lemma's final position in the surface form:
    altered = form[len(lemma) - 1]
    expected = {"k": "ğ", "p": "b", "t": "d", "ç": "c"}[lemma[-1]]
    return altered == expected


_HIGH_VOWELS = set("ıiuü")


def detect_vowel_zero(form: str, lemma: str) -> str | None:
    """Heuristic: bisyllabic stems with a high vowel in the second syllable
    drop that vowel before a vowel-initial suffix:
        ağız → ağz-      (ağzım)
        oğul → oğl-      (oğlum)
        göğüs → göğs-    (göğsü)
        burun → burn-    (burnum)
        karın → karn-    (karnım)
        akıl → akl-      (aklı)
        şehir → şehr-    (şehri)

    Returns the deleted-vowel variant if `form` is `lemma` with the
    penultimate vowel deleted, else None.
    """
    if len(lemma) < 4:
        return None
    if lemma[-2] not in _HIGH_VOWELS:
        return None
    # The penult must be a vowel, the final char must be a consonant.
    if lemma[-1] in "aeıioöuü":
        return None
    # Construct the alternate stem: lemma with penult vowel removed.
    alt = lemma[:-2] + lemma[-1]
    # Form must start with the alt stem and be longer (suffix follows).
    if not form.startswith(alt) or len(form) <= len(alt):
        return None
    # Reject if form actually starts with the full lemma (no alternation).
    if form.startswith(lemma):
        return None
    # The character right after the alt stem must be a vowel (we're at the
    # start of a vowel-initial suffix); else this is some other coincidence.
    if form[len(alt)] not in "aeıioöuüy":
        return None
    return alt


def main(argv):
    # Parse CLI: extract_lexicon.py <ud_dir> <out.json> [--split train|all]
    args = list(argv[1:])
    split = "all"
    if "--split" in args:
        idx = args.index("--split")
        split = args[idx + 1]
        if split not in ("train", "all"):
            print(f"--split must be 'train' or 'all', got {split!r}")
            return 1
        del args[idx:idx + 2]
    if len(args) != 2:
        print(f"Usage: {argv[0]} <UD_Turkish-IMST/dir> <output.json> [--split train|all]")
        return 1
    in_dir, out_path = Path(args[0]), Path(args[1])

    # (lemma, upos) → frequency
    freq = defaultdict(int)
    # (lemma, upos) → True if any inflected form shows softening
    softens = defaultdict(bool)
    # (lemma, upos) → set of vowel-zero variant stems observed
    vowel_zero_variants = defaultdict(set)

    files = sorted(in_dir.glob("*.conllu"))
    if split == "train":
        files = [f for f in files if "train" in f.name]
    print(f"Reading {len(files)} file(s): {[f.name for f in files]}")
    for conllu in files:
        for form, lemma, upos in parse_conllu(conllu):
            if upos not in OPEN_CLASS:
                continue
            lemma = lemma.lower()
            form_l = form.lower()
            if not is_clean_lemma(lemma):
                continue
            key = (lemma, upos)
            freq[key] += 1
            if not softens[key] and detect_softens(form_l, lemma):
                softens[key] = True
            vz = detect_vowel_zero(form_l, lemma)
            if vz is not None:
                vowel_zero_variants[key].add(vz)

    # Promote PROPN → NOUN for our purposes (UD distinguishes proper nouns;
    # morphologically they decline like common nouns).
    merged = defaultdict(int)
    merged_softens = defaultdict(bool)
    merged_variants = defaultdict(set)
    for (lemma, upos), count in freq.items():
        wc = "NOUN" if upos == "PROPN" else upos
        merged[(lemma, wc)] += count
        if softens[(lemma, upos)]:
            merged_softens[(lemma, wc)] = True
        merged_variants[(lemma, wc)] |= vowel_zero_variants[(lemma, upos)]

    # Emit lexicon. Sort by frequency descending so the file is
    # human-scannable from most to least common.
    entries = []
    for (lemma, wc), count in sorted(merged.items(),
                                     key=lambda x: (-x[1], x[0])):
        # Filter out single-letter "noise" entries that the UD treebank
        # tokenizer produces for things like initialisms (A.B.C.) or
        # OCR artifacts. They cause spurious parses like 'mi' →
        # 'm+POSS_3SG' (the m comes from a stray treebank token).
        # The only legitimate single-letter root is 'o' (3sg/that
        # pronoun), which we'll inject via irregulars instead.
        if len(lemma) == 1:
            continue
        entry = {
            "form":       lemma,
            "class":      wc,
            "frequency":  count,
            "soften":     merged_softens[(lemma, wc)],
        }
        if merged_variants[(lemma, wc)]:
            entry["variants"] = sorted(merged_variants[(lemma, wc)])
        entries.append(entry)

    # Merge in manual overrides for irregulars (variants the corpus extraction
    # can't detect automatically). Two modes: (a) update existing entries if
    # the (form, class) pair already exists, (b) add brand-new entries if not.
    # Adding is used for high-frequency function words like demonstratives
    # (bu, şu, o) that may be tokenized as PRON in UD and thus absent from
    # the NOUN extraction.
    irregulars_path = in_dir.parent / "irregulars.json"
    if irregulars_path.exists():
        with open(irregulars_path, "r", encoding="utf-8") as f:
            irregulars = json.load(f)
        applied = 0
        added = 0
        for override in irregulars.get("overrides", []):
            # Skip comment-only section markers (no form/class).
            if "form" not in override or "class" not in override:
                continue
            form, wc = override["form"], override["class"]
            payload = {k: v for k, v in override.items()
                       if not k.startswith("_") and k not in ("form", "class")}
            updated = False
            for entry in entries:
                if entry["form"] == form and entry["class"] == wc:
                    entry.update(payload)
                    applied += 1
                    updated = True
                    break
            if not updated:
                # Add a fresh entry. Defaults for required fields.
                new_entry = {
                    "form": form,
                    "class": wc,
                    "frequency": payload.get("frequency", 0),
                    "soften": payload.get("soften", False),
                }
                new_entry.update(payload)
                entries.append(new_entry)
                added += 1
        if applied or added:
            print(f"  (applied {applied} irregular overrides, added {added} "
                  f"new entries from {irregulars_path.name})")

    # Apply lexicon overrides.
    overrides_path = in_dir.parent / "lexicon_overrides.json"
    if overrides_path.exists():
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)

        # (a) Prune: remove derived-form entries that the rule machinery
        # handles automatically (pure inflectional Pass/Cau forms whose
        # gold lemma is the base, plus V→V lexicalized derivations that
        # the tokenizer should decompose).
        prune_set = {(o["form"], o["class"])
                     for o in overrides.get("prune", [])
                     if "form" in o and "class" in o}
        before = len(entries)
        entries = [e for e in entries
                   if (e["form"], e["class"]) not in prune_set]
        pruned = before - len(entries)
        print(f"  (pruned {pruned} derived-form entries via {overrides_path.name})")

        # (b) Derivations: tag base verbs with caus_deriv / pass_deriv
        # flags whose values are the template strings.
        deriv_applied = 0
        for deriv in overrides.get("derivations", []):
            if "form" not in deriv or "class" not in deriv:
                continue
            for entry in entries:
                if entry["form"] == deriv["form"] and entry["class"] == deriv["class"]:
                    for k, v in deriv.items():
                        if k in ("caus_deriv", "pass_deriv"):
                            entry[k] = v
                    deriv_applied += 1
                    break
        if deriv_applied:
            print(f"  (applied {deriv_applied} derivational flags via {overrides_path.name})")

    out = {
        "version":   "0.1",
        "source":    "UD_Turkish-IMST",
        "total":     len(entries),
        "entries":   entries,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # Print summary.
    by_class = defaultdict(int)
    for e in entries:
        by_class[e["class"]] += 1
    print(f"Extracted {len(entries)} entries:")
    for wc, n in sorted(by_class.items()):
        print(f"  {wc}: {n}")
    softening = sum(1 for e in entries if e["soften"])
    print(f"  (of which {softening} stems show softening)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
