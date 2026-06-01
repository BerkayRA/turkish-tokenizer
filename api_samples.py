"""
api_samples.py — A walking tour of the tr_api Tokenizer.

Run this directly to see worked examples of the JSON wire format
across the full range of Turkish morphology the parser handles.
Each section prints the result of one tokenize() call, accompanied
by a one-line explanation of what's interesting about it.

Usage:
    python api_samples.py
    python api_samples.py --json     # raw JSON only, no commentary
"""

import argparse
import json
import sys

from tr_api import Tokenizer, TokenizerConfig


# Each entry: (input_word, commentary). The commentary is shown before
# the JSON in the default mode. Inputs are grouped by what they exercise.
SAMPLES = [
    # ----- Simple nouns -----
    ("kitap",
     "Bare noun, in lexicon. Returns one morpheme (the root) and the "
     "default Case=Nom + Number=Sing + Person=3 features that UD adds "
     "to bare nominals."),

    ("kitabımı",
     "kitap + POSS_1SG + ACC. Notice the morphemes array: each suffix's "
     "surface chunk, its inventory id, and the features it emits. The "
     "softening p→b is visible in the root chunk (kitab, not kitap)."),

    ("evlerimizde",
     "Plural + possessive + locative chain. ev + PLUR + POSS_1PL + LOC. "
     "Five morphemes total including the root. All three case/possession "
     "suffixes contribute features to the result."),

    # ----- Verb inflection -----
    ("geldim",
     "Finite past tense. gel + PAST + 1SG_K. The Person=1 and Tense=Past "
     "features are emitted by the suffixes; Mood=Ind, Polarity=Pos, and "
     "Aspect=Perf are filled in as UD defaults for finite verbs."),

    ("geliyorum",
     "Progressive. gel + PROG + 1SG_Z. Notice the suffix is -iyor on the "
     "surface (vowel harmonized) while its inventory id is PROG and its "
     "template is Hyor."),

    ("gelmiyordum",
     "Negative past progressive. Chain: gel + NEG + PROG + PAST_COP + "
     "1SG_K. The NEG suffix's final -A is deleted by following PROG, so "
     "its surface chunk is just 'm'."),

    # ----- Mood: optative and imperative -----
    ("geleyim",
     "Optative 1sg. gel + OPT + OPT_1SG. The -(y)A optative inserts a "
     "buffer y after the vowel-final stem, giving 'gele', then -(y)Hm "
     "agreement gives 'geleyim'. Mood=Opt is emitted."),

    ("gelmeyeyim",
     "Negative optative 1sg. gel + NEG + OPT + OPT_1SG. Reads as 'let "
     "me not come'."),

    ("gelsin",
     "Imperative 3sg. gel + IMP_3SG. Treebank tags this Mood=Imp, "
     "Person=3 — a separate paradigm from optative 3sg."),

    # ----- Voice: inflectional CAUS/PASS -----
    ("yaptırdı",
     "Inflectional causative. yap + CAUS + PAST. Voice=Cau is emitted. "
     "The lemma is yap; gold doesn't move the lemma to the derived form."),

    ("yapıldı",
     "Inflectional passive. yap + PASS + PAST. Voice=Pass."),

    ("yaptırıldı",
     "Causative + passive together. Voice=CauPass (a derived combined "
     "feature in ud_feats())."),

    # ----- V→V derivational suffixes (the architectural choice) -----
    ("çıkardı",
     "V→V derivation. By design, this decomposes to çık + CAUS_DERIV + "
     "PAST, root=çık (not çıkar, even though UD-IMST treats çıkar as "
     "the gold lemma). No Voice feature is emitted — CAUS_DERIV is a "
     "morphological boundary marker, not an inflectional voice."),

    ("geçirdi",
     "Another V→V derivation. geç + CAUS_DERIV + PAST. The lex flag on "
     "geç specifies template='Hr', producing 'geçir' as the derived stem."),

    ("bulundu",
     "PASS-shaped V→V derivation. bul + PASS_DERIV + PAST. Distinct from "
     "the inflectional passive of bul which would be 'bulun' too but with "
     "Voice=Pass. The derivational reading wins because bul carries a "
     "pass_deriv flag."),

    # ----- Tense compounds: Pluperfect -----
    ("söylemişti",
     "Pluperfect. söyle + EVID + PAST_COP. The two-morpheme combination "
     "is collapsed in ud_feats() to a single Tense=Pqp feature, matching "
     "UD-IMST's convention."),

    # ----- Participles and verbal nouns -----
    ("yazmak",
     "Verbal noun (-mAk). yaz + NMZ_INF. UD-IMST keeps UPOS=VERB but "
     "marks VerbForm=Vnoun and assigns Case=Nom (verbal nouns ARE nominal "
     "morphologically). Person/Number defaults are NOT added (the 948 "
     "Vnoun tokens in train all lack them)."),

    ("olduğu",
     "Participial form. ol + DHK + POSS_3SG. VerbForm=Part. Notice the "
     "Person[psor]/Number[psor] features from POSS, but NO plain "
     "Person/Number defaults — participial paradigm differs from finite."),

    # ----- Genuine ambiguity -----
    ("evi",
     "Ambiguous: 'his house' (ev + POSS_3SG) vs 'the house [acc]' "
     "(ev + ACC). Both have the same score; the top is whichever the "
     "scoring picks first, alternatives shows the other."),

    # ----- OOV handling -----
    ("Muammer",
     "Proper noun. With apostrophe-free input, parsed as OOV root with "
     "no suffixes. The per-character OOV bonus keeps the parser from "
     "shaving off 'Muamme' + POSS_2SG[-r dropped]."),

    ("Osman'ın",
     "Apostrophized proper noun + -(H)n. The apostrophe is stripped at "
     "the entry point of parse(), so Osman is recognized as the root and "
     "the suffix is parsed (POSS_2SG and GEN have the same surface here; "
     "the parser picks one)."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="print only JSON, no commentary")
    ap.add_argument("--no-alts", action="store_true",
                    help="don't include alternatives in output")
    args = ap.parse_args()

    cfg = TokenizerConfig(include_alternatives=not args.no_alts)
    tok = Tokenizer(cfg)

    for word, note in SAMPLES:
        result = tok.tokenize(word)
        if args.json:
            print(json.dumps({"input": word, "result": result},
                             ensure_ascii=False))
        else:
            print("=" * 76)
            print(f"INPUT: {word}")
            print(f"NOTE:  {note}")
            print("-" * 76)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print()


if __name__ == "__main__":
    main()
