"""
tr_pretokenize.py — Word-level pre-tokenization for clitics that are
officially written with a space but very commonly attached in casual
Turkish writing.

Currently handles the interrogative particle mi / mı / mu / mü (plus any
trailing copular / agreement suffixes): the official orthography is
"gelecek mi", "gelecek misin", but text in the wild routinely writes
"gelecekmi", "gelecekmisin". UD-Turkish-IMST treats the particle as its
own token, so splitting it off both matches the treebank and lets the
parser analyse each piece correctly.

The split is deliberately CONSERVATIVE. Many ordinary words legitimately
end in -mi / -mı / -mu / -mü as part of their own morphology
("resmi" = resim+ACC, "ölümü" = ölüm+ACC, "kalemi" = kalem+ACC). To avoid
mangling those, a word is only split when ALL of the following hold:

  1. The whole word has no clean in-lexicon analysis (its best parse is
     OOV). A word that parses cleanly as-is is trusted and never split —
     this is what protects resmi / ölümü / kalemi / adamı.
  2. A trailing segment m + <harmonising high vowel> (+ optional agreement
     suffixes) parses as a complete interrogative-particle cluster.
  3. The remaining head is itself a real in-lexicon word.

Vowel harmony is enforced on the particle vowel (gelecek -> "mi", okudun
-> "mu", hasta -> "mı"), which prevents splitting at the wrong "m" inside
the word ("gelecekmisin" -> gelecek + misin, never gelecekm + isin).

Known limitation: a word whose attached-particle reading collides with a
valid in-lexicon analysis is left intact (rule 1). For example
"hastamısın" parses in-lexicon as hasta+POSS_1SG+..., so it is not split.
Disambiguating that needs sentence context and is out of scope here.
"""

from __future__ import annotations

from typing import List

from tr_phonology import tr_lower, last_vowel


# The bare interrogative particle, in all four harmonic shapes. A valid
# clitic tail is one of these optionally followed by agreement/copular
# suffixes (misin, mısın, musun, miyim, miydi, midir, ...), which the
# parser already analyses with the particle as the root.
PARTICLES = frozenset({"mi", "mı", "mu", "mü"})

# High-vowel harmony: the particle vowel is determined by the last vowel
# of the preceding stem.
_HARMONY = {
    "a": "ı", "ı": "ı",
    "e": "i", "i": "i",
    "o": "u", "u": "u",
    "ö": "ü", "ü": "ü",
}

# A word must be at least this long to even consider a stem + particle
# split (need at least a 1-char head plus a 2-char particle).
_MIN_SPLITTABLE_LEN = 3


def _expected_particle_vowel(head_lower: str):
    """The harmonising particle vowel for a given (lowercased) head, or
    None if the head has no vowel to harmonise with."""
    lv = last_vowel(head_lower)
    if lv is None:
        return None
    return _HARMONY.get(lv)


def split_question_clitic(word: str, parser) -> List[str]:
    """Split an attached interrogative particle off `word`.

    Returns ``[word]`` unchanged, or ``[head, clitic]`` when `word` is a
    stem with an attached question particle. Original casing is preserved
    by slicing the input string (the parser is only consulted on a
    lowercased copy).
    """
    wl = tr_lower(word)
    if len(wl) < _MIN_SPLITTABLE_LEN:
        return [word]

    # Rule 1: a word that already has a clean in-lexicon parse is trusted.
    whole = parser.parse(wl)
    if whole and not whole[0].oov:
        return [word]

    # Rules 2 & 3: find an 'm' that starts a harmonising particle cluster
    # whose head is a real word. Among valid candidates prefer the longest
    # head (i.e. the shortest particle), which is the conservative split.
    best_i = None
    for i in range(1, len(wl) - 1):
        if wl[i] != "m":
            continue
        head = wl[:i]
        expected_vowel = _expected_particle_vowel(head)
        if expected_vowel is None or wl[i + 1] != expected_vowel:
            continue
        tail = wl[i:]
        tail_an = parser.parse(tail)
        if not tail_an or tail_an[0].oov or tail_an[0].root not in PARTICLES:
            continue
        head_an = parser.parse(head)
        if not head_an or head_an[0].oov:
            continue
        if best_i is None or i > best_i:
            best_i = i

    if best_i is None:
        return [word]
    return [word[:best_i], word[best_i:]]
