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

There are two split regimes:

  - Particle + person/copular agreement (misin, mısın, mıydı, midir,
    miyim, ...) is an UNAMBIGUOUS attached question and is split even when
    the whole word also has a clean in-lexicon parse — so "hastamısın" ->
    hasta + mısın every time. These copular/agreement suffixes never attach
    to an ordinary noun the way a bare particle's look-alike can, so the
    override is safe. (Crucially, particle + POSSESSIVE — "mim" = mi+POSS_1SG
    — is NOT in the agreement whitelist, so a clean word like "adamım" is
    never mis-split into ada + mım.)
  - A BARE particle (mi/mı/mu/mü, optionally + possessive) collides with
    ordinary noun/adjective morphology (resmi, ölümü, kalemi), so it is only
    split off when the whole word has no clean in-lexicon reading.
"""

from __future__ import annotations

from typing import List

from tr_phonology import tr_lower, last_vowel


# The bare interrogative particle, in all four harmonic shapes. A valid
# clitic tail is one of these optionally followed by agreement/copular
# suffixes (misin, mısın, musun, miyim, miydi, midir, ...), which the
# parser already analyses with the particle as the root.
PARTICLES = frozenset({"mi", "mı", "mu", "mü"})

# Suffix IDs that mark the particle as a predicate carrying person/copular
# agreement: the z-type person endings plus the copular markers. Their
# presence in the trailing cluster makes an attached question unambiguous.
# POSS_* is deliberately excluded — "mim" (mi+POSS_1SG) must NOT count, so
# that ordinary words like "adamım" are never split.
COPULAR_AGREEMENT_IDS = frozenset({
    "1SG_Z", "2SG_Z", "1PL_Z", "2PL_Z", "3PL_Z",
    "PAST_COP", "COND_COP", "COP_DHR", "COP_EVID",
})

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

    whole = parser.parse(wl)
    whole_clean = bool(whole) and not whole[0].oov

    # Find every 'm' that starts a harmonising particle cluster on a
    # plausible stem. Classify by whether the cluster carries person/copular
    # agreement (an unambiguous question) or is a bare particle (ambiguous
    # with noun morphology). Head index starts at 2 — a one-letter stem
    # before a question particle is implausible.
    agreement_splits = []
    bare_splits = []
    for i in range(2, len(wl) - 1):
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
        if not head_an:
            continue
        tail_suffix_ids = {m.suffix_id for m in tail_an[0].morphemes[1:]}
        if tail_suffix_ids & COPULAR_AGREEMENT_IDS:
            # Particle + person/copular agreement (misin, mısınız, mıydı) is
            # unambiguous, so the stem only has to be a plausible word — it may
            # carry an OOV root, e.g. the unknown proper-noun base in
            # "Çekoslovakyalılaştır…mısınız". Bare particles below still
            # require an in-lexicon stem to avoid false splits (resmi, ölümü).
            agreement_splits.append(i)
        elif not head_an[0].oov:
            bare_splits.append(i)

    # Unambiguous attached question (particle + person/copular agreement):
    # split even over a clean whole-word parse. hastamısın -> hasta + mısın.
    if agreement_splits:
        i = max(agreement_splits)
        return [word[:i], word[i:]]
    # Bare particle: only split when the whole word has no clean reading.
    if bare_splits and not whole_clean:
        i = max(bare_splits)
        return [word[:i], word[i:]]
    return [word]
