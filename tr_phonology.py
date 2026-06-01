"""
tr_phonology.py — Turkish phonology toolkit.

A standalone module with no external dependencies. Implements:
  - Turkish-aware case conversion (handles dotted/dotless I correctly)
  - Vowel and consonant classification
  - Vowel harmony (backness, backness + rounding)
  - Consonant alternations (softening, voicing assimilation)
  - Buffer-consonant insertion
  - Surface-form generation from abstract suffix templates

Template syntax for suffixes (Oflazer convention):
    A     → low vowel: a or e (backness harmony only)
    H     → high vowel: ı, i, u, or ü (backness + rounding harmony)
    D     → t or d (voicing assimilation)
    C     → ç or c (voicing assimilation)
    (X)   → optional buffer consonant X, realized only after vowel-final stems.
            Common: (y), (n), (s).
    any other character → literal

Examples:
    apply_suffix("gel",   "DHm")    → "geldim"     past 1sg
    apply_suffix("git",   "DHm")    → "gittim"     voicing assim. (t after voiceless)
    apply_suffix("oku",   "DHm")    → "okudum"     voiced after vowel
    apply_suffix("kitap", "Hm")     → "kitabım"    p → b softening + harmony
    apply_suffix("araba", "(y)A")   → "arabaya"    buffer y; dative
    apply_suffix("ev",    "(y)A")   → "eve"        buffer suppressed
    apply_suffix("araba", "(s)H")   → "arabası"    buffer s; 3sg poss
"""

from typing import Optional

# -----------------------------------------------------------------------------
# Alphabet constants
# -----------------------------------------------------------------------------

VOWELS              = set("aeıioöuü")
CONSONANTS          = set("bcçdfgğhjklmnprsştvyz")

BACK_VOWELS         = set("aıou")
FRONT_VOWELS        = set("eiöü")
ROUNDED_VOWELS      = set("oöuü")
UNROUNDED_VOWELS    = set("aeıi")
HIGH_VOWELS         = set("ıiuü")
LOW_VOWELS          = set("aeoö")

# "Fıstıkçı Şahap" mnemonic — the voiceless (sert) consonants.
VOICELESS_CONSONANTS = set("pçtkfsşh")
VOICED_CONSONANTS    = CONSONANTS - VOICELESS_CONSONANTS

# Final-consonant softening map (applied when a vowel-initial suffix follows).
SOFTEN = {"k": "ğ", "p": "b", "t": "d", "ç": "c"}
HARDEN = {v: k for k, v in SOFTEN.items()}


# -----------------------------------------------------------------------------
# Case conversion (Turkish-aware)
# -----------------------------------------------------------------------------
# Python's default str.lower() / str.upper() do the WRONG thing for Turkish:
#   "İSTANBUL".lower()  → "i̇stanbul"   (i with combining dot above — broken)
#   "ışık".upper()      → "IŞIK"        (correct here, by luck)
#   "iyi".upper()       → "IYI"         (WRONG — should be "İYİ")
#
# We use explicit translation tables.

_TR_LOWER_MAP = str.maketrans({
    "İ": "i", "I": "ı",
    "Ç": "ç", "Ğ": "ğ", "Ö": "ö", "Ş": "ş", "Ü": "ü",
    "A": "a", "B": "b", "C": "c", "D": "d", "E": "e",
    "F": "f", "G": "g", "H": "h", "J": "j", "K": "k",
    "L": "l", "M": "m", "N": "n", "O": "o", "P": "p",
    "R": "r", "S": "s", "T": "t", "U": "u", "V": "v",
    "Y": "y", "Z": "z",
})

_TR_UPPER_MAP = str.maketrans({
    "i": "İ", "ı": "I",
    "ç": "Ç", "ğ": "Ğ", "ö": "Ö", "ş": "Ş", "ü": "Ü",
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
    "f": "F", "g": "G", "h": "H", "j": "J", "k": "K",
    "l": "L", "m": "M", "n": "N", "o": "O", "p": "P",
    "r": "R", "s": "S", "t": "T", "u": "U", "v": "V",
    "y": "Y", "z": "Z",
})


def tr_lower(s: str) -> str:
    """Lowercase a Turkish string. Handles İ → i and I → ı correctly."""
    return s.translate(_TR_LOWER_MAP)


def tr_upper(s: str) -> str:
    """Uppercase a Turkish string. Handles i → İ and ı → I correctly."""
    return s.translate(_TR_UPPER_MAP)


# -----------------------------------------------------------------------------
# Character classification
# -----------------------------------------------------------------------------

def is_vowel(c: str) -> bool:
    return c in VOWELS

def is_consonant(c: str) -> bool:
    return c in CONSONANTS

def is_back(c: str) -> bool:
    return c in BACK_VOWELS

def is_front(c: str) -> bool:
    return c in FRONT_VOWELS

def is_rounded(c: str) -> bool:
    return c in ROUNDED_VOWELS

def is_high(c: str) -> bool:
    return c in HIGH_VOWELS

def is_voiceless(c: str) -> bool:
    return c in VOICELESS_CONSONANTS


# -----------------------------------------------------------------------------
# Stem inspection
# -----------------------------------------------------------------------------

def last_vowel(stem: str) -> Optional[str]:
    """Return the last vowel of `stem`, or None if there is no vowel."""
    for c in reversed(stem):
        if c in VOWELS:
            return c
    return None


def last_letter(stem: str) -> str:
    """Return the last letter of `stem`, or '' if empty."""
    return stem[-1] if stem else ""


def ends_in_vowel(stem: str) -> bool:
    return bool(stem) and stem[-1] in VOWELS


def count_syllables(stem: str) -> int:
    """Count syllables in `stem` (= number of vowels). Turkish has no
    syllabic consonants and no diphthongs, so vowel count equals
    syllable count."""
    return sum(1 for c in stem if c in VOWELS)


def is_monosyllabic(stem: str) -> bool:
    """True if `stem` has exactly one vowel."""
    return count_syllables(stem) == 1


# -----------------------------------------------------------------------------
# Archiphoneme resolution
# -----------------------------------------------------------------------------

def resolve_A(stem: str) -> str:
    """A → 'a' (back stem) or 'e' (front stem). Defaults to 'e' if no vowel."""
    v = last_vowel(stem)
    if v is None:
        return "e"
    return "a" if v in BACK_VOWELS else "e"


def resolve_H(stem: str) -> str:
    """H → 'ı', 'i', 'u', or 'ü' by backness + rounding harmony.
    Defaults to 'i' if no vowel."""
    v = last_vowel(stem)
    if v is None:
        return "i"
    back    = v in BACK_VOWELS
    rounded = v in ROUNDED_VOWELS
    if     back and not rounded: return "ı"
    if     back and     rounded: return "u"
    if not back and not rounded: return "i"
    return "ü"  # front + rounded


def resolve_D(stem: str) -> str:
    """D → 't' after a voiceless consonant, otherwise 'd'."""
    return "t" if last_letter(stem) in VOICELESS_CONSONANTS else "d"


def resolve_C(stem: str) -> str:
    """C → 'ç' after a voiceless consonant, otherwise 'c'."""
    return "ç" if last_letter(stem) in VOICELESS_CONSONANTS else "c"


def resolve_G(stem: str) -> str:
    """G → 'k' after a voiceless consonant, otherwise 'g'.

    Used by suffixes like -GHn (`-gın`/`-kın`), where the initial
    consonant of the suffix assimilates in voicing to the stem-final
    consonant: kız+gın (voiced z) vs. küs+kün (voiceless s) vs.
    sus+kun (voiceless s)."""
    return "k" if last_letter(stem) in VOICELESS_CONSONANTS else "g"


# -----------------------------------------------------------------------------
# Consonant alternations
# -----------------------------------------------------------------------------

def soften_final(stem: str) -> str:
    """Apply final-consonant softening (k→ğ, p→b, t→d, ç→c).

    This is triggered when a vowel-initial suffix follows. Not all stems
    soften — many monosyllabic stems and loanwords are exceptions
    (at → atı, tek → teki). Those will be flagged in the lexicon; this
    function applies the rule unconditionally and is meant to be called
    only when the lexicon says it should be.
    """
    if not stem:
        return stem
    last = stem[-1]
    if last in SOFTEN:
        return stem[:-1] + SOFTEN[last]
    return stem


def harden_final(stem: str) -> str:
    """Inverse of soften_final, used during parsing to undo the alternation
    (ğ→k, b→p, d→t, c→ç)."""
    if not stem:
        return stem
    last = stem[-1]
    if last in HARDEN:
        return stem[:-1] + HARDEN[last]
    return stem


# -----------------------------------------------------------------------------
# Surface-form generation
# -----------------------------------------------------------------------------

def _suffix_starts_with_vowel(template: str, stem: str) -> bool:
    """Will the realized suffix begin with a vowel? Used to decide whether
    stem-final softening fires."""
    if not template:
        return False

    i = 0
    if template[0] == "(":
        # If stem ends in vowel, buffer is realized → suffix starts with the
        # buffer consonant (not a vowel).
        if ends_in_vowel(stem):
            return False
        # Buffer suppressed; check what comes after the ")".
        end = template.index(")", i)
        i = end + 1
        if i >= len(template):
            return False

    ch = template[i]
    if ch in "AH":
        return True
    return ch in VOWELS


def apply_suffix(stem: str, template: str, soften: bool = True) -> str:
    """Realize a suffix template against a stem.

    Resolves archiphonemes (A, H, D, C), inserts/suppresses buffer consonants,
    and applies final-consonant softening when appropriate.

    Args:
        stem: the stem to attach to.
        template: the abstract suffix form (see module docstring).
        soften: whether the stem undergoes final-consonant softening before
            a vowel-initial suffix. This is a per-stem lexical property in
            Turkish (kitap → kitabım, but at → atım), so callers with
            lexicon knowledge should pass the appropriate value. Default is
            True, matching the most common case.
    """
    if not template:
        return stem

    starts_with_vowel = _suffix_starts_with_vowel(template, stem)
    working_stem = soften_final(stem) if (starts_with_vowel and soften) else stem

    out = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]

        if ch == "(":
            end = template.index(")", i)
            buffer_char = template[i + 1:end]
            # Realize buffer only if what precedes it is a vowel.
            preceding = out[-1] if out else (working_stem[-1] if working_stem else "")
            if preceding in VOWELS:
                out.append(buffer_char)
            i = end + 1
            continue

        # Harmony / assimilation context is the stem plus everything emitted so far.
        context = working_stem + "".join(out)

        if   ch == "A": out.append(resolve_A(context))
        elif ch == "H": out.append(resolve_H(context))
        elif ch == "D": out.append(resolve_D(context))
        elif ch == "C": out.append(resolve_C(context))
        elif ch == "G": out.append(resolve_G(context))
        else:           out.append(ch)
        i += 1

    return working_stem + "".join(out)
