"""
test_tr_phonology.py — Unit tests for tr_phonology.

Run with:    python -m unittest test_tr_phonology -v
"""

import unittest

from tr_phonology import (
    tr_lower, tr_upper,
    is_vowel, is_consonant, is_back, is_front, is_rounded, is_high, is_voiceless,
    last_vowel, last_letter, ends_in_vowel,
    resolve_A, resolve_H, resolve_D, resolve_C,
    soften_final, harden_final,
    apply_suffix,
)


class TestCaseConversion(unittest.TestCase):
    """The Turkish dotted/dotless I is the most common source of bugs in
    Turkish text processing. These tests guard against the default Python
    behavior, which is wrong for Turkish."""

    def test_dotted_dotless_I_lowercase(self):
        self.assertEqual(tr_lower("İSTANBUL"), "istanbul")
        self.assertEqual(tr_lower("IŞIK"),     "ışık")
        self.assertEqual(tr_lower("İYİ"),      "iyi")

    def test_dotted_dotless_I_uppercase(self):
        self.assertEqual(tr_upper("istanbul"), "İSTANBUL")
        self.assertEqual(tr_upper("ışık"),     "IŞIK")
        self.assertEqual(tr_upper("iyi"),      "İYİ")

    def test_special_letters_lower(self):
        self.assertEqual(tr_lower("ÇAĞRI"),  "çağrı")
        self.assertEqual(tr_lower("GÜZEL"),  "güzel")
        self.assertEqual(tr_lower("ÖĞRENCİ"),"öğrenci")
        self.assertEqual(tr_lower("ŞAPKA"),  "şapka")

    def test_special_letters_upper(self):
        self.assertEqual(tr_upper("çağrı"),   "ÇAĞRI")
        self.assertEqual(tr_upper("güzel"),   "GÜZEL")
        self.assertEqual(tr_upper("öğrenci"), "ÖĞRENCİ")
        self.assertEqual(tr_upper("şapka"),   "ŞAPKA")

    def test_roundtrip(self):
        for word in ["İstanbul", "Türkiye", "ışık", "öğretmen", "çocuk", "ağaç"]:
            self.assertEqual(tr_upper(tr_lower(word)), tr_upper(word))
            self.assertEqual(tr_lower(tr_upper(word)), tr_lower(word))

    def test_differs_from_python_default(self):
        # Sanity check that the bug we're protecting against is real.
        # Default Python turns "İ" into "i" + combining dot above (2 chars).
        self.assertNotEqual("İ".lower(), "i")
        # Ours gives a clean single-character "i".
        self.assertEqual(tr_lower("İ"), "i")
        self.assertEqual(len(tr_lower("İ")), 1)


class TestClassification(unittest.TestCase):

    def test_is_vowel(self):
        for c in "aeıioöuü":
            self.assertTrue(is_vowel(c), f"{c} should be a vowel")
        for c in "bcçdfgğhjklmnprsştvyz":
            self.assertFalse(is_vowel(c), f"{c} should not be a vowel")

    def test_is_consonant(self):
        self.assertTrue(is_consonant("k"))
        self.assertTrue(is_consonant("ğ"))
        self.assertFalse(is_consonant("a"))

    def test_backness(self):
        self.assertTrue(is_back("a"));   self.assertTrue(is_back("u"))
        self.assertFalse(is_back("e"));  self.assertFalse(is_back("ü"))
        self.assertTrue(is_front("e"));  self.assertTrue(is_front("i"))
        self.assertFalse(is_front("a")); self.assertFalse(is_front("o"))

    def test_rounding(self):
        for c in "oöuü":
            self.assertTrue(is_rounded(c), f"{c} should be rounded")
        for c in "aeıi":
            self.assertFalse(is_rounded(c), f"{c} should not be rounded")

    def test_height(self):
        for c in "ıiuü":
            self.assertTrue(is_high(c), f"{c} should be high")
        for c in "aeoö":
            self.assertFalse(is_high(c), f"{c} should not be high")

    def test_voicelessness(self):
        # The "Fıstıkçı Şahap" set.
        for c in "pçtkfsşh":
            self.assertTrue(is_voiceless(c), f"{c} should be voiceless")
        for c in "bcdgğjlmnrvyz":
            self.assertFalse(is_voiceless(c), f"{c} should not be voiceless")


class TestStemInspection(unittest.TestCase):

    def test_last_vowel(self):
        self.assertEqual(last_vowel("gel"),    "e")
        self.assertEqual(last_vowel("kitap"),  "a")
        self.assertEqual(last_vowel("öğretmen"), "e")
        self.assertEqual(last_vowel("oku"),    "u")
        self.assertEqual(last_vowel("gör"),    "ö")
        self.assertEqual(last_vowel("ışık"),   "ı")

    def test_last_vowel_none(self):
        self.assertIsNone(last_vowel(""))
        self.assertIsNone(last_vowel("xyz"))  # no Turkish vowels

    def test_ends_in_vowel(self):
        self.assertTrue(ends_in_vowel("araba"))
        self.assertTrue(ends_in_vowel("oku"))
        self.assertFalse(ends_in_vowel("ev"))
        self.assertFalse(ends_in_vowel("kitap"))
        self.assertFalse(ends_in_vowel(""))

    def test_last_letter(self):
        self.assertEqual(last_letter("gel"), "l")
        self.assertEqual(last_letter(""),    "")


class TestArchiphonemeA(unittest.TestCase):
    """A → 'a' (back) or 'e' (front)."""

    def test_back_stems(self):
        for stem in ["kitap", "araba", "okul", "kapı", "soru"]:
            self.assertEqual(resolve_A(stem), "a", f"A after {stem!r}")

    def test_front_stems(self):
        for stem in ["ev", "göz", "üzüm", "şehir", "köy"]:
            self.assertEqual(resolve_A(stem), "e", f"A after {stem!r}")

    def test_no_vowel_default(self):
        self.assertEqual(resolve_A(""), "e")
        self.assertEqual(resolve_A("xyz"), "e")


class TestArchiphonemeH(unittest.TestCase):
    """H covers all four high vowels by full harmony."""

    def test_back_unrounded(self):
        # Last vowel back + unrounded → ı
        for stem in ["kitap", "kız", "arı", "kapı"]:
            self.assertEqual(resolve_H(stem), "ı", f"H after {stem!r}")

    def test_back_rounded(self):
        # Last vowel back + rounded → u
        for stem in ["okul", "kol", "burun", "oyun"]:
            self.assertEqual(resolve_H(stem), "u", f"H after {stem!r}")

    def test_front_unrounded(self):
        # Last vowel front + unrounded → i
        for stem in ["ev", "şehir", "ekmek", "gemi"]:
            self.assertEqual(resolve_H(stem), "i", f"H after {stem!r}")

    def test_front_rounded(self):
        # Last vowel front + rounded → ü
        for stem in ["göz", "üzüm", "söz", "köy"]:
            self.assertEqual(resolve_H(stem), "ü", f"H after {stem!r}")


class TestArchiphonemeDC(unittest.TestCase):
    """D and C undergo voicing assimilation: voiceless after voiceless."""

    def test_D_after_voiceless(self):
        # git, kitap, ağaç, çocuk all end in voiceless → D = t
        for stem in ["git", "kitap", "ağaç", "çocuk", "yat", "kork"]:
            self.assertEqual(resolve_D(stem), "t", f"D after {stem!r}")

    def test_D_after_voiced(self):
        # gel, gör, yaz, ev — voiced consonants → D = d
        for stem in ["gel", "gör", "yaz", "ev"]:
            self.assertEqual(resolve_D(stem), "d", f"D after {stem!r}")

    def test_D_after_vowel(self):
        # Vowels are voiced → D = d
        for stem in ["oku", "ye", "araba"]:
            self.assertEqual(resolve_D(stem), "d", f"D after {stem!r}")

    def test_C_after_voiceless(self):
        for stem in ["git", "kitap", "çocuk"]:
            self.assertEqual(resolve_C(stem), "ç", f"C after {stem!r}")

    def test_C_after_voiced_or_vowel(self):
        for stem in ["gel", "ev", "araba"]:
            self.assertEqual(resolve_C(stem), "c", f"C after {stem!r}")


class TestConsonantAlternation(unittest.TestCase):

    def test_soften_k(self):
        self.assertEqual(soften_final("ekmek"), "ekmeğ")
        self.assertEqual(soften_final("çocuk"), "çocuğ")

    def test_soften_p(self):
        self.assertEqual(soften_final("kitap"), "kitab")
        self.assertEqual(soften_final("dolap"), "dolab")

    def test_soften_t(self):
        self.assertEqual(soften_final("kanat"), "kanad")
        self.assertEqual(soften_final("kağıt"), "kağıd")

    def test_soften_ç(self):
        self.assertEqual(soften_final("ağaç"), "ağac")
        self.assertEqual(soften_final("amaç"), "amac")

    def test_no_softening(self):
        # Stems ending in non-softening consonants or vowels.
        for stem in ["ev", "gel", "araba", "kız"]:
            self.assertEqual(soften_final(stem), stem)

    def test_harden_round_trip(self):
        for stem in ["kitab", "ekmeğ", "kanad", "ağac"]:
            self.assertEqual(soften_final(harden_final(stem)), stem)

    def test_empty(self):
        self.assertEqual(soften_final(""), "")
        self.assertEqual(harden_final(""), "")


class TestApplySuffix(unittest.TestCase):
    """Integration: realized forms of stem + suffix template."""

    # --- Past tense -DH + person ---
    def test_past_1sg(self):
        self.assertEqual(apply_suffix("gel",   "DHm"), "geldim")
        self.assertEqual(apply_suffix("git",   "DHm"), "gittim")   # t after voiceless
        self.assertEqual(apply_suffix("oku",   "DHm"), "okudum")
        self.assertEqual(apply_suffix("gör",   "DHm"), "gördüm")
        self.assertEqual(apply_suffix("yaz",   "DHm"), "yazdım")
        self.assertEqual(apply_suffix("sor",   "DHm"), "sordum")
        self.assertEqual(apply_suffix("öl",    "DHm"), "öldüm")
        self.assertEqual(apply_suffix("bak",   "DHm"), "baktım")   # t after voiceless

    def test_past_1pl(self):
        self.assertEqual(apply_suffix("gel", "DHk"), "geldik")
        self.assertEqual(apply_suffix("git", "DHk"), "gittik")
        self.assertEqual(apply_suffix("oku", "DHk"), "okuduk")

    # --- 1sg possessive -Hm (vowel-initial, triggers softening) ---
    def test_possessive_softens_p(self):
        self.assertEqual(apply_suffix("kitap", "Hm"), "kitabım")

    def test_possessive_softens_ç(self):
        self.assertEqual(apply_suffix("ağaç", "Hm"), "ağacım")

    def test_possessive_softens_t(self):
        self.assertEqual(apply_suffix("kanat", "Hm"), "kanadım")

    def test_possessive_softens_k(self):
        # ekmek → ekmeğim (1sg poss)
        self.assertEqual(apply_suffix("ekmek", "Hm"), "ekmeğim")

    def test_possessive_no_softening_needed(self):
        self.assertEqual(apply_suffix("ev",  "Hm"), "evim")
        self.assertEqual(apply_suffix("göz", "Hm"), "gözüm")

    # --- Dative -(y)A: buffer y after vowels ---
    def test_dative_after_consonant(self):
        self.assertEqual(apply_suffix("ev",    "(y)A"), "eve")
        self.assertEqual(apply_suffix("okul",  "(y)A"), "okula")
        self.assertEqual(apply_suffix("göz",   "(y)A"), "göze")

    def test_dative_after_vowel(self):
        self.assertEqual(apply_suffix("araba", "(y)A"), "arabaya")
        self.assertEqual(apply_suffix("oku",   "(y)A"), "okuya")
        self.assertEqual(apply_suffix("kapı",  "(y)A"), "kapıya")

    # --- 3sg possessive -(s)H: buffer s after vowels ---
    def test_3sg_poss_after_vowel(self):
        self.assertEqual(apply_suffix("araba", "(s)H"), "arabası")
        self.assertEqual(apply_suffix("oku",   "(s)H"), "okusu")  # if it were a noun
        self.assertEqual(apply_suffix("kapı",  "(s)H"), "kapısı")
        self.assertEqual(apply_suffix("köprü", "(s)H"), "köprüsü")

    def test_3sg_poss_after_consonant(self):
        # No buffer; softening fires on stem-final voiceless stop.
        self.assertEqual(apply_suffix("ev",     "(s)H"), "evi")
        self.assertEqual(apply_suffix("kitap",  "(s)H"), "kitabı")
        self.assertEqual(apply_suffix("göz",    "(s)H"), "gözü")
        self.assertEqual(apply_suffix("kanat",  "(s)H"), "kanadı")

    # --- Accusative -(y)H ---
    def test_accusative(self):
        self.assertEqual(apply_suffix("ev",    "(y)H"), "evi")
        self.assertEqual(apply_suffix("araba", "(y)H"), "arabayı")
        self.assertEqual(apply_suffix("kitap", "(y)H"), "kitabı")
        self.assertEqual(apply_suffix("göz",   "(y)H"), "gözü")
        self.assertEqual(apply_suffix("okul",  "(y)H"), "okulu")

    # --- Locative -DA ---
    def test_locative(self):
        self.assertEqual(apply_suffix("ev",     "DA"), "evde")
        self.assertEqual(apply_suffix("okul",   "DA"), "okulda")
        self.assertEqual(apply_suffix("kitap",  "DA"), "kitapta")  # voiceless-stem
        self.assertEqual(apply_suffix("ağaç",   "DA"), "ağaçta")
        self.assertEqual(apply_suffix("araba",  "DA"), "arabada")

    # --- Ablative -DAn ---
    def test_ablative(self):
        self.assertEqual(apply_suffix("ev",    "DAn"), "evden")
        self.assertEqual(apply_suffix("okul",  "DAn"), "okuldan")
        self.assertEqual(apply_suffix("kitap", "DAn"), "kitaptan")
        self.assertEqual(apply_suffix("araba", "DAn"), "arabadan")

    # --- Plural -lAr ---
    def test_plural(self):
        self.assertEqual(apply_suffix("ev",     "lAr"), "evler")
        self.assertEqual(apply_suffix("kitap",  "lAr"), "kitaplar")
        self.assertEqual(apply_suffix("göz",    "lAr"), "gözler")
        self.assertEqual(apply_suffix("araba",  "lAr"), "arabalar")

    # --- Empty / degenerate inputs ---
    def test_empty_template(self):
        self.assertEqual(apply_suffix("gel", ""), "gel")

    def test_empty_stem(self):
        # Falls back to defaults; output is just the realized template.
        self.assertEqual(apply_suffix("", "DHm"), "dim")

    # --- Multiple archiphonemes resolve in context ---
    def test_chained_resolution(self):
        # -DHr (causative on monosyllables, also 3sg general etc.)
        # gel + DHr → geldir? actually this template is for the *causative* of some
        # roots; using it as a phonology probe only.
        self.assertEqual(apply_suffix("gel", "DHr"), "geldir")
        self.assertEqual(apply_suffix("kal", "DHr"), "kaldır")
        self.assertEqual(apply_suffix("öl",  "DHr"), "öldür")
        self.assertEqual(apply_suffix("bul", "DHr"), "buldur")


if __name__ == "__main__":
    unittest.main(verbosity=2)
