"""
test_tr_suggest.py — OOV suggestions and morphology-aware correction.

Run with:    python -m unittest test_tr_suggest -v
"""

import unittest

from tr_api import Tokenizer, TokenizerConfig


class TestOOVSuggestions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def top_suggestion(self, word):
        s = self.tok.tokenize(word).get("suggestions", [])
        return s[0]["word"] if s else None

    def test_typo_gets_intended_word_first(self):
        self.assertEqual(self.top_suggestion("kitp"), "kitap")       # deletion
        self.assertEqual(self.top_suggestion("mektob"), "mektup")    # 2 subs
        self.assertEqual(self.top_suggestion("arabba"), "araba")     # insertion
        self.assertEqual(self.top_suggestion("öğretmenn"), "öğretmen")

    def test_valid_word_has_no_suggestions(self):
        for w in ("kitap", "geldim", "mekân", "ilmi", "kitabımı"):
            with self.subTest(word=w):
                self.assertNotIn("suggestions", self.tok.tokenize(w))

    def test_garbage_yields_empty_list_not_crash(self):
        r = self.tok.tokenize("xyzqwz")
        self.assertEqual(r.get("suggestions"), [])

    def test_can_be_disabled(self):
        tok = Tokenizer(TokenizerConfig(suggest_on_oov=False))
        self.assertNotIn("suggestions", tok.tokenize("kitp"))

    def test_suggest_method_direct(self):
        out = self.tok.suggest("kitp")
        self.assertTrue(out)
        self.assertEqual(out[0]["word"], "kitap")
        self.assertIn("distance", out[0])

    def test_suggestions_capped(self):
        tok = Tokenizer(TokenizerConfig(max_suggestions=2))
        self.assertLessEqual(len(tok.tokenize("kitp").get("suggestions", [])), 2)


class TestMorphologyAwareCorrection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_corrects_stem_typo_in_inflected_word(self):
        # 'okllarda' (missing u) -> okullarda; the stem is corrected and the
        # suffix chain (-lar-da) re-attaches, so the lemma is recovered.
        out = self.tok.correct("okllarda")
        self.assertTrue(out)
        self.assertEqual(out[0]["lemma"], "okul")
        self.assertEqual(out[0]["word"], "okullarda")

    def test_correction_entry_shape(self):
        out = self.tok.correct("kitp")
        self.assertTrue(out)
        for key in ("word", "lemma", "split", "distance"):
            self.assertIn(key, out[0])

    def test_valid_word_not_corrected(self):
        # A word that already parses in-lexicon yields no correction.
        for w in ("okullarda", "kitaplarım", "geldim", "öğretmenleri"):
            with self.subTest(word=w):
                self.assertEqual(self.tok.correct(w), [])

    def test_correction_surfaces_in_tokenize_suggestions(self):
        # tokenize() prefers morphology-aware corrections for inflected OOV.
        sugg = self.tok.tokenize("okllarda").get("suggestions", [])
        self.assertTrue(sugg)
        self.assertEqual(sugg[0]["word"], "okullarda")
        self.assertEqual(sugg[0]["lemma"], "okul")

    def test_garbage_not_corrected(self):
        self.assertEqual(self.tok.correct("zzzqww"), [])

    def test_corrects_typo_at_softened_stem_boundary(self):
        # The softened stem form (kitap->kitab, mektup->mektub) is in the
        # fuzzy index, so a typo next to the boundary is repaired and the
        # surface is reconstructed correctly.
        k = self.tok.correct("kitebımı")
        self.assertTrue(k)
        self.assertEqual(k[0]["lemma"], "kitap")
        self.assertEqual(k[0]["word"], "kitabımı")
        m = self.tok.correct("mektobu")
        self.assertTrue(m)
        self.assertEqual(m[0]["lemma"], "mektup")
        self.assertEqual(m[0]["word"], "mektubu")

    def test_bare_correction_shows_canonical_not_softened(self):
        # A bare correction must surface the citation form (mektup), never
        # the softened stem (mektub).
        sugg = self.tok.tokenize("mektob").get("suggestions", [])
        self.assertTrue(sugg)
        self.assertEqual(sugg[0]["word"], "mektup")
        self.assertNotEqual(sugg[0]["word"], "mektub")


class TestTailRepair(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_repairs_suffix_transposition(self):
        # evlerimizdne -> evlerimizden (n/e transposed in the case suffix);
        # stem repair can't fix this since the stem (ev) is correct.
        out = self.tok.correct("evlerimizdne")
        self.assertTrue(out)
        self.assertEqual(out[0]["word"], "evlerimizden")
        self.assertEqual(out[0]["lemma"], "ev")

    def test_repairs_missing_suffix_letter(self):
        # A correct repair must appear among the candidates.
        words = [x["word"] for x in self.tok.correct("kitaplarn")]
        self.assertTrue(any(w in ("kitapları", "kitapların") for w in words))

    def test_tail_repair_can_be_disabled(self):
        tok = Tokenizer(TokenizerConfig(correct_tail_typos=False))
        # With tail repair off, a pure suffix typo is no longer corrected.
        self.assertEqual(tok.correct("evlerimizdne"), [])
        # ...but stem repair still works.
        self.assertTrue(tok.correct("kitp"))

    def test_valid_word_still_not_corrected(self):
        for w in ("evlerimizden", "kitaplarım", "geldim"):
            with self.subTest(word=w):
                self.assertEqual(self.tok.correct(w), [])


class TestPerCallOverrides(unittest.TestCase):
    """Per-call kwargs override the config without rebuilding the tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_suggest_override(self):
        self.assertIn("suggestions", self.tok.tokenize("kitp", suggest=True))
        self.assertNotIn("suggestions", self.tok.tokenize("kitp", suggest=False))

    def test_tail_repair_override(self):
        # Off -> a pure suffix typo is not corrected; on -> it is.
        self.assertEqual(
            self.tok.tokenize("evlerimizdne", tail_repair=False).get("suggestions"), [])
        on = self.tok.tokenize("evlerimizdne", tail_repair=True).get("suggestions", [])
        self.assertTrue(any(s["word"] == "evlerimizden" for s in on))

    def test_alternatives_override(self):
        self.assertNotIn("alternatives", self.tok.tokenize("yüzü", alternatives=False))
        self.assertIn("alternatives", self.tok.tokenize("yüzü", alternatives=True))

    def test_text_overrides_propagate(self):
        off = self.tok.tokenize_text("gelecekmisin", split_clitics=False)
        on = self.tok.tokenize_text("gelecekmisin", split_clitics=True)
        words_off = [t["surface"] for t in off["tokens"] if t["kind"] == "word"]
        words_on = [t["surface"] for t in on["tokens"] if t["kind"] == "word"]
        self.assertEqual(words_off, ["gelecekmisin"])
        self.assertEqual(words_on, ["gelecek", "misin"])


if __name__ == "__main__":
    unittest.main()
