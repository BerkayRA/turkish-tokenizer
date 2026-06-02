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


if __name__ == "__main__":
    unittest.main()
