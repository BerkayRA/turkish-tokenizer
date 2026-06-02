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


if __name__ == "__main__":
    unittest.main()
