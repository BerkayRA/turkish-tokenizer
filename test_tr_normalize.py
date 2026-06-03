"""
test_tr_normalize.py — batch corpus normalizer output modes.

Run with:    python -m unittest test_tr_normalize -v
"""

import json
import unittest

from tr_api import Tokenizer, TokenizerConfig
import tr_normalize as N


class TestNormalizeModes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer(TokenizerConfig(
            suggest_on_oov=False, include_alternatives=False))

    def r(self, text):
        return self.tok.tokenize_text(
            text, suggest=False, tail_repair=False, alternatives=False)

    def test_surface_splits_attached_clitic(self):
        self.assertEqual(
            N.render_surface(self.r("Yarın gelecekmisin?"), False),
            "Yarın gelecek misin?")

    def test_surface_fold_diacritics(self):
        self.assertEqual(N.render_surface(self.r("mekân"), True), "mekan")
        self.assertEqual(N.render_surface(self.r("mekân"), False), "mekân")

    def test_lemma_mode(self):
        self.assertEqual(
            N.render_lemma(self.r("kitabımı okudum"), "▁"), "kitap oku")

    def test_morphemes_mode(self):
        self.assertEqual(
            N.render_morphemes(self.r("kitabımı"), "|"), "kitab|ım|ı")
        self.assertEqual(
            N.render_morphemes(self.r("gelecekmisin"), "|"),
            "gel|ecek mi|sin")

    def test_jsonl_mode(self):
        d = json.loads(N.render_jsonl("kitabımı", self.r("kitabımı")))
        self.assertEqual(d["text"], "kitabımı")
        self.assertEqual(d["tokens"][0]["lemma"], "kitap")
        self.assertEqual(d["tokens"][0]["pos"], "NOUN")


if __name__ == "__main__":
    unittest.main()
