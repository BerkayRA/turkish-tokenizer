"""
test_tr_fertility.py — fertility / boundary-alignment metric.

Run with:    python -m unittest test_tr_fertility -v
"""

import unittest

import tr_fertility as F


class TestAdapters(unittest.TestCase):

    def test_whitespace(self):
        self.assertEqual(F.WhitespaceAdapter().encode("kitap"), ["kitap"])

    def test_char(self):
        self.assertEqual(F.CharAdapter().encode("kitap"), list("kitap"))


class TestBoundaryHelpers(unittest.TestCase):

    def test_clean_strips_markers(self):
        self.assertEqual(F._clean("▁kitap"), "kitap")
        self.assertEqual(F._clean("##lar"), "lar")

    def test_internal_boundaries(self):
        self.assertEqual(F._internal_boundaries(["ki", "tap"], "kitap"), {2})
        # marker-prefixed pieces still reconstruct
        self.assertEqual(F._internal_boundaries(["▁ki", "tap"], "kitap"), {2})
        # single piece -> no internal boundary
        self.assertEqual(F._internal_boundaries(["kitap"], "kitap"), set())

    def test_internal_boundaries_unreconstructible(self):
        # pieces that don't join back to the word -> cannot align
        self.assertIsNone(F._internal_boundaries(["xy", "z"], "kitap"))

    def test_morpheme_boundaries(self):
        analysis = {"morphemes": [{"chunk": "kitab"}, {"chunk": "ım"},
                                  {"chunk": "ı"}]}
        # kitab|ım|ı -> cuts at 5 and 7
        self.assertEqual(F._morpheme_boundaries(analysis), {5, 7})

    def test_score_corpus_reusable(self):
        from tr_api import Tokenizer, TokenizerConfig
        tok = Tokenizer(TokenizerConfig(
            suggest_on_oov=False, include_alternatives=False))
        # whitespace -> fertility 1.0, no internal boundaries
        ws = F.score_corpus(F.WhitespaceAdapter(), ["kitabımı okudum"], tok)
        self.assertEqual(ws["fertility"], 1.0)
        self.assertEqual(ws["words"], 2)
        # char -> captures every morpheme boundary (recall 1.0)
        ch = F.score_corpus(F.CharAdapter(), ["kitabımı okudum"], tok)
        self.assertEqual(ch["boundary_recall"], 1.0)
        self.assertGreater(ch["fertility"], 1.0)

    def test_char_boundaries_superset_of_morphemes(self):
        # The char tokenizer cuts at every position, so it must capture every
        # morpheme boundary (recall = 1.0 by construction).
        word = "kitabımı"
        char_b = F._internal_boundaries(list(word), word)
        morph_b = F._morpheme_boundaries(
            {"morphemes": [{"chunk": "kitab"}, {"chunk": "ım"}, {"chunk": "ı"}]})
        self.assertTrue(morph_b <= char_b)


if __name__ == "__main__":
    unittest.main()
