"""
test_tr_diacritics.py — Circumflex-insensitive matching.

Turkish loanwords carry circumflex vowels (â/î/û) that are routinely dropped
in everyday writing (mekân/mekan, resmî/resmi, kâr/kar, ilmî/ilmi). Both
spellings must resolve to the same lemma. Run with:

    python -m unittest test_tr_diacritics -v
"""

import unittest
from pathlib import Path

from tr_phonology    import fold_diacritics
from tr_lexicon      import load_lexicon
from tr_api          import Tokenizer


HERE = Path(__file__).parent


class TestFoldDiacritics(unittest.TestCase):

    def test_folds_circumflex_vowels(self):
        self.assertEqual(fold_diacritics("mekân"), "mekan")
        self.assertEqual(fold_diacritics("resmî"), "resmi")
        self.assertEqual(fold_diacritics("kâr"),   "kar")
        self.assertEqual(fold_diacritics("ilmî"),  "ilmi")

    def test_length_preserving(self):
        for w in ("mekân", "resmî", "kâğıt", "lâzım"):
            self.assertEqual(len(fold_diacritics(w)), len(w))

    def test_plain_text_unchanged(self):
        self.assertEqual(fold_diacritics("kitap"), "kitap")


class TestLexiconDiacriticInsensitive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lex = load_lexicon(HERE / "lexicon_full.json")

    def test_membership_either_spelling(self):
        # lexicon_full stores the circumflex form; both spellings must hit.
        self.assertIn("ilmî", self.lex)
        self.assertIn("ilmi", self.lex)
        self.assertIn("mekan", self.lex)
        self.assertIn("mekân", self.lex)

    def test_get_returns_same_roots(self):
        self.assertEqual(
            {r.form for r in self.lex.get("ilmi")},
            {r.form for r in self.lex.get("ilmî")},
        )


class TestTokenizerDiacritics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def root(self, word):
        return self.tok.tokenize(word)["root"]

    def test_diacritic_and_plain_match(self):
        for plain, circ in [("mekan", "mekân"), ("resmi", "resmî"),
                            ("ilmi", "ilmî"), ("kar", "kâr")]:
            with self.subTest(word=plain):
                self.assertEqual(self.root(plain), self.root(circ))

    def test_diacritic_word_is_in_lexicon(self):
        # 'ilmi' used to fall through to OOV (it was only stored as 'ilmî').
        self.assertFalse(self.tok.tokenize("ilmi")["oov"])

    def test_harmony_correct_on_inflected_loanword(self):
        # mekân + LOC: harmony must treat the folded 'a' as back -> -da.
        r = self.tok.tokenize("mekânda")
        self.assertEqual(r["root"], self.root("mekan"))
        self.assertEqual(r["split"], "mekan-da")

    def test_circumflex_word_is_one_token_in_text(self):
        # Circumflex letters are word characters, so a loanword like 'mekân'
        # is a single token in sentence mode (not split into mek + â + n).
        r = self.tok.tokenize_text("mekân çok güzeldi")
        words = [t["surface"] for t in r["tokens"] if t["kind"] == "word"]
        self.assertEqual(words, ["mekân", "çok", "güzeldi"])

    def test_recognised_loanword_not_split_as_clitic(self):
        # 'ilmi' must NOT be split into il + mi now that it resolves in-lex.
        r = self.tok.tokenize_text("ilmi")
        words = [t["surface"] for t in r["tokens"] if t["kind"] == "word"]
        self.assertEqual(words, ["ilmi"])


if __name__ == "__main__":
    unittest.main()
