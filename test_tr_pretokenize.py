"""
test_tr_pretokenize.py — Tests for attached-clitic pre-tokenization.

Covers the interrogative-particle splitter (gelecekmisin -> gelecek +
misin) and its conservative guards against splitting ordinary words that
merely end in -mi / -mu. Run with:

    python -m unittest test_tr_pretokenize -v
"""

import unittest
from pathlib import Path

from tr_inventory     import load_inventory
from tr_lexicon       import load_lexicon
from tr_morphotactics import load_graph
from tr_parse         import Parser
from tr_pretokenize   import split_question_clitic
from tr_api           import Tokenizer, TokenizerConfig


HERE = Path(__file__).parent


def build_parser():
    inv   = load_inventory(HERE / "inventory.json")
    graph = load_graph(HERE / "morphotactics.json")
    # The production-default lexicon (what tr_api.Tokenizer loads), so the
    # in-lex head/tail checks match real usage.
    lex   = load_lexicon(HERE / "lexicon_full.json")
    return Parser(lex, inv, graph)


class TestSplitQuestionClitic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = build_parser()

    def split(self, word):
        return split_question_clitic(word, self.parser)

    def test_splits_attached_particle(self):
        cases = {
            "gelecekmi":    ["gelecek", "mi"],
            "gelecekmisin": ["gelecek", "misin"],
            "geldimi":      ["geldi", "mi"],
            "güzelmi":      ["güzel", "mi"],
            "evdemi":       ["evde", "mi"],
            "okudunmu":     ["okudun", "mu"],
            "geliyormu":    ["geliyor", "mu"],
            "aldınmı":      ["aldın", "mı"],
            "kitapmı":      ["kitap", "mı"],
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(self.split(word), expected)

    def test_particle_vowel_harmony(self):
        """The split point respects harmony, not just the first 'm'."""
        # gelecek (front) -> mi, okudun (back rounded) -> mu, aldın -> mı
        self.assertEqual(self.split("gelecekmisin"), ["gelecek", "misin"])
        self.assertEqual(self.split("okudunmu"),     ["okudun", "mu"])

    def test_does_not_split_ordinary_words(self):
        """Words that parse cleanly in-lexicon are never split."""
        for word in ("resmi", "ölümü", "yarımı", "kalemi", "adamı",
                     "kremi", "filmi", "kitabımı", "geldim"):
            with self.subTest(word=word):
                self.assertEqual(self.split(word), [word])

    def test_short_words_not_split(self):
        for word in ("mi", "mu", "am", "ev"):
            with self.subTest(word=word):
                self.assertEqual(self.split(word), [word])

    def test_oov_head_not_split(self):
        """A junk head must not be split off even with a particle-shaped tail."""
        # 'zzz' is not a word; 'zzzmı' should stay whole.
        self.assertEqual(self.split("zzzmı"), ["zzzmı"])

    def test_in_lex_collision_left_intact(self):
        """Documented limitation: an attached reading that collides with a
        clean in-lex parse is left whole (needs sentence context)."""
        self.assertEqual(self.split("hastamısın"), ["hastamısın"])


class TestTokenizerClitics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_tokenize_text_splits_clitic(self):
        r = self.tok.tokenize_text("Yarın gelecekmisin?")
        words = [t["surface"] for t in r["tokens"] if t["kind"] == "word"]
        self.assertEqual(words, ["Yarın", "gelecek", "misin"])

    def test_reconstruction_preserved_after_split(self):
        text = "Sen de gelecekmisin?"
        r = self.tok.tokenize_text(text)
        self.assertEqual("".join(t["surface"] for t in r["tokens"]), text)

    def test_split_pieces_are_analysed(self):
        r = self.tok.tokenize_text("gelecekmisin")
        word_tokens = [t for t in r["tokens"] if t["kind"] == "word"]
        # The particle piece is rooted at the interrogative particle.
        particle = word_tokens[-1]["analysis"]
        self.assertEqual(particle["root"], "mi")

    def test_can_be_disabled(self):
        tok = Tokenizer(TokenizerConfig(split_clitics=False))
        r = tok.tokenize_text("gelecekmisin?")
        words = [t["surface"] for t in r["tokens"] if t["kind"] == "word"]
        self.assertEqual(words, ["gelecekmisin"])


if __name__ == "__main__":
    unittest.main()
