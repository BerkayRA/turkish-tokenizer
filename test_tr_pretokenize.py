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

    def test_splits_particle_plus_agreement_over_clean_parse(self):
        """Particle + person/copular agreement is an unambiguous question and
        is split even when the whole word has a clean in-lex parse."""
        cases = {
            "hastamısın":  ["hasta", "mısın"],
            "hastamıyım":  ["hasta", "mıyım"],
            "zenginmiyiz": ["zengin", "miyiz"],
            "güzelmiydi":  ["güzel", "miydi"],
            "doktormudur": ["doktor", "mudur"],
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(self.split(word), expected)

    def test_splits_agreement_cluster_over_oov_stem(self):
        # Particle + agreement (mısınız) is unambiguous, so it splits even
        # when the stem carries an OOV root (Çekoslovakya is not in the
        # lexicon). The bare-particle path still requires an in-lex stem.
        canary = "çekoslovakyalılaştıramadıklarımızdanmısınız"
        self.assertEqual(
            self.split(canary),
            ["çekoslovakyalılaştıramadıklarımızdan", "mısınız"])

    def test_evidential_not_split(self):
        """-mış/-miş + agreement (okumuşsun) must NOT be read as a particle."""
        for word in ("okumuşsun", "görmüşsün", "tanımışım", "gelmişsin"):
            with self.subTest(word=word):
                self.assertEqual(self.split(word), [word])

    def test_possessive_lookalike_not_split(self):
        """particle+POSSESSIVE (mim = mi+POSS_1SG) is excluded from the
        agreement whitelist, so 'adamım' is never split into ada + mım."""
        for word in ("adamım", "kalemim", "selamım", "kalemin"):
            with self.subTest(word=word):
                self.assertEqual(self.split(word), [word])


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

    # --- single-word mode clitic splitting ---

    def test_word_mode_returns_segments(self):
        r = self.tok.tokenize("gelecekmisin")
        self.assertIn("segments", r)
        self.assertEqual([s["surface"] for s in r["segments"]],
                         ["gelecek", "misin"])
        self.assertEqual(r["surface"], "gelecekmisin")

    def test_word_mode_canary_splits_and_decomposes(self):
        r = self.tok.tokenize("çekoslovakyalılaştıramadıklarımızdanmısınız")
        self.assertIn("segments", r)
        surfaces = [s["surface"] for s in r["segments"]]
        self.assertEqual(surfaces,
                         ["çekoslovakyalılaştıramadıklarımızdan", "mısınız"])
        # The particle segment is rooted at the interrogative particle.
        self.assertEqual(r["segments"][1]["root"], "mı")

    def test_word_mode_split_off_is_flat(self):
        r = self.tok.tokenize("gelecekmisin", split_clitics=False)
        self.assertNotIn("segments", r)

    def test_word_mode_plain_word_is_flat(self):
        r = self.tok.tokenize("kitabımı")
        self.assertNotIn("segments", r)
        self.assertEqual(r["split"], "kitab-ım-ı")


if __name__ == "__main__":
    unittest.main()
