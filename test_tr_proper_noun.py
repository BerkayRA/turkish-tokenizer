"""
test_tr_proper_noun.py — Tests for proper-noun handling in the parser.

Covers the apostrophe boundary and title-case nudges wired into the
scorer (see ParseConfig.apostrophe_boundary_bonus / proper_noun_inlex_bonus
/ proper_noun_oov_bonus). Run with:

    python -m unittest test_tr_proper_noun -v
"""

import unittest
from pathlib import Path

from tr_inventory     import load_inventory
from tr_lexicon       import load_lexicon
from tr_morphotactics import load_graph
from tr_parse         import Parser, ParseConfig


HERE = Path(__file__).parent

CANARY = "muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine"


def build_parser(**kwargs):
    inv   = load_inventory(HERE / "inventory.json")
    graph = load_graph(HERE / "morphotactics.json")
    # Use the production-default lexicon (what tr_api.Tokenizer loads): it
    # contains the competing 'ha' root that makes the Hacı -> ha+cı vs hacı
    # contest real, so the title-case nudge is actually exercised.
    lex   = load_lexicon(HERE / "lexicon_full.json")
    cfg   = ParseConfig(**kwargs) if kwargs else None
    return Parser(lex, inv, graph, cfg)


class TestProperNoun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = build_parser()

    def top(self, word):
        analyses = self.parser.parse(word)
        self.assertTrue(analyses, f"no parse for {word!r}")
        return analyses[0]

    # --- apostrophe handling ---------------------------------------------

    def test_apostrophe_is_stripped(self):
        """The apostrophe is zero-width morphologically; the stem is found."""
        a = self.top("Muammer'in")
        self.assertEqual(a.root, "muammer")
        # No apostrophe survives into any morpheme chunk.
        self.assertNotIn("'", "".join(m.chunk for m in a.morphemes))

    def test_compound_proper_noun_possessive_before_apostrophe(self):
        """Parkı'ndan = park + POSS_3SG | ABL.

        The apostrophe boundary falls AFTER the embedded possessive, not at
        the root|suffix split. The bonus must not demote this correct
        reading down to a spurious 'parkı' root.
        """
        a = self.top("Parkı'ndan")
        self.assertEqual(a.root, "park")
        self.assertEqual(a.split(), "park-ı-ndan")

    def test_compound_proper_noun_kiraathane(self):
        """Kıraathanesi'nin = kıraathane + POSS_3SG | GEN."""
        a = self.top("Kıraathanesi'nin")
        self.assertEqual(a.root, "kıraathane")

    def test_apostrophe_boundary_aligns_with_a_morpheme_break(self):
        """The winning analysis places a morpheme boundary at the apostrophe."""
        # "Parkı'ndan": apostrophe after "parkı" (index 5). Cumulative
        # morpheme spans must hit 5 exactly.
        a = self.top("Parkı'ndan")
        cum, boundaries = 0, set()
        for m in a.morphemes:
            cum += len(m.chunk)
            boundaries.add(cum)
        self.assertIn(5, boundaries)

    # --- title-case in-lexicon nudge -------------------------------------

    def test_title_case_inlex_beats_spurious_derivation(self):
        """Hacı -> bare hacı, not the spurious ha+cı (NDER_CH)."""
        a = self.top("Hacı")
        self.assertEqual(a.root, "hacı")
        self.assertEqual(len(a.morphemes), 1)

    def test_inlex_bonus_stays_below_productivity(self):
        """A capitalized productive derivation still decomposes.

        proper_noun_inlex_bonus (1.0) < productivity_bonus (2.5), so
        Heyecanlı still parses as heyecan+ADJZ_LH rather than a bare lemma.
        """
        a = self.top("Heyecanlı")
        self.assertEqual(a.root, "heyecan")
        self.assertGreater(len(a.morphemes), 1)

    def test_inlex_bonus_is_title_case_gated(self):
        """The in-lex nudge only fires on title case.

        Lowercase 'hacı' keeps its (pre-existing) decomposed reading; only
        the capitalized form is nudged to the bare lemma. This guards that
        the bonus is conditioned on casing, not applied unconditionally.
        """
        cap   = self.top("Hacı")
        lower = self.top("hacı")
        self.assertEqual(cap.root, "hacı")
        self.assertNotEqual(lower.root, cap.root)

    # --- regression guards -----------------------------------------------

    def test_inflected_proper_noun_with_apostrophe(self):
        a = self.top("Kerem'in")
        self.assertEqual(a.root, "kerem")
        self.assertEqual(a.split(), "kerem-in")

    def test_canary_unaffected(self):
        """The 18-morpheme trick word must still decompose end-to-end."""
        a = self.top(CANARY)
        self.assertEqual(len(a.morphemes), 18)


class TestProperNounBonusIsolation(unittest.TestCase):
    """The bonuses are real dials: turning them off changes the ranking."""

    def test_inlex_bonus_off_changes_hacı(self):
        off = build_parser(proper_noun_inlex_bonus=0.0)
        a = off.parse("Hacı")[0]
        # Without the nudge, the spurious derivation wins again.
        self.assertNotEqual(a.root, "hacı")


if __name__ == "__main__":
    unittest.main()
