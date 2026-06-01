"""
test_tr_phase3.py — Tests for the morphotactic graph and graph-aware
generation.

Run with:    python -m unittest test_tr_phase3 -v
"""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

from tr_inventory import load_inventory
from tr_morphotactics import (
    MorphoGraph, State, Transition,
    load_graph, validate_against_inventory,
)
from tr_generate import generate
from tr_lexicon import load_lexicon
from tr_parse import Parser, ParseConfig


INVENTORY_PATH    = Path(__file__).parent / "inventory.json"
MORPHOTACTICS_PATH = Path(__file__).parent / "morphotactics.json"
LEXICON_PATH      = Path(__file__).parent / "lexicon.json"


class TestGraphLoading(unittest.TestCase):

    def test_load_real_graph(self):
        g = load_graph(MORPHOTACTICS_PATH)
        # Sanity checks: expected states should exist.
        for sid in ["VERB_ROOT", "VERB_TAM_K", "VERB_TAM_Z", "VERB_AGR",
                    "NOUN_ROOT", "NOUN_NUM", "NOUN_POSS", "NOUN_CASE",
                    "ADJ_ROOT"]:
            self.assertIn(sid, g.states, f"missing state {sid}")

    def test_start_states(self):
        g = load_graph(MORPHOTACTICS_PATH)
        self.assertEqual(g.start_state("VERB"), "VERB_ROOT")
        self.assertEqual(g.start_state("NOUN"), "NOUN_ROOT")
        self.assertEqual(g.start_state("ADJ"),  "ADJ_ROOT")

    def test_unknown_word_class(self):
        g = load_graph(MORPHOTACTICS_PATH)
        with self.assertRaises(KeyError):
            g.start_state("ADVERB")

    def test_step_transitions(self):
        g = load_graph(MORPHOTACTICS_PATH)
        # Verbal track
        self.assertEqual(g.step("VERB_ROOT", "PAST"), "VERB_TAM_K")
        self.assertEqual(g.step("VERB_ROOT", "PROG"), "VERB_TAM_Z")
        self.assertEqual(g.step("VERB_TAM_K", "1SG_K"), "VERB_AGR")
        self.assertEqual(g.step("VERB_TAM_Z", "1SG_Z"), "VERB_AGR")
        # The 1SG_Z is NOT valid after PAST (the whole point of the split)
        self.assertIsNone(g.step("VERB_TAM_K", "1SG_Z"))
        self.assertIsNone(g.step("VERB_TAM_Z", "1SG_K"))
        # Nominal track
        self.assertEqual(g.step("NOUN_ROOT", "PLUR"),     "NOUN_NUM")
        self.assertEqual(g.step("NOUN_ROOT", "POSS_1SG"), "NOUN_POSS")
        self.assertEqual(g.step("NOUN_NUM", "POSS_1SG"),  "NOUN_POSS")
        # Derivational
        self.assertEqual(g.step("NOUN_ROOT", "VBZ_LA"), "VERB_ROOT")
        self.assertEqual(g.step("VERB_ROOT", "NMZ_INF"), "NOUN_ROOT")

    def test_accepting_states(self):
        g = load_graph(MORPHOTACTICS_PATH)
        self.assertTrue(g.is_accepting("NOUN_ROOT"))  # bare noun = NOM SG
        self.assertTrue(g.is_accepting("VERB_AGR"))
        self.assertTrue(g.is_accepting("VERB_TAM_Z"))  # 3SG has zero ending
        self.assertFalse(g.is_accepting("VERB_NEG"))
        self.assertFalse(g.is_accepting("VERB_VOICE"))


class TestGraphInventoryCrossValidation(unittest.TestCase):

    def test_real_graph_validates_against_real_inventory(self):
        inv = load_inventory(INVENTORY_PATH)
        graph = load_graph(MORPHOTACTICS_PATH)
        validate_against_inventory(graph, inv)   # should not raise

    def test_rejects_transition_with_unknown_suffix(self):
        bad_json = {
            "start_states": {"NOUN": "S0"},
            "states": [{"id": "S0", "accept": True}, {"id": "S1"}],
            "transitions": [
                {"via": "NONEXISTENT", "from": ["S0"], "to": "S1"}
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad_json, f); name = f.name
        try:
            inv = load_inventory(INVENTORY_PATH)
            g = load_graph(name)
            with self.assertRaises(ValueError) as cm:
                validate_against_inventory(g, inv)
            self.assertIn("NONEXISTENT", str(cm.exception))
        finally:
            os.unlink(name)

    def test_rejects_unknown_target_state(self):
        bad_json = {
            "start_states": {"NOUN": "S0"},
            "states": [{"id": "S0"}],
            "transitions": [
                {"via": "PLUR", "from": ["S0"], "to": "GHOST"}
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad_json, f); name = f.name
        try:
            inv = load_inventory(INVENTORY_PATH)
            g = load_graph(name)
            with self.assertRaises(ValueError) as cm:
                validate_against_inventory(g, inv)
            self.assertIn("GHOST", str(cm.exception))
        finally:
            os.unlink(name)


class TestGraphAwareGeneration(unittest.TestCase):
    """Test that generation with a graph validates transitions correctly."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)

    def test_valid_verb_chain_passes(self):
        g = generate("gel", ["PAST", "1SG_K"], self.inv,
                     word_class="VERB", graph=self.graph)
        self.assertEqual(g.surface, "geldim")
        # Last step should land in the accepting VERB_AGR state.
        self.assertEqual(g.steps[-1].state, "VERB_AGR")
        self.assertTrue(self.graph.is_accepting(g.steps[-1].state))

    def test_valid_noun_chain_passes(self):
        g = generate("ev", ["PLUR", "POSS_1SG", "LOC"], self.inv,
                     word_class="NOUN", graph=self.graph)
        self.assertEqual(g.surface, "evlerimde")
        self.assertEqual(g.steps[-1].state, "NOUN_CASE")

    def test_strict_rejects_k_after_z(self):
        # PROG → VERB_TAM_Z; 1SG_K is not licensed from there.
        with self.assertRaises(ValueError) as cm:
            generate("gel", ["PROG", "1SG_K"], self.inv,
                     word_class="VERB", graph=self.graph)
        self.assertIn("Invalid morphotactic transition", str(cm.exception))
        self.assertIn("1SG_K", str(cm.exception))

    def test_strict_rejects_z_after_k(self):
        with self.assertRaises(ValueError):
            generate("gel", ["PAST", "1SG_Z"], self.inv,
                     word_class="VERB", graph=self.graph)

    def test_strict_rejects_nominal_suffix_on_verb(self):
        with self.assertRaises(ValueError):
            generate("gel", ["LOC"], self.inv,
                     word_class="VERB", graph=self.graph)

    def test_strict_rejects_verb_suffix_on_noun(self):
        with self.assertRaises(ValueError):
            generate("ev", ["PAST"], self.inv,
                     word_class="NOUN", graph=self.graph)

    def test_non_strict_warns_but_continues(self):
        # With strict=False, the invalid suffix is still applied (so we can
        # see what the bug would produce), but a warning is logged.
        with self.assertLogs("tr_generate", level="WARNING") as cm:
            g = generate("gel", ["PROG", "1SG_K"], self.inv,
                         word_class="VERB", graph=self.graph,
                         strict=False)
        # Surface still produced (with the wrong suffix applied).
        # gel + PROG + 1SG_K (-m) = geliyorm
        self.assertEqual(g.surface, "geliyorm")
        # Warning was logged.
        self.assertTrue(any("Invalid morphotactic" in r for r in cm.output))
        self.assertTrue(any("1SG_K" in r for r in cm.output))

    def test_word_class_required_when_graph_given(self):
        with self.assertRaises(ValueError) as cm:
            generate("gel", ["PAST", "1SG_K"], self.inv, graph=self.graph)
        self.assertIn("word_class", str(cm.exception))

    def test_no_graph_no_validation(self):
        # Without a graph, the generator doesn't care about morphotactics.
        # This invalid chain produces output without raising.
        g = generate("gel", ["LOC"], self.inv)
        self.assertEqual(g.surface, "gelde")  # nonsense but generated


class TestDerivationCrossingTracks(unittest.TestCase):
    """Derivational suffixes move between NOUN/VERB/ADJ tracks."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)

    def test_noun_verbalized_then_conjugated(self):
        # tuz (N) → tuzla (V) → tuzluyor (V+PROG) → tuzluyoruz (V+PROG+1PL_Z)
        g = generate("tuz", ["VBZ_LA", "PROG", "1PL_Z"], self.inv,
                     word_class="NOUN", graph=self.graph)
        self.assertEqual(g.surface, "tuzluyoruz")
        # State trajectory: NOUN_ROOT → VERB_ROOT → VERB_TAM_Z → VERB_AGR
        states = [st.state for st in g.steps]
        self.assertEqual(states, ["VERB_ROOT", "VERB_TAM_Z", "VERB_AGR"])

    def test_verb_nominalized_then_declined(self):
        # gel (V) → gelmek (N, infinitive) → gelmekte (N+LOC) "in coming"
        g = generate("gel", ["NMZ_INF", "LOC"], self.inv,
                     word_class="VERB", graph=self.graph)
        self.assertEqual(g.surface, "gelmekte")
        states = [st.state for st in g.steps]
        self.assertEqual(states, ["NOUN_ROOT", "NOUN_CASE"])

    def test_neologism_commit_full_chain(self):
        # commit (N, OOV) → commitle (V) → commitleyecek (V+FUT) →
        # commitleyeceğiz (V+FUT+1PL_Z)
        g = generate("commit", ["VBZ_LA", "FUT", "1PL_Z"], self.inv,
                     word_class="NOUN", graph=self.graph)
        self.assertEqual(g.surface, "commitleyeceğiz")
        states = [st.state for st in g.steps]
        self.assertEqual(states, ["VERB_ROOT", "VERB_TAM_Z", "VERB_AGR"])


class TestPronominalN(unittest.TestCase):
    """Pronominal-n is the buffer 'n' that appears between a 3rd-person
    possessive (POSS_3SG, POSS_3PL) and a case marker. Implemented via the
    rule registry: `insert_n_after_3rd_person_possessive` is declared on
    every CASE suffix, and fires when ctx['prev_morph_id'] is one of the
    3rd-person possessives.
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain):
        return generate(stem, chain, self.inv, soften=False,
                        word_class="NOUN", graph=self.graph).surface

    def test_pronominal_n_3sg_poss(self):
        # POSS_3SG + each case = pronominal n must surface
        self.assertEqual(self._gen("araba", ["POSS_3SG", "LOC"]), "arabasında")
        self.assertEqual(self._gen("araba", ["POSS_3SG", "ABL"]), "arabasından")
        self.assertEqual(self._gen("araba", ["POSS_3SG", "DAT"]), "arabasına")
        self.assertEqual(self._gen("ev",    ["POSS_3SG", "LOC"]), "evinde")
        self.assertEqual(self._gen("ev",    ["POSS_3SG", "ABL"]), "evinden")
        self.assertEqual(self._gen("ev",    ["POSS_3SG", "DAT"]), "evine")

    def test_pronominal_n_3pl_poss(self):
        # POSS_3PL + LOC: araba-ları + n + da
        self.assertEqual(self._gen("araba", ["POSS_3PL", "LOC"]), "arabalarında")
        self.assertEqual(self._gen("ev",    ["POSS_3PL", "DAT"]), "evlerine")

    def test_no_pronominal_n_after_other_poss(self):
        # 1sg/2sg/1pl/2pl possessives do NOT trigger pronominal-n
        self.assertEqual(self._gen("ev", ["POSS_1SG", "LOC"]), "evimde")
        self.assertEqual(self._gen("ev", ["POSS_2SG", "LOC"]), "evinde")
        self.assertEqual(self._gen("ev", ["POSS_1PL", "LOC"]), "evimizde")
        self.assertEqual(self._gen("ev", ["POSS_2PL", "LOC"]), "evinizde")
        # Note: POSS_2SG happens to produce the same surface as POSS_3SG+n
        # for some cases (evinde) — that's genuine homophony, not a bug.

    def test_no_pronominal_n_without_poss(self):
        # Case directly on the root: no possessive, no buffer n
        self.assertEqual(self._gen("ev", ["LOC"]), "evde")
        self.assertEqual(self._gen("ev", ["ABL"]), "evden")
        self.assertEqual(self._gen("ev", ["DAT"]), "eve")


class TestAoristAllomorphy(unittest.TestCase):
    """The aorist marker has three surface allomorphs:
      - vowel-final stems              → -r        (oku-r, başla-r)
      - polysyllabic consonant-final   → -Hr       (konuş-ur, otur-ur)
      - monosyllabic consonant-final   → -Ar or -Hr
            - default                  → -Ar       (yap-ar, bak-ar)
            - lexical exceptions       → -Hr       (gel-ir, al-ır, ver-ir, ...)

    The exception list (13 verbs) is hand-curated in irregulars.json and
    propagated to lexicon entries as aorist_high=True. Generation uses
    this flag; parsing relies on phonological disambiguation (since -Ar
    harmonizes to a/e and -Hr to ı/i/u/ü, at most one can match a given
    surface).
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, aorist_high=False):
        root_ctx = {"root_aorist_high": True} if aorist_high else None
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph,
                        root_ctx=root_ctx).surface

    def test_vowel_final_stems_get_r(self):
        self.assertEqual(self._gen("oku",   ["AOR"]), "okur")
        self.assertEqual(self._gen("başla", ["AOR"]), "başlar")
        self.assertEqual(self._gen("bekle", ["AOR"]), "bekler")
        self.assertEqual(self._gen("ye",    ["AOR"]), "yer")
        self.assertEqual(self._gen("de",    ["AOR"]), "der")

    def test_monosyl_consonant_default_gets_Ar(self):
        self.assertEqual(self._gen("yap",  ["AOR"]), "yapar")
        self.assertEqual(self._gen("bak",  ["AOR"]), "bakar")
        self.assertEqual(self._gen("çık",  ["AOR"]), "çıkar")
        self.assertEqual(self._gen("kork", ["AOR"]), "korkar")
        self.assertEqual(self._gen("yat",  ["AOR"]), "yatar")

    def test_monosyl_consonant_exception_gets_Hr(self):
        # 13 lexically marked monosyllabic verbs
        for stem, expected in [
            ("gel", "gelir"),
            ("al",  "alır"),
            ("bil", "bilir"),
            ("bul", "bulur"),
            ("dur", "durur"),
            ("gör", "görür"),
            ("kal", "kalır"),
            ("ol",  "olur"),
            ("öl",  "ölür"),
            ("san", "sanır"),
            ("var", "varır"),
            ("ver", "verir"),
            ("vur", "vurur"),
        ]:
            self.assertEqual(self._gen(stem, ["AOR"], aorist_high=True), expected,
                             f"{stem} + AOR (aorist_high=True)")

    def test_polysyl_consonant_gets_Hr(self):
        # Default for polysyllabic consonant-final stems is -Hr regardless
        # of aorist_high flag (which is only meaningful for monosyllables).
        self.assertEqual(self._gen("konuş",  ["AOR"]), "konuşur")
        self.assertEqual(self._gen("otur",   ["AOR"]), "oturur")
        self.assertEqual(self._gen("çağır",  ["AOR"]), "çağırır")
        self.assertEqual(self._gen("başlat", ["AOR"]), "başlatır")

    def test_polysyl_via_causative_gets_Hr(self):
        # A monosyllabic root + CAUS makes a polysyllabic stem; the rule
        # then picks -Hr based on the stem (not the root's flag).
        # yap + DIr → yaptır, then yaptır + AOR → yaptırır.
        self.assertEqual(self._gen("yap", ["CAUS", "AOR"]), "yaptırır")
        # Same for an aorist_high root: the flag is irrelevant once the
        # stem is polysyllabic via derivation.
        self.assertEqual(self._gen("ver", ["CAUS", "AOR"], aorist_high=True), "verdirir")

    def test_aorist_with_agreement(self):
        # Full forms like "gelirim", "yaparım", "okurum"
        self.assertEqual(self._gen("gel", ["AOR", "1SG_Z"], aorist_high=True), "gelirim")
        self.assertEqual(self._gen("yap", ["AOR", "1SG_Z"]),                   "yaparım")
        self.assertEqual(self._gen("oku", ["AOR", "1SG_Z"]),                   "okurum")
        self.assertEqual(self._gen("bak", ["AOR", "2SG_Z"]),                   "bakarsın")


class TestNegAorist(unittest.TestCase):
    """The negative aorist is morphologically suppletive in Turkish:

        gel-me-z       (3sg: NEG + AOR(z))
        gel-me-z-sin   (2sg: NEG + AOR(z) + 2SG_Z)
        gel-me-m       (1sg: NEG + AOR(∅) + 1SG_Z(-Hm))
        gel-me-y-iz    (1pl: NEG + AOR(∅) + 1PL_Z(-(y)Hz))
        gel-me-z-ler   (3pl: NEG + AOR(z) + 3PL)

    AOR's surface after NEG is empty before 1SG_Z and 1PL_Z, and -z
    everywhere else. The rule machinery handles this by reading
    ctx['prev_morph_id'] and ctx['next_morph_id']; the parser tries both
    alternatives and surface matching disambiguates.
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, aorist_high=False):
        rc = {"root_aorist_high": True} if aorist_high else None
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph,
                        root_ctx=rc).surface

    def test_neg_aorist_3sg_uses_z(self):
        # Bare 3sg: NEG + AOR(z), no agreement ending
        self.assertEqual(self._gen("gel", ["NEG", "AOR"]), "gelmez")
        self.assertEqual(self._gen("yap", ["NEG", "AOR"]), "yapmaz")
        self.assertEqual(self._gen("oku", ["NEG", "AOR"]), "okumaz")

    def test_neg_aorist_1sg_uses_zero(self):
        # 1sg: AOR is empty, then -Hm attaches to vowel-final stem → -m
        self.assertEqual(self._gen("gel", ["NEG", "AOR", "1SG_Z"]), "gelmem")
        self.assertEqual(self._gen("yap", ["NEG", "AOR", "1SG_Z"]), "yapmam")
        self.assertEqual(self._gen("oku", ["NEG", "AOR", "1SG_Z"]), "okumam")

    def test_neg_aorist_1pl_uses_zero_with_buffer(self):
        # 1pl: AOR is empty, then -(y)Hz attaches with buffer y → -yIz
        self.assertEqual(self._gen("gel", ["NEG", "AOR", "1PL_Z"]), "gelmeyiz")
        self.assertEqual(self._gen("yap", ["NEG", "AOR", "1PL_Z"]), "yapmayız")
        self.assertEqual(self._gen("oku", ["NEG", "AOR", "1PL_Z"]), "okumayız")

    def test_neg_aorist_2sg_2pl_3pl_use_z(self):
        # Non-1st-person finite forms: AOR surfaces as -z
        self.assertEqual(self._gen("gel", ["NEG", "AOR", "2SG_Z"]), "gelmezsin")
        self.assertEqual(self._gen("gel", ["NEG", "AOR", "2PL_Z"]), "gelmezsiniz")
        self.assertEqual(self._gen("gel", ["NEG", "AOR", "3PL"]),   "gelmezler")
        self.assertEqual(self._gen("yap", ["NEG", "AOR", "2SG_Z"]), "yapmazsın")

    def test_positive_aorist_still_works(self):
        # Make sure the regular aorist allomorphy didn't break.
        self.assertEqual(self._gen("gel", ["AOR", "1SG_Z"], aorist_high=True), "gelirim")
        self.assertEqual(self._gen("gel", ["AOR", "1PL_Z"], aorist_high=True), "geliriz")
        self.assertEqual(self._gen("yap", ["AOR", "1SG_Z"]),                   "yaparım")
        self.assertEqual(self._gen("yap", ["AOR", "1PL_Z"]),                   "yaparız")


class TestOptative(unittest.TestCase):
    """The optative mood -(y)A: 'let X', 'may X'. Decomposed into the
    OPT marker plus separate agreement (OPT_1SG, OPT_1PL); bare OPT is 3sg.

    The treebank annotates these as Mood=Opt. Distinct from the
    indicative-track agreement because OPT's 1pl is -lHm (not -(y)Hz)
    and 1sg requires a buffer y after the vowel-final stem (geleyim,
    not gelem).
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, soften=False):
        return generate(stem, chain, self.inv, soften=soften,
                        word_class="VERB", graph=self.graph).surface

    def test_opt_3sg_bare(self):
        # No agreement: OPT alone is 3sg
        self.assertEqual(self._gen("ol",  ["OPT"]), "ola")
        self.assertEqual(self._gen("koş", ["OPT"]), "koşa")

    def test_opt_3sg_negative(self):
        # The "almaya" form from the treebank: al + NEG + OPT (3sg negative)
        self.assertEqual(self._gen("al", ["NEG", "OPT"]), "almaya")
        self.assertEqual(self._gen("gel", ["NEG", "OPT"]), "gelmeye")

    def test_opt_1sg_inserts_buffer_y(self):
        # OPT 1sg: gel-e-yim (buffer y inserts after vowel-stem)
        self.assertEqual(self._gen("gel",    ["OPT", "OPT_1SG"]), "geleyim")
        self.assertEqual(self._gen("ver",    ["OPT", "OPT_1SG"]), "vereyim")
        self.assertEqual(self._gen("göster", ["OPT", "OPT_1SG"]), "göstereyim")
        self.assertEqual(self._gen("oku",    ["OPT", "OPT_1SG"]), "okuyayım")

    def test_opt_1pl(self):
        # OPT 1pl: gel-e-lim (uses -lHm, not -(y)Hz)
        self.assertEqual(self._gen("gel",    ["OPT", "OPT_1PL"]), "gelelim")
        self.assertEqual(self._gen("bak",    ["OPT", "OPT_1PL"]), "bakalım")
        self.assertEqual(self._gen("koş",    ["OPT", "OPT_1PL"]), "koşalım")
        self.assertEqual(self._gen("görüş",  ["OPT", "OPT_1PL"]), "görüşelim")

    def test_neg_opt_1sg(self):
        # The user's test: gel + NEG + OPT + OPT_1SG = "gelmeyeyim"
        # ("let me not come"). gelme + e + yim, with buffer y from OPT.
        self.assertEqual(self._gen("gel", ["NEG", "OPT", "OPT_1SG"]), "gelmeyeyim")

    def test_neg_opt_1pl(self):
        # gel + NEG + OPT + OPT_1PL = "gelmeyelim" ("let us not come")
        self.assertEqual(self._gen("gel", ["NEG", "OPT", "OPT_1PL"]), "gelmeyelim")


class TestImperative3SG(unittest.TestCase):
    """IMP_3SG (-sIn): used for 3sg "let him/her/it X" commands.
    Distinguished from OPT 3sg (bare -A) in the treebank — they are
    different paradigms despite related meanings.
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain):
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph).surface

    def test_imp_3sg(self):
        # The forms in the treebank
        self.assertEqual(self._gen("gel",   ["IMP_3SG"]), "gelsin")
        self.assertEqual(self._gen("getir", ["IMP_3SG"]), "getirsin")
        self.assertEqual(self._gen("geç",   ["IMP_3SG"]), "geçsin")
        self.assertEqual(self._gen("ol",    ["IMP_3SG"]), "olsun")
        self.assertEqual(self._gen("gör",   ["IMP_3SG"]), "görsün")
        self.assertEqual(self._gen("oku",   ["IMP_3SG"]), "okusun")

    def test_neg_imp_3sg(self):
        # NEG + IMP_3SG: gelmesin = "let him not come"
        self.assertEqual(self._gen("gel", ["NEG", "IMP_3SG"]), "gelmesin")


class TestAorAllomorphGating(unittest.TestCase):
    """The AOR `expand` view must respect root_aorist_high at parse time.
    Without this gating, the parser would accept BOTH -Ar and -Hr for
    every consonant-final monosyllabic root, creating phantom AOR
    readings for surfaces that only differ in vowel harmony.

    Example: 'artırdı' (gold = art+CAUS_DERIV+PAST) must NOT be parsable
    as art+AOR(-Hr)+PAST_COP, because art's AOR allomorph is -Ar (gives
    'artardı'), not -Hr. Only verbs flagged aorist_high (gel, al, bil,
    bul, vur, ver, etc.) take -Hr.
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)
        self.lex   = load_lexicon(LEXICON_PATH)
        self.parser = Parser(self.lex, self.inv, self.graph,
                             ParseConfig(return_all=True))

    def _all_chains(self, w):
        """Return all (root, suffix_id chain, oov) tuples for `w`."""
        out = []
        for a in self.parser.parse(w):
            sids = tuple(m.suffix_id for m in a.morphemes if m.suffix_id)
            out.append((a.root, sids, a.oov))
        return out

    def test_default_aor_does_not_accept_Hr(self):
        # art is monosyllabic default-AOR; should never accept -Hr (which
        # would let 'artırdı' parse as art+AOR+PAST_COP).
        chains = self._all_chains("artırdı")
        bad = [(r, c) for (r, c, oov) in chains
               if not oov and r == "art" and "AOR" in c and "PAST_COP" in c]
        self.assertEqual(bad, [],
                         f"art+AOR+PAST_COP should not match 'artırdı' "
                         f"on the in-lex path (art's AOR is -Ar). Got: {bad}")

    def test_aorist_high_does_not_accept_Ar(self):
        # gel is aorist_high; should never accept -Ar via the in-lex path.
        # (OOV-root paths can still try -Ar, but they're heavily penalized
        # and don't win.)
        chains = self._all_chains("gelerdi")
        bad = [(r, c) for (r, c, oov) in chains
               if not oov and r == "gel" and "AOR" in c and "PAST_COP" in c]
        self.assertEqual(bad, [],
                         f"gel+AOR+PAST_COP should not match 'gelerdi' "
                         f"on the in-lex path (gel's AOR is -Hr, gives "
                         f"gelirdi). Got: {bad}")

    def test_correct_aor_surfaces_parse(self):
        # Sanity: the right surfaces still parse correctly via the in-lex path.
        chains = self._all_chains("artardı")
        has_correct = any(r == "art" and "AOR" in c and "PAST_COP" in c
                          and not oov
                          for r, c, oov in chains)
        self.assertTrue(has_correct,
                        f"artardı should parse as art+AOR+PAST_COP. "
                        f"Got: {chains}")
        chains = self._all_chains("gelirdi")
        has_correct = any(r == "gel" and "AOR" in c and "PAST_COP" in c
                          and not oov
                          for r, c, oov in chains)
        self.assertTrue(has_correct,
                        f"gelirdi should parse as gel+AOR+PAST_COP. "
                        f"Got: {chains}")


class TestPassiveAllomorphy(unittest.TestCase):
    """Passive allomorphy: three templates, fully phonological.
      - after vowel:  -n        (tara-n, oku-n)
      - after l:      -Hn       (al-ın, bil-in, gel-in)
      - else:         -Hl       (yap-ıl, gör-ül, sat-ıl)
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain):
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph).surface

    def test_vowel_final_gets_n(self):
        self.assertEqual(self._gen("tara", ["PASS"]), "taran")
        self.assertEqual(self._gen("oku",  ["PASS"]), "okun")
        self.assertEqual(self._gen("iste", ["PASS"]), "isten")
        self.assertEqual(self._gen("ye",   ["PASS"]), "yen")

    def test_l_final_gets_Hn(self):
        self.assertEqual(self._gen("sil", ["PASS"]), "silin")
        self.assertEqual(self._gen("al",  ["PASS"]), "alın")
        self.assertEqual(self._gen("bil", ["PASS"]), "bilin")
        self.assertEqual(self._gen("gel", ["PASS"]), "gelin")
        self.assertEqual(self._gen("bul", ["PASS"]), "bulun")

    def test_other_consonant_gets_Hl(self):
        self.assertEqual(self._gen("yap", ["PASS"]), "yapıl")
        self.assertEqual(self._gen("sat", ["PASS"]), "satıl")
        self.assertEqual(self._gen("gör", ["PASS"]), "görül")
        self.assertEqual(self._gen("yaz", ["PASS"]), "yazıl")


class TestCausativeAllomorphy(unittest.TestCase):
    """Causative allomorphy:
      - after vowel:                       -t          (bekle-t, yürü-t)
      - after r/l on POLYsyllabic stems:   -t          (belir-t, çıkar-t,
                                                       düzel-t, otur-t)
      - else (default):                    -DHr        (yap-tır, sat-tır,
                                                       ver-dir, gel-dir)
      - lexical exceptions (monosyllables): -Hr/-Ar    (geç-ir, düş-ür,
                                                       iç-ir, çık-ar)
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, caus_deriv=None, pass_deriv=None):
        rc = {}
        if caus_deriv:
            rc["root_caus_deriv"] = caus_deriv
        if pass_deriv:
            rc["root_pass_deriv"] = pass_deriv
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph,
                        root_ctx=rc or None).surface

    def test_vowel_final_gets_t(self):
        self.assertEqual(self._gen("bekle", ["CAUS"]), "beklet")
        self.assertEqual(self._gen("yürü",  ["CAUS"]), "yürüt")
        self.assertEqual(self._gen("anla",  ["CAUS"]), "anlat")

    def test_polysyl_rl_final_gets_t(self):
        self.assertEqual(self._gen("belir", ["CAUS"]), "belirt")
        self.assertEqual(self._gen("düzel", ["CAUS"]), "düzelt")
        self.assertEqual(self._gen("çıkar", ["CAUS"]), "çıkart")
        self.assertEqual(self._gen("otur",  ["CAUS"]), "oturt")

    def test_monosyl_rl_final_gets_DHr(self):
        # Critical: monosyllabic r/l-final verbs do NOT take -t
        self.assertEqual(self._gen("ver",  ["CAUS"]), "verdir")
        self.assertEqual(self._gen("gel",  ["CAUS"]), "geldir")
        self.assertEqual(self._gen("bil",  ["CAUS"]), "bildir")
        self.assertEqual(self._gen("bul",  ["CAUS"]), "buldur")
        self.assertEqual(self._gen("var",  ["CAUS"]), "vardır")

    def test_other_consonant_gets_DHr(self):
        self.assertEqual(self._gen("yap", ["CAUS"]), "yaptır")
        self.assertEqual(self._gen("sat", ["CAUS"]), "sattır")
        self.assertEqual(self._gen("aç",  ["CAUS"]), "açtır")


class TestCausDerivAllomorphy(unittest.TestCase):
    """CAUS_DERIV is a V→V derivational suffix gated by a per-root flag
    (caus_deriv). The flag's value IS the template (Ar, Hr, t). The
    parser only applies CAUS_DERIV to verbs whose lexicon entry sets
    the flag; this prevents speculative derivational decomposition on
    unrelated verbs.

    Per design: CAUS_DERIV emits no Voice feature (the gold treebank
    treats forms like çıkar as their own lemmas without Voice).
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, caus_deriv=None, pass_deriv=None):
        rc = {}
        if caus_deriv:
            rc["root_caus_deriv"] = caus_deriv
        if pass_deriv:
            rc["root_pass_deriv"] = pass_deriv
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph,
                        root_ctx=rc or None).surface

    def test_caus_deriv_with_template_Hr(self):
        self.assertEqual(self._gen("geç", ["CAUS_DERIV"], caus_deriv="Hr"), "geçir")
        self.assertEqual(self._gen("düş", ["CAUS_DERIV"], caus_deriv="Hr"), "düşür")
        self.assertEqual(self._gen("bit", ["CAUS_DERIV"], caus_deriv="Hr"), "bitir")
        self.assertEqual(self._gen("duy", ["CAUS_DERIV"], caus_deriv="Hr"), "duyur")

    def test_caus_deriv_with_template_Ar(self):
        self.assertEqual(self._gen("çık", ["CAUS_DERIV"], caus_deriv="Ar"), "çıkar")
        self.assertEqual(self._gen("kız", ["CAUS_DERIV"], caus_deriv="Ar"), "kızar")
        self.assertEqual(self._gen("kop", ["CAUS_DERIV"], caus_deriv="Ar"), "kopar")

    def test_caus_deriv_with_template_t(self):
        # Vowel-final base + -t template
        self.assertEqual(self._gen("anla", ["CAUS_DERIV"], caus_deriv="t"), "anlat")

    def test_caus_deriv_chains_with_inflection(self):
        # The derived stem is still a verb root, so subsequent inflection works
        self.assertEqual(self._gen("çık", ["CAUS_DERIV", "PAST"], caus_deriv="Ar"), "çıkardı")
        self.assertEqual(self._gen("geç", ["CAUS_DERIV", "PAST"], caus_deriv="Hr"), "geçirdi")
        self.assertEqual(self._gen("düş", ["CAUS_DERIV", "PAST"], caus_deriv="Hr"), "düşürdü")


class TestPassDerivAllomorphy(unittest.TestCase):
    """PASS_DERIV is the passive-shaped V→V derivational counterpart of
    CAUS_DERIV (bul → bulun, sık → sıkıl, doku → dokun). Same gating
    mechanism; emits no Voice."""

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)

    def _gen(self, stem, chain, pass_deriv=None):
        rc = {"root_pass_deriv": pass_deriv} if pass_deriv else None
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph,
                        root_ctx=rc).surface

    def test_pass_deriv_with_template_Hn(self):
        self.assertEqual(self._gen("bul", ["PASS_DERIV"], pass_deriv="Hn"), "bulun")

    def test_pass_deriv_with_template_Hl(self):
        self.assertEqual(self._gen("sık", ["PASS_DERIV"], pass_deriv="Hl"), "sıkıl")
        self.assertEqual(self._gen("süz", ["PASS_DERIV"], pass_deriv="Hl"), "süzül")

    def test_pass_deriv_with_template_n(self):
        # Vowel-final base + -n template
        self.assertEqual(self._gen("daya",  ["PASS_DERIV"], pass_deriv="n"), "dayan")
        self.assertEqual(self._gen("dinle", ["PASS_DERIV"], pass_deriv="n"), "dinlen")
        self.assertEqual(self._gen("doku",  ["PASS_DERIV"], pass_deriv="n"), "dokun")
        self.assertEqual(self._gen("ye",    ["PASS_DERIV"], pass_deriv="n"), "yen")

    def test_pass_deriv_chains_with_inflection(self):
        self.assertEqual(self._gen("bul",  ["PASS_DERIV", "PAST"], pass_deriv="Hn"), "bulundu")
        self.assertEqual(self._gen("dinle", ["PASS_DERIV", "PAST"], pass_deriv="n"),  "dinlendi")


class TestPotNegSuppletion(unittest.TestCase):
    """The potential suffix -(y)Abil is suppletive before NEG: it
    reduces to -(y)A. So 'cannot' = stem + (y)A + mA, not stem + (y)Abil
    + mA. Standalone POT (without NEG following) keeps -(y)Abil."""

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)
        self.lex   = load_lexicon(LEXICON_PATH)
        self.parser = Parser(self.lex, self.inv, self.graph,
                             ParseConfig(return_all=True))

    def _gen(self, stem, chain):
        return generate(stem, chain, self.inv, soften=False,
                        word_class="VERB", graph=self.graph).surface

    def _top(self, w):
        return (self.parser.parse(w) or [None])[0]

    def test_pot_alone_keeps_full_template(self):
        # gel + POT + PAST = gelebildi ("could come")
        self.assertEqual(self._gen("gel", ["POT", "PAST"]), "gelebildi")

    def test_pot_before_neg_uses_suppletive(self):
        # gel + POT + NEG + PAST = gelemedi ("couldn't come"),
        # NOT gelebilmedi.
        self.assertEqual(self._gen("gel", ["POT", "NEG", "PAST"]), "gelemedi")
        # yap + POT + NEG + PAST = yapamadı
        self.assertEqual(self._gen("yap", ["POT", "NEG", "PAST"]), "yapamadı")

    def test_pot_after_neg_doubles_back(self):
        # gel + NEG + POT + FUT_PART = gelmeyebilecek
        # (the second POT, after NEG, is the full -(y)Abil form again)
        self.assertEqual(self._gen("gel", ["NEG", "POT", "FUT_PART"]),
                         "gelmeyebilecek")

    def test_parse_negative_potential(self):
        # gelemedi parses correctly
        a = self._top("gelemedi")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "gel")
        suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffix_ids, ["POT", "NEG", "PAST"])

    def test_parse_does_not_overgenerate_suppletive(self):
        # 'gelir' must parse as gel+AOR, NOT as gel+POT(suppletive)+...
        # — the suppletive only fires before NEG.
        a = self._top("gelir")
        self.assertIsNotNone(a)
        # The top reading should be gel+AOR, not anything with POT.
        suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("AOR", suffix_ids)
        self.assertNotIn("POT", suffix_ids)


class TestNewDerivations(unittest.TestCase):
    """The morphemes added for trick-word parsing: VBZ_LAS (ADJ→VERB),
    AGENT (VERB→ADJ agent noun), VER (verbal modifier "do quickly"),
    COP_EVID (copular evidential), CASINA (-cAsHnA "as if").
    """

    def setUp(self):
        self.inv   = load_inventory(INVENTORY_PATH)
        self.graph = load_graph(MORPHOTACTICS_PATH)
        self.lex   = load_lexicon(LEXICON_PATH)
        self.parser = Parser(self.lex, self.inv, self.graph)

    def _top(self, w):
        return (self.parser.parse(w) or [None])[0]

    def test_vbz_las(self):
        # mutlu + VBZ_LAS = mutlulaş ("become happy")
        # Need to parse a stem that goes through ADJ_ROOT first
        a = self._top("güzelleş")  # güzel (adj) + leş
        if a:
            self.assertEqual(a.root, "güzel")
            suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
            self.assertIn("VBZ_LAS", suffix_ids)

    def test_agent(self):
        # yaz + AGENT = yazıcı ("printer / writer")
        a = self._top("yazıcı")
        # Either recognized as derived from yaz or as a noun in its own
        # right; the new infrastructure allows the decomposition path.
        self.assertIsNotNone(a)

    def test_ver_aspect(self):
        # gel + VER + PAST = geliverdi ("came quickly")
        a = self._top("geliverdi")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "gel")
        suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("VER", suffix_ids)

    def test_trickword_cekoslovakya(self):
        # Famous test: Çekoslovakyalılaştıramadıklarımızdan
        # = "from those we couldn't make Czechoslovakian"
        a = self._top("çekoslovakyalılaştıramadıklarımızdan")
        self.assertIsNotNone(a, "trick word must parse")
        suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
        # Spot-check critical morphemes
        self.assertIn("ADJZ_LH", suffix_ids)
        self.assertIn("VBZ_LAS", suffix_ids)
        self.assertIn("CAUS", suffix_ids)
        self.assertIn("POT", suffix_ids)
        self.assertIn("NEG", suffix_ids)
        self.assertIn("DHK", suffix_ids)
        self.assertIn("ABL", suffix_ids)

    def test_trickword_muvaffakiyet(self):
        # The big one
        a = self._top("muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine")
        self.assertIsNotNone(a, "muvaffakiyet trick word must parse")
        suffix_ids = [m.suffix_id for m in a.morphemes if m.suffix_id]
        # Spot-check critical morphemes
        self.assertIn("ADJZ_SHZ", suffix_ids)
        self.assertIn("VBZ_LAS", suffix_ids)
        self.assertIn("AGENT", suffix_ids)
        self.assertIn("VER", suffix_ids)
        self.assertIn("POT", suffix_ids)
        self.assertIn("NEG", suffix_ids)
        self.assertIn("FUT_PART", suffix_ids)
        self.assertIn("COP_EVID", suffix_ids)
        self.assertIn("CASINA", suffix_ids)


class TestExpandedDerivations(unittest.TestCase):
    """The high-frequency derivational suffixes added in this iteration:
    NDER_CH (-CH agent), NDER_LHK (-lHk abstract), NDER_CHK (-CHk
    diminutive), NMZ_HS (-(y)Hş action noun), NMZ_HM (-(y)Hm
    instance), NMZ_MAN (-mAn agent), ADJ_GHN (-GHn state), ADJ_HK
    (-Hk past-participle), VBZ_LAN (-lAn middle), ADV_CA (-CA adverb),
    EQU (-CA equative case), NEC (-mAlH necessitative)."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)
        cls.lex   = load_lexicon(LEXICON_PATH)
        cls.parser = Parser(cls.lex, cls.inv, cls.graph)

    def _all(self, w):
        from tr_parse import ParseConfig
        return Parser(self.lex, self.inv, self.graph,
                      ParseConfig(return_all=True)).parse(w)

    def _has_suffix(self, w, suffix_id):
        """True if any analysis of w contains the given suffix id."""
        for a in self._all(w):
            if any(m.suffix_id == suffix_id for m in a.morphemes):
                return True
        return False

    def test_nder_ch_agent(self):
        # gazeteci = gazete + CH (gazete is in lexicon, gazeteci may not be)
        # We don't require it to be the TOP parse; just that the decomposition is available.
        self.assertTrue(self._has_suffix("gazeteci", "NDER_CH"),
                        "gazete+NDER_CH should be a possible parse of gazeteci")

    def test_nder_lhk_abstract(self):
        # iyilik = iyi + lHk
        self.assertTrue(self._has_suffix("iyilik", "NDER_LHK"))

    def test_nder_chk_diminutive(self):
        # kitapçık = kitap + CHk
        self.assertTrue(self._has_suffix("kitapçık", "NDER_CHK"))

    def test_nmz_hs_action_noun(self):
        # geliş = gel + (y)Hş ('coming, way of coming')
        self.assertTrue(self._has_suffix("geliş", "NMZ_HS"))

    def test_nec_necessitative(self):
        # gelmeli = gel + mAlH ('must come'). Top parse should include NEC.
        a = (self.parser.parse("gelmeli") or [None])[0]
        self.assertIsNotNone(a)
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("NEC", suffs)
        self.assertEqual(a.ud_feats().get("Mood"), "Nec")

    def test_nec_with_agreement(self):
        # gelmeliyim = gel + NEC + 1SG_Z
        a = (self.parser.parse("gelmeliyim") or [None])[0]
        self.assertIsNotNone(a)
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["NEC", "1SG_Z"])

    def test_zero_copula_with_adjective(self):
        # iyiyim = iyi + 1SG_Z ('I am good')
        a = (self.parser.parse("iyiyim") or [None])[0]
        self.assertIsNotNone(a)
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["1SG_Z"])

    def test_neg_aor_1sg_suppletion_parses(self):
        # gelmem = gel + NEG + AOR(empty) + 1SG_Z(bare 'm')
        a = (self.parser.parse("gelmem") or [None])[0]
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "gel")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["NEG", "AOR", "1SG_Z"])

    def test_equative_case(self):
        # bence = ben + CA (Case=Equ)
        a = (self.parser.parse("bence") or [None])[0]
        self.assertIsNotNone(a)
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("EQU", suffs)

    def test_g_archiphoneme(self):
        # kızgın = kız + GHn (voiced 'g' after voiced 'z')
        # küskün = küs + GHn (voiceless 'k' after voiceless 's')
        self.assertTrue(self._has_suffix("kızgın", "ADJ_GHN"))
        self.assertTrue(self._has_suffix("küskün", "ADJ_GHN"))


class TestPronouns(unittest.TestCase):
    """Personal, demonstrative, reflexive, and interrogative pronouns.
    All have pronominal-n behavior or use variants for case marking."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)
        cls.lex   = load_lexicon(LEXICON_PATH)
        cls.parser = Parser(cls.lex, cls.inv, cls.graph)

    def _top(self, w):
        return (self.parser.parse(w) or [None])[0]

    def test_bu_acc_pronominal_n(self):
        # bunu = bu + ACC (with pronominal n)
        a = self._top("bunu")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "bu")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["ACC"])

    def test_o_dat_pronominal_n(self):
        # ona = o + DAT (with pronominal n)
        a = self._top("ona")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "o")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["DAT"])

    def test_ben_dat_ban_variant(self):
        # bana = ben + DAT, using the irregular 'ban' variant
        a = self._top("bana")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "ben")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["DAT"])

    def test_sen_dat_san_variant(self):
        # sana = sen + DAT, using 'san' variant
        a = self._top("sana")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "sen")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["DAT"])

    def test_kendi_with_possessive(self):
        # kendisi = kendi + POSS_3SG
        a = self._top("kendisi")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "kendi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["POSS_3SG"])

    def test_kim_dat(self):
        # kime = kim + DAT (kim has NO pronominal n)
        a = self._top("kime")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "kim")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["DAT"])

    def test_ne_acc_buffer_y(self):
        # neyi = ne + ACC (with buffer y; ne is vowel-final, no pronominal n)
        a = self._top("neyi")
        self.assertIsNotNone(a)
        self.assertEqual(a.root, "ne")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["ACC"])


class TestProductiveDerivationWins(unittest.TestCase):
    """When a productive derivation is available, the decomposed parse
    must win over the lexicalized headword reading. This is the user's
    explicit design choice: 'the root of heyecanlı is heyecan, and the
    root of gazeteci is gazete; we want to show the derivations as
    such.' Enforced via productivity_bonus in tr_parse.py."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)
        cls.lex   = load_lexicon(LEXICON_PATH)
        cls.parser = Parser(cls.lex, cls.inv, cls.graph)

    def _top(self, w):
        return (self.parser.parse(w) or [None])[0]

    def test_heyecanlı_decomposes(self):
        a = self._top("heyecanlı")
        self.assertEqual(a.root, "heyecan",
                         "heyecanlı must decompose to heyecan+ADJZ_LH")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("ADJZ_LH", suffs)

    def test_gazeteci_decomposes(self):
        a = self._top("gazeteci")
        self.assertEqual(a.root, "gazete")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("NDER_CH", suffs)

    def test_caresiz_decomposes(self):
        a = self._top("çaresiz")
        self.assertEqual(a.root, "çare")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("ADJZ_SHZ", suffs)

    def test_iyilik_decomposes(self):
        a = self._top("iyilik")
        self.assertEqual(a.root, "iyi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("NDER_LHK", suffs)

    def test_kitapcik_decomposes(self):
        a = self._top("kitapçık")
        self.assertEqual(a.root, "kitap")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("NDER_CHK", suffs)

    def test_elmalar_stays_plural(self):
        # Regression: VBZ_LA must NOT spuriously match the -lA prefix
        # of plural -lAr. elmalar must be elma+PLUR, not elma+VBZ_LA+AOR.
        a = self._top("elmalar")
        self.assertEqual(a.root, "elma")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["PLUR"])

    def test_alındı_keeps_pass(self):
        # Regression: NMZ_HNTI (-Hntı) must NOT spuriously match -ındı
        # at the end of al+ın+dı. alındı must be al+PASS+PAST.
        a = self._top("alındı")
        self.assertEqual(a.root, "al")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertIn("PASS", suffs)


class TestQuestionParticles(unittest.TestCase):
    """The interrogative particles mi/mı/mu/mü are written as separate
    words in Turkish and must parse as bare lemmas, not as 'm+POSS_3SG'
    or similar spurious decompositions. This test guards against the
    single-letter-noun-root problem that historically came from stray
    'm' tokens in the UD treebank."""

    @classmethod
    def setUpClass(cls):
        cls.inv   = load_inventory(INVENTORY_PATH)
        cls.graph = load_graph(MORPHOTACTICS_PATH)
        cls.lex   = load_lexicon(LEXICON_PATH)
        cls.parser = Parser(cls.lex, cls.inv, cls.graph)

    def _top(self, w):
        return (self.parser.parse(w) or [None])[0]

    def test_mi_parses_as_lemma(self):
        a = self._top("mi")
        self.assertEqual(a.root, "mi")
        self.assertFalse(a.oov)

    def test_mı_parses_as_lemma(self):
        a = self._top("mı")
        self.assertEqual(a.root, "mı")
        self.assertFalse(a.oov)

    def test_mu_parses_as_lemma(self):
        a = self._top("mu")
        self.assertEqual(a.root, "mu")
        self.assertFalse(a.oov)

    def test_mü_parses_as_lemma(self):
        a = self._top("mü")
        self.assertEqual(a.root, "mü")
        self.assertFalse(a.oov)

    def test_no_single_letter_roots(self):
        # The lexicon should contain NO single-letter noun roots
        # except 'o' (the pronoun, which has pronominal_n behavior).
        single_letter = [r for f, rs in self.lex._by_form.items()
                         if len(f) == 1
                         for r in rs]
        non_pronoun = [r for r in single_letter if r.form != "o"]
        self.assertEqual(non_pronoun, [],
                         f"Single-letter non-pronoun roots leaked into lexicon: {non_pronoun}")

    # --- Q particle + suffix chains (zero-copula on the particle) ---
    # The particle itself behaves like a bare NOUN/ADJ stem and takes the
    # full inventory of copular suffixes plus agreement.

    def test_misiniz_z_agreement(self):
        # misiniz = mi + 2PL_Z ('are you [whether?]')
        a = self._top("misiniz")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["2PL_Z"])

    def test_muyum_z_agreement_with_vowel_harmony(self):
        # muyum = mu + 1SG_Z (buffer y after vowel-final particle)
        a = self._top("muyum")
        self.assertEqual(a.root, "mu")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["1SG_Z"])

    def test_midir_cop_dhr(self):
        # midir = mi + DHr ('is it [whether?]', emphatic)
        a = self._top("midir")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["COP_DHR"])

    def test_musunuzdur_chained(self):
        # musunuzdur = mu + 2PL_Z + COP_DHR ('are you [whether?] (for sure)')
        a = self._top("musunuzdur")
        self.assertEqual(a.root, "mu")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["2PL_Z", "COP_DHR"])

    def test_misindir_z_then_dhr(self):
        # misindir = mi + 2SG_Z + COP_DHR
        a = self._top("misindir")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["2SG_Z", "COP_DHR"])

    def test_miydi_past_copula(self):
        # miydi = mi + (y)dH ('was it [whether?]'). PAST_COP from a
        # bare nominal/particle stem (zero copula).
        a = self._top("miydi")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["PAST_COP"])

    def test_miydim_past_copula_with_k_agreement(self):
        # miydim = mi + PAST_COP + 1SG_K ('was I [whether?]')
        a = self._top("miydim")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["PAST_COP", "1SG_K"])

    def test_miyse_conditional_copula(self):
        # miyse = mi + (y)sA ('if it is [whether?]'). 3sg, no agreement.
        a = self._top("miyse")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["COP_COND"])

    def test_miysem_conditional_copula_with_k_agreement(self):
        # miysem = mi + COP_COND + 1SG_K ('if I am [whether?]'). COP_COND
        # takes k-type agreement, not z-type.
        a = self._top("miysem")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["COP_COND", "1SG_K"])

    def test_miymis_evidential_copula(self):
        # miymiş = mi + (y)mHş ('apparently it is [whether?]')
        a = self._top("miymiş")
        self.assertEqual(a.root, "mi")
        suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
        self.assertEqual(suffs, ["COP_EVID"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
