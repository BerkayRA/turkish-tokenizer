"""
test_tr_phase2.py — Tests for the inventory, rules, and generator
(no morphotactic graph; graph tests are in test_tr_phase3.py).

Run with:    python -m unittest test_tr_phase2 -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from tr_inventory import Inventory, Suffix, load_inventory
from tr_generate import generate
from tr_rules import get_forward, get_inverse


INVENTORY_PATH = Path(__file__).parent / "inventory.json"


class TestInventoryLoading(unittest.TestCase):

    def test_load_real_inventory(self):
        inv = load_inventory(INVENTORY_PATH)
        for sid in ["PROG", "PAST", "FUT", "ACC", "DAT", "LOC",
                    "POSS_1SG", "VBZ_LA", "ADJZ_LH", "NEG",
                    "1SG_Z", "1SG_K", "2SG_Z", "2SG_K", "3PL"]:
            self.assertIn(sid, inv, f"expected {sid} in inventory")

    def test_inventory_count(self):
        inv = load_inventory(INVENTORY_PATH)
        # The inventory has grown substantially as derivational coverage
        # has expanded. Set a wide range to allow continued growth.
        self.assertGreater(len(inv), 30)
        self.assertLess(len(inv), 150)

    def test_load_rejects_unknown_rule(self):
        bad = {"suffixes": [{
            "id": "X", "template": "Hm",
            "rules": ["nonexistent_rule"]
        }]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad, f); name = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                load_inventory(name)
            self.assertIn("unknown rule", str(cm.exception))
        finally:
            os.unlink(name)

    def test_load_rejects_duplicate_id(self):
        bad = {"suffixes": [
            {"id": "X", "template": "Hm"},
            {"id": "X", "template": "lAr"},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad, f); name = f.name
        try:
            with self.assertRaises(ValueError):
                load_inventory(name)
        finally:
            os.unlink(name)

    def test_load_rejects_malformed_template(self):
        bad = {"suffixes": [{"id": "X", "template": "(yz)A"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad, f); name = f.name
        try:
            with self.assertRaises(ValueError):
                load_inventory(name)
        finally:
            os.unlink(name)


class TestRuleRegistry(unittest.TestCase):

    def test_drop_H_after_vowel_stem(self):
        fn = get_forward("drop_initial_H_after_vowel_stem")
        s, t, _ = fn("ev", "Hm", {});       self.assertEqual((s, t), ("ev", "Hm"))
        s, t, _ = fn("araba", "Hm", {});    self.assertEqual((s, t), ("araba", "m"))
        s, t, _ = fn("araba", "lAr", {});   self.assertEqual((s, t), ("araba", "lAr"))

    def test_delete_stem_final_low_vowel(self):
        fn = get_forward("delete_stem_final_low_vowel")
        s, t, _ = fn("gelme", "Hyor", {});  self.assertEqual((s, t), ("gelm", "Hyor"))
        s, t, _ = fn("yapma", "Hyor", {});  self.assertEqual((s, t), ("yapm", "Hyor"))
        s, t, _ = fn("oku",   "Hyor", {});  self.assertEqual((s, t), ("oku",  "Hyor"))
        s, t, _ = fn("gel",   "Hyor", {});  self.assertEqual((s, t), ("gel",  "Hyor"))

    def test_delete_stem_final_low_vowel_inverse(self):
        inv = get_inverse("delete_stem_final_low_vowel")
        # gelm could be the original "gel" + ... or original "gelme" with
        # 'e' deleted; for "gelm" the inverse should propose both "gelm"
        # (no change) and "gelme" (e re-attached by front harmony).
        out = inv(["gelm"], {})
        self.assertIn("gelm",  out)
        self.assertIn("gelme", out)
        # Back-harmony case: yapm → yapm or yapma.
        out = inv(["yapm"], {})
        self.assertIn("yapm",  out)
        self.assertIn("yapma", out)


class TestVerbConjugation(unittest.TestCase):
    """gel- (to come), yap- (to do), oku- (to read) — covers all stem shapes."""

    @classmethod
    def setUpClass(cls):
        cls.inv = load_inventory(INVENTORY_PATH)

    # Set of monosyllabic verbs that take the -Hr (high vowel) aorist
    # instead of the default -Ar. Mirrors the irregulars.json entries.
    AORIST_HIGH = {
        "al", "bil", "bul", "dur", "gel", "gör", "kal", "ol", "öl",
        "san", "var", "ver", "vur",
    }

    def gen(self, stem, chain, soften=False):
        root_ctx = {"root_aorist_high": True} if stem in self.AORIST_HIGH else None
        return generate(stem, chain, self.inv, soften=soften, root_ctx=root_ctx).surface

    # --- gel- with z-type agreement (PROG, FUT, AOR, EVID) ---
    def test_gel_progressive(self):
        self.assertEqual(self.gen("gel", ["PROG", "1SG_Z"]), "geliyorum")
        self.assertEqual(self.gen("gel", ["PROG", "2SG_Z"]), "geliyorsun")
        self.assertEqual(self.gen("gel", ["PROG", "1PL_Z"]), "geliyoruz")
        self.assertEqual(self.gen("gel", ["PROG", "2PL_Z"]), "geliyorsunuz")
        self.assertEqual(self.gen("gel", ["PROG", "3PL"]),   "geliyorlar")

    def test_gel_future(self):
        # FUT alone (3sg, no person ending) = gelecek
        self.assertEqual(self.gen("gel", ["FUT"]), "gelecek")
        # FUT + 1SG_Z: k → ğ softening before -Hm
        self.assertEqual(self.gen("gel", ["FUT", "1SG_Z"], soften=True), "geleceğim")
        self.assertEqual(self.gen("gel", ["FUT", "2SG_Z"], soften=True), "geleceksin")
        self.assertEqual(self.gen("gel", ["FUT", "1PL_Z"], soften=True), "geleceğiz")
        self.assertEqual(self.gen("gel", ["FUT", "3PL"],   soften=True), "gelecekler")

    def test_gel_aorist(self):
        self.assertEqual(self.gen("gel", ["AOR", "1SG_Z"]), "gelirim")
        self.assertEqual(self.gen("gel", ["AOR", "2SG_Z"]), "gelirsin")
        self.assertEqual(self.gen("gel", ["AOR", "3PL"]),   "gelirler")

    def test_gel_evidential(self):
        self.assertEqual(self.gen("gel", ["EVID", "1SG_Z"]), "gelmişim")
        self.assertEqual(self.gen("gel", ["EVID", "2SG_Z"]), "gelmişsin")

    # --- gel- with k-type agreement (PAST, COND) ---
    def test_gel_past_correct_forms(self):
        # These are the standard forms — the k-type split makes them work.
        self.assertEqual(self.gen("gel", ["PAST", "1SG_K"]), "geldim")
        self.assertEqual(self.gen("gel", ["PAST", "2SG_K"]), "geldin")
        self.assertEqual(self.gen("gel", ["PAST", "1PL_K"]), "geldik")
        self.assertEqual(self.gen("gel", ["PAST", "2PL_K"]), "geldiniz")
        self.assertEqual(self.gen("gel", ["PAST", "3PL"]),   "geldiler")

    def test_gel_conditional(self):
        # COND -sA also takes k-type agreement
        self.assertEqual(self.gen("gel", ["COND", "1SG_K"]), "gelsem")
        self.assertEqual(self.gen("gel", ["COND", "2SG_K"]), "gelsen")
        self.assertEqual(self.gen("gel", ["COND", "1PL_K"]), "gelsek")

    def test_gel_negation_with_prog(self):
        self.assertEqual(self.gen("gel", ["NEG", "PROG", "1SG_Z"]), "gelmiyorum")

    # --- yap- (no soften) ---
    def test_yap_prog(self):
        self.assertEqual(self.gen("yap", ["PROG"]),           "yapıyor")
        self.assertEqual(self.gen("yap", ["PROG", "1SG_Z"]), "yapıyorum")
        self.assertEqual(self.gen("yap", ["NEG", "PROG"]),    "yapmıyor")

    def test_yap_past(self):
        # yap + PAST: D → t (after p), H = ı → yaptı
        # + 1SG_K (-m): yaptım. No softening fires (m isn't vowel-initial).
        self.assertEqual(self.gen("yap", ["PAST", "1SG_K"]), "yaptım")
        self.assertEqual(self.gen("yap", ["PAST", "2SG_K"]), "yaptın")
        self.assertEqual(self.gen("yap", ["PAST", "1PL_K"]), "yaptık")

    def test_yap_future(self):
        # yap + FUT alone — yap stays p (no softening across (y) buffer-as-vowel).
        self.assertEqual(self.gen("yap", ["FUT"]), "yapacak")
        # yap + FUT + 1SG_Z: built in two steps to respect the per-stem
        # no-soften rule on yap. In Phase 4 the lexicon will make this
        # transparent.
        step1 = generate("yap", ["FUT"], self.inv, soften=False).surface
        step2 = generate(step1, ["1SG_Z"], self.inv, soften=True).surface
        self.assertEqual(step1, "yapacak")
        self.assertEqual(step2, "yapacağım")

    # --- oku- (vowel-final stem) ---
    def test_oku_prog(self):
        self.assertEqual(self.gen("oku", ["PROG"]),           "okuyor")
        self.assertEqual(self.gen("oku", ["PROG", "1SG_Z"]), "okuyorum")
        self.assertEqual(self.gen("oku", ["NEG", "PROG"]),    "okumuyor")

    def test_oku_past(self):
        # oku + PAST: D = d after vowel, H = u. → okudu
        # + 1SG_K (-m) → okudum
        self.assertEqual(self.gen("oku", ["PAST"]),           "okudu")
        self.assertEqual(self.gen("oku", ["PAST", "1SG_K"]), "okudum")
        self.assertEqual(self.gen("oku", ["PAST", "1PL_K"]), "okuduk")


class TestNounDeclension(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inv = load_inventory(INVENTORY_PATH)

    def gen(self, stem, chain, soften=True):
        return generate(stem, chain, self.inv, soften=soften).surface

    def test_ev_basic_cases(self):
        self.assertEqual(self.gen("ev", ["NOM"]), "ev")
        self.assertEqual(self.gen("ev", ["ACC"]), "evi")
        self.assertEqual(self.gen("ev", ["DAT"]), "eve")
        self.assertEqual(self.gen("ev", ["LOC"]), "evde")
        self.assertEqual(self.gen("ev", ["ABL"]), "evden")
        self.assertEqual(self.gen("ev", ["GEN"]), "evin")

    def test_ev_plural_and_compound(self):
        self.assertEqual(self.gen("ev", ["PLUR"]),                       "evler")
        self.assertEqual(self.gen("ev", ["PLUR", "ACC"]),                "evleri")
        self.assertEqual(self.gen("ev", ["PLUR", "POSS_1SG", "LOC"]),    "evlerimde")

    def test_kitap_softening(self):
        self.assertEqual(self.gen("kitap", ["ACC"]),       "kitabı")
        self.assertEqual(self.gen("kitap", ["DAT"]),       "kitaba")
        self.assertEqual(self.gen("kitap", ["POSS_1SG"]),  "kitabım")
        self.assertEqual(self.gen("kitap", ["LOC"]),       "kitapta")
        self.assertEqual(self.gen("kitap", ["PLUR"]),      "kitaplar")

    def test_araba_basic(self):
        self.assertEqual(self.gen("araba", ["ACC"]), "arabayı")
        self.assertEqual(self.gen("araba", ["DAT"]), "arabaya")
        self.assertEqual(self.gen("araba", ["LOC"]), "arabada")
        self.assertEqual(self.gen("araba", ["GEN"]), "arabanın")
        self.assertEqual(self.gen("araba", ["INS"]), "arabayla")

    def test_araba_possessive(self):
        self.assertEqual(self.gen("araba", ["POSS_1SG"]), "arabam")
        self.assertEqual(self.gen("araba", ["POSS_3SG"]), "arabası")
        self.assertEqual(self.gen("araba", ["PLUR", "POSS_1PL"]), "arabalarımız")
        self.assertEqual(self.gen("araba", ["PLUR", "POSS_1PL", "LOC"]),
                         "arabalarımızda")


class TestDerivation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inv = load_inventory(INVENTORY_PATH)

    def gen(self, stem, chain, soften=True):
        return generate(stem, chain, self.inv, soften=soften).surface

    def test_noun_to_adj(self):
        self.assertEqual(self.gen("göz", ["ADJZ_LH"]),  "gözlü")
        self.assertEqual(self.gen("tuz", ["ADJZ_SHZ"]), "tuzsuz")

    def test_noun_to_verb_to_progressive(self):
        self.assertEqual(self.gen("tuz", ["VBZ_LA", "PROG"]), "tuzluyor")

    def test_neologism_selfie(self):
        self.assertEqual(self.gen("selfie", ["PLUR", "POSS_1SG", "ACC"]),
                         "selfielerimi")

    def test_neologism_commit(self):
        self.assertEqual(self.gen("commit", ["VBZ_LA", "FUT", "1PL_Z"]),
                         "commitleyeceğiz")


class TestOutputFormats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inv = load_inventory(INVENTORY_PATH)

    def test_split_output(self):
        g = generate("ev", ["PLUR", "POSS_1SG", "LOC"], self.inv)
        self.assertEqual(g.surface, "evlerimde")
        self.assertEqual(g.split(), "ev-ler-im-de")

    def test_tagged_output(self):
        g = generate("gel", ["PAST", "1SG_K"], self.inv)
        self.assertEqual(g.surface, "geldim")
        tagged = g.tagged()
        self.assertIn("PAST",        tagged)
        self.assertIn("1SG_K",       tagged)
        self.assertIn("Tense=Past",  tagged)
        self.assertIn("Person=1",    tagged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
