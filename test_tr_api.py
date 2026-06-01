"""
test_tr_api.py — Tests for the high-level Tokenizer API.

Smoke-tests rather than exhaustive coverage; the parser itself is tested
elsewhere. The point of these tests is to verify the wire format is
stable and the convenience functions work as documented.
"""

import json
import unittest

from tr_api import Tokenizer, TokenizerConfig, tokenize


class TestTokenizerAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_returns_dict(self):
        r = self.tok.tokenize("kitap")
        self.assertIsInstance(r, dict)
        self.assertTrue(r["parsed"])

    def test_json_serializable(self):
        # The wire format must round-trip through json.
        r = self.tok.tokenize("kitabımı")
        s = json.dumps(r, ensure_ascii=False)
        r2 = json.loads(s)
        self.assertEqual(r2["root"], "kitap")
        self.assertEqual(r2["split"], "kitab-ım-ı")

    def test_required_keys_present(self):
        r = self.tok.tokenize("geldim")
        for key in ("surface", "parsed", "root", "root_class", "final_class",
                    "morphemes", "split", "tagged", "features",
                    "emitted_features", "score", "oov"):
            self.assertIn(key, r, f"missing key: {key}")

    def test_morpheme_shape(self):
        r = self.tok.tokenize("kitabımı")
        for m in r["morphemes"]:
            self.assertIn("chunk", m)
            self.assertIn("id", m)
            self.assertIn("feats", m)
            self.assertIn("is_root", m)
            self.assertIsInstance(m["feats"], dict)
        # First morpheme is the root.
        self.assertTrue(r["morphemes"][0]["is_root"])
        self.assertIsNone(r["morphemes"][0]["id"])

    def test_empty_input(self):
        r = self.tok.tokenize("")
        self.assertFalse(r["parsed"])
        self.assertIn("error", r)

    def test_whitespace_input(self):
        r = self.tok.tokenize("   ")
        self.assertFalse(r["parsed"])

    def test_basic_noun(self):
        r = self.tok.tokenize("kitap")
        self.assertEqual(r["root"], "kitap")
        self.assertEqual(r["root_class"], "NOUN")
        self.assertEqual(len(r["morphemes"]), 1)
        self.assertFalse(r["oov"])

    def test_inflected_noun(self):
        r = self.tok.tokenize("kitabımı")
        self.assertEqual(r["root"], "kitap")
        self.assertEqual(r["split"], "kitab-ım-ı")
        self.assertEqual(r["features"]["Case"], "Acc")
        suffix_ids = [m["id"] for m in r["morphemes"] if m["id"]]
        self.assertEqual(suffix_ids, ["POSS_1SG", "ACC"])

    def test_finite_verb(self):
        r = self.tok.tokenize("geldim")
        self.assertEqual(r["root"], "gel")
        self.assertEqual(r["root_class"], "VERB")
        self.assertEqual(r["features"]["Tense"], "Past")
        self.assertEqual(r["features"]["Person"], "1")

    def test_v_to_v_derivation(self):
        # Per design: çıkardı decomposes to çık+CAUS_DERIV+PAST
        r = self.tok.tokenize("çıkardı")
        self.assertEqual(r["root"], "çık")
        suffix_ids = [m["id"] for m in r["morphemes"] if m["id"]]
        self.assertIn("CAUS_DERIV", suffix_ids)
        # No Voice feature on V→V derivations.
        self.assertNotIn("Voice", r["features"])

    def test_passive_emits_voice(self):
        # alındı = al+PASS+PAST, Voice=Pass
        r = self.tok.tokenize("alındı")
        self.assertEqual(r["root"], "al")
        self.assertEqual(r["features"].get("Voice"), "Pass")

    def test_alternatives_excluded_by_default_in_oov(self):
        # OOV alternatives should not pollute the alts list
        r = self.tok.tokenize("kitabımı")
        for alt in r.get("alternatives", []):
            self.assertFalse(alt["oov"],
                             f"OOV alt should be excluded: {alt['root']}")

    def test_module_level_tokenize(self):
        # The lazy module-level function should work
        r = tokenize("kitap")
        self.assertEqual(r["root"], "kitap")

    def test_batch(self):
        results = self.tok.tokenize_batch(["kitap", "gel", "çıkardı"])
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["root"], "kitap")
        self.assertEqual(results[1]["root"], "gel")
        self.assertEqual(results[2]["root"], "çık")

    def test_tokenize_text(self):
        r = self.tok.tokenize_text("Kitabımı gördüm.")
        # Token kinds: word, space, word, punct
        kinds = [t["kind"] for t in r["tokens"]]
        self.assertEqual(kinds, ["word", "space", "word", "punct"])
        # First word analyzed
        self.assertEqual(r["tokens"][0]["analysis"]["root"], "kitap")
        self.assertEqual(r["tokens"][2]["analysis"]["root"], "gör")
        # Non-word tokens have null analysis
        self.assertIsNone(r["tokens"][1]["analysis"])
        self.assertIsNone(r["tokens"][3]["analysis"])

    def test_tokenize_text_reconstruction(self):
        # The token surfaces concatenated must equal the input exactly
        text = "Çıkardı, sonra geçirdi.\nYarın da gelecek."
        r = self.tok.tokenize_text(text)
        reconstructed = "".join(t["surface"] for t in r["tokens"])
        self.assertEqual(reconstructed, text)

    def test_tokenize_text_empty(self):
        r = self.tok.tokenize_text("")
        self.assertEqual(r["tokens"], [])

    def test_tokenize_text_handles_apostrophe(self):
        # Apostrophe is part of the word (proper-noun separator)
        r = self.tok.tokenize_text("Osman'ın kitabı")
        words = [t for t in r["tokens"] if t["kind"] == "word"]
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0]["surface"], "Osman'ın")


class TestTokenizerConfig(unittest.TestCase):

    def test_disable_alternatives(self):
        tok = Tokenizer(TokenizerConfig(include_alternatives=False))
        r = tok.tokenize("evi")
        self.assertNotIn("alternatives", r)

    def test_train_only_lexicon(self):
        from pathlib import Path
        train_path = Path(__file__).parent / "lexicon_train.json"
        if not train_path.exists():
            self.skipTest("lexicon_train.json not present")
        tok = Tokenizer(TokenizerConfig(lexicon_path=train_path))
        r = tok.tokenize("kitap")
        self.assertEqual(r["root"], "kitap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
