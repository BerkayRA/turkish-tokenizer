"""
test_tr_additions.py — curated lexicon additions (proper-noun gazetteer).

Run with:    python -m unittest test_tr_additions -v
"""

import unittest
from pathlib import Path

from tr_api import Tokenizer, TokenizerConfig
from apply_additions import load_additions, merge_into_entries

HERE = Path(__file__).parent


class TestGazetteerResolves(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer()

    def test_country_and_city_names_are_in_lex(self):
        for w in ("çekoslovakya", "japonya", "azerbaycan", "trabzon",
                  "gaziantep", "yunanistan"):
            with self.subTest(word=w):
                r = self.tok.tokenize(w)
                self.assertFalse(r["oov"], f"{w} should be in-lexicon")
                self.assertEqual(r["root"], w)

    def test_gazetteer_root_inflects(self):
        # A place name takes the full derivational/inflectional chain.
        r = self.tok.tokenize("japonyalılaştıramadıklarımız")
        self.assertEqual(r["root"], "japonya")
        self.assertFalse(r["oov"])

    def test_canary_resolves_in_lex(self):
        r = self.tok.tokenize("çekoslovakyalılaştıramadıklarımızdanmısınız")
        self.assertEqual(r["segments"][0]["root"], "çekoslovakya")
        self.assertFalse(r["segments"][0]["oov"])


class TestApplyAdditions(unittest.TestCase):

    def test_additions_present_in_production_lexicons(self):
        adds = {a["form"] for a in load_additions(HERE / "lexicon_overrides.json")}
        self.assertIn("çekoslovakya", adds)
        # Every curated form must be present in the shipped production lexicon.
        tok = Tokenizer(TokenizerConfig(lexicon_path=HERE / "lexicon_full.json"))
        for form in adds:
            with self.subTest(form=form):
                self.assertIn(form, tok._lex)

    def test_merge_is_idempotent(self):
        additions = [{"form": "testlandiya", "class": "NOUN"}]
        entries = [{"form": "kitap", "class": "NOUN"}]
        self.assertEqual(merge_into_entries(entries, additions), 1)
        # Second merge adds nothing.
        self.assertEqual(merge_into_entries(entries, additions), 0)
        self.assertEqual(len(entries), 2)

    def test_train_lexicon_left_pristine(self):
        # The no-leakage eval lexicon must NOT carry the gazetteer.
        tok = Tokenizer(TokenizerConfig(lexicon_path=HERE / "lexicon_train.json"))
        self.assertNotIn("çekoslovakya", tok._lex)


if __name__ == "__main__":
    unittest.main()
