"""
test_tr_fuzzy.py — Tests for the BK-tree fuzzy-lookup primitive.

Run with:    python -m unittest test_tr_fuzzy -v
"""

import unittest

from tr_fuzzy import FuzzyIndex, levenshtein, weighted_distance


class TestLevenshtein(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(levenshtein("kitap", "kitap"), 0)
        self.assertEqual(levenshtein("kitap", "kitp"), 1)     # deletion
        self.assertEqual(levenshtein("kitap", "kitabp"), 1)   # insertion
        self.assertEqual(levenshtein("kitap", "ketop"), 2)    # two subs
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_transposition_is_two_under_levenshtein(self):
        # A single adjacent transposition is two substitutions under plain
        # Levenshtein — which is why the BK radius defaults to 2.
        self.assertEqual(levenshtein("kitap", "iktap"), 2)

    def test_is_symmetric(self):
        self.assertEqual(levenshtein("mektup", "mektip"),
                         levenshtein("mektip", "mektup"))


class TestWeightedDistance(unittest.TestCase):

    def test_cheap_confusion_pair(self):
        # i/ı is a cheap confusion, so it costs less than a generic sub.
        self.assertLess(weighted_distance("kapı", "kapi"),
                        weighted_distance("kapı", "kapo"))

    def test_transposition_cheaper_than_two_subs(self):
        self.assertLess(weighted_distance("kitap", "iktap"), 2.0)


class TestFuzzyIndex(unittest.TestCase):

    def setUp(self):
        self.idx = FuzzyIndex({
            "kitap": 1200, "katip": 90, "kâtip": 5,
            "kitaplık": 30, "kapı": 800, "kapu": 1,
            "mektup": 400, "araba": 950,
        })

    def test_len(self):
        self.assertEqual(len(self.idx), 8)

    def test_exact_match_distance_zero(self):
        res = self.idx.nearest("kitap")
        self.assertEqual(res[0][0], "kitap")
        self.assertEqual(res[0][1], 0)

    def test_typo_suggests_intended_word(self):
        # one deletion
        self.assertEqual(self.idx.nearest("kitp")[0][0], "kitap")
        # one substitution
        self.assertEqual(self.idx.nearest("mektip")[0][0], "mektup")

    def test_frequency_breaks_ties(self):
        # "kapi" is distance 1 from both "kapı" (i/ı, freq 800) and "kapu"
        # (freq 1). The frequent, cheap-confusion candidate must win.
        self.assertEqual(self.idx.nearest("kapi")[0][0], "kapı")

    def test_respects_max_distance(self):
        self.assertEqual(self.idx.nearest("xyz", max_distance=1), [])

    def test_limit(self):
        self.assertLessEqual(len(self.idx.nearest("kitap", limit=2)), 2)

    def test_empty_index(self):
        self.assertEqual(FuzzyIndex().nearest("kitap"), [])

    def test_add_keeps_higher_frequency(self):
        idx = FuzzyIndex()
        idx.add("kitap", 5)
        idx.add("kitap", 50)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx.nearest("kitap")[0][2], 50)


if __name__ == "__main__":
    unittest.main()
