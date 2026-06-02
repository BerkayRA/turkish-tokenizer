"""
tr_fuzzy.py — Approximate string matching for Turkish NLP.

A reusable, dependency-free fuzzy-lookup primitive intended to be shared
across the tool suite (tokenizer, normalizer, search, autocomplete). A
vocabulary is indexed in a BK-tree under unit-cost Levenshtein distance —
a true metric, so the triangle inequality prunes the search to a small
fraction of the terms. Candidates found within the radius are then
re-ranked with a Turkish-aware weighted cost (the common diacritic/letter
confusions are cheap, adjacent transpositions are cheap) and term
frequency, so the most plausible correction comes first.

Design notes:
  - The TREE metric is plain Levenshtein. It is a genuine metric, which the
    BK-tree relies on for correctness. A radius of 2 still captures a single
    transposition (which has Levenshtein distance 2).
  - The RANKING cost is a separate weighted Damerau function. It need not be
    a metric, so it can charge less for likely confusions without breaking
    the tree.

Typical usage:

    from tr_fuzzy import FuzzyIndex
    idx = FuzzyIndex({"kitap": 1200, "katip": 90, "kitaplık": 30})
    idx.nearest("kitp")        # -> [("kitap", 1, 1200), ...]
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union


# Turkish letter-confusion classes. A substitution WITHIN one of these pairs
# is a common typo (the diacritic was dropped, or the dotted/dotless i was
# mistyped) and so costs less than a generic substitution when ranking.
_CONFUSION_PAIRS = (
    {"i", "ı"}, {"o", "ö"}, {"u", "ü"},
    {"c", "ç"}, {"s", "ş"}, {"g", "ğ"},
)
_CHEAP_SUB_COST = 0.4
_GENERIC_SUB_COST = 1.0
_TRANSPOSE_COST = 0.9
_INDEL_COST = 1.0


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    pair = {a, b}
    for p in _CONFUSION_PAIRS:
        if pair == p:
            return _CHEAP_SUB_COST
    return _GENERIC_SUB_COST


# -----------------------------------------------------------------------------
# Distances
# -----------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Unit-cost Levenshtein edit distance (a true metric)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = cur
    return prev[-1]


def weighted_distance(a: str, b: str) -> float:
    """Turkish-weighted Damerau edit distance, for RANKING only.

    Charges less for common Turkish confusions (i/ı, o/ö, ...) and for
    adjacent transpositions. Not guaranteed to be a metric, so it is never
    used for BK-tree structure — only to order candidates.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb * _INDEL_COST
    if lb == 0:
        return la * _INDEL_COST
    # Optimal string alignment DP with weighted costs.
    d = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        d[i][0] = i * _INDEL_COST
    for j in range(1, lb + 1):
        d[0][j] = j * _INDEL_COST
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            best = min(
                d[i - 1][j] + _INDEL_COST,
                d[i][j - 1] + _INDEL_COST,
                d[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1]),
            )
            if (i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                best = min(best, d[i - 2][j - 2] + _TRANSPOSE_COST)
            d[i][j] = best
    return d[la][lb]


# -----------------------------------------------------------------------------
# BK-tree index
# -----------------------------------------------------------------------------

class _BKNode:
    __slots__ = ("term", "freq", "children")

    def __init__(self, term: str, freq: int):
        self.term = term
        self.freq = freq
        self.children: Dict[int, "_BKNode"] = {}


Term = Union[str, Tuple[str, int]]


class FuzzyIndex:
    """A BK-tree over a vocabulary, supporting nearest-neighbour lookup by
    edit distance. Construct from an iterable of terms or (term, frequency)
    pairs, or a {term: frequency} mapping.
    """

    def __init__(self, terms: Union[Iterable[Term], Dict[str, int]] = ()):
        self._root: "_BKNode | None" = None
        self._size = 0
        items: Iterable[Term]
        if isinstance(terms, dict):
            items = terms.items()
        else:
            items = terms
        for it in items:
            if isinstance(it, str):
                self.add(it, 0)
            else:
                term, freq = it
                self.add(term, freq)

    def __len__(self) -> int:
        return self._size

    def add(self, term: str, freq: int = 0) -> None:
        """Insert a term (idempotent on the term; keeps the higher freq)."""
        if not term:
            return
        if self._root is None:
            self._root = _BKNode(term, freq)
            self._size = 1
            return
        node = self._root
        while True:
            dist = levenshtein(term, node.term)
            if dist == 0:
                node.freq = max(node.freq, freq)
                return
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = _BKNode(term, freq)
                self._size += 1
                return
            node = child

    def nearest(self, word: str, max_distance: int = 2,
                limit: int = 5) -> List[Tuple[str, int, int]]:
        """Return up to `limit` vocabulary terms within `max_distance`
        (unit-cost Levenshtein) of `word`.

        Each result is (term, distance, frequency). Results are ranked by
        the Turkish-weighted cost, then by descending frequency, then by the
        raw distance — so the most plausible correction is first.
        """
        if self._root is None or not word:
            return []
        found: List[Tuple[str, int, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            dist = levenshtein(word, node.term)
            if dist <= max_distance:
                found.append((node.term, dist, node.freq))
            lo, hi = dist - max_distance, dist + max_distance
            for child_dist, child in node.children.items():
                if lo <= child_dist <= hi:
                    stack.append(child)
        found.sort(key=lambda r: (weighted_distance(word, r[0]),
                                  -r[2], r[1], r[0]))
        return found[:limit]


__all__ = ["FuzzyIndex", "levenshtein", "weighted_distance"]
