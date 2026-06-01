"""
tr_lexicon.py — Turkish root lexicon with prefix-trie lookup.

A Root represents a single lemma with its word class and phonological flags.
The Lexicon stores roots indexed for efficient prefix matching against a
surface form: given the start of a word, return all roots that could be
that prefix (including the canonical form and any allomorphic variants).

For 5000+ entries a trie is sized appropriately; for 100k+ we'd want a
DAWG, but the JSON load + lookup pattern handles the current scale easily.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tr_phonology import SOFTEN as SOFTEN_MAP


@dataclass(frozen=True)
class Root:
    form:         str           # canonical surface form
    word_class:   str           # "VERB" | "NOUN" | "ADJ"
    soften:       bool = False  # final consonant softens before vowel-initial suffix
    variants:     tuple = ()    # alternative surface forms (e.g., "gid" for "git")
    frequency:    int = 0       # corpus frequency, for ranker tiebreak
    aorist_high:  bool = False  # this verb takes -Hr aorist (al, gel, bil, etc.);
                                #   only meaningful for monosyllabic VERB roots
    caus_deriv:   str = ""      # if non-empty, the template ("Ar", "Hr", "t")
                                #   this verb takes for CAUS_DERIV (a V→V
                                #   lexicalized derivation). E.g., "çık"
                                #   has caus_deriv="Ar" → produces çıkar.
    pass_deriv:   str = ""      # similarly for PASS_DERIV (values: "Hn",
                                #   "Hl", "n"). E.g., "bul" → bulun.
    pronominal_n: bool = False  # demonstrative/personal pronouns (bu, şu,
                                #   o) insert -n- before case markers:
                                #   bu+ACC → bunu, o+DAT → ona, şu+LOC → şunda.
                                #   The same buffer-n that follows POSS_3SG.

    def all_forms(self) -> tuple:
        """Canonical + variants as a tuple."""
        return (self.form,) + self.variants

    def root_ctx(self) -> dict:
        """Return a context dict for the rule machinery, carrying any
        root-level flags that downstream rules might inspect."""
        ctx = {}
        if self.aorist_high:
            ctx["root_aorist_high"] = True
        if self.caus_deriv:
            ctx["root_caus_deriv"] = self.caus_deriv
        if self.pass_deriv:
            ctx["root_pass_deriv"] = self.pass_deriv
        if self.pronominal_n:
            ctx["root_pronominal_n"] = True
        return ctx


class _TrieNode:
    __slots__ = ("children", "roots")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        # Roots whose surface form (canonical OR variant) ends at this node.
        self.roots: List[tuple] = []   # (Root, surface_form_used)


class Lexicon:
    """Indexed root lexicon. Supports prefix matching against a surface word."""

    def __init__(self, roots: List[Root]):
        self._roots: List[Root] = list(roots)
        self._trie = _TrieNode()
        self._by_form: Dict[str, List[Root]] = {}

        for r in self._roots:
            # Build the full set of surface forms to index:
            # - canonical
            # - all explicit variants
            # - if soften=True: auto-add the softened-final-consonant form.
            #   This makes the parser find "kitab..." → "kitap" without
            #   hand-curating every softening pair.
            # - if it's a VERB ending in a low vowel (a/e): auto-add the
            #   truncated form too. The PROG suffix -Hyor deletes the
            #   stem's final low vowel (başla + Hyor → başlıyor), so
            #   indexing 'başl' alongside 'başla' lets the parser find
            #   the root inside surfaces like 'başlıyordum'.
            forms = list(r.all_forms())
            if r.soften and r.form and r.form[-1] in SOFTEN_MAP:
                softened = r.form[:-1] + SOFTEN_MAP[r.form[-1]]
                if softened not in forms:
                    forms.append(softened)
            # NOTE: We do NOT auto-index a truncated form for vowel-final
            # verbs (e.g. başla → başl) here. That kind of stem truncation
            # is triggered by specific suffixes (PROG -Hyor deletes the
            # stem's final low vowel) and shouldn't be treated as a general
            # lexicon variant; the parser handles it via the suffix rule
            # mechanism instead (see PROG's expand rule).

            for surf in forms:
                self._by_form.setdefault(surf, []).append(r)
                node = self._trie
                for ch in surf:
                    node = node.children.setdefault(ch, _TrieNode())
                node.roots.append((r, surf))

    def __len__(self) -> int:
        return len(self._roots)

    def __contains__(self, form: str) -> bool:
        return form in self._by_form

    def get(self, form: str) -> List[Root]:
        """All Roots with the given (canonical or variant) surface form."""
        return list(self._by_form.get(form, []))

    def prefix_match(self, word: str) -> List[tuple]:
        """Return all (Root, prefix_used, prefix_len) where the surface
        form starts the given word.

        For "geliyorum" the trie walk will hit "gel" and return
        (Root(gel, VERB), "gel", 3) at minimum. If "geliyor" or some other
        prefix were also a root, it would be returned too.
        """
        results = []
        node = self._trie
        for i, ch in enumerate(word, start=1):
            if ch not in node.children:
                break
            node = node.children[ch]
            for (root, surf) in node.roots:
                results.append((root, surf, i))
        return results

    def all_roots(self) -> List[Root]:
        return list(self._roots)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_lexicon(path: str | Path) -> Lexicon:
    """Load a lexicon JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "entries" not in data:
        raise ValueError(f"{path}: missing 'entries'")

    roots = []
    for entry in data["entries"]:
        for required in ("form", "class"):
            if required not in entry:
                raise ValueError(f"{path}: entry missing {required}: {entry}")
        roots.append(Root(
            form         = entry["form"],
            word_class   = entry["class"],
            soften       = bool(entry.get("soften", False)),
            variants     = tuple(entry.get("variants", [])),
            frequency    = int(entry.get("frequency", 0)),
            aorist_high  = bool(entry.get("aorist_high", False)),
            caus_deriv   = str(entry.get("caus_deriv", "")),
            pass_deriv   = str(entry.get("pass_deriv", "")),
            pronominal_n = bool(entry.get("pronominal_n", False)),
        ))
    return Lexicon(roots)
