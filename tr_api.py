"""
tr_api.py — High-level API for the Turkish morphological tokenizer.

This is the entry point for downstream callers (web demos, batch scripts,
third-party integrations) who don't want to deal with the parser's
internals. It loads the inventory, morphotactics, and lexicon once at
construction time and exposes JSON-serializable analyses.

Typical usage:

    from tr_api import Tokenizer
    tok = Tokenizer()              # loads default data files
    result = tok.tokenize("kitabımı")
    # result = {
    #   "surface": "kitabımı",
    #   "root": "kitap",
    #   "lemma": "kitap",              # alias for root (downstream convenience)
    #   "root_class": "NOUN",
    #   "final_class": "NOUN",
    #   "split": "kitab-ım-ı",
    #   "tagged": "kitab+NOUN-ım+POSS_1SG[...]-ı+ACC[...]",
    #   "morphemes": [
    #     {"chunk": "kitab", "id": null,        "feats": {...}, "is_root": true},
    #     {"chunk": "ım",    "id": "POSS_1SG",  "feats": {...}, "is_root": false},
    #     {"chunk": "ı",     "id": "ACC",       "feats": {...}, "is_root": false},
    #   ],
    #   "features": {"Case": "Acc", "Number": "Sing", ...},   # ud_feats
    #   "score": ...,
    #   "oov": false,
    #   "alternatives": [...]                                  # other analyses
    # }

The Tokenizer is constructed once and reused: per-word tokenize() calls
are stateless and thread-safe (the underlying Parser doesn't mutate
state across calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tr_inventory     import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon       import load_lexicon
from tr_parse         import Parser, ParseConfig, Analysis
from tr_pretokenize   import split_question_clitic
from tr_phonology      import fold_diacritics
from tr_fuzzy          import FuzzyIndex


HERE = Path(__file__).parent

# Morphology-aware stem-correction bounds (see Tokenizer.correct). Stems are
# fuzzy-corrected only within this length window, with a capped number of
# near roots per split, and each edit costs a score penalty so a cleaner
# parse with fewer edits wins.
_MIN_STEM_LEN = 2
_MAX_STEM_LEN = 10
_NEAR_ROOTS_PER_STEM = 4
_STEM_EDIT_PENALTY = 3.0
# Stem queries run once per candidate stem length, so they use a tight
# edit radius (one typo) to stay fast; multi-edit cases fall back to the
# single root-level suggestion query at the configured distance.
_STEM_MAX_DISTANCE = 1


@dataclass
class TokenizerConfig:
    """Configuration for the Tokenizer.

    `lexicon_path` defaults to lexicon_full.json (UD-extracted entries
    merged with the TDK Turkish dictionary headwords for broad coverage,
    ~62K entries). For honest benchmarking against test-set data use
    lexicon_train.json (UD-train only, no leakage).

    `include_alternatives` controls whether tokenize() also returns
    lower-scored analyses (capped at `max_alternatives`).
    """
    inventory_path:      Path = HERE / "inventory.json"
    morphotactics_path:  Path = HERE / "morphotactics.json"
    lexicon_path:        Path = HERE / "lexicon_full.json"
    include_alternatives: bool = True
    max_alternatives:    int = 5
    parser_config:       Optional[ParseConfig] = None
    # When True, tokenize_text() splits an attached interrogative particle
    # (gelecekmisin -> gelecek + misin) into separate tokens before
    # analysing. See tr_pretokenize for the (conservative) split rule.
    split_clitics:       bool = True
    # When True, an out-of-vocabulary word gets a "suggestions" list of
    # near in-lexicon corrections (typo / spelling help) via a BK-tree over
    # the root lexicon. The index is built lazily on first use.
    suggest_on_oov:      bool = True
    suggestion_max_distance: int = 2      # max edit distance for suggestions
    max_suggestions:     int = 3          # cap on suggestions returned


class Tokenizer:
    """High-level wrapper around Parser. Constructs the data layers
    once and exposes a JSON-friendly tokenize() entry point.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._inv   = load_inventory(self.config.inventory_path)
        self._graph = load_graph(self.config.morphotactics_path)
        self._lex   = load_lexicon(self.config.lexicon_path)
        # Construct two parser instances: one for top-only, one for
        # all-analyses. This avoids reconfiguring per-call.
        pc_top = self.config.parser_config or ParseConfig()
        pc_all = ParseConfig(**{**pc_top.__dict__, "return_all": True})
        self._parser_top = Parser(self._lex, self._inv, self._graph, pc_top)
        self._parser_all = Parser(self._lex, self._inv, self._graph, pc_all)
        # Fuzzy index over root forms, built lazily on the first suggestion
        # request (so tokenizing never pays for it unless OOV help is used).
        self._fuzzy: Optional[FuzzyIndex] = None
        # Maps each indexed (folded) form back to its canonical lemma, so
        # suggestions surface citation forms even though the index also holds
        # softened/variant stems (e.g. "kitab" -> "kitap").
        self._form_lemma: Dict[str, str] = {}

    def _fuzzy_index(self) -> FuzzyIndex:
        """Lazily build (and cache) the BK-tree over all indexed surface
        forms — canonical, variants, and softened stems — keyed on
        circumflex-folded forms. Indexing the softened/variant stems lets
        the morphology-aware corrector fix typos that sit next to a stem
        boundary (kitebımı -> kitabımı)."""
        if self._fuzzy is None:
            terms: Dict[str, int] = {}
            for key, lemma, freq in self._lex.stem_forms():
                if len(key) < 2:
                    continue
                if key not in terms or freq > terms[key]:
                    terms[key] = freq
                    self._form_lemma[key] = lemma
            self._fuzzy = FuzzyIndex(terms)
        return self._fuzzy

    def suggest(self, word: str) -> List[Dict[str, Any]]:
        """Near in-lexicon root corrections for `word`, best first.

        Returns a list of {"word", "distance"} dicts. The edit-distance
        ceiling is tightened for short words so a 3-4 letter word is not
        matched against everything two edits away.
        """
        word = fold_diacritics((word or "").strip().lower())
        if len(word) < 2:
            return []
        # Short words tolerate fewer edits (a 2-edit window on a 4-letter
        # word matches almost anything).
        max_d = 1 if len(word) <= 4 else self.config.suggestion_max_distance
        # Over-fetch, then map indexed forms (which include softened/variant
        # stems) back to canonical lemmas and de-duplicate, so the surfaced
        # suggestions are citation forms.
        hits = self._fuzzy_index().nearest(
            word, max_distance=max_d, limit=self.config.max_suggestions * 3)
        out: List[Dict[str, Any]] = []
        seen = set()
        for term, dist, _freq in hits:
            lemma = self._form_lemma.get(term, term)
            if lemma in seen:
                continue
            seen.add(lemma)
            out.append({"word": lemma, "distance": dist})
            if len(out) >= self.config.max_suggestions:
                break
        return out

    def correct(self, word: str) -> List[Dict[str, Any]]:
        """Morphology-aware correction of an OOV word.

        Turkish is agglutinative, so a typo usually sits in the STEM of a
        fully-inflected word ("mektobumu"). Matching the whole surface
        against root forms is hopeless; instead, for each plausible stem
        length we fuzzy-correct the prefix to a near in-lexicon root, glue
        the untouched suffix tail back on, and RE-PARSE. A candidate is
        kept only if the corrected word parses cleanly in-lexicon — the
        parser is the oracle that rejects illegal suffix chains. Candidates
        are ranked by parse score minus an edit-distance penalty.

        Returns full-word corrections, best first, each a dict with the
        corrected surface ("word"), its lemma, split, and the edit distance.
        """
        folded = fold_diacritics((word or "").strip().lower())
        if len(folded) < _MIN_STEM_LEN + 1:
            return []
        # A word that already parses in-lexicon needs no correction.
        whole = self._parser_top.parse(folded)
        if whole and not whole[0].oov:
            return []
        fuzzy = self._fuzzy_index()
        # corrected surface -> (score, Analysis, distance)
        best: Dict[str, tuple] = {}
        max_stem = min(len(folded), _MAX_STEM_LEN)
        for r in range(_MIN_STEM_LEN, max_stem + 1):
            stem, rest = folded[:r], folded[r:]
            for cand, dist, _freq in fuzzy.nearest(
                    stem, max_distance=_STEM_MAX_DISTANCE,
                    limit=_NEAR_ROOTS_PER_STEM):
                if cand == stem:
                    continue  # no stem typo at this split
                corrected = cand + rest
                analyses = self._parser_top.parse(corrected)
                if not analyses or analyses[0].oov:
                    continue
                top = analyses[0]
                score = top.score - dist * _STEM_EDIT_PENALTY
                prev = best.get(corrected)
                if prev is None or score > prev[0]:
                    best[corrected] = (score, top, dist)
        ranked = sorted(best.values(), key=lambda v: -v[0])[:self.config.max_suggestions]
        out: List[Dict[str, Any]] = []
        for score, top, dist in ranked:
            d = self._analysis_to_dict(top)
            # For a bare (single-morpheme) correction, show the canonical
            # lemma rather than the matched surface, which may be a softened
            # stem form (mektub) that is not a valid standalone word.
            corrected_word = top.root if len(top.morphemes) == 1 else top.surface
            out.append({
                "word":     corrected_word,
                "lemma":    top.root,
                "split":    d["split"],
                "tagged":   d["tagged"],
                "distance": dist,
            })
        return out

    def _oov_suggestions(self, word: str) -> List[Dict[str, Any]]:
        """Best available OOV help: morphology-aware corrections when the
        word inflects around a mistyped stem, else root-level suggestions."""
        corrections = self.correct(word)
        return corrections if corrections else self.suggest(word)

    def tokenize(self, word: str) -> Dict[str, Any]:
        """Tokenize a single word.

        Returns a JSON-serializable dict describing the top analysis
        plus any alternatives (if configured). If no parse succeeds
        (extremely rare — only on input the parser truly can't handle),
        returns a `parsed: False` shell with the surface preserved.
        """
        word = (word or "").strip()
        if not word:
            return {"surface": "", "parsed": False, "error": "empty input"}

        # Use the all-analyses parser when alternatives are wanted,
        # else top-only (faster).
        parser = (self._parser_all
                  if self.config.include_alternatives
                  else self._parser_top)
        analyses = parser.parse(word)
        if not analyses:
            shell = {"surface": word, "parsed": False, "error": "no parse"}
            if self.config.suggest_on_oov:
                shell["suggestions"] = self._oov_suggestions(word)
            return shell

        top = analyses[0]
        result = self._analysis_to_dict(top)
        result["parsed"] = True
        result["surface"] = word

        # Out-of-vocabulary: the parser fell back to an OOV root, so offer
        # near in-lexicon corrections (likely a typo or unknown word).
        if top.oov and self.config.suggest_on_oov:
            result["suggestions"] = self._oov_suggestions(word)

        if self.config.include_alternatives:
            # Only include alternatives that are reasonably competitive
            # with the top: same order of magnitude, in-lexicon, and
            # not duplicating the top's root + suffix shape.
            top_root_class = (top.root, top.root_class,
                              tuple(m.suffix_id for m in top.morphemes))
            alts = []
            for a in analyses[1:]:
                if a.oov:
                    continue
                # Skip near-duplicates of the top.
                sig = (a.root, a.root_class,
                       tuple(m.suffix_id for m in a.morphemes))
                if sig == top_root_class:
                    continue
                # Cap at max_alternatives once filtered.
                alts.append(self._analysis_to_dict(a))
                if len(alts) >= self.config.max_alternatives:
                    break
            result["alternatives"] = alts

        return result

    def tokenize_batch(self, words: List[str]) -> List[Dict[str, Any]]:
        """Tokenize a list of words. Convenience wrapper over tokenize()."""
        return [self.tokenize(w) for w in words]

    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """Tokenize a full sentence or paragraph.

        Splits `text` into tokens on whitespace and punctuation,
        preserves the punctuation in order, and runs tokenize() on each
        word-shaped token. Returns:

            {
              "text": "...",
              "tokens": [
                {"kind": "word", "surface": "...", "analysis": {...}},
                {"kind": "punct", "surface": ",", "analysis": null},
                {"kind": "space", "surface": " ",  "analysis": null},
                ...
              ]
            }

        The token list preserves the original text exactly (concatenating
        all surfaces reconstructs `text`), which makes it easy for a
        renderer to lay out the morpheme-level breakdown inline with
        the original punctuation and spacing.
        """
        text = text or ""
        tokens: List[Dict[str, Any]] = []
        for surface, kind in _split_text(text):
            if kind == "word":
                # Split an attached interrogative particle into its own
                # token (gelecekmisin -> gelecek + misin). The pieces
                # concatenate back to `surface`, so text reconstruction is
                # preserved. Disabled words yield a single piece.
                pieces = (split_question_clitic(surface, self._parser_top)
                          if self.config.split_clitics else [surface])
                for piece in pieces:
                    tokens.append({
                        "kind": "word",
                        "surface": piece,
                        "analysis": self.tokenize(piece),
                    })
            else:
                tokens.append({
                    "kind": kind,
                    "surface": surface,
                    "analysis": None,
                })
        return {"text": text, "tokens": tokens}

    @staticmethod
    def _analysis_to_dict(a: Analysis) -> Dict[str, Any]:
        """Convert an Analysis to a JSON-serializable dict.

        Includes both `features` (UD-compliant with defaults) and
        `emitted_features` (raw morpheme-emitted, no defaults), so
        downstream code can pick. Morpheme entries include their
        suffix id, surface chunk, and the feature pairs they emit.
        """
        morphemes = []
        for m in a.morphemes:
            morphemes.append({
                "chunk":   m.chunk,
                "id":      m.suffix_id,
                "feats":   dict(m.feats),
                "is_root": m.suffix_id is None,
            })
        return {
            "root":             a.root,
            "lemma":            a.root,   # alias: the bare dictionary form,
                                          # convenient for downstream callers
                                          # that only want the lemma string.
            "root_class":       a.root_class,
            "final_class":      a.final_class,
            "morphemes":        morphemes,
            "split":            a.split(),
            "tagged":           a.tagged(),
            "features":         a.ud_feats(),
            "emitted_features": a.emitted_feats(),
            "score":            round(a.score, 3),
            "oov":              a.oov,
        }


# Module-level default instance for one-line use:
#     from tr_api import tokenize
#     tokenize("kitabımı")
# Constructed lazily on first call.

_default_tokenizer: Optional[Tokenizer] = None


def tokenize(word: str) -> Dict[str, Any]:
    """Tokenize a word using a module-level default Tokenizer. Lazy
    initialization: the first call constructs the Tokenizer (loading
    data files); subsequent calls reuse it.
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()
    return _default_tokenizer.tokenize(word)


__all__ = ["Tokenizer", "TokenizerConfig", "tokenize"]


# ---------------------------------------------------------------------------
# Text tokenization: split a string into (surface, kind) chunks where kind
# is "word", "space", or "punct". Preserves order and exact surface so the
# original text can be reconstructed by concatenating all surfaces.
# ---------------------------------------------------------------------------

# Turkish word character set: ASCII letters + Turkish-specific letters
# (lowercase and uppercase) + digits + apostrophe (proper-noun separator).
_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "çğıöşüÇĞİÖŞÜ"
                  "0123456789"
                  "'’")  # both straight and curly apostrophe


def _classify(ch: str) -> str:
    if ch in _WORD_CHARS:
        return "word"
    if ch.isspace():
        return "space"
    return "punct"


def _split_text(text: str):
    """Yield (chunk, kind) pairs covering `text` exactly.

    Adjacent characters of the same kind form one chunk; the boundary
    between word/space/punct kinds is where a new chunk starts. So
    'Kitabımı gördüm.' yields:
        ('Kitabımı', 'word')
        (' ', 'space')
        ('gördüm', 'word')
        ('.', 'punct')
    """
    if not text:
        return
    buf = [text[0]]
    cur_kind = _classify(text[0])
    for ch in text[1:]:
        k = _classify(ch)
        if k == cur_kind:
            buf.append(ch)
        else:
            yield "".join(buf), cur_kind
            buf = [ch]
            cur_kind = k
    yield "".join(buf), cur_kind
