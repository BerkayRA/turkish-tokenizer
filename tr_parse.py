"""
tr_parse.py — Turkish morphological parser.

Algorithm: chart-style dynamic programming.

The chart is indexed by position in the surface word: chart[i] holds all
parse fragments that consume exactly word[0:i]. Each fragment carries:
  - the current FSM state (where in the morphotactic graph)
  - the current word_class (the class as of the most recent derivation step)
  - the running list of morphemes applied so far (root + suffix steps)
  - the OOV flag (True if the root was guessed rather than looked up)

Seeding: for each prefix of the surface word, check whether it matches a
lexicon entry (canonical or variant). Each match seeds chart[len(prefix)]
with the corresponding root analysis.

Also seed OOV roots: for each prefix length 1..L, seed an "unknown root" at
each position. These start with a low score and only succeed if downstream
suffix-matching consumes the rest of the word into an accepting state.

Extension: for each chart[i] cell, look at outgoing transitions from its
state in the morphotactic graph. For each candidate suffix, try to match its
surface realization against word[i:i+k] for plausible k. A match advances
the chart to position i+k with the new state.

Suffix matching uses a unification approach: we walk the template, allowing
archiphonemes (A/H/D/C) to match harmonically-appropriate characters in
the surface, and handling buffer consonants and stem-final softening.

Output: collect all chart[len(word)] entries whose state is accepting and
whose word_class is consistent with the (overall) parse.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

from tr_inventory  import Inventory, Suffix
from tr_lexicon    import Lexicon, Root
from tr_morphotactics import MorphoGraph
from tr_phonology  import (
    BACK_VOWELS, FRONT_VOWELS, HIGH_VOWELS, LOW_VOWELS,
    ROUNDED_VOWELS, UNROUNDED_VOWELS, VOICELESS_CONSONANTS, VOWELS,
    HARDEN, SOFTEN,
    is_vowel, is_voiceless, last_vowel, tr_lower,
)
from tr_rules import get_expand


# -----------------------------------------------------------------------------
# Parse fragment + final analysis
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Morpheme:
    """One morpheme in a parse: either a root (suffix_id is None) or a
    suffix application."""
    chunk:       str
    suffix_id:   Optional[str] = None  # None for root morphemes
    root_form:   Optional[str] = None  # set only on root morphemes
    word_class:  Optional[str] = None  # set on root and on derivation transitions
    oov:         bool = False
    feats:       Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ParseFragment:
    """A partial parse: covers word[0:end], ends in `state`, with current
    word_class `word_class`.

    `root_ctx` carries root-level flags (aorist_high, caus_deriv, etc.)
    that need to reach context-sensitive rules later in the chain. It's
    populated when the fragment is seeded from a lexicon root.

    `next_must_be` constrains what the NEXT suffix can be. When set
    (non-empty), only the listed suffix ids are allowed as continuations.
    Used by rules that produce suppletive forms only valid in specific
    contexts — currently POT's -(y)A suppletive realization which only
    surfaces before NEG.
    """
    end:           int
    state:         str
    word_class:    str
    morphemes:     Tuple[Morpheme, ...]
    has_oov:       bool = False
    score:         float = 0.0
    root_ctx:      Tuple[Tuple[str, str], ...] = ()   # frozen for hashability
    next_must_be:  Tuple[str, ...] = ()                # constrained continuations

    def with_suffix(self, suffix_id: str, chunk: str,
                    new_end: int, new_state: str,
                    new_word_class: str,
                    feats: Tuple[Tuple[str, str], ...],
                    score_delta: float,
                    next_must_be: Tuple[str, ...] = ()) -> "ParseFragment":
        return replace(
            self,
            end=new_end,
            state=new_state,
            word_class=new_word_class,
            morphemes=self.morphemes + (Morpheme(
                chunk=chunk, suffix_id=suffix_id,
                word_class=new_word_class,
                feats=feats,
            ),),
            score=self.score + score_delta,
            next_must_be=next_must_be,
        )


@dataclass
class Analysis:
    """A complete parse of a word."""
    surface:     str
    root:        str
    root_class:  str       # word class of the root (NOUN/VERB/ADJ)
    final_class: str       # word class AFTER all derivations have applied;
                           # equals root_class if no derivation happened.
                           # This is what ud_feats() uses to pick defaults,
                           # since e.g. a verbal noun (yazmak = yaz+NMZ_INF)
                           # surfaces as a noun and takes nominal features.
    morphemes:   List[Morpheme]
    oov:         bool
    score:       float

    def split(self) -> str:
        pieces = [m.chunk for m in self.morphemes if m.chunk]
        return "-".join(pieces)

    def tagged(self) -> str:
        out = []
        for m in self.morphemes:
            if m.suffix_id is None:
                # Root
                tag = f"{m.word_class}"
                if m.oov:
                    tag += "[UNK]"
                out.append(f"{m.chunk}+{tag}")
            else:
                feat_str = ",".join(f"{k}={v}" for k, v in m.feats)
                tag = f"{m.suffix_id}[{feat_str}]" if feat_str else m.suffix_id
                out.append(f"{m.chunk}+{tag}" if m.chunk else f"∅+{tag}")
        return "-".join(out)

    def emitted_feats(self) -> Dict[str, str]:
        """Union of features emitted by the morphemes in this parse. No
        defaults applied. This is the morphological-faithful view: a feature
        is present iff some suffix emitted it."""
        out: Dict[str, str] = {}
        for m in self.morphemes:
            for k, v in m.feats:
                if k.startswith("_"):
                    # Internal markers (e.g., _derivation) — skip.
                    continue
                out[k] = v
        return out

    def ud_feats(self) -> Dict[str, str]:
        """UD-compliant feature dict: fills in defaults the way UD-Turkish
        annotates them. Drops Evident=Fh (UD only marks Nfh).

        Three default paradigms:
          - VERB defaults (Mood=Ind, Polarity=Pos, etc.): apply when the
            morphologically-final class is VERB, OR when the parse is
            participial (VerbForm in {Part, Vnoun, Conv, Inf}) — UD-IMST
            tags participial forms as UPOS=VERB with verbal features.
          - NOUN defaults (Case=Nom, Number=Sing, Person=3): apply only
            for true nominal final-class WITHOUT VerbForm. Participial
            forms get any explicit Case/POSS features but no nominal
            defaults.
          - ADJ: like NOUN if nominal features are present.

        Derived multi-morpheme features:
          - Tense=Pqp: when EVID and PAST_COP both fire (söylemişti
            "had said"), UD-IMST uses Tense=Pqp (pluperfect).
          - Voice=CauPass: when CAUS and PASS both fire.
        """
        out = self.emitted_feats()
        if out.get("Evident") == "Fh":
            del out["Evident"]

        # Derived multi-morpheme features.
        suffix_ids = [m.suffix_id for m in self.morphemes if m.suffix_id]
        if "EVID" in suffix_ids and "PAST_COP" in suffix_ids:
            out["Tense"] = "Pqp"
        if "CAUS" in suffix_ids and "PASS" in suffix_ids:
            out["Voice"] = "CauPass"

        is_participial = out.get("VerbForm") in ("Part", "Vnoun", "Conv", "Inf")

        if self.final_class == "VERB" or is_participial:
            # Verbal defaults. These apply to both finite verbs and
            # participial forms; UD-IMST tags both as UPOS=VERB.
            out.setdefault("Mood",     "Ind")
            out.setdefault("Polarity", "Pos")
            out.setdefault("Aspect",   "Perf")
            out.setdefault("Tense",    "Pres")
            if is_participial:
                # Verbal nouns (Vnoun) ARE nouns morphologically: they
                # always take a case marker (Nom by default). Other
                # participial forms (Part, Conv) don't default Case;
                # they only have it when an explicit case suffix
                # attached.
                if out.get("VerbForm") == "Vnoun":
                    out.setdefault("Case", "Nom")
            else:
                # Finite verb: person/number defaults.
                out.setdefault("Person", "3")
                out.setdefault("Number", "Sing")
        elif self.final_class == "NOUN":
            out.setdefault("Case",     "Nom")
            out.setdefault("Number",   "Sing")
            out.setdefault("Person",   "3")
        elif self.final_class == "ADJ":
            nominal = {"Case", "Number", "Person",
                       "Number[psor]", "Person[psor]"}
            if any(k in out for k in nominal):
                out.setdefault("Case",   "Nom")
                out.setdefault("Number", "Sing")
                out.setdefault("Person", "3")
        return out


# -----------------------------------------------------------------------------
# Suffix matching against a surface slice
# -----------------------------------------------------------------------------

def _matches_A(c: str, last_v: Optional[str]) -> bool:
    """A matches a/e by backness harmony."""
    if c == "a":
        return last_v is None or last_v in BACK_VOWELS
    if c == "e":
        return last_v is None or last_v in FRONT_VOWELS
    return False


def _matches_H(c: str, last_v: Optional[str]) -> bool:
    """H matches one of ı/i/u/ü by full harmony."""
    if c not in HIGH_VOWELS:
        return False
    back    = (last_v in BACK_VOWELS)    if last_v else None
    rounded = (last_v in ROUNDED_VOWELS) if last_v else None
    if last_v is None:
        return True   # no context: accept any
    if back and not rounded:    return c == "ı"
    if back and     rounded:    return c == "u"
    if not back and not rounded: return c == "i"
    return c == "ü"


def _matches_D(c: str, prev: str) -> bool:
    if c == "t": return prev in VOICELESS_CONSONANTS
    if c == "d": return prev not in VOICELESS_CONSONANTS
    return False


def _matches_C(c: str, prev: str) -> bool:
    if c == "ç": return prev in VOICELESS_CONSONANTS
    if c == "c": return prev not in VOICELESS_CONSONANTS
    return False


def _matches_G(c: str, prev: str) -> bool:
    """G → 'k' after voiceless, 'g' otherwise."""
    if c == "k": return prev in VOICELESS_CONSONANTS
    if c == "g": return prev not in VOICELESS_CONSONANTS
    return False


def match_suffix(
    surface:     str,
    template:    str,
    pos:         int,
    running:     str,
    can_soften:  bool,
    a_deletable: bool = False,
) -> List[Tuple[int, str, bool]]:
    """Try to match `template` against surface[pos:].

    Returns a list of (new_pos, matched_chunk, softening_happened) tuples.
    Multiple results are possible due to:
      - buffer realized vs suppressed
      - retroactive low-vowel deletion (when this suffix is followed by
        one whose rules include `delete_stem_final_low_vowel`, the final
        A of this suffix's template is absent in the surface). Only
        attempted when `a_deletable=True` — without that flag every
        A-final template would spuriously match empty, polluting parses
        of forms like "süre" with "sür+OPT" (OPT consuming nothing).

    Note: initial-H drop after vowel-final stems (e.g., araba+Hm → arabam)
    is handled upstream by the rule registry. The parser calls each
    suffix's forward rules (including `drop_initial_H_after_vowel_stem`)
    before invoking this function, so the template passed here already
    has its leading H removed when applicable. There's no separate
    fallback in this function.
    """
    results: List[Tuple[int, str, bool]] = []
    results.extend(_match_template_once(surface, template, pos, running))
    # Retroactive deletion alternative: only fires for suffixes whose
    # final low vowel can actually be deleted by a following suffix
    # (currently just NEG -mA before PROG -Hyor).
    if a_deletable and template.endswith("A") and len(template) > 1:
        results.extend(_match_template_once(surface, template[:-1], pos, running))
    return results


def _match_template_once(
    surface:   str,
    template:  str,
    pos:       int,
    running:   str,
) -> List[Tuple[int, str, bool]]:
    """Core matching of one template (no retroactive-deletion variants)."""
    results: List[Tuple[int, str, bool]] = []
    start_last_v = last_vowel(running)
    start_prev = running[-1] if running else ""

    branches: List[Tuple[int, int, Optional[str], str, bool]] = [
        (0, pos, start_last_v, start_prev, False)
    ]

    while branches:
        ti, si, last_v, prev, softened = branches.pop()

        if ti >= len(template):
            # Successfully consumed the whole template.
            chunk = surface[pos:si]
            results.append((si, chunk, softened))
            continue

        tc = template[ti]

        # Handle softening at the very start: if template starts with a
        # vowel (or with a buffer that suppresses and reveals a vowel), the
        # stem-final consonant might have softened. We try both:
        # the surface shows the softened consonant in the running stem already;
        # the parser sees that the running stem ended with the softened char,
        # so when undoing we need to harden it back. This is handled at the
        # parse_word level rather than here, by adjusting the root.

        # Buffer group (X) at template start or after a vowel:
        if tc == "(":
            # Find the closing paren and the buffer char inside.
            end = template.index(")", ti)
            buffer_c = template[ti + 1:end]
            next_ti = end + 1

            # Buffer realization is phonologically deterministic:
            #   - after vowel-final stem: buffer MUST be realized
            #   - after consonant-final stem: buffer MUST be suppressed
            # `prev` is the last surface char of the stem at this position.
            # If `prev` is empty (start of word), we can't tell — accept both.
            if not prev:
                # Edge: template-initial buffer with no stem context.
                if si < len(surface) and surface[si] == buffer_c:
                    branches.append((next_ti, si + 1, last_v, buffer_c, softened))
                branches.append((next_ti, si, last_v, prev, softened))
            elif is_vowel(prev):
                # Vowel-final stem: buffer MUST be realized.
                if si < len(surface) and surface[si] == buffer_c:
                    branches.append((next_ti, si + 1, last_v, buffer_c, softened))
                # Note: NO suppression branch. Suppressing the buffer after
                # a vowel-final stem would let a template like (y)A match
                # the empty string against any vowel-final stem, polluting
                # POT-suppletive matches and similar.
            else:
                # Consonant-final stem: buffer MUST be suppressed.
                branches.append((next_ti, si, last_v, prev, softened))
            continue

        # Ran out of surface but still have template: dead branch.
        if si >= len(surface):
            continue

        sc = surface[si]

        if tc == "A":
            if _matches_A(sc, last_v):
                branches.append((ti + 1, si + 1, sc, sc, softened))
        elif tc == "H":
            if _matches_H(sc, last_v):
                branches.append((ti + 1, si + 1, sc, sc, softened))
        elif tc == "D":
            if _matches_D(sc, prev):
                branches.append((ti + 1, si + 1, last_v, sc, softened))
        elif tc == "C":
            if _matches_C(sc, prev):
                branches.append((ti + 1, si + 1, last_v, sc, softened))
        elif tc == "G":
            if _matches_G(sc, prev):
                branches.append((ti + 1, si + 1, last_v, sc, softened))
        else:
            # Literal character.
            if sc == tc:
                nv = sc if sc in VOWELS else last_v
                branches.append((ti + 1, si + 1, nv, sc, softened))
            # Allow stem/suffix-final softening: template wants 'k' but
            # surface shows 'ğ' (etc.) when a vowel-initial suffix follows.
            # We try this whenever the template char is one of k/p/t/ç, and
            # let downstream filtering reject parses where no vowel-initial
            # suffix actually follows.
            elif tc in SOFTEN and sc == SOFTEN[tc]:
                nv = last_v   # softened chars are consonants
                branches.append((ti + 1, si + 1, nv, sc, True))

    return results


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------

@dataclass
class ParseConfig:
    """Parser configuration knobs.

    Scoring model (designed so different decompositions compete fairly):
      - Per-suffix chunk: +0.5 per matched character − 0.2 baseline. So
        suffixes are rewarded by how much surface they consume, not by
        count alone. Empty chunks (e.g., phantom NOM after POSS) cost
        −0.2.
      - In-lexicon root: log1p(frequency) + len(root)*inlex_char_bonus.
        The per-character term lets a longer in-lex lemma compete fairly
        against a shorter (possibly more frequent) root + derivation
        chain that covers the same surface.
      - In-lexicon BARE root (no suffixes attached): an extra
        bare_root_bonus is added to the final analysis score.
      - OOV root: oov_penalty + len(root) * oov_char_bonus. The
        per-character term ensures longer OOV roots are preferred when
        they could otherwise be shaved down by adding phantom suffixes
        (e.g., "Osman" beating "osma + POSS_2SG[H-dropped]").
      - Derivation: derivation_penalty per cross-class step.
      - Voice derivation: voice_penalty per CAUS or PASS application.
        Biases against speculative voice-decomposition when a longer
        in-lex lemma would cover the same surface (e.g., preventing
        'çıkardı' = 'çık+CAUS+PAST' from beating 'çıkar+PAST', since
        çıkar is a lexicalized verb in its own right).
    """
    return_all:           bool = False    # if False, return only top-scoring
    max_oov_root_len:     int = 12         # cap OOV root candidate length
    min_root_len:         int = 1
    oov_penalty:          float = -50.0    # base score for OOV root
    oov_char_bonus:       float = 0.35     # per-char bonus on OOV roots
    inlex_char_bonus:     float = 0.15     # per-char bonus on in-lex roots
    bare_root_bonus:      float = 0.5      # bonus for in-lex root + no suffixes
    derivation_penalty:   float = -2.0     # per derivation step
    # Some derivations are extremely productive in modern Turkish: their
    # decomposition is essentially predictable from the surface and the
    # base is clearly visible. For those, we add a bonus that more than
    # offsets the generic derivation_penalty, so the decomposed parse
    # wins over a lexicalized headword for forms like heyecanlı, çaresiz,
    # gazeteci, iyilik, kitapçık. The bonus is tuned to make the
    # decomposed parse beat the lexicalized one even when the lexicalized
    # surface is in TDK (and thus has a frequency-weighted lex score).
    # Note: VBZ_LA, VBZ_LAS, VBZ_LAN intentionally NOT here even though
    # they're productive — including them causes elma+lAr (plural) to
    # spuriously parse as elma+VBZ_LA+AOR (-r). The verbalizing
    # derivations are kept at the default penalty so plural always wins.
    productive_derivations: tuple = ("ADJZ_LH", "ADJZ_SHZ", "ADJZ_MSH",
                                     "NDER_CH", "NDER_LHK", "NDER_CHK",
                                     "NDER_DAS", "ADV_CA")
    productivity_bonus:   float = 2.5      # offsets derivation_penalty
    # Conversely, some derivations are RARE (only a few lexicalized
    # examples) and including them in the parser's search space too
    # eagerly causes spurious decompositions. NMZ_HNTI (-Hntı residue)
    # is the worst offender: it has ~10 lexicalized forms in Turkish
    # but the template Hntı matches the end of many regular surfaces
    # (al+ın+dı looks like al+ındı under NMZ_HNTI). For these, we add
    # an extra penalty on top of derivation_penalty so they only fire
    # when no better parse exists.
    rare_derivations:     tuple = ("NMZ_HNTI", "NMZ_MACA", "NMZ_DHKCE",
                                   "ADJZ_MTRK", "ADJZ_CHL", "ADJ_GAN",
                                   "ADJ_MAZ", "VBZ_DA", "VBZ_HMSA",
                                   "VMOD_GEL", "VMOD_DUR", "VMOD_YAZ",
                                   "NDER_CAGIZ", "NDER_GHL")
    rare_derivation_extra_penalty: float = -3.0
    voice_penalty:        float = -2.2     # per CAUS or PASS application
                                            # (set slightly lower than
                                            # compound_tense_penalty so
                                            # CAUS+PAST wins over
                                            # AOR+PAST_COP in surface
                                            # ties for pruned forms)
    compound_tense_penalty: float = -2.5   # per PAST_COP application
                                            # (TAM+copula compound past;
                                            # slightly stronger than
                                            # voice_penalty so CAUS+PAST
                                            # wins over AOR+PAST_COP in
                                            # surface-tied cases)
    suffix_bonus:         float = 1.0      # legacy; superseded by per-char scoring


class Parser:
    """Turkish morphological parser. Initialize once with the lexicon,
    inventory, and graph; then call parse(word) repeatedly."""

    def __init__(
        self,
        lexicon:    Lexicon,
        inventory:  Inventory,
        graph:      MorphoGraph,
        config:     Optional[ParseConfig] = None,
    ):
        self.lex   = lexicon
        self.inv   = inventory
        self.graph = graph
        self.cfg   = config or ParseConfig()

        # Pre-build: for each FSM state, list outgoing (suffix_id, to_state).
        self._out: Dict[str, List[Tuple[str, str]]] = {}
        for t in graph.all_transitions():
            for fs in t.from_states:
                self._out.setdefault(fs, []).append((t.via, t.to_state))

    # --- public API ---

    def parse(self, word: str) -> List[Analysis]:
        """Parse `word` and return all analyses, ranked by score.

        With return_all=False (default), only top-scoring analyses are
        returned (ties broken by frequency).
        """
        word = tr_lower(word)
        # Turkish uses apostrophe to separate proper-noun stems from their
        # inflectional suffixes ("Muammer'in", "Parkı'ndan"). The
        # apostrophe is zero-width morphologically; strip it before parsing.
        for apo in ("'", "\u2019", "\u2032"):
            if apo in word:
                word = word.replace(apo, "")
        chart = self._fill_chart(word)
        analyses = self._collect(chart, word)
        analyses.sort(key=lambda a: -a.score)
        if not self.cfg.return_all and analyses:
            top = analyses[0].score
            analyses = [a for a in analyses if a.score >= top]
        return analyses

    # --- chart filling ---

    def _fill_chart(self, word: str) -> Dict[int, List[ParseFragment]]:
        chart: Dict[int, List[ParseFragment]] = {}

        # --- Seed: lexicon prefix matches ---
        for root, surf, prefix_len in self.lex.prefix_match(word):
            start_state = self.graph.start_state(root.word_class)
            morpheme = Morpheme(
                chunk=surf, suffix_id=None,
                root_form=root.form,
                word_class=root.word_class,
                feats=(),
            )
            frag = ParseFragment(
                end=prefix_len,
                state=start_state,
                word_class=root.word_class,
                morphemes=(morpheme,),
                has_oov=False,
                # Score: log-frequency + per-character bonus. The per-char
                # term lets a longer in-lex lemma compete with shorter
                # roots + decomposition that cover the same surface.
                score=(_freq_score(root.frequency)
                       + prefix_len * self.cfg.inlex_char_bonus),
                root_ctx=tuple(root.root_ctx().items()),
            )
            chart.setdefault(prefix_len, []).append(frag)

        # --- Seed: PROG-truncated vowel-final verbs ---
        # PROG's -Hyor suffix deletes the stem's final low vowel: başla +
        # Hyor → başlıyor (not *başlayıyor). The lexicon trie indexes the
        # canonical form 'başla', but the surface 'başlıyor' starts with
        # 'başl' (truncated stem) + 'ıyor'. To find the verb, look up roots
        # whose form is `word[:k] + 'a'` or `word[:k] + 'e'` for some k,
        # and seed them with the constraint that PROG must follow.
        for plen in range(2, min(self.cfg.max_oov_root_len, len(word))):
            # Surface position `plen` is the truncated stem's end.
            for low_v in ("a", "e"):
                candidate = word[:plen] + low_v
                roots = self.lex.get(candidate)
                for root in roots:
                    if root.word_class != "VERB":
                        continue
                    start_state = self.graph.start_state("VERB")
                    morpheme = Morpheme(
                        chunk=word[:plen], suffix_id=None,
                        root_form=root.form,
                        word_class="VERB",
                        feats=(),
                    )
                    frag = ParseFragment(
                        end=plen,
                        state=start_state,
                        word_class="VERB",
                        morphemes=(morpheme,),
                        has_oov=False,
                        score=(_freq_score(root.frequency)
                               + plen * self.cfg.inlex_char_bonus),
                        root_ctx=tuple(root.root_ctx().items()),
                        next_must_be=("PROG",),
                    )
                    chart.setdefault(plen, []).append(frag)

        # --- Seed: OOV roots ---
        # Try each prefix length (within bounds) for each plausible class.
        # Class is chosen based on whether subsequent suffixes will license
        # it — we seed all three and let the graph filter.
        for plen in range(self.cfg.min_root_len,
                          min(self.cfg.max_oov_root_len, len(word)) + 1):
            prefix = word[:plen]
            # Skip if a lexicon match already covers this position (less
            # ambitious — but lexicon matches are still preferred via score).
            for word_class in ("NOUN", "VERB", "ADJ"):
                start_state = self.graph.start_state(word_class)
                morpheme = Morpheme(
                    chunk=prefix, suffix_id=None,
                    root_form=prefix,
                    word_class=word_class,
                    oov=True,
                    feats=(),
                )
                frag = ParseFragment(
                    end=plen,
                    state=start_state,
                    word_class=word_class,
                    morphemes=(morpheme,),
                    has_oov=True,
                    score=self.cfg.oov_penalty + plen * self.cfg.oov_char_bonus,
                )
                chart.setdefault(plen, []).append(frag)

        # --- Extension: propagate through suffix transitions ---
        # Process cells in increasing position order. Each cell may extend
        # to higher positions. We loop until no changes.
        changed = True
        while changed:
            changed = False
            positions = sorted(chart.keys())
            for pos in positions:
                # Snapshot the cell since we may add to other cells.
                fragments = list(chart[pos])
                for frag in fragments:
                    for suffix_id, to_state in self._out.get(frag.state, []):
                        # Constraint check: if this fragment requires its
                        # next morpheme to be in a specific set (set by a
                        # suppletion rule on the previous step), filter
                        # accordingly.
                        if frag.next_must_be and suffix_id not in frag.next_must_be:
                            continue
                        suffix = self.inv.get(suffix_id)
                        # Apply this suffix's rules in their parse-time
                        # (expand) view. Most rules produce ONE template
                        # outcome (deterministic, like H-drop or pronominal-n);
                        # some produce multiple alternatives (e.g., aorist
                        # allomorphy returns both -Ar and -Hr for consonant-
                        # final stems). We try each alternative.
                        running = word[:pos]
                        prev_morph = frag.morphemes[-1] if frag.morphemes else None
                        prev_morph_id = prev_morph.suffix_id if prev_morph else None
                        prev_morph_chunk = prev_morph.chunk if prev_morph else None
                        # Second-back morpheme (for two-step suppletion checks
                        # like NEG → AOR → 1SG_Z, where 1SG_Z needs to know
                        # it's downstream of a NEG-AOR sequence).
                        prev_prev_morph = (frag.morphemes[-2]
                                           if len(frag.morphemes) >= 2 else None)
                        prev_prev_morph_id = (prev_prev_morph.suffix_id
                                              if prev_prev_morph else None)
                        # Build ctx from root-level flags + previous-morpheme info.
                        base_ctx = dict(frag.root_ctx)
                        base_ctx["prev_morph_id"] = prev_morph_id
                        base_ctx["prev_morph_chunk"] = prev_morph_chunk
                        base_ctx["prev_prev_morph_id"] = prev_prev_morph_id
                        # alternatives is a list of (template, ctx) pairs.
                        alternatives = [(suffix.template, base_ctx)]
                        for rule_name in suffix.rules:
                            expand_fn = get_expand(rule_name)
                            next_alts = []
                            for tmpl, c in alternatives:
                                next_alts.extend(expand_fn(running, tmpl, c))
                            alternatives = next_alts
                        # Try to match each surviving alternative template.
                        # Track which alternative's ctx produced each match so
                        # we can thread constraints (like _next_must_be) onto
                        # the resulting fragment.
                        matches = []  # list of (new_pos, chunk, soft, alt_ctx)
                        # If this suffix is one that deletes the stem's
                        # final low vowel (currently only PROG -Hyor), and
                        # the running stem ends in a/e, ALSO try matching
                        # with the running stem virtually truncated. The
                        # matched chunk in that case includes the position
                        # consumed in the surface, which already represents
                        # the deleted vowel + the suffix realization.
                        does_low_vowel_delete = "delete_stem_final_low_vowel" in suffix.rules
                        for tmpl, alt_ctx in alternatives:
                            for new_pos, chunk, soft in match_suffix(
                                surface=word,
                                template=tmpl,
                                pos=pos,
                                running=running,
                                can_soften=False,
                                a_deletable=suffix.a_deletable,
                            ):
                                matches.append((new_pos, chunk, soft, alt_ctx))
                            # Stem-final low-vowel deletion alternative:
                            # if the running stem ends in 'a'/'e' AND the
                            # next surface char is the buffer's spot that
                            # would normally hold the deleted vowel, try
                            # treating the stem as truncated. The 'chunk'
                            # then absorbs the position normally occupied
                            # by the deleted vowel onward.
                            if (does_low_vowel_delete
                                    and running and running[-1] in ("a", "e")):
                                # Virtual truncation: pretend running ended one char earlier.
                                trunc_running = running[:-1]
                                for new_pos, chunk, soft in match_suffix(
                                    surface=word,
                                    template=tmpl,
                                    pos=pos,
                                    running=trunc_running,
                                    can_soften=False,
                                    a_deletable=suffix.a_deletable,
                                ):
                                    matches.append((new_pos, chunk, soft, alt_ctx))
                        for new_pos, chunk, _soft, alt_ctx in matches:
                            new_class = suffix.class_out or frag.word_class
                            # Per-character coverage bonus (so total reward
                            # for covering a span is roughly independent of
                            # how many morphemes it's split across). Empty
                            # chunks get a small dispreference (kills
                            # phantom NOM-after-POSS chains).
                            if chunk:
                                score_delta = 0.5 * len(chunk) - 0.2
                            else:
                                score_delta = -0.2
                            if suffix.class_out and suffix.class_out != frag.word_class:
                                score_delta += self.cfg.derivation_penalty
                            # Productivity bonus: extremely productive
                            # derivations get a bonus that more than
                            # offsets the generic derivation_penalty, so
                            # forms like heyecanlı → heyecan+ADJZ_LH win
                            # over the lexicalized headword reading.
                            if suffix_id in self.cfg.productive_derivations:
                                score_delta += self.cfg.productivity_bonus
                            # Rare-derivation extra penalty: rarely-attested
                            # derivations (NMZ_HNTI etc.) shouldn't fire
                            # speculatively. Only let them win when no
                            # other reading exists.
                            if suffix_id in self.cfg.rare_derivations:
                                score_delta += self.cfg.rare_derivation_extra_penalty
                            # Voice derivation (CAUS/PASS): pay a penalty
                            # so the parser doesn't speculatively decompose
                            # lexicalized verbs like 'çıkar' as 'çık+CAUS'
                            # when both readings are available.
                            if suffix_id in ("CAUS", "PASS"):
                                score_delta += self.cfg.voice_penalty
                            # Compound tense (TAM + PAST_COP "used to X /
                            # would X"): pay a penalty so plain Past
                            # reading wins for ambiguous surfaces ending
                            # in -ardı/-irdi/etc. (treebank distribution
                            # favors plain Past 2:1 over Hab+Past).
                            if suffix_id == "PAST_COP":
                                score_delta += self.cfg.compound_tense_penalty
                            # Suppletion constraint: if the matched
                            # alternative was tagged with _next_must_be in
                            # ctx, the resulting fragment can only extend
                            # via the listed suffixes.
                            constraint = alt_ctx.get("_next_must_be", ()) if alt_ctx else ()
                            new_frag = frag.with_suffix(
                                suffix_id=suffix_id,
                                chunk=chunk,
                                new_end=new_pos,
                                new_state=to_state,
                                new_word_class=new_class,
                                feats=tuple(sorted(suffix.feats.items())),
                                score_delta=score_delta,
                                next_must_be=tuple(constraint),
                            )
                            if not _already_in(chart, new_pos, new_frag):
                                chart.setdefault(new_pos, []).append(new_frag)
                                changed = True
        return chart

    # --- collection ---

    def _collect(self, chart: Dict[int, List[ParseFragment]],
                 word: str) -> List[Analysis]:
        N = len(word)
        out: List[Analysis] = []
        seen = set()
        for frag in chart.get(N, []):
            if not self.graph.is_accepting(frag.state):
                continue
            # Build the Analysis.
            morphemes = list(frag.morphemes)
            root_morpheme = morphemes[0]
            sig = (
                root_morpheme.root_form,
                root_morpheme.word_class,
                tuple((m.suffix_id, m.chunk) for m in morphemes[1:]),
            )
            if sig in seen:
                continue
            seen.add(sig)
            # Bare-root bonus: when the whole word IS the in-lexicon
            # root (no suffixes attached), nudge the score upward. This
            # makes "the unmarked reading of a known lemma is the lemma
            # itself" the default, preventing speculative suffix-peeling
            # from a more-frequent verb root that happens to be a
            # prefix of a noun lemma (e.g., süre, takım, durum).
            score = frag.score
            if not root_morpheme.oov and len(morphemes) == 1:
                score += self.cfg.bare_root_bonus
            out.append(Analysis(
                surface=word,
                root=root_morpheme.root_form,
                root_class=root_morpheme.word_class,
                final_class=frag.word_class,
                morphemes=morphemes,
                oov=root_morpheme.oov,
                score=score,
            ))
        return out


def _freq_score(freq: int) -> float:
    """Convert raw frequency to a log-scaled bonus."""
    import math
    return math.log1p(max(0, freq))


def _already_in(chart, pos, frag):
    """Avoid adding identical fragments to the same chart cell."""
    for existing in chart.get(pos, []):
        if (existing.state == frag.state
            and existing.word_class == frag.word_class
            and existing.morphemes == frag.morphemes):
            return True
    return False
