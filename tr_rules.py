"""
tr_rules.py — Rule registry for per-suffix phonological adjustments.

Each rule has two views:
  - forward (generation): given a stem and a suffix template, possibly
    modify them before realization.
  - inverse (parsing):    given a candidate stem after suffix-stripping,
    possibly modify it to recover the "true" pre-suffixation stem.

The two views are co-located because they describe the same linguistic
operation. JSON inventories reference rules by name; both views are
recoverable from that name.

Forward signature:
    forward(stem, template, ctx) -> (stem', template', ctx')

Inverse signature:
    inverse(stem_candidates, ctx) -> stem_candidates'

Inverse takes and returns a SET (or list) of candidate stems, because the
inverse may not be deterministic — for example, given a stem ending in 'm',
we don't know whether the H of -Hm was dropped (after a vowel-final stem)
or never existed (the stem was consonant-final). The inverse therefore
returns both possibilities.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tr_phonology import (
    BACK_VOWELS, FRONT_VOWELS, HIGH_VOWELS, LOW_VOWELS, ROUNDED_VOWELS,
    ends_in_vowel, is_monosyllabic, last_vowel,
)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

ForwardFn = Callable[[str, str, Dict[str, Any]], Tuple[str, str, Dict[str, Any]]]
InverseFn = Callable[[List[str], Dict[str, Any]], List[str]]
ExpandFn  = Callable[[str, str, Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]


class _RuleEntry:
    __slots__ = ("name", "forward", "inverse", "expand")
    def __init__(self, name: str):
        self.name = name
        self.forward: ForwardFn | None = None
        self.inverse: InverseFn | None = None
        self.expand:  ExpandFn  | None = None


RULES: Dict[str, _RuleEntry] = {}


def _ensure(name: str) -> _RuleEntry:
    if name not in RULES:
        RULES[name] = _RuleEntry(name)
    return RULES[name]


def forward(name: str) -> Callable[[ForwardFn], ForwardFn]:
    """Register the forward (generation) view of a rule. The forward
    view is deterministic: it picks one template/stem outcome based on
    context, suitable for surface realization."""
    def decorator(fn: ForwardFn) -> ForwardFn:
        entry = _ensure(name)
        if entry.forward is not None:
            raise ValueError(f"Duplicate forward rule: {name}")
        entry.forward = fn
        return fn
    return decorator


def inverse(name: str) -> Callable[[InverseFn], InverseFn]:
    """Register the inverse (parsing) view of a rule. Operates on
    stem candidates, expanding them with possible un-deletions etc.
    Use this when the rule deleted material from the stem; otherwise
    prefer `expand` for template-modifying rules."""
    def decorator(fn: InverseFn) -> InverseFn:
        entry = _ensure(name)
        if entry.inverse is not None:
            raise ValueError(f"Duplicate inverse rule: {name}")
        entry.inverse = fn
        return fn
    return decorator


def expand(name: str) -> Callable[[ExpandFn], ExpandFn]:
    """Register the parse-time template-expansion view of a rule.

    Unlike `forward`, which returns ONE outcome, `expand` returns a list
    of (template, ctx) alternatives the parser should try. Use this for
    rules where the surface might be produced by any of several possible
    template variants and the parser needs to try them all (e.g., AOR's
    -Ar vs -Hr allomorphy on consonant-final stems).

    If a rule has no `expand` view but does have `forward`, the parser
    falls back to using `forward` and treats it as the single template
    to try."""
    def decorator(fn: ExpandFn) -> ExpandFn:
        entry = _ensure(name)
        if entry.expand is not None:
            raise ValueError(f"Duplicate expand rule: {name}")
        entry.expand = fn
        return fn
    return decorator


def get_forward(name: str) -> ForwardFn:
    entry = RULES.get(name)
    if entry is None or entry.forward is None:
        raise KeyError(f"No forward rule registered for {name!r}")
    return entry.forward


def get_inverse(name: str) -> InverseFn:
    entry = RULES.get(name)
    if entry is None:
        raise KeyError(f"No rule registered for {name!r}")
    # Inverse is optional — some rules don't reshape the stem (only the
    # template), so there's nothing for the parser to undo.
    return entry.inverse or (lambda stems, ctx: stems)


def get_expand(name: str) -> ExpandFn:
    """Return the rule's expand view, or a fallback that uses forward
    (yielding the single deterministic template)."""
    entry = RULES.get(name)
    if entry is None:
        raise KeyError(f"No rule registered for {name!r}")
    if entry.expand is not None:
        return entry.expand
    # Fall back to forward: wrap its single result as a one-element list.
    if entry.forward is not None:
        fwd = entry.forward
        def _from_forward(stem, template, ctx):
            _s, new_template, new_ctx = fwd(stem, template, ctx)
            return [(new_template, new_ctx)]
        return _from_forward
    # No views registered: identity.
    return lambda stem, template, ctx: [(template, ctx)]


# -----------------------------------------------------------------------------
# Rules
# -----------------------------------------------------------------------------

@forward("drop_initial_H_after_vowel_stem")
def _(stem, template, ctx):
    """Drop the initial H of the template when the stem ends in a vowel.

        ev    + Hm → evim
        araba + Hm → arabam
    """
    if template.startswith("H") and ends_in_vowel(stem):
        return stem, template[1:], ctx
    return stem, template, ctx

# No inverse rule: this rule only changes the template, not the stem.
# The parser's job is to recognize that an "m" suffix at the end of a vowel-
# final running stem could be either -Hm-with-H-dropped or just plain -m.
# That's handled by the suffix-matching machinery, not by stem rewriting.


@forward("delete_stem_final_low_vowel")
def _(stem, template, ctx):
    """Delete the stem-final low vowel (a or e). Used by -Hyor.

        gelme + Hyor → gelm + Hyor → gelmiyor
        okuma + Hyor → okum + Hyor → okumuyor
    """
    if stem and stem[-1] in LOW_VOWELS:
        return stem[:-1], template, ctx
    return stem, template, ctx


@inverse("delete_stem_final_low_vowel")
def _(stem_candidates, ctx):
    """Inverse: a stem given to the parser as (e.g.) "gelm" might have
    originally been "gelme" or "gelma". The parser must consider both
    possibilities; backness harmony narrows it down.

    For each candidate stem, return:
      - the stem as-is (rule didn't fire), AND
      - the stem with a harmonic low vowel re-attached (rule did fire)

    Backness harmony: the last vowel of the original stem determines a/e.
    """
    expanded = []
    for s in stem_candidates:
        expanded.append(s)  # rule didn't fire
        # Rule fired: re-attach the low vowel that was deleted.
        # The vowel preceding the deletion site is now the last vowel of `s`.
        v = last_vowel(s)
        if v is None:
            # No vowel context, try both.
            expanded.append(s + "a")
            expanded.append(s + "e")
        else:
            expanded.append(s + ("a" if v in BACK_VOWELS else "e"))
    # De-duplicate while preserving order.
    seen, out = set(), []
    for s in expanded:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


@forward("insert_n_after_3rd_person_possessive")
def _(stem, template, ctx):
    """Insert pronominal 'n' before a case marker. Two trigger conditions:

    (1) After a 3rd-person possessive (POSS_3SG or POSS_3PL). Mandatory
        whenever a case marker follows:
            araba-sı + DA   → arabasında    (3sg poss + locative)
            ev-i     + DAn  → evinden       (3sg poss + ablative)
            araba-lar-ı + DA → arabalarında (3pl poss + locative)

    (2) On demonstrative/personal pronouns (bu, şu, o) directly before
        any case marker. Same buffer-n morphophonologically:
            bu + ACC → bunu       (this + acc)
            o  + DAT → ona        (3sg pronoun + dative)
            şu + LOC → şunda      (that + locative)

    The flag `root_pronominal_n` is carried on the root entry and reaches
    here via ParseFragment.root_ctx threaded through during chart fill.

    Both triggers prepend 'n' to the template. The case markers (-DA,
    -DAn, -A, -H, etc.) are all that this rule can validly fire before.
    """
    prev = ctx.get("prev_morph_id")
    if prev in ("POSS_3SG", "POSS_3PL", "REL_KI"):
        return stem, "n" + template, ctx
    if ctx.get("root_pronominal_n") and prev is None:
        # On a bare demonstrative pronoun (no intervening morpheme),
        # case markers take a pronominal n.
        return stem, "n" + template, ctx
    return stem, template, ctx

# No inverse rule: this rule only modifies the template, not the stem,
# and its forward direction is deterministic given the context. Parsing
# just invokes the forward rule with the previous-morpheme context, gets
# the n-prepended template back, and matches that.


# -----------------------------------------------------------------------------
# Aorist allomorphy
# -----------------------------------------------------------------------------
#
# Turkish aorist has three surface allomorphs:
#   - after vowel-final stems:            -r          (oku-r, başla-r)
#   - after polysyllabic consonant-final: -Hr         (konuş-ur, otur-ur)
#   - after monosyllabic consonant-final: -Ar or -Hr  (yap-ar vs gel-ir)
#
# The choice between -Ar and -Hr for monosyllabic consonant-final stems
# is LEXICAL: most take -Ar (yap, bak, kork, çık, ...), but ~13 take -Hr
# (al, bil, bul, dur, gel, gör, kal, ol, öl, san, var, ver, vur). The
# exceptional verbs are marked in the lexicon with aorist_high=True.
#
# Forward (generation): picks one outcome based on stem and lexical flag.
# Expand (parsing): yields all phonologically possible templates; the
# parser tries each, and surface matching naturally disambiguates (because
# -Ar harmonizes to a/e while -Hr harmonizes to ı/i/u/ü, they can't both
# match the same surface).

@forward("aorist_allomorphy")
def _(stem, template, ctx):
    """Pick the right aorist allomorph for generation.

    Three patterns:

    1. After NEG (suppletive zero pattern):
         NEG + AOR + 1SG_Z → gelme + ∅ + m       → gelmem
         NEG + AOR + 1PL_Z → gelme + ∅ + yiz     → gelmeyiz
         NEG + AOR + anything else → gelme + z   → gelmez

    2. Lexically high-vowel monosyllables (al, gel, ver, ...):
         signaled by ctx['root_aorist_high'] → -Hr

    3. Default phonological pattern:
         vowel-final stem → -r       (oku-r)
         monosyllabic     → -Ar      (yap-ar)
         polysyllabic     → -Hr      (konuş-ur)
    """
    # Suppletive NEG-AOR: surface depends on the following agreement.
    if ctx.get("prev_morph_id") == "NEG":
        if ctx.get("next_morph_id") in ("1SG_Z", "1PL_Z"):
            return stem, "", ctx
        return stem, "z", ctx
    # Regular allomorphy.
    if ends_in_vowel(stem):
        return stem, "r", ctx
    if is_monosyllabic(stem) and not ctx.get("root_aorist_high"):
        return stem, "Ar", ctx
    return stem, "Hr", ctx


@expand("aorist_allomorphy")
def _(stem, template, ctx):
    """Yield phonologically plausible aorist templates for parsing.

    After NEG: the two suppletive alternatives are -z and ∅. The parser
    tries both; surface matching picks the right one (a NEG-stem of
    "gelme" followed by "m" or "yiz" requires the ∅ template; followed
    by "z" + anything else requires "z").

    Elsewhere:
      - vowel-final stem:                          only -r   (oku-r)
      - polysyllabic consonant-final:              only -Hr  (konuş-ur,
                                                              gelin-ir)
      - monosyllabic consonant-final, aorist_high: only -Hr  (gel-ir,
                                                              al-ır)
      - monosyllabic consonant-final, default:     only -Ar  (yap-ar,
                                                              art-ar)

    The aorist_high flag is set on the ~13 monosyllabic verbs that
    lexically take -Hr; reading it here keeps the parser from
    over-generating phantom AOR readings (e.g., 'artırdı' should NOT
    be parseable as art+AOR(-Hr)+PAST_COP, because art's AOR is -Ar
    not -Hr). For polysyllabic stems (after voice/derivation has
    extended the stem) the flag is irrelevant — those always take -Hr.
    """
    if ctx.get("prev_morph_id") == "NEG":
        return [("", ctx), ("z", ctx)]
    if ends_in_vowel(stem):
        return [("r", ctx)]
    # Consonant-final.
    if not is_monosyllabic(stem):
        return [("Hr", ctx)]
    # Monosyllabic: -Hr if aorist_high (lex-flagged), else -Ar.
    if ctx.get("root_aorist_high"):
        return [("Hr", ctx)]
    return [("Ar", ctx)]


# -----------------------------------------------------------------------------
# Passive allomorphy
# -----------------------------------------------------------------------------
#
# Turkish passive has three surface allomorphs, conditioned by the stem-
# final segment (no lexical exceptions):
#   - after vowel:           -n         (tara-n, oku-n, bekle-n)
#   - after l:               -Hn        (al-ın, bil-in, bul-un, gel-in)
#   - after other consonant: -Hl        (yap-ıl, sat-ıl, gör-ül)
#
# Forward (generation): picks one outcome based on stem.
# Expand (parsing): yields phonologically plausible templates. After a
# vowel, only -n is possible; after l, only -Hn (since -Hl after l would
# produce a geminate that's not the surface). After other consonants,
# only -Hl. So expand actually returns just ONE template per stem class —
# the rule is fully phonological.

@forward("passive_allomorphy")
def _(stem, template, ctx):
    """Pick the right passive allomorph for generation."""
    if ends_in_vowel(stem):
        return stem, "n", ctx
    if stem and stem[-1] == "l":
        return stem, "Hn", ctx
    return stem, "Hl", ctx


@expand("passive_allomorphy")
def _(stem, template, ctx):
    """Yield phonologically plausible passive templates for parsing.

    Each stem class admits exactly one template — passive is fully
    phonological with no lexical exceptions.
    """
    if ends_in_vowel(stem):
        return [("n", ctx)]
    if stem and stem[-1] == "l":
        return [("Hn", ctx)]
    return [("Hl", ctx)]


# -----------------------------------------------------------------------------
# Causative allomorphy (inflectional)
# -----------------------------------------------------------------------------
#
# Inflectional Turkish causative is phonologically deterministic:
#   - after vowel:                              -t   (bekle-t, yürü-t)
#   - after r/l on POLYsyllabic stems:          -t   (belir-t, çıkart-t,
#                                                    oturt, düzel-t)
#   - elsewhere (default consonant):            -DHr (yap-tır, sat-tır,
#                                                    ver-dir, gel-dir,
#                                                    var-dır)
#
# Crucial: monosyllabic r/l-final verbs take -DHr, not -t. `ver` → verdir,
# `gel` → geldir.
#
# Note: forms like geç-ir, düş-ür, çık-ar (with -Hr/-Ar templates) are NOT
# inflectional causatives. They are V→V derivational applications,
# modeled separately as CAUS_DERIV.

@forward("causative_allomorphy")
def _(stem, template, ctx):
    """Pick the right inflectional causative allomorph for generation."""
    if not stem:
        return stem, "DHr", ctx
    last = stem[-1]
    if last in "aeıioöuü":
        return stem, "t", ctx
    if last in "rl" and not is_monosyllabic(stem):
        return stem, "t", ctx
    return stem, "DHr", ctx


@expand("causative_allomorphy")
def _(stem, template, ctx):
    """Yield phonologically plausible inflectional CAUS templates for parsing.

    Phonologically deterministic; one template per stem class.
    Lexicalized -Hr/-Ar derivations are handled by CAUS_DERIV instead.
    """
    if not stem:
        return [("DHr", ctx)]
    last = stem[-1]
    if last in "aeıioöuü":
        return [("t", ctx)]
    if last in "rl" and not is_monosyllabic(stem):
        return [("t", ctx)]
    return [("DHr", ctx)]


# -----------------------------------------------------------------------------
# Derivational V→V allomorphy (CAUS_DERIV, PASS_DERIV)
# -----------------------------------------------------------------------------
#
# These are V→V derivational suffixes that produce historically/lexically
# distinct verbs:
#   - CAUS_DERIV: çık → çıkar, geç → geçir, anla → anlat, kız → kızar
#   - PASS_DERIV: bul → bulun, sık → sıkıl, doku → dokun, ye → yen
#
# Per the design, the tokenizer ALWAYS decomposes these to the base root,
# even when UD-IMST treats the derived form as the gold lemma. The
# morphemes emit no Voice features (the `_derivation` marker is stripped
# in ud_feats()).
#
# Gating: each base verb that licenses a derivational suffix carries a
# lex flag whose VALUE is the template (e.g., "Ar", "Hr", "t", "Hn",
# "Hl", "n"). The expand rule returns the specified template if the flag
# is set, else returns empty list (blocking the suffix). This makes
# CAUS_DERIV/PASS_DERIV applicable only to verbs the lexicon has flagged.

@forward("caus_deriv_allomorphy")
def _(stem, template, ctx):
    """Pick the CAUS_DERIV template from the lex flag's value."""
    flag = ctx.get("root_caus_deriv")
    if flag:
        return stem, flag, ctx
    return stem, template, ctx


@expand("caus_deriv_allomorphy")
def _(stem, template, ctx):
    """Return the CAUS_DERIV template specified by the lex flag, or block
    the suffix entirely if the flag is absent (preventing speculative
    derivational decomposition of verbs that aren't lexicalized this way)."""
    flag = ctx.get("root_caus_deriv")
    if flag:
        return [(flag, ctx)]
    return []   # Block


@forward("pass_deriv_allomorphy")
def _(stem, template, ctx):
    """Pick the PASS_DERIV template from the lex flag's value."""
    flag = ctx.get("root_pass_deriv")
    if flag:
        return stem, flag, ctx
    return stem, template, ctx


@expand("pass_deriv_allomorphy")
def _(stem, template, ctx):
    """Return the PASS_DERIV template specified by the lex flag, or block."""
    flag = ctx.get("root_pass_deriv")
    if flag:
        return [(flag, ctx)]
    return []


# -----------------------------------------------------------------------------
# Potential + negation suppletion
# -----------------------------------------------------------------------------
#
# Turkish potential (-(y)Abil "be able to") fuses with negation: instead
# of *(y)Abil + mA = "able-to + not", the negative form realizes as just
# -(y)A + mA = "-AmA" "cannot". So:
#   gel + POT + NEG + PAST → gelemedi  (NOT gelebilmedi)
#                            = gel + (y)A + mA + DH
# The standalone POT outside NEG context keeps its full -(y)Abil form.
#
# Implementation: the forward view checks next_morph_id; if NEG follows,
# the template becomes "(y)A". The expand view does the inverse:
# returns "(y)A" before NEG, the full "(y)Abil" otherwise.

@forward("potential_neg_suppletion")
def _(stem, template, ctx):
    """If NEG follows, POT realizes as -(y)A; otherwise as -(y)Abil."""
    if ctx.get("next_morph_id") == "NEG":
        return stem, "(y)A", ctx
    return stem, template, ctx


@expand("potential_neg_suppletion")
def _(stem, template, ctx):
    """During parsing, we don't know what follows yet, so return BOTH
    alternatives: the full -(y)Abil for non-NEG continuations and -(y)A
    for the NEG-following case. We constrain the -(y)A alternative via
    a `_next_must_be` marker in ctx so the parser only extends it with
    NEG (without this, the parser would over-generate -(y)A POT matches
    on stems where the next morpheme isn't actually NEG)."""
    constrained_ctx = dict(ctx)
    constrained_ctx["_next_must_be"] = ("NEG",)
    return [(template, ctx), ("(y)A", constrained_ctx)]


# -----------------------------------------------------------------------------
# NEG + AOR + 1SG agreement suppletion
# -----------------------------------------------------------------------------
#
# After NEG, AOR is suppletive: for 1SG it's zero AND 1SG_Z reduces to
# bare 'm' (gelmem, not *gelmeyim). For 1PL the standard (y)Hz holds
# (gelmeyiz). The aorist rule already handles AOR's side (returns empty
# template before 1SG/1PL). This rule handles 1SG_Z's side: when
# following an empty-AOR, drop everything before the final m.
#
# 1PL_Z does NOT need any special handling — its default (y)Hz template
# already produces gelmeyiz correctly.

@forward("neg_aor_agreement_suppletion")
def _(stem, template, ctx):
    """For 1SG_Z after a zero-AOR (NEG-suppletion context): template
    reduces to bare 'm'. For 1PL_Z or other contexts: default."""
    if (ctx.get("prev_morph_id") == "AOR"
            and ctx.get("prev_morph_chunk") == ""
            and template == "(y)Hm"):
        return stem, "m", ctx
    return stem, template, ctx


@expand("neg_aor_agreement_suppletion")
def _(stem, template, ctx):
    """Same logic in expand."""
    if (ctx.get("prev_morph_id") == "AOR"
            and ctx.get("prev_morph_chunk") == ""
            and template == "(y)Hm"):
        return [("m", ctx)]
    return [(template, ctx)]
