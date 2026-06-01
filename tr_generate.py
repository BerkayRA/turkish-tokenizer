"""
tr_generate.py — Generate surface forms from stem + suffix chain.

The generator applies suffixes one at a time, running each suffix's rules in
declared order before calling apply_suffix. It returns both the final surface
form and the per-step breakdown.

If a morphotactic graph is provided, the generator validates each transition.
By default (strict=True), an invalid transition raises ValueError. With
strict=False, the invalid transition is logged at WARNING level and the
generator continues (useful for debugging).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from tr_inventory import Inventory, Suffix
from tr_morphotactics import MorphoGraph
from tr_phonology import apply_suffix
from tr_rules import get_forward


log = logging.getLogger(__name__)


@dataclass
class Step:
    """One step in a derivation."""
    suffix:   Suffix
    chunk:    str
    surface:  str
    state:    Optional[str] = None   # post-transition state, if graph was used


@dataclass
class Generation:
    stem:    str
    surface: str
    steps:   List[Step]

    def split(self) -> str:
        pieces = [self.stem]
        for st in self.steps:
            pieces.append(st.chunk)
        return "-".join(p for p in pieces if p)

    def tagged(self) -> str:
        pieces = [self.stem]
        for st in self.steps:
            feat_str = ",".join(f"{k}={v}" for k, v in sorted(st.suffix.feats.items()))
            tag = f"{st.suffix.id}[{feat_str}]" if feat_str else st.suffix.id
            pieces.append(f"{st.chunk}+{tag}" if st.chunk else f"∅+{tag}")
        return "-".join(pieces)


def generate(
    stem:           str,
    suffix_chain:   List[str],
    inventory:      Inventory,
    soften:         bool = True,
    word_class:     Optional[str] = None,
    graph:          Optional[MorphoGraph] = None,
    strict:         bool = True,
    root_ctx:       Optional[dict] = None,
) -> Generation:
    """Apply a sequence of suffixes to a stem.

    Args:
        stem: the starting stem.
        suffix_chain: list of suffix ids, applied in order.
        inventory: the loaded Inventory.
        soften: whether stem-final consonant softening fires before
            vowel-initial suffixes. Per-stem in real Turkish; for now passed
            in; in Phase 4 it'll be read from the lexicon.
        word_class: starting word class ("VERB", "NOUN", "ADJ"). Required
            when graph is provided.
        graph: optional morphotactic graph. When provided, validates each
            transition.
        strict: when graph is provided, controls behavior on invalid
            transitions. True (default): raise ValueError. False: log a
            warning at WARNING level and continue.
        root_ctx: optional dict of root-level flags to seed the rule
            context (e.g., {"root_aorist_high": True} for monosyllabic
            verbs that take the high-vowel aorist allomorph). Callers
            holding the Root object can build this from its fields.

    Returns:
        A Generation object.
    """
    if graph is not None and word_class is None:
        raise ValueError("word_class must be provided when graph is used")

    current_state: Optional[str] = (
        graph.start_state(word_class) if graph is not None else None
    )

    steps: List[Step] = []
    current = stem
    prev_morph_id: Optional[str] = None      # immediately preceding morpheme
    prev_morph_chunk: Optional[str] = None   # its chunk (may be empty for suppletive)
    prev_prev_morph_id: Optional[str] = None # two-back, for chained suppletion
    base_ctx = dict(root_ctx) if root_ctx else {}

    for i, suffix_id in enumerate(suffix_chain):
        suffix = inventory.get(suffix_id)
        before = current
        # Look one step ahead for rules that need to know what's coming
        # (e.g., neg-aorist suppletion: AOR is empty before 1SG_Z/1PL_Z).
        next_morph_id = suffix_chain[i + 1] if i + 1 < len(suffix_chain) else None

        # --- Morphotactic check ---
        if graph is not None:
            next_state = graph.step(current_state, suffix_id)
            if next_state is None:
                msg = (f"Invalid morphotactic transition: from state "
                       f"{current_state!r} via {suffix_id!r}. "
                       f"Stem so far: {before!r}")
                log.warning(msg)
                if strict:
                    raise ValueError(msg)
                # else: continue with current_state unchanged
            else:
                current_state = next_state

        # --- Apply pre-rules ---
        # ctx carries the previous and next morpheme ids, the previous
        # morpheme's chunk, and any root-level flags (seeded from
        # root_ctx) so context-sensitive rules can inspect them. The
        # chunk-tracking is needed by suppletion-after-suppletion rules
        # like neg_aor_agreement_suppletion (1SG_Z/1PL_Z lose their
        # buffer-y when following a zero-AOR).
        ctx = dict(base_ctx)
        ctx["prev_morph_id"] = prev_morph_id
        ctx["prev_morph_chunk"] = prev_morph_chunk
        ctx["prev_prev_morph_id"] = prev_prev_morph_id
        ctx["next_morph_id"] = next_morph_id
        s, t = before, suffix.template
        for rule_name in suffix.rules:
            s, t, ctx = get_forward(rule_name)(s, t, ctx)

        # --- Realize ---
        new_surface = apply_suffix(s, t, soften=soften)

        # Chunk = whatever new_surface added past the longest common prefix
        # with `before`. Handles plain appends, rules that modify the stem,
        # and stem-final softening uniformly.
        common = 0
        for a, b in zip(before, new_surface):
            if a == b:
                common += 1
            else:
                break
        chunk = new_surface[common:]

        steps.append(Step(
            suffix=suffix, chunk=chunk, surface=new_surface,
            state=current_state,
        ))
        current = new_surface
        prev_prev_morph_id = prev_morph_id
        prev_morph_id = suffix_id
        prev_morph_chunk = chunk

    return Generation(stem=stem, surface=current, steps=steps)
