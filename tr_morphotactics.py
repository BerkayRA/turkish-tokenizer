"""
tr_morphotactics.py — Morphotactic graph for Turkish.

The graph models the constraint that not every suffix can follow every other
suffix. States are positions in the agglutinative chain (e.g., VERB_TAM_K,
NOUN_POSS); transitions are suffixes that move between them.

The graph is used in two directions:
  - Generation: validate a stem + suffix-chain before realizing it. Catches
    bugs like "PAST + 1SG_Z" (z-type agreement after past tense — wrong; PAST
    requires k-type).
  - Parsing (Phase 4+): constrain the search for valid analyses. Given a
    surface form, only suffix chains that walk the graph from a start state
    to an accepting state are considered.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tr_inventory import Inventory


@dataclass(frozen=True)
class State:
    id: str
    accept: bool = False


@dataclass(frozen=True)
class Transition:
    via: str               # suffix id from the inventory
    from_states: Tuple[str, ...]
    to_state: str


class MorphoGraph:
    """A directed graph of states and suffix-labeled transitions."""

    def __init__(self,
                 states:        List[State],
                 transitions:   List[Transition],
                 start_states:  Dict[str, str]):
        # states by id
        self.states: Dict[str, State] = {}
        for s in states:
            if s.id in self.states:
                raise ValueError(f"Duplicate state id: {s.id}")
            self.states[s.id] = s

        # word_class → starting state
        self.start_states: Dict[str, str] = dict(start_states)

        self._all_transitions: List[Transition] = list(transitions)

        # index: (from_state, suffix_id) → to_state
        self._step_index: Dict[Tuple[str, str], str] = {}
        for t in self._all_transitions:
            for fs in t.from_states:
                key = (fs, t.via)
                if key in self._step_index and self._step_index[key] != t.to_state:
                    raise ValueError(
                        f"Conflicting transitions from {fs!r} via {t.via!r}: "
                        f"{self._step_index[key]!r} vs {t.to_state!r}"
                    )
                self._step_index[key] = t.to_state

    # --- public API ---

    def start_state(self, word_class: str) -> str:
        if word_class not in self.start_states:
            raise KeyError(
                f"No start state for word class {word_class!r}. "
                f"Known: {sorted(self.start_states)}"
            )
        return self.start_states[word_class]

    def step(self, current: str, suffix_id: str) -> Optional[str]:
        """Returns the destination state if (current, suffix_id) is a valid
        transition, otherwise None."""
        return self._step_index.get((current, suffix_id))

    def is_accepting(self, state: str) -> bool:
        return self.states[state].accept

    def all_transitions(self) -> List[Transition]:
        return list(self._all_transitions)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_graph(path: str | Path) -> MorphoGraph:
    """Load a morphotactics graph from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for required in ("states", "transitions", "start_states"):
        if required not in data:
            raise ValueError(f"{path}: missing top-level key {required!r}")

    states: List[State] = []
    for entry in data["states"]:
        if "_note" in entry or "_section" in entry:
            pass  # documentation-only fields are tolerated
        if "id" not in entry:
            raise ValueError(f"{path}: state entry missing 'id': {entry}")
        states.append(State(id=entry["id"], accept=entry.get("accept", False)))

    transitions: List[Transition] = []
    for entry in data["transitions"]:
        if "_section" in entry:
            continue
        for required in ("via", "from", "to"):
            if required not in entry:
                raise ValueError(
                    f"{path}: transition entry missing {required!r}: {entry}"
                )
        from_states = entry["from"]
        if not isinstance(from_states, list) or not from_states:
            raise ValueError(
                f"{path}: transition via {entry['via']!r}: "
                f"'from' must be a non-empty list"
            )
        transitions.append(Transition(
            via=entry["via"],
            from_states=tuple(from_states),
            to_state=entry["to"],
        ))

    return MorphoGraph(states, transitions, data["start_states"])


# -----------------------------------------------------------------------------
# Cross-validation against an inventory
# -----------------------------------------------------------------------------

def validate_against_inventory(graph: MorphoGraph, inventory: Inventory) -> None:
    """Every suffix referenced in a transition must exist in the inventory.
    Every state referenced (including start states and to-states) must be
    defined."""
    state_ids = set(graph.states.keys())

    for word_class, sid in graph.start_states.items():
        if sid not in state_ids:
            raise ValueError(
                f"Start state for {word_class!r} is {sid!r}, "
                f"which is not a defined state"
            )

    for t in graph.all_transitions():
        if t.via not in inventory:
            raise ValueError(
                f"Transition via {t.via!r}: suffix not in inventory"
            )
        for fs in t.from_states:
            if fs not in state_ids:
                raise ValueError(
                    f"Transition via {t.via!r}: unknown from-state {fs!r}"
                )
        if t.to_state not in state_ids:
            raise ValueError(
                f"Transition via {t.via!r}: unknown to-state {t.to_state!r}"
            )
