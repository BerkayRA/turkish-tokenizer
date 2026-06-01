"""
tr_inventory.py — Load and validate the suffix inventory JSON.

Fail-fast validation catches data errors at load time rather than mid-parse.
The Inventory class is a thin wrapper over a dict[id, Suffix] with lookup
helpers.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tr_rules import RULES


@dataclass(frozen=True)
class Suffix:
    """A single suffix entry from the inventory."""
    id:        str
    template:  str
    feats:     Dict[str, str] = field(default_factory=dict)
    category:  str = ""
    rules:     List[str] = field(default_factory=list)
    class_in:  Optional[str] = None    # for derivational suffixes
    class_out: Optional[str] = None    # for derivational suffixes
    a_deletable: bool = False          # if True, this suffix's stem-final
                                       # low vowel can be deleted by a
                                       # following suffix (e.g., NEG -mA
                                       # before PROG → -m). Only NEG has
                                       # this in practice; the parser's
                                       # retroactive A-deletion alternative
                                       # only fires when this flag is set.

    def __repr__(self):
        return f"<Suffix {self.id} '{self.template}' {self.feats}>"


class Inventory:
    """An indexed collection of Suffix objects."""

    def __init__(self, suffixes: List[Suffix]):
        self._by_id: Dict[str, Suffix] = {}
        for s in suffixes:
            if s.id in self._by_id:
                raise ValueError(f"Duplicate suffix id: {s.id}")
            self._by_id[s.id] = s

    def get(self, suffix_id: str) -> Suffix:
        if suffix_id not in self._by_id:
            raise KeyError(f"Unknown suffix id: {suffix_id}")
        return self._by_id[suffix_id]

    def all(self) -> List[Suffix]:
        return list(self._by_id.values())

    def by_category(self, category: str) -> List[Suffix]:
        return [s for s in self._by_id.values() if s.category == category]

    def __len__(self):
        return len(self._by_id)

    def __contains__(self, suffix_id: str):
        return suffix_id in self._by_id


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

ALLOWED_TEMPLATE_CHARS = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"   # surface letters
    "AHDCG"                            # archiphonemes
    "()"                               # buffer brackets
    "ynsş"                             # buffer consonants
)


def load_inventory(path: str | Path) -> Inventory:
    """Load and validate an inventory JSON file.

    Raises ValueError on any structural problem: unknown rule, malformed
    template, duplicate id, missing required field.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "suffixes" not in data:
        raise ValueError(f"{path}: missing 'suffixes' key")

    suffixes: List[Suffix] = []
    seen_ids = set()

    for i, entry in enumerate(data["suffixes"]):
        # Section-divider entries are documentation only; skip them.
        if "_section" in entry:
            continue

        # Required fields.
        for required in ("id", "template"):
            if required not in entry:
                raise ValueError(
                    f"{path}: suffix entry {i} missing required field "
                    f"{required!r}: {entry}"
                )

        sid = entry["id"]
        if sid in seen_ids:
            raise ValueError(f"{path}: duplicate suffix id {sid!r}")
        seen_ids.add(sid)

        template = entry["template"]
        _validate_template(template, sid, path)

        rules = entry.get("rules", [])
        for r in rules:
            if r not in RULES:
                raise ValueError(
                    f"{path}: suffix {sid!r} references unknown rule {r!r}. "
                    f"Known rules: {sorted(RULES)}"
                )

        suffixes.append(Suffix(
            id          = sid,
            template    = template,
            feats       = {k: v for k, v in entry.get("feats", {}).items() if not k.startswith("_")},
            category    = entry.get("category", ""),
            rules       = rules,
            class_in    = entry.get("class_in"),
            class_out   = entry.get("class_out"),
            a_deletable = bool(entry.get("a_deletable", False)),
        ))

    return Inventory(suffixes)


def _validate_template(template: str, sid: str, path: Path) -> None:
    """Cheap syntactic check on the template string."""
    for ch in template:
        if ch not in ALLOWED_TEMPLATE_CHARS:
            raise ValueError(
                f"{path}: suffix {sid!r}: template {template!r} contains "
                f"illegal character {ch!r}"
            )
    # Balanced parentheses with exactly one inner char.
    i = 0
    while i < len(template):
        if template[i] == "(":
            end = template.find(")", i)
            if end < 0:
                raise ValueError(
                    f"{path}: suffix {sid!r}: unmatched '(' in {template!r}"
                )
            inner = template[i+1:end]
            if len(inner) != 1:
                raise ValueError(
                    f"{path}: suffix {sid!r}: buffer group {template[i:end+1]!r} "
                    f"must contain exactly one character"
                )
            i = end + 1
        elif template[i] == ")":
            raise ValueError(
                f"{path}: suffix {sid!r}: unmatched ')' in {template!r}"
            )
        else:
            i += 1
