"""
tr_evaluate.py — Evaluate the parser against UD_Turkish-IMST gold.

For each open-class token in a .conllu file:
  - run the parser
  - take the top-scoring analysis
  - compare against the gold lemma, UPOS, and feature set

Reports:
  - root accuracy             (parser_root == gold_lemma)
  - UPOS accuracy             (parser_class == gold_UPOS; PROPN normalized to NOUN)
  - feature exact-match       (parser feat set == gold feat set)
  - feature precision/recall/F1 (over (key, value) pairs)
  - parse coverage            (fraction of tokens where parser produces ANY analysis)
  - per-UPOS breakdown of all of the above
  - sample errors for manual inspection

Usage:
    python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu \\
        --lexicon lexicon_train.json \\
        --limit 1000 \\
        --show-errors 20
"""

import argparse
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tr_inventory     import load_inventory
from tr_lexicon       import load_lexicon
from tr_morphotactics import load_graph
from tr_parse         import Parser, ParseConfig, Analysis
from tr_phonology     import tr_lower


OPEN_CLASS = {"VERB", "NOUN", "PROPN", "ADJ"}


# -----------------------------------------------------------------------------
# Gold data
# -----------------------------------------------------------------------------

@dataclass
class GoldToken:
    form:  str           # surface form (kept in its original casing for display)
    lemma: str           # gold lemma (lowercased)
    upos:  str           # gold UPOS, PROPN normalized to NOUN
    feats: Dict[str, str]  # gold features as a dict (preserves [psor] etc.)
    raw_upos: str        # original (un-normalized) UPOS, for reporting


def parse_conllu(path: Path) -> List[GoldToken]:
    """Read a .conllu file, return open-class tokens with clean lemmas."""
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            form, lemma, upos, _, feats_str = cols[1], cols[2], cols[3], cols[4], cols[5]
            if upos not in OPEN_CLASS:
                continue
            if not lemma or lemma == "_":
                continue
            # Skip lemmas that aren't pure alphabetic (numbers, mixed tokens).
            if not any(c.isalpha() for c in lemma):
                continue
            feats = {}
            if feats_str and feats_str != "_":
                for pair in feats_str.split("|"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        feats[k] = v
            norm_upos = "NOUN" if upos == "PROPN" else upos
            tokens.append(GoldToken(
                form     = form,
                lemma    = tr_lower(lemma),
                upos     = norm_upos,
                feats    = feats,
                raw_upos = upos,
            ))
    return tokens


# -----------------------------------------------------------------------------
# Prediction → feature dict
# -----------------------------------------------------------------------------

def collect_predicted_feats(a: Analysis) -> Dict[str, str]:
    """UD-compliant feature dict for the given analysis. Applies the
    UD-Turkish default-feature conventions (Case=Nom on bare nouns,
    Mood=Ind on bare verbs, etc.) so the comparison against gold features
    is fair."""
    return a.ud_feats()


# -----------------------------------------------------------------------------
# Evaluation aggregation
# -----------------------------------------------------------------------------

@dataclass
class Counters:
    n_tokens:       int = 0
    n_parsed:       int = 0  # parser produced at least one analysis
    n_root_correct: int = 0
    n_upos_correct: int = 0
    n_feat_exact:   int = 0
    feat_tp:        int = 0
    feat_fp:        int = 0
    feat_fn:        int = 0
    # Track per-feature-key confusion for diagnostics.
    feat_key_tp:    Counter = field(default_factory=Counter)
    feat_key_fp:    Counter = field(default_factory=Counter)
    feat_key_fn:    Counter = field(default_factory=Counter)

    def update(self, gold: GoldToken, analysis: Optional[Analysis]):
        self.n_tokens += 1
        if analysis is None:
            # Every gold feat counts as a false negative.
            for k in gold.feats:
                self.feat_fn += 1
                self.feat_key_fn[k] += 1
            return
        self.n_parsed += 1
        pred_feats = collect_predicted_feats(analysis)
        if analysis.root == gold.lemma:
            self.n_root_correct += 1
        if analysis.root_class == gold.upos:
            self.n_upos_correct += 1
        if pred_feats == gold.feats:
            self.n_feat_exact += 1
        gold_pairs = set(gold.feats.items())
        pred_pairs = set(pred_feats.items())
        for (k, v) in gold_pairs & pred_pairs:
            self.feat_tp += 1
            self.feat_key_tp[k] += 1
        for (k, v) in pred_pairs - gold_pairs:
            self.feat_fp += 1
            self.feat_key_fp[k] += 1
        for (k, v) in gold_pairs - pred_pairs:
            self.feat_fn += 1
            self.feat_key_fn[k] += 1

    def pct(self, num: int) -> str:
        if self.n_tokens == 0:
            return "  n/a"
        return f"{100.0 * num / self.n_tokens:5.1f}%"

    def f1(self) -> Tuple[float, float, float]:
        p = self.feat_tp / (self.feat_tp + self.feat_fp) if (self.feat_tp + self.feat_fp) else 0.0
        r = self.feat_tp / (self.feat_tp + self.feat_fn) if (self.feat_tp + self.feat_fn) else 0.0
        f = 2*p*r / (p + r) if (p + r) else 0.0
        return p, r, f


# -----------------------------------------------------------------------------
# Main eval loop
# -----------------------------------------------------------------------------

def evaluate(parser: Parser, tokens: List[GoldToken],
             show_errors: int = 0) -> Tuple[Counters, Dict[str, Counters]]:
    """Run the parser over the gold tokens, return overall + per-UPOS counters."""
    overall = Counters()
    by_upos: Dict[str, Counters] = defaultdict(Counters)
    error_examples: List[Tuple[GoldToken, Optional[Analysis]]] = []

    t0 = time.time()
    for i, tok in enumerate(tokens, 1):
        analyses = parser.parse(tok.form)
        top = analyses[0] if analyses else None
        overall.update(tok, top)
        by_upos[tok.upos].update(tok, top)
        # Capture errors for later display.
        is_root_err = (top is None) or (top.root != tok.lemma)
        if is_root_err and len(error_examples) < show_errors:
            error_examples.append((tok, top))
    dt = time.time() - t0
    if tokens:
        print(f"  ({len(tokens)} tokens in {dt:.1f}s = {len(tokens)/dt:.0f} tok/s)")

    # Print errors.
    if error_examples:
        print(f"\n--- Sample root errors (first {len(error_examples)}) ---")
        for tok, top in error_examples:
            if top is None:
                print(f"  {tok.form:25} gold={tok.lemma}+{tok.upos}  pred=NO PARSE")
            else:
                marker = "*" if top.oov else " "
                pred = f"{top.root}+{top.root_class}"
                print(f"  {tok.form:25} gold={tok.lemma}+{tok.upos}  pred={marker}{pred}")

    return overall, by_upos


def print_counters(label: str, c: Counters):
    p, r, f = c.f1()
    print(f"  {label:8} n={c.n_tokens:5d}  "
          f"cover={c.pct(c.n_parsed)}  "
          f"root={c.pct(c.n_root_correct)}  "
          f"upos={c.pct(c.n_upos_correct)}  "
          f"feat_exact={c.pct(c.n_feat_exact)}  "
          f"feat_F1={f:.3f}  (P={p:.3f} R={r:.3f})")


def print_feature_diagnostics(c: Counters, top_n: int = 10):
    """Show which feature keys hurt F1 most."""
    print(f"\n--- Top feature-key error sources ---")
    print(f"  {'key':25} {'TP':>6} {'FP':>6} {'FN':>6}  (FN means we missed it; FP we hallucinated it)")
    keys = set(c.feat_key_tp) | set(c.feat_key_fp) | set(c.feat_key_fn)
    rows = []
    for k in keys:
        tp, fp, fn = c.feat_key_tp[k], c.feat_key_fp[k], c.feat_key_fn[k]
        rows.append((fp + fn, k, tp, fp, fn))
    rows.sort(reverse=True)
    for _err, k, tp, fp, fn in rows[:top_n]:
        print(f"  {k:25} {tp:6d} {fp:6d} {fn:6d}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu", help="Path to .conllu file (typically dev set)")
    ap.add_argument("--inventory",     default="inventory.json")
    ap.add_argument("--morphotactics", default="morphotactics.json")
    ap.add_argument("--lexicon",       default="lexicon.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only first N tokens")
    ap.add_argument("--show-errors", type=int, default=20)
    args = ap.parse_args(argv[1:])

    print(f"Loading parser components...")
    print(f"  inventory:     {args.inventory}")
    print(f"  morphotactics: {args.morphotactics}")
    print(f"  lexicon:       {args.lexicon}")
    inv = load_inventory(args.inventory)
    graph = load_graph(args.morphotactics)
    lex = load_lexicon(args.lexicon)
    print(f"  ({len(lex)} lexicon entries)")
    parser = Parser(lex, inv, graph)

    print(f"\nReading gold from {args.conllu}...")
    tokens = parse_conllu(Path(args.conllu))
    if args.limit:
        tokens = tokens[:args.limit]
    print(f"  {len(tokens)} open-class gold tokens")

    print(f"\nEvaluating...")
    overall, by_upos = evaluate(parser, tokens, show_errors=args.show_errors)

    print(f"\n=== Results ===")
    print_counters("OVERALL", overall)
    for upos in sorted(by_upos):
        print_counters(upos, by_upos[upos])
    print_feature_diagnostics(overall)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
