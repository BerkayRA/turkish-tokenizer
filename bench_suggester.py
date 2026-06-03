"""
bench_suggester.py — precision benchmark for the OOV spelling suggester.

Builds a *labelled* typo set automatically: it takes real, in-lexicon
Turkish words (the surface forms of the UD_Turkish-IMST dev set), applies a
single random Damerau edit to each, and keeps the ones that become
out-of-vocabulary. The original word is the gold answer. We then measure
how often the suggester recovers it.

This mirrors the standard way spelling correctors are evaluated (cf.
Norvig): synthetic single-edit noise over a real vocabulary, scored by
top-1 accuracy and recall@k.

Usage:
    python bench_suggester.py                       # tail repair on
    python bench_suggester.py --no-tail             # stem repair only
    python bench_suggester.py --sample 300 --seed 7
"""

import argparse
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from tr_api import Tokenizer, TokenizerConfig
from tr_phonology import tr_lower, fold_diacritics

ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
TAIL_WINDOW = 4   # mirror tr_api._TAIL_EDIT_WINDOW for region tagging


def read_surfaces(path: Path):
    forms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 2 or "-" in cols[0] or "." in cols[0]:
                continue
            forms.append(cols[1])
    return forms


def make_typo(word, rng):
    """Apply one random Damerau edit. Returns (typo, kind, position)."""
    i = rng.randrange(len(word))
    kind = rng.choice(["delete", "insert", "substitute", "transpose"])
    if kind == "delete" and len(word) > 3:
        return word[:i] + word[i + 1:], kind, i
    if kind == "transpose" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:], kind, i
    if kind == "substitute":
        c = rng.choice([c for c in ALPHABET if c != word[i]])
        return word[:i] + c + word[i + 1:], kind, i
    c = rng.choice(ALPHABET)                       # insert (also the fallback)
    return word[:i] + c + word[i:], "insert", i


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", default="UD_Turkish-IMST/tr_imst-ud-dev.conllu")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--no-tail", action="store_true",
                    help="disable suffix-tail repair (stem repair only)")
    args = ap.parse_args(argv[1:])

    rng = random.Random(args.seed)
    tok = Tokenizer(TokenizerConfig(correct_tail_typos=not args.no_tail))
    # A suggestion-free probe for the (frequent) in-/out-of-vocabulary checks
    # during set construction, so we don't run the expensive suggester there.
    probe = Tokenizer(TokenizerConfig(suggest_on_oov=False))

    def is_valid(word):
        r = probe.tokenize(word)
        return bool(r.get("parsed")) and not r.get("oov")

    print(f"tail_repair={'off' if args.no_tail else 'on'}  seed={args.seed}")

    # Gold set: unique, lowercased, in-lexicon dev surface forms.
    print("Collecting in-lexicon gold words...")
    gold = []
    seen = set()
    for w in read_surfaces(Path(args.conllu)):
        if not w.isalpha() or not (args.min_len <= len(w) <= args.max_len):
            continue
        wl = tr_lower(w)
        if wl in seen:
            continue
        seen.add(wl)
        if is_valid(wl):
            gold.append(wl)
    rng.shuffle(gold)
    print(f"  {len(gold)} candidate gold words")

    # Generate OOV typos until we have `sample` of them.
    cases = []           # (typo, gold, kind, region)
    collisions = 0       # edits that produced another valid (non-OOV) word
    for g in gold:
        if len(cases) >= args.sample:
            break
        typo, kind, pos = make_typo(g, rng)
        if typo == g:
            continue
        if is_valid(typo):
            collisions += 1          # edit landed on a real word; not a typo
            continue
        region = "tail" if pos >= len(g) - TAIL_WINDOW else "stem"
        cases.append((typo, g, kind, region))

    print(f"  {len(cases)} OOV typos  ({collisions} edits skipped as valid words)\n")

    # Evaluate.
    top1 = top3 = fired = 0
    by_kind = defaultdict(lambda: [0, 0, 0])    # kind -> [n, top1, top3]
    by_region = defaultdict(lambda: [0, 0, 0])
    latencies = []
    for typo, g, kind, region in cases:
        t = time.time()
        r = tok.tokenize(typo)
        latencies.append((time.time() - t) * 1000)
        sugg = r.get("suggestions", [])
        words = [fold_diacritics(tr_lower(s["word"])) for s in sugg]
        g_f = fold_diacritics(g)
        is1 = bool(words) and words[0] == g_f
        is3 = g_f in words[:3]
        fired += 1 if words else 0
        top1 += 1 if is1 else 0
        top3 += 1 if is3 else 0
        for bucket in (by_kind[kind], by_region[region]):
            bucket[0] += 1
            bucket[1] += 1 if is1 else 0
            bucket[2] += 1 if is3 else 0

    n = len(cases)
    pct = lambda x: f"{100*x/n:5.1f}%" if n else "  n/a"
    print("=== Suggester precision ===")
    print(f"  typos evaluated     {n}")
    print(f"  fired (>=1 sugg.)   {pct(fired)}")
    print(f"  top-1 accuracy      {pct(top1)}")
    print(f"  recall@3            {pct(top3)}")
    if fired:
        print(f"  top-1 | fired       {100*top1/fired:5.1f}%")
    print("\n  by edit type        n   top1   recall@3")
    for k in ("delete", "insert", "substitute", "transpose"):
        c = by_kind[k]
        if c[0]:
            print(f"    {k:11s}    {c[0]:4d}  {100*c[1]/c[0]:5.1f}%  {100*c[2]/c[0]:6.1f}%")
    print("\n  by edit region      n   top1   recall@3")
    for k in ("stem", "tail"):
        c = by_region[k]
        if c[0]:
            print(f"    {k:11s}    {c[0]:4d}  {100*c[1]/c[0]:5.1f}%  {100*c[2]/c[0]:6.1f}%")
    if latencies:
        print(f"\n  latency/word: mean {statistics.mean(latencies):.0f}ms  "
              f"median {statistics.median(latencies):.0f}ms  "
              f"max {max(latencies):.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
