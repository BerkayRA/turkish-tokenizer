"""
tr_normalize.py — batch corpus normalizer for LLM data preparation.

Streams text (a file or stdin), normalizes it line by line, and writes to
stdout (or a file). It is fast by default — the lengthy OOV suggester and
alternative analyses are turned OFF — so it scales to large corpora. The
attached interrogative particle is split (gelecekmisin -> gelecek mi),
which is the main surface normalization useful before training.

Output modes:
  surface    Clean readable text: attached clitics get a space, spacing and
             punctuation preserved. The stream you'd feed to a subword
             tokenizer trainer or train on directly. (default)
  lemma      Whitespace-joined lemmas of the word tokens (punctuation
             dropped) — for deduplication, frequency lists, classical NLP.
  morphemes  Whitespace-joined morpheme-segmented words (e.g. kitab▁ım▁ı) —
             a morphological pre-tokenization signal for vocabulary design.
  jsonl      One JSON object per input line with the full per-token
             analysis — for building annotated datasets.

Usage:
    python tr_normalize.py corpus.txt > clean.txt
    python tr_normalize.py --mode lemma corpus.txt > lemmas.txt
    cat corpus.txt | python tr_normalize.py --mode morphemes --sep '|'
    python tr_normalize.py --mode surface --fold-diacritics corpus.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tr_api import Tokenizer, TokenizerConfig
from tr_phonology import fold_diacritics


def _iter_lines(path):
    if path is None or path == "-":
        for line in sys.stdin:
            yield line.rstrip("\n")
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")


def _word_tokens(result):
    """The word tokens of a tokenize_text result, in order."""
    return [t for t in result["tokens"] if t["kind"] == "word"]


def render_surface(result, fold):
    """Reconstruct readable text. Two consecutive word tokens with no
    separator between them came from a clitic split, so insert a space."""
    out = []
    prev_word = False
    for t in result["tokens"]:
        if t["kind"] == "word":
            if prev_word:
                out.append(" ")
            surf = t["surface"]
            out.append(fold_diacritics(surf) if fold else surf)
            prev_word = True
        else:
            out.append(t["surface"])
            prev_word = False
    return "".join(out)


def _word_value(tok, field, sep):
    """lemma or morpheme-split for a word token, falling back to surface."""
    a = tok["analysis"]
    if not a or not a.get("parsed"):
        return tok["surface"]
    if field == "lemma":
        return a.get("lemma") or tok["surface"]
    split = a.get("split") or tok["surface"]
    return split.replace("-", sep) if sep != "-" else split


def render_lemma(result, sep):
    return " ".join(_word_value(t, "lemma", sep) for t in _word_tokens(result))


def render_morphemes(result, sep):
    return " ".join(_word_value(t, "morphemes", sep) for t in _word_tokens(result))


def render_jsonl(line, result):
    toks = []
    for t in _word_tokens(result):
        a = t["analysis"] or {}
        toks.append({
            "surface":   t["surface"],
            "lemma":     a.get("lemma"),
            "pos":       a.get("root_class"),
            "morphemes": [m["chunk"] for m in a.get("morphemes", [])],
            "oov":       a.get("oov"),
        })
    return json.dumps({"text": line, "tokens": toks}, ensure_ascii=False)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="-",
                    help="input text file (default: stdin)")
    ap.add_argument("-o", "--output", default="-",
                    help="output file (default: stdout)")
    ap.add_argument("--mode", choices=["surface", "lemma", "morphemes", "jsonl"],
                    default="surface")
    ap.add_argument("--sep", default="▁",
                    help="morpheme separator for --mode morphemes "
                         "(default: ▁ U+2581)")
    ap.add_argument("--fold-diacritics", action="store_true",
                    help="fold circumflex vowels in --mode surface (mekân->mekan)")
    ap.add_argument("--no-split-clitics", action="store_true",
                    help="do not split attached interrogative particles")
    ap.add_argument("--lexicon", default=None,
                    help="lexicon JSON (default: the tokenizer's bundled one)")
    args = ap.parse_args(argv[1:])

    cfg_kwargs = {"suggest_on_oov": False, "include_alternatives": False,
                  "split_clitics": not args.no_split_clitics}
    if args.lexicon:
        cfg_kwargs["lexicon_path"] = Path(args.lexicon)
    tok = Tokenizer(TokenizerConfig(**cfg_kwargs))

    out = (sys.stdout if args.output in ("-", None)
           else open(args.output, "w", encoding="utf-8"))
    n_lines = n_words = n_oov = 0
    t0 = time.time()
    try:
        for line in _iter_lines(args.input):
            if not line.strip():
                out.write("\n")
                n_lines += 1
                continue
            result = tok.tokenize_text(
                line, suggest=False, tail_repair=False, alternatives=False)
            words = _word_tokens(result)
            n_words += len(words)
            n_oov += sum(1 for t in words
                         if (t["analysis"] or {}).get("oov"))
            if args.mode == "surface":
                out.write(render_surface(result, args.fold_diacritics))
            elif args.mode == "lemma":
                out.write(render_lemma(result, args.sep))
            elif args.mode == "morphemes":
                out.write(render_morphemes(result, args.sep))
            else:
                out.write(render_jsonl(line, result))
            out.write("\n")
            n_lines += 1
    finally:
        if out is not sys.stdout:
            out.close()

    dt = time.time() - t0
    rate = n_words / dt if dt > 0 else 0
    oov_pct = (100 * n_oov / n_words) if n_words else 0
    print(f"[tr_normalize] {n_lines} lines, {n_words} words "
          f"({oov_pct:.1f}% OOV) in {dt:.1f}s = {rate:.0f} words/s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
