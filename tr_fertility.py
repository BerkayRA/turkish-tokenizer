"""
tr_fertility.py — score a subword tokenizer against Turkish morphology.

Turkish is agglutinative, so a statistically-trained subword tokenizer
(BPE / Unigram) often fragments words badly — high "fertility" (many tokens
per word) and cuts that ignore morpheme boundaries. This tool measures that
over a corpus, using THIS project's morphological analyzer as the gold
reference, so you can compare candidate tokenizers or vocab sizes when
building a Turkish LLM.

Metrics (micro-averaged over word tokens):
  - fertility            subword tokens / word   (lower is better)
  - tokens / morpheme     subword tokens / morpheme
  - morphemes / word      our analyzer's segmentation density
  - single-token words %  words emitted as exactly one subword
  - boundary alignment    precision / recall / F1 of subword-internal cut
                          points vs. true morpheme boundaries (best-effort:
                          only over words whose pieces reconstruct the word)

Subword tokenizer adapters (encode one word -> piece strings):
  whitespace   one piece per word          (baseline, fertility 1.0)
  char         one piece per character     (baseline, captures every boundary)
  --spm M.model    SentencePiece model     (needs `sentencepiece`)
  --hf NAME_OR_DIR Hugging Face tokenizer   (needs `tokenizers`/`transformers`)

Usage:
    python tr_fertility.py corpus.txt                 # whitespace baseline
    python tr_fertility.py --tokenizer char corpus.txt
    python tr_fertility.py --spm tr_bpe.model corpus.txt
    python tr_fertility.py --hf dbmdz/bert-base-turkish-cased corpus.txt
"""

import argparse
import sys
import time
from pathlib import Path

from tr_api import Tokenizer, TokenizerConfig, _split_text

# Markers various subword tokenizers prefix/suffix onto pieces. Stripping
# them lets us reconstruct the surface substring for boundary alignment.
_MARKERS = ("▁", "Ġ", "##", "</w>", "Ċ")


# -----------------------------------------------------------------------------
# Subword tokenizer adapters: encode(word) -> list[str] of piece strings.
# -----------------------------------------------------------------------------

class WhitespaceAdapter:
    name = "whitespace"
    def encode(self, word):
        return [word]


class CharAdapter:
    name = "char"
    def encode(self, word):
        return list(word)


class SpmAdapter:
    def __init__(self, model_path):
        try:
            import sentencepiece as spm
        except ImportError:
            sys.exit("--spm needs the `sentencepiece` package (pip install sentencepiece)")
        self._sp = spm.SentencePieceProcessor(model_file=model_path)
        self.name = f"spm:{Path(model_path).name}"
    def encode(self, word):
        return self._sp.encode(word, out_type=str)


class HfAdapter:
    def __init__(self, name_or_dir):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            sys.exit("--hf needs the `transformers` package (pip install transformers)")
        self._tk = AutoTokenizer.from_pretrained(name_or_dir)
        self.name = f"hf:{name_or_dir}"
    def encode(self, word):
        return self._tk.tokenize(word)


class TokenizersAdapter:
    """A Hugging Face `tokenizers` model saved as a tokenizer.json
    (e.g. a byte-level BPE trained with ByteLevelBPETokenizer)."""
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer as _HFTok
        except ImportError:
            sys.exit("--tokenizers-json needs the `tokenizers` package")
        self._tk = _HFTok.from_file(path)
        self.name = f"tokenizers:{Path(path).name}"
    def encode(self, word):
        return self._tk.encode(word).tokens


def _clean(piece):
    for m in _MARKERS:
        if piece.startswith(m):
            piece = piece[len(m):]
        if piece.endswith(m):
            piece = piece[:-len(m)]
    return piece


def _internal_boundaries(pieces, word):
    """Char offsets where the subword pieces cut `word`, or None if the
    cleaned pieces don't reconstruct it (e.g. byte-level BPE)."""
    cleaned = [_clean(p) for p in pieces]
    if "".join(cleaned) != word:
        return None
    bounds, pos = set(), 0
    for p in cleaned[:-1]:
        pos += len(p)
        if pos:
            bounds.add(pos)
    return bounds


def _morpheme_boundaries(analysis):
    """Internal char offsets between morpheme chunks."""
    bounds, pos = set(), 0
    chunks = [m["chunk"] for m in analysis.get("morphemes", [])]
    for c in chunks[:-1]:
        pos += len(c)
        if pos:
            bounds.add(pos)
    return bounds


def score_corpus(adapter, lines, tok, limit=None):
    """Score `adapter` over an iterable of text lines, using `tok` (a
    Tokenizer) as the morphological reference. Returns a metrics dict. This
    is the reusable core — the CLI and external harnesses both call it."""
    n_words = n_subtok = n_morph = n_single = 0
    b_hit = b_sub = b_morph = n_aligned = 0
    stop = False
    for line in lines:
        if stop:
            break
        for surface, kind in _split_text(line.rstrip("\n")):
            if kind != "word":
                continue
            word = surface.lower()
            pieces = adapter.encode(word)
            if not pieces:
                continue
            n_words += 1
            n_subtok += len(pieces)
            if len(pieces) == 1:
                n_single += 1
            analysis = tok.tokenize(word, suggest=False, tail_repair=False,
                                    alternatives=False, split_clitics=False)
            morphs = analysis.get("morphemes", []) if analysis.get("parsed") else []
            n_morph += max(1, len(morphs))
            mb = _morpheme_boundaries(analysis) if morphs else set()
            sb = _internal_boundaries(pieces, word)
            if sb is not None:
                n_aligned += 1
                b_sub += len(sb)
                b_morph += len(mb)
                b_hit += len(sb & mb)
            if limit and n_words >= limit:
                stop = True
                break

    prec = b_hit / b_sub if b_sub else 0.0
    rec = b_hit / b_morph if b_morph else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "name": getattr(adapter, "name", "?"),
        "words": n_words, "subword_tokens": n_subtok, "morphemes": n_morph,
        "fertility": (n_subtok / n_words) if n_words else 0.0,
        "tokens_per_morpheme": (n_subtok / n_morph) if n_morph else 0.0,
        "morphemes_per_word": (n_morph / n_words) if n_words else 0.0,
        "single_token_pct": (100 * n_single / n_words) if n_words else 0.0,
        "aligned_pct": (100 * n_aligned / n_words) if n_words else 0.0,
        "boundary_precision": prec, "boundary_recall": rec, "boundary_f1": f1,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="-", help="corpus (default: stdin)")
    ap.add_argument("--tokenizer", choices=["whitespace", "char"],
                    default="whitespace", help="built-in baseline adapter")
    ap.add_argument("--spm", help="SentencePiece .model path")
    ap.add_argument("--hf", help="Hugging Face tokenizer name or directory")
    ap.add_argument("--tokenizers-json", help="Hugging Face tokenizers tokenizer.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N word tokens")
    ap.add_argument("--lexicon", default=None)
    args = ap.parse_args(argv[1:])

    if args.spm:
        adapter = SpmAdapter(args.spm)
    elif args.hf:
        adapter = HfAdapter(args.hf)
    elif args.tokenizers_json:
        adapter = TokenizersAdapter(args.tokenizers_json)
    elif args.tokenizer == "char":
        adapter = CharAdapter()
    else:
        adapter = WhitespaceAdapter()

    cfg = {"suggest_on_oov": False, "include_alternatives": False}
    if args.lexicon:
        cfg["lexicon_path"] = Path(args.lexicon)
    tok = Tokenizer(TokenizerConfig(**cfg))

    t0 = time.time()
    lines = (sys.stdin if args.input in ("-", None)
             else open(args.input, encoding="utf-8"))
    try:
        m = score_corpus(adapter, lines, tok, limit=args.limit)
    finally:
        if lines is not sys.stdin:
            lines.close()

    if not m["words"]:
        sys.exit("no word tokens found in input")
    print(f"tokenizer: {adapter.name}")
    print(f"corpus:    {args.input}   ({m['words']} words in {time.time() - t0:.1f}s)")
    print(f"  fertility (tokens/word):   {m['fertility']:6.3f}")
    print(f"  tokens / morpheme:         {m['tokens_per_morpheme']:6.3f}")
    print(f"  morphemes / word:          {m['morphemes_per_word']:6.3f}")
    print(f"  single-token words:        {m['single_token_pct']:5.1f}%")
    print(f"  boundary alignment ({m['aligned_pct']:.0f}% of words reconstructible):")
    print(f"    precision: {m['boundary_precision']:.3f}   "
          f"recall: {m['boundary_recall']:.3f}   F1: {m['boundary_f1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
