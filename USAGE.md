# Usage guide

This guide covers the everyday operations: parsing a word, generating a form, running tests, running evaluation, and extending the morphology.

## Setup

The whole project runs on Python 3 with stdlib only — no `pip install` step. Drop the files into a directory together; modules find each other by sibling import.

For evaluation, clone the UD_Turkish-IMST treebank into the same directory:

```bash
git clone https://github.com/UniversalDependencies/UD_Turkish-IMST.git
```

This gives you `UD_Turkish-IMST/tr_imst-ud-{train,dev,test}.conllu`. The extracted `lexicon.json` and `lexicon_train.json` are checked in, so you don't need to re-extract unless you change extraction logic.

## High-level API

The `tr_api` module is the recommended entry point for downstream code (web demos, batch scripts, integrations). It loads all data files once and returns JSON-serializable analyses:

```python
from tr_api import tokenize

result = tokenize("kitabımı")
# {
#   "surface": "kitabımı",
#   "parsed": true,
#   "root": "kitap",
#   "root_class": "NOUN",
#   "final_class": "NOUN",
#   "split": "kitab-ım-ı",
#   "tagged": "kitab+NOUN-ım+POSS_1SG[...]-ı+ACC[...]",
#   "morphemes": [
#     {"chunk": "kitab", "id": null,       "feats": {}, "is_root": true},
#     {"chunk": "ım",    "id": "POSS_1SG", "feats": {"Number[psor]": "Sing", "Person[psor]": "1"}, "is_root": false},
#     {"chunk": "ı",     "id": "ACC",      "feats": {"Case": "Acc"}, "is_root": false},
#   ],
#   "features":         {"Case": "Acc", "Number": "Sing", "Person": "3", ...},
#   "emitted_features": {"Case": "Acc", "Number[psor]": "Sing", "Person[psor]": "1"},
#   "score": 4.895,
#   "oov": false,
#   "alternatives": [...]
# }
```

The module-level `tokenize()` constructs a default `Tokenizer` on first call and reuses it for subsequent ones. For more control (custom lexicon, disabling alternatives), construct a `Tokenizer` directly:

```python
from tr_api import Tokenizer, TokenizerConfig

tok = Tokenizer(TokenizerConfig(
    lexicon_path = "lexicon_train.json",   # honest, no test leakage
    include_alternatives = False,           # only top analysis
))
result = tok.tokenize("çıkardı")
results = tok.tokenize_batch(["kitap", "gel", "çıkardı"])
```

The wire format is documented in `tr_api.py`'s module docstring and is stable. JSON serialization is verified by the test suite.

## Parsing a word (lower-level)

The minimal pattern is to load the three configuration objects, construct a parser, and call `.parse()`:

```python
from tr_inventory     import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon       import load_lexicon
from tr_parse         import Parser

inv   = load_inventory("inventory.json")
graph = load_graph("morphotactics.json")
lex   = load_lexicon("lexicon_train.json")
parser = Parser(lex, inv, graph)

analyses = parser.parse("kitabımı")
top = analyses[0]
print(top.split())        # → kitab-ım-ı
print(top.tagged())       # → kitab+NOUN-ım+POSS_1SG[Person[psor]=1,Number[psor]=Sing]-ı+ACC[Case=Acc]
print(top.root)           # → kitap
print(top.root_class)     # → NOUN
print(top.ud_feats())     # → {'Case': 'Acc', 'Number': 'Sing', 'Person': '3',
                          #    'Number[psor]': 'Sing', 'Person[psor]': '1'}
print(top.emitted_feats())# → {'Case': 'Acc', 'Number[psor]': 'Sing', 'Person[psor]': '1'}
                          #   (no default-fill)
```

The first call to `Parser(...)` is fast; per-call parses run at around 500–700 tokens/second on typical hardware. The `parser` object is safe to reuse and stateless across calls.

By default, `parse()` returns only the top-scoring analyses (multiple if tied). To see every candidate the chart produced, pass a `ParseConfig` with `return_all=True`:

```python
from tr_parse import ParseConfig
parser = Parser(lex, inv, graph, ParseConfig(return_all=True))
for a in parser.parse("evi"):
    print(f"  score={a.score:6.2f}  {a.tagged()}")
```

`evi` is genuinely ambiguous between `ev+POSS_3SG` ("her house") and `ev+ACC` ("the house [acc]"); both come back with the same score.

## Generating a form

The forward direction takes a root and a list of suffix IDs:

```python
from tr_generate import generate

surface = generate("kitap", ["POSS_1SG", "ACC"], lex, inv, graph)
print(surface)            # → kitabımı

surface = generate("gel", ["PROG", "1SG_Z"], lex, inv, graph)
print(surface)            # → geliyorum

surface = generate("yap", ["NEG", "FUT", "1SG_Z"], lex, inv, graph)
print(surface)            # → yapmayacağım
```

The root must be in the lexicon (or you can construct a `Root` directly and add it). The suffix chain must form a valid path through the morphotactic graph; an invalid sequence like `["PAST", "1SG_Z"]` (z-type agreement after past tense) raises a `ValueError`.

## Running the test suite

The full suite runs in around 15 seconds (the eval regression test dominates):

```bash
python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4 test_tr_phase5
```

To skip the slow regression test:

```bash
python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4
```

That runs in about 2 seconds. The regression test auto-skips if `UD_Turkish-IMST/tr_imst-ud-dev.conllu` isn't on disk.

## Running evaluation

The eval harness has its own CLI. The standard invocation reads the dev set, runs the parser, and prints per-UPOS metrics plus the top error sources:

```bash
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu \
    --lexicon lexicon_train.json \
    --show-errors 20
```

To use the all-splits lexicon (acknowledged to have leakage):

```bash
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu --lexicon lexicon.json
```

To evaluate on the test set (treat as final — don't iterate against it):

```bash
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-test.conllu --lexicon lexicon_train.json
```

`--limit N` evaluates only the first N tokens (useful for fast smoke tests during development). `--show-errors N` prints the first N root-mismatch examples to inspect manually.

The output has three sections: per-UPOS metrics (cover/root/upos/feat_exact/feat_F1), a sample of errors, and a "top feature-key error sources" table showing which feature keys have the highest false-positive and false-negative counts. The last one is the most useful diagnostic when you're trying to figure out where features are leaking.

## Re-extracting the lexicon

The lexicon is regenerated from the UD treebank by running:

```bash
python extract_lexicon.py UD_Turkish-IMST/ lexicon_train.json --split train
python extract_lexicon.py UD_Turkish-IMST/ lexicon.json
```

The default is `--split all` (every `.conllu` in the directory). The script detects softening and vowel-zero stems automatically, then merges in any manual overrides from `irregulars.json` and writes the result.

Re-extract whenever you change the extraction logic, update the UD corpus, or add new irregular overrides. The output is deterministic.

## Extending the morphology — adding a new suffix

Adding a suffix is a three-step change. First, write the entry in `inventory.json`. Second, add a transition in `morphotactics.json` saying which states it can leave from and which state it leads to. Third (optional), if the suffix needs a phonological alternation not covered by the existing rules, add a forward/inverse pair in `tr_rules.py`.

As a worked example, here's how the `IMP_2PL` (-Hn, 2nd person plural imperative) was added in Phase 5:

In `inventory.json`:

```json
{"id": "IMP_2PL", "template": "Hn",
 "feats": {"Mood": "Imp", "Person": "2", "Number": "Plur", "Tense": "Pres"},
 "category": "IMP", "rules": []}
```

In `morphotactics.json`, under the verbal transitions section:

```json
{"via": "IMP_2PL", "from": ["VERB_ROOT", "VERB_VOICE", "VERB_NEG"], "to": "VERB_AGR"}
```

This says IMP_2PL can attach to a bare verb root, a voice-marked stem, or a negated stem, and lands the parse in the verb-agreement state (which is accepting). No new rules are needed because the H-drop after vowel stems is handled by the existing `drop_initial_H_after_vowel_stem` rule, which the parser invokes for any H-initial template.

After editing the JSONs, run the test suite. If a new test is wanted, add it to `test_tr_phase4.py` using `assertTopParse` or `assertSomeParse`. Then re-run evaluation to see if dev-set metrics moved.

## Extending the morphology — adding a new morphotactic transition

A common case is opening a new path for an existing suffix. For example, allowing `COND` (`-sA`) to follow `AOR` (`-Hr`) so that `bulursanız` parses as `bul + AOR + COND + 2PL_K` was a one-line addition in Phase 5:

```json
{"via": "COND", "from": ["VERB_ROOT", "VERB_VOICE", "VERB_POT", "VERB_NEG", "VERB_TAM_Z"], "to": "VERB_TAM_K"}
```

Adding `VERB_TAM_Z` to the `from` array lets the conditional follow any z-type TAM, which models the actual grammar.

## Tuning parser behavior

`ParseConfig` exposes the scoring and search parameters. The defaults are:

```python
ParseConfig(
    return_all              = False,    # only top-scoring analyses
    max_oov_root_len        = 12,        # cap OOV root prefix length
    min_root_len            = 1,         # minimum
    oov_penalty             = -50.0,     # base score for OOV roots
    oov_char_bonus          = 0.35,      # per-char bonus on OOV roots
    inlex_char_bonus        = 0.15,      # per-char bonus on in-lex roots
    bare_root_bonus         = 0.5,       # bonus when an in-lex root has no suffixes
    derivation_penalty      = -2.0,      # per cross-class derivation step
    voice_penalty           = -2.2,      # per CAUS or PASS application
    compound_tense_penalty  = -2.5,      # per PAST_COP application
    suffix_bonus            = 1.0,       # legacy (superseded by per-char scoring)
)
```

The knobs that matter most:

- `inlex_char_bonus` and `bare_root_bonus` together encode "the unmarked reading of a known lemma is the lemma itself." Increasing them makes the parser more reluctant to peel suffixes off an in-lex root.
- `oov_char_bonus` prevents proper nouns from being shaved (`Osman` beating `osma + POSS_2SG[H-dropped]`).
- `voice_penalty` and `compound_tense_penalty` are scoring biases that suppress speculative analyses. Both are negative because they're applied as cost per morpheme.
- `derivation_penalty` applies when a suffix's `class_out` differs from `class_in` (the parse crosses NOUN→ADJ or VERB→NOUN etc.).

`return_all=True` is useful for inspecting genuine ambiguity, or for downstream code that wants to pass multiple candidates to a tagger.

## Inspecting individual analyses

The `Analysis` dataclass exposes the parser's internal view. Each morpheme in `analysis.morphemes` is a `Morpheme(suffix_id, chunk, feats, ...)`. The `suffix_id` is `None` for the root; otherwise it's the inventory key. The `chunk` is the surface substring consumed by this morpheme (possibly empty). The `feats` is a tuple of `(key, value)` pairs.

To debug a parse, iterate the morphemes:

```python
analyses = parser.parse("geliyordu")
top = analyses[0]
print(f"root: {top.root} ({top.root_class}), oov: {top.oov}, score: {top.score:.2f}")
for m in top.morphemes:
    label = m.suffix_id if m.suffix_id else f"ROOT:{m.word_class}"
    feats = "|".join(f"{k}={v}" for k, v in m.feats) or "-"
    print(f"  {label:10}  chunk={m.chunk!r:10}  feats={feats}")
```

This is the most useful tool when a parse is wrong and you want to understand why a particular chain won the scoring competition.
