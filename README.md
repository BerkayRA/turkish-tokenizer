# Turkish Morphological Tokenizer

A dependency-free Turkish morphological tokenizer for NLP. It splits an
inflected word into a root plus an ordered chain of morpheme tags with their
features — e.g.

```
kitabımı   →  kitap + POSS_1SG + ACC        (split: kitab-ım-ı)
geliyordu  →  gel + PROG + PAST_COP         (split: geliyor-du)
```

It is pure Python (standard library only — **no third-party dependencies**),
built on an archiphoneme phonology layer, a UD-aligned suffix inventory, and a
finite-state morphotactic graph. The design favors **morphological
transparency** (recovering the productive derivation) over matching a
lexicalized dictionary lemma.

## Features

- Full open-class morphology: nouns, verbs, adjectives, and the major
  derivational suffixes between them.
- Vowel harmony, consonant softening, voicing assimilation, and buffer-consonant
  insertion handled in a dedicated, independently tested phonology layer.
- Universal Dependencies feature names; evaluated against UD_Turkish-IMST.
- Proper-noun handling: apostrophe-boundary and title-case aware
  (`Muammer'in`, `Parkı'ndan`, `Hacı`).
- Attached interrogative-particle splitting for informal text
  (`gelecekmisin` → `gelecek` + `misin`), conservative and vowel-harmony aware
  so ordinary words (`resmi`, `ölümü`) are left intact.
- Circumflex-insensitive matching (`mekân` = `mekan`, `resmî` = `resmi`).
- OOV spelling suggestions and morphology-aware correction
  (`okllarda` → `okullarda`) via a reusable BK-tree fuzzy index (`tr_fuzzy.py`).
- JSON-serializable API with a `lemma`/`surface` convenience shape and ranked
  alternative analyses.

## Requirements

- Python 3.11+ (developed on 3.14). No external packages.

## Quick start

```python
from tr_api import tokenize

result = tokenize("kitabımı")
print(result["lemma"])   # "kitap"
print(result["split"])   # "kitab-ım-ı"
print(result["tagged"])  # "kitab+NOUN-ım+POSS_1SG[...]-ı+ACC[...]"
```

For more control (custom lexicon, disabling alternatives, clitic splitting):

```python
from tr_api import Tokenizer, TokenizerConfig

tok = Tokenizer(TokenizerConfig(include_alternatives=False))
tok.tokenize("çıkardı")
tok.tokenize_text("Yarın gelecekmisin?")   # splits the attached "mi"
```

See [USAGE.md](USAGE.md) for the full API, the wire format, and the
lower-level `Parser`/`generate` entry points.

## Tests

```bash
python -m unittest discover -p "test_tr_*.py"
```

315 tests across phonology, the suffix inventory, the parser, the API,
proper-noun handling, clitic pre-tokenization, diacritic folding, and the
fuzzy suggester, plus an evaluation regression guard (`test_tr_phase5.py`)
that locks in headline metrics on the UD dev set.

## Evaluation

```bash
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu \
    --lexicon lexicon_train.json --show-errors 0
```

On the UD_Turkish-IMST dev set with the no-leakage train-only lexicon: root
~83.8%, UPOS ~89.2%, feature F1 ~0.87. (Root accuracy is intentionally
secondary to morphological correctness — see the design notes.)

> The evaluation corpus (`UD_Turkish-IMST/`) and the TDK headword dump
> (`tdk_words.json`) are external data and are **not** included in this
> repository (see `.gitignore`). `tr_evaluate.py` and lexicon re-extraction
> need them; the parser and the test suite do not.

## Documentation

| File | Contents |
|------|----------|
| [USAGE.md](USAGE.md) | API, wire format, running tests/eval, extending the morphology |
| [DESIGN.md](DESIGN.md) | What was built and why; phase-by-phase design decisions and trade-offs |
| [CODE.md](CODE.md) | Module-level code walkthrough |
| [QUICKREF.md](QUICKREF.md) | Compact reference for suffix IDs and features |
| [SUGGESTIONS.md](SUGGESTIONS.md) | Possible next directions |

## License

[MIT](LICENSE)
