# Quick reference card

Read `HANDOFF.md` first — this is just a scannable summary.

## Files at a glance

| File | What's in it |
|---|---|
| `inventory.json` | 92 suffix definitions (id, template, feats, category, rules) |
| `morphotactics.json` | State machine: ~140 transitions, ~14 states |
| `irregulars.json` | Manual lex overrides + pronouns + Q particles |
| `lexicon.json` | UD-extracted (5,478 entries, all splits) |
| `lexicon_train.json` | UD train-only (4,370 entries) |
| `lexicon_full.json` | UD + TDK (61,693 entries) — used by default |
| `tdk_words.json` | Raw TDK dictionary (92,406 headwords) |
| `tr_phonology.py` | apply_suffix, archiphonemes A/H/D/C/G |
| `tr_parse.py` | The parser; biggest and most subtle |
| `tr_rules.py` | @forward / @expand rule registry |
| `tr_generate.py` | generate(stem, chain) → surface |
| `tr_lexicon.py` | Root dataclass, Lexicon trie |
| `tr_inventory.py` | Loads inventory.json |
| `tr_morphotactics.py` | Loads morphotactics.json |
| `tr_api.py` | High-level Tokenizer class |
| `tr_server.py` | stdlib HTTP server |
| `tr_demo.html` | Single-file demo UI |
| `tr_evaluate.py` | Eval against UD CoNLL-U |
| `extract_lexicon.py` | UD → lexicon |
| `build_expanded_lexicon.py` | UD + TDK → lexicon_full |

## Commands

```bash
# Restore state (always first)
cp /mnt/user-data/outputs/*.py /mnt/user-data/outputs/*.json /mnt/user-data/outputs/*.md /mnt/user-data/outputs/*.html /home/claude/

# Re-fetch external data if missing
[ ! -d UD_Turkish-IMST ] && git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-IMST.git
[ ! -f tdk_words.json ] && (cd /tmp && git clone --depth 1 https://github.com/bilalozdemir/tr-word-list.git && cp tr-word-list/files/words.json /home/claude/tdk_words.json)

# Test suite
python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4 test_tr_phase5 test_tr_api

# Eval (full lex)
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu --lexicon lexicon_full.json --show-errors 0

# Re-extract lexicons (after editing irregulars or extract_lexicon)
python extract_lexicon.py UD_Turkish-IMST/ lexicon.json
python extract_lexicon.py UD_Turkish-IMST/ lexicon_train.json --split train
python build_expanded_lexicon.py --base lexicon.json --tdk tdk_words.json --out lexicon_full.json

# Parse one word
python -c "
from tr_inventory import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon import load_lexicon
from tr_parse import Parser, ParseConfig
p = Parser(load_lexicon('lexicon_full.json'),
           load_inventory('inventory.json'),
           load_graph('morphotactics.json'),
           ParseConfig(return_all=True))
for a in p.parse('WORD')[:5]:
    s = '+'.join(m.suffix_id for m in a.morphemes if m.suffix_id) or 'BARE'
    c = '-'.join(m.chunk if m.chunk else '∅' for m in a.morphemes)
    print(f'{a.score:6.2f} {\"*\" if a.oov else \" \"}{a.root}+{s}  ({c})')
"
```

## When you finish work

```bash
# Copy all changed files to outputs
cp inventory.json morphotactics.json irregulars.json lexicon*.json tr_*.py test_*.py *.md *.html /mnt/user-data/outputs/

# Re-bundle
rm -f /mnt/user-data/outputs/turkish-tokenizer.zip
zip -j /mnt/user-data/outputs/turkish-tokenizer.zip \
  tr_*.py test_*.py \
  inventory.json morphotactics.json irregulars.json \
  lexicon.json lexicon_train.json lexicon_full.json lexicon_overrides.json \
  tr_demo.html *.md
```

## Sanity-check parses

These should all parse correctly. If one breaks, something fundamental regressed:

| Word | Should produce |
|---|---|
| `kitap` | `kitap+BARE` |
| `kitabımı` | `kitap+POSS_1SG+ACC` |
| `geldi` | `gel+PAST` |
| `gelmem` | `gel+NEG+AOR+1SG_Z` (suppletion) |
| `gelmeyiz` | `gel+NEG+AOR+1PL_Z` |
| `gelmeliyim` | `gel+NEC+1SG_Z` |
| `geliyorum` | `gel+PROG+1SG_Z` |
| `oynuyorlardı` | `oyna+PROG+3PL+PAST_COP` |
| `başlıyor` | `başla+PROG` |
| `iyiyim` | `iyi+1SG_Z` (zero copula) |
| `evdeydim` | `ev+LOC+PAST_COP+1SG_K` |
| `heyecanlı` | `heyecan+ADJZ_LH` (productive decomp) |
| `gazeteci` | `gazete+NDER_CH` |
| `çaresiz` | `çare+ADJZ_SHZ` |
| `iyilik` | `iyi+NDER_LHK` |
| `kitapçık` | `kitap+NDER_CHK` |
| `elmalar` | `elma+PLUR` (NOT VBZ_LA+AOR) |
| `alındı` | `al+PASS+PAST` (NOT al+NMZ_HNTI) |
| `Bunu` | `bu+ACC` (pronominal-n) |
| `bana` | `ben+DAT` (irregular variant) |
| `mi` | `mi+BARE` (NOT m+POSS_3SG) |
| `misiniz` | `mi+2PL_Z` |
| `musunuzdur` | `mu+2PL_Z+COP_DHR` |
| `miydim` | `mi+PAST_COP+1SG_K` |
| `miysem` | `mi+COP_COND+1SG_K` |
| `vermeyince` | `ver+NEG+INCE` |
| `topallamazmış` | `topalla+NEG+AOR+COP_EVID` |
| `yapmışımdır` | `yap+EVID+1SG_Z+COP_DHR` |
| `yaptığımızsa` | `yap+DHK+POSS_1PL+COP_COND` |
| `hayatsızlarınkinde` | `hayat+ADJZ_SHZ+...+REL_KI+LOC` |
| `Çekoslovakyalılaştıramadıklarımızdan` | 11 morphemes |
| `muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine` | 17 morphemes |

## Eval baselines (dev set, current state)

| Lex | OVERALL root | UPOS | feat_F1 |
|---|---|---|---|
| train-only | 83.7% | 89.2% | 0.871 |
| full | 87.3% | 92.3% | 0.881 |

Floors in `test_tr_phase5.py` are calibrated to these. If you regress on UPOS or feat_F1 by more than ~1pp, investigate before bumping floors down.

Root accuracy will continue dropping with each new productive derivation added. This is the design intent. Just bump the floor.

## When you add a new derivational suffix

1. Add to `inventory.json` with id, template, feats, category, class_in, class_out.
2. Add a transition in `morphotactics.json` from the appropriate source state(s) to the target state.
3. Decide: is it productive (decomposition should always win)? Add to `productive_derivations` in `tr_parse.py`. Is it rare (only fires when no better parse exists)? Add to `rare_derivations`. Default: neither.
4. Add a test in `test_tr_phase3.py` (`TestExpandedDerivations` or a new class).
5. Test that the trick word still parses.
6. Add the ID to the JS `CATEGORY` map in `tr_demo.html` so it gets colored properly.
7. Update the inventory snapshot table in `HANDOFF.md`.

## When you add a new lexicon entry

For pronouns / particles / function words / irregular variants:

1. Add to `irregulars.json` in the appropriate section.
2. Re-extract: `python extract_lexicon.py UD_Turkish-IMST/ lexicon.json && python extract_lexicon.py UD_Turkish-IMST/ lexicon_train.json --split train && python build_expanded_lexicon.py --base lexicon.json --tdk tdk_words.json --out lexicon_full.json`
3. Test the new entry: `python -c "from tr_lexicon import load_lexicon; print(list(load_lexicon('lexicon_full.json').prefix_match('YOUR_FORM')))"`

## Sanity-check trick word after EVERY parser change

```python
from tr_inventory import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon import load_lexicon
from tr_parse import Parser
p = Parser(load_lexicon('lexicon_full.json'),
           load_inventory('inventory.json'),
           load_graph('morphotactics.json'))
w = 'muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine'
a = p.parse(w)[0]
assert a is not None and not a.oov, "BROKE THE TRICK WORD"
print(f'  {len(a.morphemes)} morphemes ✓')
```

Expected: 18 morphemes (1 root + 17 suffixes).
