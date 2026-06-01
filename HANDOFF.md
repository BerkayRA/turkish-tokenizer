# Handoff: Turkish morphological tokenizer

You are picking up an in-progress Turkish morphological tokenizer/parser. This document gives you everything needed to continue the work without re-deriving context. Read it end-to-end before touching code.

---

## TL;DR — what this is

A pure-Python (stdlib only) **Turkish morphological tokenizer and generator** built from grammar references (Göksel & Kerslake 2005, Lewis 1967, Goknel 2012, Cromwell list, Wikibooks). Given a Turkish surface form, it returns the root lemma plus the chain of morphemes that derives it, with UD-compatible feature tags. Also generates surfaces from (stem, chain) pairs. Used via Python API, HTTP server, and a no-framework HTML demo.

Current state: **259 tests passing**, 92 suffixes in inventory (44 derivational), 61,693-entry lexicon (UD + TDK), evaluation against UD_Turkish-IMST giving overall root 87.3% / UPOS 92.3% / feat_F1 0.881 on dev with the full lex.

The design intent: **morphological transparency over UD-lemma matching**. Forms like `heyecanlı` decompose to `heyecan+ADJZ_LH` even though UD records the lemma as `heyecanlı`. This is deliberate — the user has consistently chosen morphological correctness over eval scores. Don't reverse this when you see eval drops.

---

## How the user works

The user is fluent in Turkish and has been steering the design through specific failing examples. Their cadence:

- Sometimes they say "Continue from where we left off" and expect you to resume the in-progress task. The transcript is in `/mnt/transcripts/` — read the journal first to find the latest.
- Sometimes they give a short list of misses (e.g. "Bunu parses wrong; oynuyorlardı doesn't parse"). Reproduce each, diagnose individually, fix carefully, regress-test.
- Sometimes they direct architectural changes ("implement ALL derivational suffixes; the root of heyecanlı should be heyecan").
- They prefer Q:/A: button-style clarification when significant decisions exist, but generally trust you to proceed if direction is clear.
- They will correct your Turkish if you get it wrong — e.g. they corrected `gel+AOR+PAST_COP = gelirdi not gelerdi`. Take corrections seriously.
- They accept eval drops when justified by design intent. Don't apologize for them; just note them.

When they ask for something with "all", scope it: compile a comprehensive reference list first (search the web if needed), audit current coverage, identify gaps, add them, test.

---

## Filesystem layout

The project files live in `/home/claude/`. Between sessions the filesystem resets, so the latest state is mirrored to `/mnt/user-data/outputs/`. **First thing to do in a new session: copy outputs back to `/home/claude/`.**

```bash
cp /mnt/user-data/outputs/*.py /mnt/user-data/outputs/*.json /mnt/user-data/outputs/*.md /mnt/user-data/outputs/*.html /home/claude/
```

External data the project depends on:
- `UD_Turkish-IMST/` — clone of `https://github.com/UniversalDependencies/UD_Turkish-IMST.git`
- `tdk_words.json` — copy of `https://github.com/bilalozdemir/tr-word-list/files/words.json` (92,406 Turkish dictionary headwords from TDK under CC BY-SA 4.0)

To restore both:
```bash
cd /home/claude && git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-IMST.git
cd /tmp && git clone --depth 1 https://github.com/bilalozdemir/tr-word-list.git && cp tr-word-list/files/words.json /home/claude/tdk_words.json
```

### Project files

**Core modules** (in dependency order):
- `tr_phonology.py` — archiphoneme resolution (A, H, D, C, G), `apply_suffix(stem, template)`, vowel harmony, softening
- `tr_rules.py` — registry of `@forward` (generate-direction) and `@expand` (parse-direction) rules. Examples: `aorist_allomorphy`, `pronominal_n`, `potential_neg_suppletion`, `neg_aor_agreement_suppletion`
- `tr_inventory.py` — loads/validates `inventory.json`. `ALLOWED_TEMPLATE_CHARS` (Turkish letters + archiphonemes + buffers)
- `tr_morphotactics.py` — loads `morphotactics.json`. State machine with `start_state(class)` and `step(state, suffix)`
- `tr_lexicon.py` — Root dataclass + Lexicon trie. Auto-indexes softening variants. Loads from `lexicon*.json`
- `tr_generate.py` — `generate(stem, suffix_chain, ...) → Generation` (forward direction)
- `tr_parse.py` — Parser, ParseConfig, `match_suffix`. Chart-based parser; this is the biggest and most subtle file
- `tr_api.py` — `Tokenizer` class (high-level entry point), `tokenize(word)`, `tokenize_text(text)`
- `tr_server.py` — stdlib `http.server` exposing `/api/tokenize?word=X` and `POST /api/tokenize_text`
- `tr_demo.html` — single-file demo UI (no framework). CSS in `<style>`, JS at the bottom. Hovering a morpheme shows its features.

**Data files**:
- `inventory.json` — all suffix definitions (id, template, feats, category, class_in/out, rules)
- `morphotactics.json` — state graph (states list, transitions list with via/from/to)
- `irregulars.json` — manual lexicon overrides + extra entries (pronouns, demonstratives, question particles)
- `lexicon.json` — UD-extracted lex (5,478 entries, all UD splits)
- `lexicon_train.json` — UD-extracted, train split only (4,370 entries, used for honest eval)
- `lexicon_full.json` — UD + TDK merged (61,693 entries, used by default in the API)
- `lexicon_overrides.json` — derivational flag overrides (caus_deriv/pass_deriv, lexicalized-derivation pruning)
- `tdk_words.json` — raw TDK dictionary source (92,406 headwords)

**Build / eval scripts**:
- `extract_lexicon.py` — UD treebank → lexicon. Filters single-letter junk, applies irregulars overrides
- `build_expanded_lexicon.py` — Merge UD lex with TDK headwords → lexicon_full.json
- `tr_evaluate.py` — Evaluate against UD CoNLL-U. Reports overall + per-UPOS root/upos/feat_exact/feat_F1
- `api_samples.py` — Examples of using the Tokenizer API

**Tests** (`unittest`):
- `test_tr_phonology.py` — archiphonemes, apply_suffix, harmony, softening
- `test_tr_phase2.py` — inventory loading, validation
- `test_tr_phase3.py` — phonology + generation + parsing of inflectional + derivational + pronouns + Q particle. Largest test file
- `test_tr_phase4.py` — parser known-words + lexicon pruning + V→V derivation behavior
- `test_tr_phase5.py` — eval-floor regression tests (the "is the system as good as before?" test)
- `test_tr_api.py` — API end-to-end

**Docs**:
- `CODE.md`, `DESIGN.md`, `USAGE.md` — original design docs. Read them after this handoff.

---

## Required first action in a new session

```bash
# 1. Restore project to /home/claude/
cp /mnt/user-data/outputs/*.py /mnt/user-data/outputs/*.json /mnt/user-data/outputs/*.md /mnt/user-data/outputs/*.html /home/claude/

# 2. Re-clone external data if missing
ls /home/claude/UD_Turkish-IMST/*.conllu 2>/dev/null || \
  (cd /home/claude && git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-IMST.git)
ls /home/claude/tdk_words.json 2>/dev/null || \
  (cd /tmp && git clone --depth 1 https://github.com/bilalozdemir/tr-word-list.git && cp tr-word-list/files/words.json /home/claude/tdk_words.json)

# 3. Sanity check the parser
cd /home/claude && python -c "
from tr_inventory import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon import load_lexicon
from tr_parse import Parser
p = Parser(load_lexicon('lexicon_full.json'),
           load_inventory('inventory.json'),
           load_graph('morphotactics.json'))
for w in ['heyecanlı', 'gazeteci', 'mi', 'musunuzdur', 'gelmem',
          'muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine']:
    a = (p.parse(w) or [None])[0]
    if a:
        suffs = '+'.join(m.suffix_id for m in a.morphemes if m.suffix_id) or 'BARE'
        print(f'  {w[:40]:40} → {a.root}+{suffs}')
"

# 4. Run the full test suite
cd /home/claude && python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4 test_tr_phase5 test_tr_api
```

If all 259 tests pass, you're good. If they don't, something got mis-restored — diff against `/mnt/user-data/outputs/` and fix.

---

## Architecture overview

### Phonology

Turkish suffixes are templates over archiphonemes — abstract symbols that resolve to specific surface letters based on context:

- `A` → `a` or `e` (low-vowel harmony)
- `H` → `ı`, `i`, `u`, `ü` (high-vowel harmony)
- `D` → `d` or `t` (voicing assimilation: `t` after voiceless)
- `C` → `c` or `ç` (voicing assimilation)
- `G` → `g` or `k` (voicing assimilation; only used in `-GHn`, `-GAn`)

Buffer consonants in parens `(y)`, `(s)`, `(n)`, `(ş)`: realized after vowel-final stems, suppressed after consonant-final. E.g. `-(y)Hm` = `iyim` after consonant, `yim` after vowel.

Final-consonant softening: `p/ç/t/k → b/c/d/ğ` before vowel-initial suffix on stems flagged `soften=True`.

All implemented in `tr_phonology.py:apply_suffix` (generate) and `tr_parse.py:match_suffix` (parse). The two directions MUST stay consistent — when you change one, change the other.

### Morphotactics

`morphotactics.json` lists states and transitions. States include `VERB_ROOT`, `VERB_VOICE`, `VERB_NEG`, `VERB_POT`, `VERB_TAM_Z`, `VERB_TAM_K`, `VERB_AGR`, `NOUN_ROOT`, `NOUN_NUM`, `NOUN_POSS`, `NOUN_CASE`, `ADJ_ROOT`, `COP_FIN`, `ADV_TERM`. States marked `"accept": true` are terminal (a parse can stop there).

The parser fills a chart: each fragment is `(end-position-in-surface, current-state, root-info, morpheme-chain, score)`. New fragments come from (a) seeding with lexicon prefix matches, (b) extending an existing fragment via a transition + matching the suffix template against the remaining surface.

### Scoring (CRITICAL — this is where tuning lives)

In `tr_parse.py:ParseConfig`:

```
oov_penalty:            -50.0   # base score for OOV root candidate
oov_char_bonus:           0.35  # per-char bonus on OOV roots
inlex_char_bonus:         0.15  # per-char bonus on in-lex roots
bare_root_bonus:          0.5   # bonus for in-lex + no suffixes
derivation_penalty:      -2.0   # per cross-class step
productive_derivations:  ("ADJZ_LH", "ADJZ_SHZ", "ADJZ_MSH",
                          "NDER_CH", "NDER_LHK", "NDER_CHK",
                          "NDER_DAS", "ADV_CA")
productivity_bonus:       2.5   # offsets derivation_penalty for productive
rare_derivations:        (NMZ_HNTI, NMZ_MACA, NMZ_DHKCE, ADJZ_MTRK,
                          ADJZ_CHL, ADJ_GAN, ADJ_MAZ, VBZ_DA, VBZ_HMSA,
                          VMOD_GEL, VMOD_DUR, VMOD_YAZ, NDER_CAGIZ, NDER_GHL)
rare_derivation_extra_penalty: -3.0
voice_penalty:           -2.2   # per CAUS or PASS application
compound_tense_penalty:  -2.5   # per PAST_COP
```

Per-suffix-match: `score_delta = 0.5 * len(chunk) - 0.2` (or `-0.2` if chunk empty), plus the various penalties.

**Adding a new productive suffix can break tests.** When you add one to `productive_derivations`, check that it doesn't make wrong decompositions win — VBZ_LA was deliberately excluded because its `lA` template eats the start of plural `-lAr` (`elma+VBZ_LA+AOR` would beat `elma+PLUR`).

When adding a new rare/lexicalized suffix, default to including it in `rare_derivations` unless you've tested it doesn't over-fire.

### Constraint propagation

A few rules need to "lock in" what suffix can come next. The mechanism: an `expand` rule can set `_next_must_be: tuple` in the ctx of an alternative it returns; the parser then refuses to extend that fragment with anything not in the tuple.

Used by:
- POT suppletion (`potential_neg_suppletion` in `tr_rules.py`): the suppletive `(y)A` form of POT can only be followed by NEG
- PROG truncation (in `tr_parse.py:_fill_chart`): when seeding a truncated vowel-final verb stem, the next suffix must be PROG

This isn't documented elsewhere yet — putting it in `DESIGN.md` is on the TODO list.

### Rule registry

`tr_rules.py` has two registries: `_forward` (used in generation) and `_expand` (used in parsing). `@forward("rule_name")` and `@expand("rule_name")` decorators register handlers. Each rule has signature `(stem, template, ctx) -> (stem, template, ctx)` for forward; `(stem, template, ctx) -> list of (template, ctx)` for expand (multiple alternatives).

Ctx is a dict carrying `prev_morph_id`, `prev_morph_chunk`, `prev_prev_morph_id`, `next_morph_id`, and root-level flags (`root_aorist_high`, `root_caus_deriv`, `root_pass_deriv`, `root_pronominal_n`). The chart-fill thread these through; if you add a context-dependent rule, you may need to extend the ctx setup.

If a rule needs to delete a stem character (like PROG's stem-final low-vowel deletion), it should NOT do it in the rule — that's handled at the seeding stage in `_fill_chart` instead. See the PROG seeding pass for the pattern.

### Lexicon

`tr_lexicon.py:Root` dataclass: `form, word_class, soften, variants, frequency, aorist_high, caus_deriv, pass_deriv, pronominal_n`.

The Lexicon trie indexes all canonical forms AND their explicit `variants`, AND auto-generates softening variants for `soften=True` nouns. It does NOT auto-generate truncated forms (PROG handling lives in the parser's seeding code, not here).

`Lexicon.prefix_match(surface)` returns all roots whose canonical-or-variant is a prefix of the surface.

`Root.root_ctx()` returns the dict of root-level flags to seed into ctx when seeding from this root.

### Irregulars

`irregulars.json` has two roles:
- **Update** existing lexicon entries with flags they need (e.g. `gel` gets `aorist_high: True`)
- **Add** brand-new entries not present in UD extraction (e.g. pronouns `bu/şu/o`, question particles `mi/mı/mu/mü`, function words `değil/var/yok`)

`extract_lexicon.py` applies these during extraction. To add a new pronoun or function word, edit `irregulars.json` and re-run `extract_lexicon.py` followed by `build_expanded_lexicon.py`.

---

## The big trick word

The user introduced this as a benchmark:

```
muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine
```

Translation: "as if you might be among those whom we will (apparently) not be able to quickly make into ones who make [others] unsuccessful." Top parse:

```
muvaffakiyet + ADJZ_SHZ + VBZ_LAS + CAUS + AGENT + VBZ_LAS + CAUS + VER
            + POT + NEG + POT + FUT_PART + PLUR + POSS_1PL + ABL
            + COP_EVID + 2PL_Z + CASINA
```

That's 17 morphemes. **If your changes break this parse, you've broken something fundamental.** Always test this word.

Similar test word: `Çekoslovakyalılaştıramadıklarımızdan` (11 morphemes including POT-before-NEG suppletion).

---

## Pitfalls and gotchas

### Don't break vowel-final verb + PROG

Verbs ending in `a`/`e` (`başla`, `oyna`, `bekle`) plus PROG `-Hyor` delete the stem-final low vowel. `başla + Hyor = başlıyor`. This is handled by a dedicated seeding pass in `tr_parse.py:_fill_chart` that scans for `word[:k] + 'a'` and `word[:k] + 'e'` in the lexicon, seeding the truncated stem with a `next_must_be=("PROG",)` constraint. If you mess with seeding, run `python -c "import ...; p.parse('oynuyorlardı')"` to confirm.

### NEG-AOR-1SG suppletion

`gel + NEG + AOR + 1SG_Z = gelmem` (not `*gelmeyim`). Handled by `neg_aor_agreement_suppletion` rule. The NEG → AOR step produces an empty chunk; the 1SG_Z rule sees `prev_morph_id == "AOR"` and `prev_morph_chunk == ""` and switches its template from `(y)Hm` to bare `m`.

1PL_Z does NOT have a corresponding suppletion: `gelmeyiz` works with the default `(y)Hz`.

### Buffer matching is strict

`tr_parse.py:match_suffix` enforces:
- After vowel-final stem: buffer MUST be realized
- After consonant-final stem: buffer MUST be suppressed

If you relax this, weird spurious decompositions appear (POT-suppletive `(y)A` matching empty after vowel-final stems being a famous example).

### Lexicon order matters for some parses

When two roots match the same prefix (e.g. `kere` and `kerem`), the longer wins by per-char bonus. But if their frequencies differ a lot, the shorter+higher-freq can win on the frequency score. Be aware when adding new pronouns/function words — their frequency value (in irregulars.json) directly determines whether they win against shorter common nouns.

### Morphotactic transitions need both inflection AND adjective routes

When you add a derivational suffix like `ADJZ_LH` that produces ADJ output, you need to verify the ADJ_ROOT entry-state has all the right outgoing transitions (PLUR, POSS_*, all CASE, all z-agreement, the copular suffixes). Otherwise things like `hayatsızlarınkinde` fail at the nominal-track step.

In the current state: ADJ_ROOT participates in the full nominal track (PLUR/POSS/CASE) plus the verbal track at 1SG_Z/2SG_Z/1PL_Z/2PL_Z (zero copula) plus the copular suffixes plus REL_KI. Verify these transitions stay intact when you edit morphotactics.json.

### Trans-class derivation chains

When VBZ_LAS goes ADJ→VERB, the verbal-track suffixes (NEG, POT, TAM, etc.) need to fire from VERB_ROOT — they do. Conversely, V→ADJ via AGENT (`yazıcı`) needs ADJ_ROOT's transitions to work. The pattern is "transitions are defined by the target state, not where you came from." Check by tracing a chain end-to-end.

### Single-letter roots are noise

UD treebank tokenizer produces stray 1-letter NOUN entries (`l, m, n, c, d, j, w, i, e`) from initialisms and OCR artifacts. `extract_lexicon.py` filters them out at extraction. The only legitimate single-letter root is `o` (pronoun), which is injected via `irregulars.json`. There's a regression test (`test_no_single_letter_roots`) that fails if any non-`o` single-letter root sneaks in.

### TDK has tons of lexicalized derivations as headwords

The TDK dictionary lists `gazeteci`, `heyecanlı`, `iyilik`, `kitapçık`, etc. as their own entries. The user explicitly wants these to DECOMPOSE — that's what `productive_derivations` and `productivity_bonus` are for. When you add a new productive derivation, add the suffix ID to `productive_derivations` AND verify the user's target forms decompose.

### Eval drops are expected when adding productive derivations

UD records lexicalized forms. We decompose. Adding `ADJZ_LH` to `productive_derivations` will drop NOUN root accuracy by 5-6pp because tokens UD says are `heyecanlı` we now say are `heyecan`. The user has accepted this trade-off repeatedly. **Don't reverse it.** Update floors in `test_tr_phase5.py` and move on. Note the drop in the summary so the user sees it.

---

## Coverage gaps and TODO items the user might pick up

Things I noticed but didn't fully address. Each is a candidate next direction.

### 1. Sentence-level Q particle attached form

In casual Turkish text, `mi/mı/mu/mü` can be written attached: `gelecekmisin` (= `gelecek misin`). The current parser doesn't split this. Implementing a pre-tokenizer pass that detects `*m{i,ı,u,ü}{si,si,...}` at the end and splits it off would catch this. UD-Turkish-IMST has both conventions in its data.

### 2. Proper noun handling

`Kerem` (proper name) consistently parses as `kere+...` (the common noun "time/occasion") because the trie picks up the shorter root. Adding a `proper_noun_bonus` for capitalized words (or making the OOV path stronger when the surface is title-case) would help. Eval errors include 5+ instances of this for `Kerem` alone.

### 3. NEG combined with NEC, EVID, FUT

I added NEC but didn't verify that `gelmemeli` ("must not come") chains correctly. Same for `gelmemiş` (NEG+EVID), `gelmeyecek` (NEG+FUT) — these might already work because NEG → VERB_NEG and those tense markers fire from VERB_NEG, but worth confirming with explicit tests.

### 4. POSS_2SG vs pronominal-n ambiguity

`evdekinde` parses correctly as `ev+LOC+REL_KI+LOC` (with `n` rolled into the LOC chunk), but related forms like `hayatsızlarınkinde` go through a possessive interpretation. The pronominal-n insertion rule (`tr_rules.py`) covers POSS_3SG/POSS_3PL/REL_KI; it doesn't cover the spurious POSS_2SG reading that the parser sometimes picks. A scoring tweak could fix this.

### 5. Reduplication

`çok çok`, `koşa koşa`, `yavaş yavaş` (intensification/adverbial). Currently UD treats these as separate tokens so it doesn't break anything; but a real Turkish NLP system might want a special "reduplication" relationship.

### 6. -(H)kİ vs -(H)yken (temporal converb)

`gelirken` ("while coming") = `gel+AOR+(y)kAn`. The temporal converb `-(y)kAn` isn't in the inventory. This is a real gap.

### 7. Documentation

`DESIGN.md` doesn't explain constraint propagation (`_next_must_be`). `CODE.md` doesn't explain the productivity bonus / rare derivation penalty. Worth updating.

### 8. Scoring is brittle

The interaction between `productivity_bonus`, `rare_derivation_extra_penalty`, `voice_penalty`, `compound_tense_penalty`, `bare_root_bonus`, `inlex_char_bonus`, `oov_char_bonus`, `derivation_penalty` is delicate. Each new suffix added requires re-running tests and sometimes adjusting one of these dials. A grid-search over a curated test set (or learned weights) would be more robust.

### 9. Layering violation in tests

`test_no_single_letter_roots` reaches into `lex._by_form`. Adding a public `Lexicon.all_forms()` API would be cleaner.

### 10. Better eval

Eval against UD penalizes the deliberate over-decomposition policy. A hand-curated "morphologically correct decomposition" eval set would be more honest. The user might want this — flag it.

### 11. Confidence scores

The current `score` is comparative (higher = better among alternatives), not interpretable as a probability. A calibrated confidence would be useful for downstream tasks.

### 12. OOV class inference

When the parser falls back to OOV (no in-lex match), it picks a default class. Suffix patterns could disambiguate ("-mAk" → VERB-citation form, "-lAr" → noun plural, etc.). The current code has stubs for this in `tr_parse.py:_oov_inference` but it's rudimentary.

### 13. Lemmatization output

The API returns root + morpheme chain. Many downstream tasks want just the lemma string. Adding a `tokenize(word).lemma` property would be trivial and helpful.

### 14. Generation needs all the same rules

When you add a new context-dependent parsing rule, the corresponding forward rule for generation usually needs the same. Mismatches between parse and generate can cause `generate(parse(word))` round-trip failures. Test round-trips on common forms after edits.

---

## Working method (style notes)

- **Reproduce before fixing.** When the user reports a miss, run the parser on it first to see the exact analyses returned. Don't fix from memory.
- **Diagnose individually.** Each miss usually has a specific cause (missing suffix, wrong transition, scoring imbalance). Don't bundle diagnoses.
- **Test the trick word after every change.** `muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine` is the canary.
- **Run the full test suite before bundling.** All 259 should pass (or expected new ones added).
- **Bundle to `/mnt/user-data/outputs/`** — the zip + individual files. The user inspects both.
- **Summary structure** the user likes: what was fixed, what changed architecturally, what tests were added, what the eval impact was, what's in the bundle. Note open questions.
- **Don't apologize.** When eval drops because of intentional design, just state it and move on.

---

## Commands cheatsheet

```bash
# Sanity check parser
python -c "from tr_inventory import load_inventory; from tr_morphotactics import load_graph; from tr_lexicon import load_lexicon; from tr_parse import Parser; p = Parser(load_lexicon('lexicon_full.json'), load_inventory('inventory.json'), load_graph('morphotactics.json')); print(p.parse('SOME_WORD'))"

# Re-extract lexicons after editing irregulars.json or extract_lexicon.py
python extract_lexicon.py UD_Turkish-IMST/ lexicon.json
python extract_lexicon.py UD_Turkish-IMST/ lexicon_train.json --split train
python build_expanded_lexicon.py --base lexicon.json --tdk tdk_words.json --out lexicon_full.json

# Run a specific test
python -m unittest test_tr_phase3.TestQuestionParticles -v

# Full test suite
python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4 test_tr_phase5 test_tr_api

# Eval on dev set (full lex — what users see)
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu --lexicon lexicon_full.json --show-errors 0

# Eval on dev set (train-only lex — honest, no leakage)
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu --lexicon lexicon_train.json --show-errors 0

# See sample errors during eval (useful for debugging)
python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu --lexicon lexicon_full.json --show-errors 30

# Start the HTTP server (for the demo)
python tr_server.py --port 8766 &
curl "http://127.0.0.1:8766/api/tokenize?word=gazeteci"
curl -X POST -H "Content-Type: application/json" -d '{"text":"Geldin mi?"}' http://127.0.0.1:8766/api/tokenize_text

# Bundle outputs
cp inventory.json morphotactics.json irregulars.json lexicon*.json tr_*.py test_*.py *.md *.html /mnt/user-data/outputs/
cd /home/claude && rm -f /mnt/user-data/outputs/turkish-tokenizer.zip && zip -j /mnt/user-data/outputs/turkish-tokenizer.zip tr_*.py test_*.py inventory.json morphotactics.json irregulars.json lexicon*.json *.md tr_demo.html
```

---

## Current inventory snapshot (for reference)

92 suffixes total. 44 derivational:

| ID | Template | In → Out | Use |
|---|---|---|---|
| `VBZ_LA` | `lA` | N→V | productive (kanla "to bleed") |
| `VBZ_LAS` | `lAş` | ADJ→V | productive (güzelleş "to become beautiful") |
| `VBZ_LAN` | `lAn` | N→V | inchoative (evlen "to marry") |
| `VBZ_A` | `A` | N→V | yaş→yaşa |
| `VBZ_AL` | `Al` | ADJ→V | çoğal "to multiply" |
| `VBZ_AR` | `Ar` | ADJ→V | morar "to become purple" |
| `VBZ_HMSA` | `(H)msA` | N→V | "consider as X" (rare) |
| `VBZ_DA` | `DA` | N→V | onomatopoeic repeated (rare) |
| `ADJZ_LH` | `lH` | N→ADJ | **productive** (with-X) |
| `ADJZ_SHZ` | `sHz` | N→ADJ | **productive** (without-X) |
| `ADJZ_MSH` | `(H)msH` | N→ADJ | **productive** (X-like) |
| `ADJZ_MTRK` | `(H)mtırak` | ADJ→ADJ | color-ish (rare) |
| `ADJZ_CHL` | `CHl` | N→ADJ | color/material (rare) |
| `ADJ_GHN` | `GHn` | V→ADJ | state (kızgın, durgun) |
| `ADJ_HK` | `Hk` | V→ADJ | past-participle (yanık) |
| `ADJ_GAN` | `GAn` | V→ADJ | habitual (çalışkan) (rare) |
| `ADJ_MAZ` | `mAz` | V→ADJ | neg-aorist part. (rare) |
| `AGENT` | `(y)HcH` | V→ADJ | yazıcı |
| `NDER_CH` | `CH` | N→N | **productive** (agent: balıkçı) |
| `NDER_LHK` | `lHk` | N→N | **productive** (abstract: iyilik) |
| `NDER_CHK` | `CHk` | N→N | **productive** (diminutive: kitapçık) |
| `NDER_CAGIZ` | `CAğHz` | N→N | affectionate dim. (rare) |
| `NDER_DAS` | `DAş` | N→N | **productive** (companion: vatandaş) |
| `NDER_GHL` | `gHl` | N→N | family/place (rare) |
| `NMZ_INF` | `mAk` | V→N | infinitive |
| `NMZ_GER` | `mA` | V→N | gerund/action noun |
| `NMZ_HS` | `(y)Hş` | V→N | action noun (geliş) |
| `NMZ_HM` | `(y)Hm` | V→N | instance (bilim) |
| `NMZ_MAN` | `mAn` | V→N | agent (öğretmen) |
| `NMZ_HNTI` | `Hntı` | V→N | residue (rare) |
| `NMZ_MACA` | `mAcA` | V→N | game (rare) |
| `NMZ_DHKCE` | `DHkçA` | V→N | "the more X" (rare) |
| `DHK` | `DHk` | V→N | nominalizer/participle |
| `FUT_PART` | `(y)AcAk` | V→N | future participle |
| `ADV_CA` | `CA` | ADJ→ADJ | **productive** (adverb: güzelce) |
| `VER` | `(y)Hver` | V→V | quick aspect |
| `RECP` | `(H)ş` | V→V | reciprocal |
| `VMOD_GEL` | `(y)Agel` | V→V | continual (rare) |
| `VMOD_DUR` | `(y)Adur` | V→V | keep-on (rare) |
| `VMOD_YAZ` | `(y)Ayaz` | V→V | almost-Xed (rare) |
| `CAUS_DERIV` | `Hr` (etc) | V→V | lexicalized CAUS, per-root |
| `PASS_DERIV` | `Hn` (etc) | V→V | lexicalized PASS, per-root |
| `CASINA` | `cAsHnA` | ADJ→ADV | "as if" |
| `REL_KI` | `ki` | NOUN_CASE→ADJ | relational |

Bold = in `productive_derivations` (winning bonus). Rare-marked = in `rare_derivations` (extra penalty).

---

## Reference data for the user's questions

If the user says "I noticed X parses wrong":

1. **Reproduce in `return_all=True` mode** to see all candidates and their scores. This tells you whether the right parse is even AVAILABLE.
2. If not available: the issue is in inventory or morphotactics. Check that the suffix exists; check that the morphotactic transitions allow the chain.
3. If available but losing on score: the issue is in scoring. Check `productive_derivations` membership, `rare_derivations` membership, or per-suffix penalties.

Common Turkish trip-ups in our parser:
- `Kerem` (proper noun) → wants OOV, gets `kere+something`. Proper-noun handling is open.
- Compound words / borrowings → variable. We can't always know.
- Verb stem-final low-vowel deletion before PROG (`başla → başl + ıyor`) — handled by the dedicated seeding pass.
- The `-mAlH` necessitative chain — works but I haven't tested NEG before it (`gelmemeli`).

---

## Reference: previous transcripts

Located in `/mnt/transcripts/`. Read the journal (`/mnt/transcripts/journal.txt`) to find the latest. Each transcript covers a session's work. The most recent should be from June 1, 2026.

Useful prior sessions in date order:
- `2026-05-14-*` — phases 1-5 of the original build
- `2026-05-20-*` — V→V derivation (CAUS_DERIV/PASS_DERIV) added
- `2026-05-21-*` — trick-word infrastructure (VBZ_LAS, AGENT, VER, COP_EVID, CASINA, POT+NEG suppletion)
- Current session (whichever date) — derivational expansion + pronouns + Q particle + Q-particle-chain support

---

## Final notes

- The project is in good shape. 259 passing tests, comprehensive coverage of inflectional + derivational + copular + Q-particle morphology.
- The biggest weakness is the scoring tuning — fragile when adding new suffixes.
- The biggest gap is proper-noun handling — capitalized words consistently misparse.
- The user values transparency, decomposition, and concrete examples. They will tell you what's broken; you fix it; you verify the trick word still parses.

Good luck.
