# Code reference

This document is a map of the codebase: what each file does, what it exports, and how the pieces fit together. Modules are listed in dependency order; each builds on the ones above it.

## The dependency stack

At the base sits `tr_phonology.py`, which knows only about Turkish vowels, consonants, and surface-level alternations — no morphology. Above it, `tr_rules.py` packages the optional phonological alternations into a registry of named forward/inverse rule pairs. `tr_inventory.py` and `tr_morphotactics.py` then load the two JSON data files that define the morphology (what suffixes exist, what can follow what). `tr_lexicon.py` loads the root inventory. With those four layers in place, `tr_generate.py` can take a root plus a list of suffix IDs and produce a surface form; `tr_parse.py` does the inverse, taking a surface form and recovering the root + suffixes. `tr_evaluate.py` wraps the parser in a UD-corpus eval harness, and `extract_lexicon.py` is the offline tool that produced `lexicon.json` from the UD treebank.

Everything is pure Python with no external dependencies.

## tr_phonology.py — alphabet, harmony, alternations

The foundation. This module knows that Turkish has eight vowels split into four backness/rounding classes, that voiceless consonants are `çfhkpsşt`, and that final consonants soften before vowel-initial suffixes (`p→b`, `t→d`, `ç→c`, `k→ğ`). It does not know what a "suffix" is.

The most important export is `apply_suffix(stem, template) → surface_form`. The template uses Oflazer-style archiphonemes: `A` is a low vowel that resolves to `a` or `e` by backness harmony; `H` is a high vowel that resolves to `ı`/`i`/`u`/`ü` by backness *and* rounding harmony; `D` is a coronal stop that resolves to `t` or `d` by voicing assimilation; `C` likewise for `ç`/`c`. Parentheses mark optional buffer consonants — `(y)A` realizes as `ya` after a vowel and just `a` after a consonant. So `apply_suffix("kitap", "Hm") = "kitabım"` (high vowel resolves to ı, final p softens to b) and `apply_suffix("araba", "(y)A") = "arabaya"` (buffer y inserted, A resolves to a).

Other useful exports: `tr_lower` and `tr_upper`, which handle the dotted/dotless I correctly (`İ → i`, not `İ → i̇`); the constant sets `VOWELS`, `BACK_VOWELS`, `HIGH_VOWELS`, `VOICELESS_CONSONANTS`; the lookup dicts `SOFTEN` and `HARDEN`; and predicates like `is_vowel`, `is_voiceless`, `last_vowel`.

The module is heavily tested (53 unit tests in `test_tr_phonology.py`) because everything downstream assumes it's correct.

## tr_rules.py — optional phonological alternations

Some Turkish alternations apply only with specific suffixes, not universally. A canonical example is the high vowel of `-Hyor` (progressive) deleting the final low vowel of a preceding stem: `başla + Hyor → başlıyor`, not `*başlayıyor`. The alternation is real but it doesn't happen with every suffix.

This module defines a `RULES` registry that maps rule names like `"delete_stem_final_low_vowel"` to a `Rule` object holding both directions: a `forward` function applied during generation and an `expand` function applied during parsing. (Parsing produces zero or more candidate templates per rule; "expand" describes the inversion better than "inverse" because some rules return multiple alternatives.) Suffixes in `inventory.json` list which rules apply via a `"rules": ["..."]` field.

Two decorators, `@forward(name)` and `@expand(name)`, register functions into the global registry. New rules are added by writing a paired function with both decorators and a stem-mutating logic. Inventory validation reads this registry to fail-fast if a suffix declares a rule that doesn't exist.

The currently registered rules:
- `delete_stem_final_low_vowel` — used by PROG; deletes the stem's final low vowel before -Hyor.
- `drop_initial_H_after_vowel_stem` — used by H-initial agreement and possessive suffixes.
- `insert_n_after_3rd_person_possessive` — the pronominal-n that surfaces between POSS_3SG/POSS_3PL and case markers (`araba-sı-n-da`); reads `prev_morph_id` from ctx.
- `aorist_allomorphy` — picks among -r (vowel-final), -Ar (default monosyllabic consonant-final), and -Hr (polysyllabic OR lexically aorist_high). Also handles the NEG-AOR suppletion: empty before 1SG_Z/1PL_Z, -z elsewhere.
- `passive_allomorphy` — fully phonological: -n after vowels, -Hn after l, -Hl after other consonants.
- `causative_allomorphy` — fully phonological for inflectional CAUS: -t after vowels and after polysyllabic r/l-final stems, -DHr elsewhere.
- `caus_deriv_allomorphy` / `pass_deriv_allomorphy` — derivational V→V suffixes. Gated by lex flags (`root_caus_deriv`, `root_pass_deriv`) whose value IS the template string; the expand view returns the templated alternative if the flag is set, else returns empty list (blocking the suffix).

Some rules read context beyond just the stem: aorist allomorphy reads `root_aorist_high` (lex flag) and `prev_morph_id` (for NEG suppletion) and `next_morph_id` (to pick between -z and ∅). The context is threaded through by both the generator and the parser; root-level flags are stored on `ParseFragment.root_ctx` at seed time and merged into per-morpheme ctx during chart fill.

## tr_inventory.py — the suffix inventory loader

A thin loader and validator for `inventory.json`. The data class `Suffix(id, template, feats, category, rules, class_in, class_out, a_deletable)` is the runtime representation of a single suffix entry. `class Inventory` is a dict-like wrapper exposing `.get(id)`, `.all()`, and `.by_category(cat)`.

The function `load_inventory(path) → Inventory` reads the JSON, validates that every declared rule name exists in the `RULES` registry, that every suffix ID is unique, and that derivational suffixes specify both `class_in` and `class_out`. It raises on any inconsistency rather than silently tolerating bad data.

The `a_deletable` flag (currently set only on NEG) marks suffixes whose stem-final low vowel can be deleted by a following suffix. The parser's retroactive A-deletion alternative only fires when this flag is set, which prevents OPT's `(y)A` template from spuriously matching empty.

There is one quirk worth knowing: rows in `inventory.json` that start with `"_section"` are comments, ignored by the loader. They exist to make the JSON readable when scrolling.

## tr_morphotactics.py — what suffixes follow what

The morphotactic graph encodes a hard constraint of Turkish morphology: not every suffix can follow every other suffix. After `PAST` you can have k-type agreement (`-m`, `-n`, `-k`, `-nız`) but not z-type (`-Hm`, `-sHn`, `-Hz`, `-sHnHz`); after `PROG` it's the opposite. Voice (`CAUS`, `PASS`) precedes negation, which precedes TAM (tense/aspect/mood), which precedes agreement. The OPT mood (-(y)A) has its own dedicated agreement set (OPT_1SG, OPT_1PL).

The graph has states as nodes and suffixes as edge labels. `State(id, accept)` and `Transition(via, from_states, to_state)` are the dataclasses. `MorphoGraph` indexes them for two lookups the parser and generator need: given a state, what transitions leave it, and which states accept (i.e., end the chain validly). The current graph has 14 states (4 nominal, 8 verbal, 1 adjectival, plus a starts-by-class map) and around 60 transitions.

The V→V derivational transitions (CAUS_DERIV, PASS_DERIV) loop from VERB_ROOT back to VERB_ROOT, so a derived stem is structurally identical to a fresh root and all subsequent inflection works on it.

`load_graph(path) → MorphoGraph` reads `morphotactics.json` and validates that every `via` references a real suffix ID, every `from` state and every `to` state is declared, and that every word class has a designated start state. The class also exposes `step(state, suffix_id) → next_state | None` for stepping the graph one transition at a time.

## tr_lexicon.py — root inventory with trie lookup

The root lexicon holds roots (lemmas), each tagged with its word class and metadata that affects how it inflects. The `Root` dataclass carries `form` (the canonical lowercase lemma), `word_class` (`NOUN` / `VERB` / `ADJ`), a `soften` flag (true if the final consonant softens before vowel-initial suffixes), an optional tuple of `variants` (alternate surface forms like `oğl` for `oğul`), a corpus `frequency` for tiebreaking, an `aorist_high` flag (lexically-marked monosyllabic verbs that take -Hr aorist), and `caus_deriv` / `pass_deriv` fields whose VALUE is the template string that licenses a V→V derivational suffix (e.g., `caus_deriv="Ar"` on `çık` lets the parser produce `çık+CAUS_DERIV → çıkar`).

The `root_ctx()` method returns a dict of root-level flags that downstream rules can read. The parser propagates this via `ParseFragment.root_ctx` so context-sensitive rules deep in a suffix chain can still consult the root's lexical properties.

`Lexicon(roots)` builds a character-by-character trie keyed on every surface form of every root, including auto-generated softened-final-consonant variants. So `kitap` with `soften=True` is indexed under both `kitap` and `kitab`, even though only `kitap` appears in the JSON. The `.prefix_match(word)` method walks the trie one character at a time and returns every `(Root, surface_form_used, prefix_length)` tuple where some surface form of some root is a prefix of `word`. The parser uses this to seed its chart with candidate root spans.

`load_lexicon(path) → Lexicon` reads `lexicon.json` (or `lexicon_train.json`) and constructs the index. Both files have the same schema; the train-only one excludes dev/test leakage and is used for honest evaluation.

## tr_generate.py — root + suffix list → surface form

The forward direction. The single public function is `generate(root, suffix_ids, lex, inv, graph) → str`. It walks the morphotactic graph along the requested suffix sequence, applies each suffix's template to the running stem (with archiphoneme resolution from `tr_phonology`), runs any declared forward rules from `tr_rules`, and returns the final surface form. If the suffix chain doesn't form a valid path in the graph, generation raises.

The function `apply_suffix(stem, template)` is `tr_phonology`'s realizer; `generate` is the wrapper that knows about suffix IDs, rules, and morphotactic validation.

## tr_parse.py — surface form → root + suffix list

The inverse direction and the largest module. The class `Parser(lex, inv, graph, config=ParseConfig())` exposes a single method, `parse(word) → List[Analysis]`, returning analyses ranked by score. The top result is the parser's best guess; with `ParseConfig(return_all=True)` you get all candidates produced by the chart.

Internally it runs chart-style dynamic programming. The chart is keyed on (position-in-word, morphotactic-state) and holds `ParseFragment` records that bundle a partial analysis: how far into the surface form it has consumed, what morphotactic state it's in, the morphemes accumulated so far, an OOV flag, a score, and a `root_ctx` tuple of root-level flags (aorist_high, caus_deriv, pass_deriv) that downstream context-sensitive rules need to consult. The chart is seeded by trying every prefix of the surface against the lexicon (in-lexicon roots) and against the OOV root range (any prefix between configured min and max lengths). It then iteratively extends fragments by trying every outgoing transition from each fragment's state, matching the suffix template against the remaining surface via `match_suffix`. Analyses are completed when a fragment reaches the end of the surface in an accepting state.

`match_suffix` is where most of the inversion happens. It calls into `tr_phonology` to resolve archiphonemes given the running stem, then tries the template against the surface. It handles two alternatives beyond plain matching: buffer-consonant realization (the parenthesized buffer may or may not be present, e.g. `(y)A` matches both `ya` and `a`), and retroactive low-vowel deletion (when the template ends in `A` and the suffix is `a_deletable` — currently only NEG). Initial-H drop and pronominal-n insertion are handled by per-suffix rules in `tr_rules.py`, not by special cases in `match_suffix`.

Apostrophes are stripped at the entry point of `parse()` (Turkish writes `Muammer'in` to separate proper nouns from inflection; morphologically the apostrophe is zero-width).

Scoring is the parser's ranker over multiple valid analyses. Each suffix earns `0.5 × len(chunk) − 0.2` per character covered. Empty chunks (like NOM, or H-dropped templates that match zero characters) cost `−0.2`. In-lexicon roots score `log1p(frequency) + len(root) × inlex_char_bonus`; the per-character term lets longer in-lex lemmas compete fairly with shorter root + decomposition chains. A bare in-lexicon root (no suffixes attached) gets an extra `bare_root_bonus`. OOV roots score `oov_penalty + len(root) × oov_char_bonus`. Cross-class derivation steps cost `derivation_penalty`. CAUS or PASS application costs `voice_penalty`. PAST_COP application costs `compound_tense_penalty` (helps biases against speculative compound-tense readings when a simpler analysis exists).

The `Analysis` dataclass holds `root` (the morphological root form), `root_class` (the root's word class — what UD-IMST's UPOS column reflects), `final_class` (the word class AFTER all derivations, used to pick feature defaults), `morphemes` (the chain), `oov` (whether the root was a lexicon hit), and `score`. It exposes four output methods: `split()` returns a hyphen-joined surface chunking (`"kitab-ım-ı"`), `tagged()` returns the same with morpheme tags, `emitted_feats()` returns the morphologically-faithful feature dict from just the morphemes that emit features, and `ud_feats()` returns the UD-Turkish-compliant version that applies defaults (Mood=Ind on finite verbs, Case=Nom on bare nouns and on verbal nouns, etc.), distinguishes participial vs finite for default selection, and drops internal markers like `_derivation` and `Evident=Fh`. Two derived multi-morpheme features are also computed in `ud_feats()`: `Tense=Pqp` when EVID+PAST_COP both fire (the pluperfect), and `Voice=CauPass` when CAUS+PASS both fire.

## tr_evaluate.py — UD-corpus evaluation harness

The eval pipeline. `parse_conllu(path) → List[GoldToken]` reads a CoNLL-U file and yields open-class tokens (NOUN, PROPN, VERB, ADJ), with PROPN normalized to NOUN. Each token carries form, lemma, UPOS, and the gold feature dict.

`evaluate(parser, tokens, show_errors)` runs the parser over every gold token, takes the top-scoring analysis, and compares its root against the gold lemma (root accuracy), its UPOS against the gold UPOS (UPOS accuracy), and its `ud_feats()` against the gold feature dict (feature exact-match and feature P/R/F1 over key=value pairs). It returns overall and per-UPOS `Counters` records.

The CLI takes a path to a `.conllu` file plus `--inventory`, `--morphotactics`, and `--lexicon` overrides, an optional `--limit`, and `--show-errors N` to dump the first N root-error examples for manual inspection.

## extract_lexicon.py — CoNLL-U → lexicon.json

The offline tool that produces `lexicon.json` from UD treebank files. Reads every `.conllu` in the supplied directory, collects (lemma, UPOS) pairs with frequencies, and detects two stem properties automatically: final-consonant softening (by spotting forms where the lemma's `k/p/t/ç` appears as `ğ/b/d/c`) and vowel-zero alternation (by spotting forms where a bisyllabic stem with a high vowel in the second syllable has that vowel deleted, like `oğul → oğl-`).

It then merges in two override files. `irregulars.json` adds per-stem properties the automatic detector can't infer (irregular stems like `de → di`, aorist_high flags for monosyllabic -Hr verbs). `lexicon_overrides.json` does two things: `prune` removes entries (pure passives like `alın` whose gold lemma is the base, and V→V lexicalized derivations like `çıkar`, `geçir`, `bulun`), and `derivations` flags base verbs with `caus_deriv` or `pass_deriv` templates so the parser can decompose surfaces like `çıkardı` into `çık+CAUS_DERIV+PAST`.

The `--split train|all` flag selects whether to read all `.conllu` files in the directory or just the training one. The all-splits lexicon has slightly higher coverage on the dev set, but it leaks dev/test lemmas into "known roots," so honest evaluation uses the train-only lexicon.

## The JSON data files

`inventory.json` holds the suffix catalog (currently around 50 entries). Each entry has an `id`, a template in archiphoneme notation, a `feats` dict (UD-style features, some prefixed `_` for internal use only), a `category` tag for organization, an array of `rules` to apply, and (for derivational suffixes) `class_in`/`class_out` fields. NEG carries an extra `a_deletable: true` flag.

`morphotactics.json` holds the state graph. The `states` array lists state IDs with `accept` flags. The `transitions` array lists allowed (via, from, to) triples. The `start_states` map says which state a parse or generation begins in for each word class.

`irregulars.json` holds hand-curated per-stem overrides for properties the automatic extractor can't infer — irregular stems (`de → di`, `ye → yi`) and aorist_high flags for ~13 monosyllabic verbs that lexically take -Hr (al, gel, bil, bul, vur, etc.).

`lexicon_overrides.json` holds the prune list (31 derived-form entries to remove) and the derivations list (23 base verbs to flag with caus_deriv or pass_deriv templates).

`lexicon.json` (5,471 entries from all UD splits) and `lexicon_train.json` (4,346 entries from training only) are the extracted root lexicons.

## The test files

`test_tr_phonology.py` covers vowel/consonant classification, harmony, alternations, archiphoneme resolution, and the Oflazer template realization.

`test_tr_phase2.py` covers loading and validating the inventory.

`test_tr_phase3.py` covers loading and validating the morphotactic graph, generation roundtrips, and all the suffix-specific allomorphy rules (AOR, NEG-AOR suppletion, OPT, IMP_3SG, PASS, inflectional CAUS, CAUS_DERIV, PASS_DERIV).

`test_tr_phase4.py` covers the parser on known nominal and verbal forms, the bare-root bonus, the V→V derivational decomposition (pruned + deriv-flagged forms), the AOR-allomorph gating, and the UD feature emission (Pqp, CauPass, participial-vs-finite defaults, Vnoun).

`test_tr_phase5.py` is the regression test that locks the eval-floor metrics; it auto-skips if the UD corpus isn't on disk.

Total suite: 190 tests, currently all passing.
