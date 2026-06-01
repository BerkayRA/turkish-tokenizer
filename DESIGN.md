# Design and implementation decisions

This document records what we built, why we built it that way, and the trade-offs that shaped each decision. It's organized by the five phases the project went through, but the threads in any given phase often refer back to earlier phases — so it reads as much as a history as a reference.

## What the project is and isn't

The goal was a Turkish morphological tokenizer that splits inflected words into a root plus a chain of morpheme tags. Concretely, given `kitabımı` we want to recover `kitap` + `POSS_1SG` + `ACC` (with associated features), and given `geliyordu` we want `gel` + `PROG` + `PAST_COP` (with associated features). The output format is configurable: a `.split()` view that shows surface chunking (`"kitab-ım-ı"`) and a `.tagged()` view that shows morpheme labels and features.

The project covers full open-class morphology — nouns, verbs, adjectives, and the major derivational suffixes between them — but explicitly does not handle closed-class words (pronouns, postpositions, particles, conjunctions). Closed-class words in Turkish are mostly invariant and a stoplist suffices; investing in morphology there would have added complexity for little benefit on the eval metrics that matter.

It's a single-process Python library with no external dependencies. The decision to constrain ourselves to stdlib was deliberate: Turkish morphology is a closed system that should be tractable without ML, the dependency-free constraint forces a clean architecture, and the resulting code is easy to inspect, modify, and embed. If we needed neural disambiguation we'd add it as an optional layer.

## Phase 1: phonology

The first decision was where to draw the line between phonology and morphology. Turkish has rich surface alternations — vowel harmony, consonant softening, voicing assimilation, buffer-consonant insertion — and these can either live in the suffix data (every suffix specifies its full surface form in every context) or in a phonology layer (every suffix has an abstract template that the layer realizes).

We chose the second approach with archiphoneme templates, following the Oflazer convention familiar from earlier Turkish NLP tools. A template like `Hm` for the 1sg possessive encodes "high vowel matching the stem's harmony, then m," and the phonology layer resolves the `H` to `ı`/`i`/`u`/`ü` based on the stem's last vowel. The alternative — listing four surface variants per suffix — would have made the inventory four times larger, harder to maintain, and impossible to validate (no way to check that all variants of one suffix are mutually consistent). The archiphoneme approach also makes phonological inversion (during parsing) feasible: instead of matching against fixed strings, we can ask "could this surface chunk be a realization of this template under harmony?"

The phonology module deliberately knows nothing about morphology. It exposes primitives — `apply_suffix(stem, template)`, vowel classification, voicing predicates — and is exhaustively tested independently of the rest of the system. This kept the inevitable bugs in early phases isolated; if a vowel-harmony test was failing, the issue was always inside `tr_phonology.py`.

Two surprises in this phase shaped later decisions. First, Python's `str.lower()` mishandles Turkish capitals — `İ` lowercases to `i̇` with a combining dot, and `I` lowercases to `i` instead of `ı`. We wrote `tr_lower` and `tr_upper` to handle this and routed everything through them. Second, Turkish has irregular roots where the citation form alternates with a different stem before vowel-initial suffixes (`de → di`, `ye → yi`); these can't be derived by rule and need a manual override mechanism, which became `irregulars.json`.

## Phase 2: suffix inventory

The inventory grew incrementally with each new test case, but its overall shape was settled early. Each suffix entry has an ID (used everywhere else as the canonical reference), a template, a feature dict, a category tag, and a list of rule names. Derivational suffixes also have `class_in` and `class_out` to denote class transitions.

We standardized on Universal Dependencies feature names because UD is the lingua franca of dependency parsing and it gave us a corpus to evaluate against (UD_Turkish-IMST). Some morphological distinctions — like our `Evident=Fh` for the witnessed past — aren't standard UD and we drop them in the UD-compliant output view; others — like UD's `Polite=Form` on -sInIz forms — we don't currently emit and just accept as a feat-level miss in evaluation.

The k/z agreement split is worth flagging because it generates eight suffix IDs from four conceptual slots. Turkish has two surface forms for each non-3sg/3pl person agreement, conditioned on which TAM marker precedes: the "z-type" set (`-Hm`, `-sHn`, `-Hz`, `-sHnHz`) follows PROG/FUT/AOR/EVID, and the "k-type" set (`-m`, `-n`, `-k`, `-nHz`) follows PAST/COND. We chose to model these as eight separate suffix IDs (`1SG_Z`, `1SG_K`, etc.) rather than one unified ID with conditional realization. The eight-ID approach pushes the constraint into the morphotactic graph instead of the phonology, which is cleaner: the graph already enforces that k-type can only follow PAST/COND. The shared 3PL suffix (`-lAr`, identical in both contexts) needed only one ID, and the graph routes it from both TAM_K and TAM_Z to the agreement state.

A late discovery was that not every suffix-rule pairing is universal. Some rules like the H-drop in `-Hyor` (deleting a preceding low vowel) only fire with specific TAM markers. We made rules opt-in per suffix via the `rules` field, which kept the phonology layer free of suffix-specific exceptions. The validation in `load_inventory` checks that every named rule exists in the registry, which has caught typos several times.

## Phase 3: the morphotactic graph

The morphotactic graph encodes the constraint that not every suffix order is valid. We considered three representations: a regular expression per word class (too rigid for the cross-cutting transitions Turkish allows), a finite-state transducer (the right tool but a deep dependency to reimplement, and we'd be using maybe 10% of FST machinery), or a hand-rolled directed graph with states-as-positions. We went with the graph, modeled as a JSON file because the data is small (~50 transitions) and a JSON edit is the most direct way to extend it.

The states correspond to chain positions, not suffix categories. So `VERB_TAM_K` doesn't mean "after a TAM marker" generically — it specifically means "after a TAM marker that takes k-type agreement," and the only transitions out of it are the k-type agreement suffixes and 3PL (shared). This made the constraint that PAST can't be followed by z-type agreement automatic: PAST lands in TAM_K, and z-type transitions don't leave TAM_K.

Phase 5 added a few states that didn't exist originally — `VERB_POT` for the ability/potential -Abil suffix — and reshuffled some `from` lists to allow new compound chains (AOR followed by COND, PROG followed by PAST_COP). Each of these landed as a JSON edit plus a corresponding eval improvement; the graph absorbs the structural complexity that would otherwise live in special-case code.

The graph is also used in the generator (validates a requested chain before realizing it) and in the parser (constrains chart expansion), which gave us one source of truth and let bugs surface as failures in either direction.

## Phase 4: the parser

This was the largest phase. The core problem in parsing Turkish is that the surface form has lost information: vowels have harmonized, consonants have softened or assimilated, buffers have been inserted or suppressed. To recover the underlying morphemes, the parser has to consider multiple inversions at every position and search over the resulting space efficiently.

We chose chart-style dynamic programming bounded by lexicon prefix matching. The chart is keyed on (position in surface, morphotactic state), so partial parses that reach the same point through different paths can share downstream work. The lexicon trie is the search bound: only positions where some root prefix ends can be chart seeds, so we never speculate that, say, `geliy` is a root. This keeps the search small in practice — most words have only a handful of viable prefixes.

For unknown words, we fall back to OOV-root candidates: any prefix between configured min and max lengths becomes a candidate root with an `oov_penalty` score. The word class is guessed from the suffix pattern (decision recorded earlier: "OOV: Guess word class from suffix pattern"). In practice the OOV path produces useful analyses for proper nouns and novel verbs as long as the inflection is regular.

Several specific inversions live inside `match_suffix`. Buffer consonants are matched both ways (the buffer may or may not be present in the surface, depending on stem-final consonant or vowel). Retroactive low-vowel deletion is handled by trying the template with its final `A` removed when the next suffix is one that triggers deletion in generation. Initial-H drop after vowel stems (added in Phase 5) is handled similarly. These three alternatives multiply the search by a small factor but keep the parser correct.

Pronominal-n — the buffer that surfaces between 3rd-person possessives and case markers (`araba-sı-n-da`) — is hardcoded into the chart fill: when a CASE suffix transitions from POSS_3SG or POSS_3PL, the parser prepends a literal `n` to the case template before matching. This is architecturally inconsistent with the rule registry (which would be the right home for it), but the cost of routing it through the registry would have been a refactor disproportionate to the benefit. The same applies to apostrophe stripping (Phase 5), which is a string normalization at the entry point of `parse()`.

Scoring went through three iterations. Originally we awarded a flat `+1.0` per matched suffix, which biased the parser toward longer chains and broke when H-drop made it possible to peel off a phantom suffix from an OOV root (`Osman → osma + POSS_2SG`). We switched to per-character coverage scoring (`0.5 × len(chunk) − 0.2`) so the total reward for covering a span is roughly equal regardless of how many morphemes split it. The OOV root score was also given a per-character bonus (`oov_char_bonus = 0.35`) so that longer OOV roots beat short OOV roots padded with phantom suffixes. The original `suffix_bonus` config field is retained for backward compatibility but no longer used by the live code.

The Analysis class deliberately exposes both `emitted_feats()` (morphologically faithful — features come only from suffixes that emit them) and `ud_feats()` (UD-compliant — defaults filled in). This was a Phase 5 decision: the morpheme-level view is what the parser actually produces, but UD's annotation convention is that features are present-with-default rather than absent-when-unspecified. Rather than baking defaults into the inventory (which would have created spurious features on every analysis), we layered the UD view on top as a post-processor.

## Phase 5: evaluation and improvement

The final phase was about measuring what we had and closing the most impactful gaps. The eval harness runs the parser over UD_Turkish-IMST and reports root accuracy, UPOS accuracy, feature exact-match, and feature P/R/F1. We split the lexicon into a train-only version and an all-splits version: the all-splits one has slightly better coverage but contains lemmas from dev and test, so honest evaluation uses the train-only one and treats the lift from the all-splits version as a measure of OOV impact.

The baseline numbers (root 70.7%, feature F1 0.35) revealed that most "feature errors" were systematic: UD-Turkish marks every verb with Mood/Aspect/Tense/Polarity/Person/Number, including default values that our null morphemes don't emit. The cheapest fix was the `ud_feats()` post-processor, which lifted feature F1 from 0.35 to 0.76 without touching the parser at all. We did this first because it cleared the noise and let real errors become visible.

The verb root accuracy (48.1% baseline) revealed structural gaps: POT (-Abil), PAST_COP (-DH copula), IMP_2PL, converbs (-ArAk, -Hp), participles (-An, -DHk), and the future participle (-AcAk used participially) were all missing or unreachable in the graph. Adding them in Phase 5 lifted verb root accuracy to 77.6%. The future participle case is worth flagging because `-AcAk` is genuinely ambiguous between a finite future tense and a participle that takes possessive/case marking. We resolved this by adding a separate `FUT_PART` suffix with `class_out=NOUN`, so the derivation penalty (-2.0) naturally biases toward the finite reading when both are structurally available, and the participial reading wins when the finite is structurally impossible. This kept the disambiguation in the scoring system rather than in a special case.

The long-tail fixes (vowel-zero stems and apostrophes) followed the same principle of pushing complexity into the right layer. Vowel-zero detection lives in `extract_lexicon.py` because it's a per-stem property that can be inferred from training data; the parser just sees the variants and matches them through the existing trie. Apostrophe stripping lives in `parse()` because apostrophes are a string-level convention, not a morphological one.

The regression test (`test_tr_phase5.py`) locks in the Phase 5 metrics as floors using `assertGreaterEqual`. The semantics are deliberate: legitimate improvements should rewrite the floors in the same commit as the parser change, so the regression history is a record of intentional steps forward. The test auto-skips if the UD corpus isn't on disk, so it's safe to run in environments without the treebank.

## Phase 6: post-Phase-5 refactor

Phase 5 closed with the parser at 70.7% root accuracy (overall) and 0.354 feat F1. The work afterward — substantial enough that it's effectively a separate phase — was structured around closing concrete eval gaps and cleaning up architectural shortcuts that had accumulated. It was less about adding morphology and more about getting what we already had to work correctly.

The two largest interventions were the allomorphy refactor and the V→V derivational architecture. The allomorphy refactor moved AOR/PASS/CAUS surface variation out of hardcoded special cases and into a single pattern (forward picks one outcome; expand returns the phonologically plausible alternatives) parametrized by lex flags. The NEG-AOR suppletion (`gelmem` vs `gelmeyiz` vs `gelmez`) fell into the same framework: the rule reads `prev_morph_id` and `next_morph_id` to choose between -z, -∅, and the other variants. A latent bug in 1PL_Z's template (`Hz` with H-drop, instead of `(y)Hz` with buffer insertion) surfaced during this work and was fixed.

The V→V derivational architecture was the biggest design change. Forms like `çıkar` (from `çık`), `geçir` (from `geç`), `bulun` (from `bul`) are lexicalized derivations — they have their own meanings as standalone verbs. The treebank annotates them as their own lemmas. But the tokenizer's job is morphological decomposition: we want `çıkardı → çık+CAUS_DERIV+PAST`, not `çıkar+PAST`. The architecture: a separate suffix per derivational class (`CAUS_DERIV`, `PASS_DERIV`) gated by per-root lex flags whose value IS the template (e.g., `çık` carries `caus_deriv="Ar"`). The `expand` view returns the templated alternative if the flag is set, else returns empty list — blocking the suffix for verbs that aren't lexicalized this way. The morphemes emit no Voice features (the gold lemmas have none either). This deliberately costs ~3.6pp of VERB root accuracy in eval (gold lemma `çıkar` vs our root `çık`), accepted as the cost of linguistic correctness. Lexicalized derivations are pruned from the lexicon to avoid scoring conflicts.

OPT mood (-(y)A) and IMP_3SG (-sIn) were added as two separate paradigms after treebank inspection showed they're tagged differently — OPT 3sg is bare `-A` (`ola`), IMP 3sg is `-sIn` (`gelsin`). OPT carries its own agreement set (`OPT_1SG = -(y)Hm`, `OPT_1PL = -lHm`) distinct from the indicative-track AGR because OPT's 1pl is `-lHm`, not `-(y)Hz`, and the agreement attaches with buffer-y after the vowel-final OPT marker (`geleyim`, not `gelem`).

The feature-emission revamp lifted overall feat_exact from 8% to 67%. The key insight was the distinction between `root_class` (the root's word class, what UD's UPOS column reflects) and `final_class` (the class AFTER derivation, what determines feature defaults). A verbal noun like `yazmak` has `root_class=VERB` and `final_class=NOUN`, so UPOS comparison uses VERB but feature defaults use NOUN paradigm (Case=Nom). Further refinement distinguished participial forms (VerbForm=Part, Conv) from finite verbs and from verbal nouns (Vnoun): participials get verbal Mood/Tense but not nominal Case/Person/Number defaults. Two compound-feature derivations were added: Tense=Pqp from EVID+PAST_COP, Voice=CauPass from CAUS+PASS.

The bare-root bonus solved a class of bugs where a more-frequent verb root would steal the parse of a noun lemma by attaching speculative suffixes (`süre` being read as `sür+OPT` because sür is more frequent). A small bonus (+0.5) applied only when the entire surface IS an in-lexicon root (no suffixes attached) captures the intuition that the unmarked reading of a known lemma is the lemma itself. The companion fix was gating retroactive A-deletion on a new `a_deletable` flag (set only on NEG, where the alternative is real); without it, every A-final template — most notably OPT's `(y)A` — was spuriously matching the empty string.

The pronominal-n routing — moving the `n` between POSS_3SG/POSS_3PL and case markers out of a hardcoded chart-fill step and into the rule registry — closed one of the two architectural shortcuts noted at the end of Phase 5. The other (apostrophe stripping) is still at the entry point of `parse()`, since it's a string-level normalization and not really a morphological operation.

Lexicon pruning was a hand-curated cleanup of derived-form lemmas that the corpus extractor had pulled out as if they were base verbs. Two kinds: pure inflectional Pass/Cau forms whose gold lemma is the base (`alın`, `hazırlan`, `vurul`), which decompose via the new PASS/CAUS allomorphy rules and emit Voice features; and V→V lexicalized derivations (`çıkar`, `geçir`, `bulun`), which the parser decomposes via CAUS_DERIV/PASS_DERIV without emitting Voice. The user annotated the candidate list manually; the file went into `lexicon_overrides.json` with prune entries and derivation flags.

A subtle bug surfaced late: AOR's `expand` view was returning BOTH `-Ar` and `-Hr` alternatives for every consonant-final stem, regardless of the `root_aorist_high` flag, allowing phantom parses like `artırdı` → `art+AOR(-Hr)+PAST_COP` (which would surface as `artırdı` via vowel harmony but isn't the right AOR for `art`, whose AOR is `-Ar` giving `artardı`). The fix mirrors the forward rule: for monosyllabic consonant-final stems, return `-Hr` only if aorist_high is set, else `-Ar` only.

## Final numbers

After Phase 6, on the dev set with the train-only lexicon:

```
OVERALL   n=6363   cover=99.8%   root=86.0%   upos=88.5%   feat_exact=66.9%   feat_F1=0.863
ADJ       n=1059   cover=99.9%   root=84.8%   upos=53.0%   feat_exact=63.0%   feat_F1=0.648
NOUN      n=3255   cover=99.7%   root=88.5%   upos=96.7%   feat_exact=76.5%   feat_F1=0.885
VERB      n=2049   cover=99.9%   root=82.7%   upos=93.7%   feat_exact=53.7%   feat_F1=0.874
```

On the held-out test set with the same train-only lexicon (this is the honest benchmark number, run once at the end):

```
OVERALL   n=5690   cover=99.9%   root=86.7%   upos=88.7%   feat_exact=67.2%   feat_F1=0.863
ADJ       n=958    cover=100%    root=84.2%   upos=55.4%   feat_exact=64.3%   feat_F1=0.644
NOUN      n=2804   cover=99.9%   root=89.9%   upos=97.2%   feat_exact=77.7%   feat_F1=0.886
VERB      n=1928   cover=99.9%   root=83.5%   upos=92.8%   feat_exact=53.3%   feat_F1=0.874
```

Test-set numbers track dev closely, confirming the model isn't dev-tuned in any pathological way. The all-splits-lexicon variant (which includes test lemmas, so it's an upper bound rather than an honest number) reaches 91.0% root and 91.7% UPOS — the ~4pp gap shows that lexicon coverage is the dominant remaining bottleneck for in-vocabulary cases.

Cumulative deltas since the Phase 5 starting point (root 70.7%, upos 78.7%, feat_F1 0.354):

```
Overall root        70.7 → 86.0    +15.3
Overall upos        78.7 → 88.5    +9.8
Overall feat_F1     0.354 → 0.863  +0.51
Overall feat_exact  8%   → 67%     +59pp
VERB root           48.1 → 82.7    +34.6
VERB upos           63.1 → 93.7    +30.6
VERB feat_exact     0%   → 54%     +54pp
```

## Known limitations

A handful of issues are on the floor that we chose not to chase. ADJ UPOS accuracy stays around 53% because UD-Turkish tags many forms as ADJ that morphologically behave like nouns; resolving this needs either a lookup table or a feature-based UPOS classifier, neither of which is purely morphological. VERB root is held back by ~3.6pp through deliberate design — V→V derivations like `çıkardı` decompose to base+CAUS_DERIV rather than staying as the gold lemma `çıkar`, because morphological correctness was prioritized over the eval metric. The `Polite` feature (~250 dev tokens want `Polite=Infm`/`Polite=Form`) is unmodeled; adding it would help recall but cost precision. A small number of compound chains involving 3PL+PAST_COP still parse imperfectly. None of these are blockers; each is a discrete addition if the eval needs to move further.

## What we would change

The biggest structural cleanup, if we were starting over, would be to design the rule system around per-suffix context dicts from the start — both `prev_morph_id`/`next_morph_id` and root-level flags — instead of retrofitting them as suffix-by-suffix needs arose. The current setup works but it took several phases to settle on. We'd also separate scoring knobs into two layers: phonological/structural penalties (compound_tense_penalty, voice_penalty) that encode "this analysis is less likely a priori," and lexical penalties (oov_penalty, derivation_penalty) that encode "this analysis costs information." Today these are jumbled together in ParseConfig.

The lexicon-overrides file (prune + derivations) ended up being a fairly clean place for hand-curated linguistic facts the corpus extractor can't recover. If we were generalizing to other Turkic languages, this would be the right interface to grow.
