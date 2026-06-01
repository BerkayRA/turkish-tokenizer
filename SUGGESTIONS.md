# Suggested next directions

Things I considered worth flagging but didn't pursue. Ordered roughly by user-visible value vs implementation cost. These are suggestions; the user gets the final call.

## High value, low/medium cost

### 1. Sentence-level pre-tokenizer for attached `mi`

In Turkish text the question particle is *officially* written with a space (`gelecek mi`) but in casual writing very commonly without (`gelecekmi`, `gelecekmisin`). Our parser currently fails on the attached form. UD-Turkish-IMST has both conventions.

Implementation: add a pre-tokenizer pass that scans for word-final `m{i,ı,u,ü}` followed by an optional agreement/copular suffix sequence (`-sin`, `-siniz`, `-yim`, `-dir`, `-ydi`, etc.) and splits it off as a separate token. Probably 30 lines in `tr_api.py` or a new `tr_pretokenize.py`. The split logic needs vowel-harmony awareness — `gelecekmisin` splits as `gelecek` + `misin` (not `gelecekm` + `isin`).

Cost: medium (needs careful regex/scanning logic). Value: high — covers a real and frequent text pattern.

### 2. Proper-noun handling

`Kerem` (proper name) parses as `kere+...` (the common noun "time/occasion"). The eval errors include 5+ instances of this for `Kerem` alone, and many more for other proper nouns. A capitalization-aware bonus would fix this cleanly:

```python
# In tr_parse.py:_fill_chart, when seeding:
if word[0].isupper() and root.form == word.lower():
    # Bonus for "the surface is the proper-noun form of this lemma"
    score += self.cfg.proper_noun_bonus
```

Plus an OOV path that strongly prefers a capitalized-surface match over decomposition.

Cost: low (1-2 hours). Value: high — proper nouns are common and consistently misparse.

### 3. Lemmatization output in the API

Currently `tokenize(word)` returns a `Token` with a morpheme chain. Most downstream tasks want just the lemma string. Adding `token.lemma → str` and `token.surface → str` as direct properties would be trivial and useful.

Cost: minimal (10 minutes). Value: high — makes the API one-line usable for common tasks.

### 4. NEG before NEC, EVID, FUT — verify

I added NEC but didn't write tests for `gelmemeli` ("must not come"), `gelmemiş` ("hasn't come, apparently"), `gelmeyecek` ("won't come"). These probably already work because NEG → VERB_NEG and those tense markers fire from VERB_NEG, but worth verifying with explicit tests. If broken, fixing is just a `from` list extension.

Cost: low (verify; 0-1 hour). Value: medium — basic forms that users will hit.

## Medium value, medium cost

### 5. `-(y)kAn` temporal converb

`gelirken` ("while coming") = `gel+AOR+(y)kAn`. The temporal converb isn't in the inventory. Implementation: add a `CONV_KEN` suffix, transition it from TAM_Z, mark as terminal (no further inflection). Test cases: `gelirken`, `yaparken`, `çalışırken`.

Cost: low (1 hour). Value: medium — common form in narrative Turkish text.

### 6. Better OOV class inference

When the parser falls back to OOV (no in-lex match), it currently picks NOUN by default. Suffix patterns are diagnostic:
- Ends in `-mAk` / `-mEk` → VERB (citation form)
- Ends in `-lAr` after consonant → NOUN with plural
- Ends in `-DH` / `-mHş` / `-(y)AcAk` after consonant → VERB with TAM
- Capitalized → likely proper NOUN

Implementation: in `tr_parse.py:_fill_chart`, when seeding an OOV candidate, look at the trailing chars of the surface to bias the class. The stub `_oov_inference` exists; extend it.

Cost: medium (a few hours; needs careful design to not bias toward false POS for ambiguous suffixes). Value: medium-high — would improve UPOS accuracy for OOV tokens, which is ~5% of the corpus.

### 7. Hand-curated "morphological correctness" eval

Eval against UD penalizes our deliberate decomposition policy. The user has been clear that eval drops are acceptable, but it would be much clearer to have a separate eval that measures what we actually optimize for.

Implementation: build a CSV of ~500-1000 (surface, expected-decomposition) pairs covering the productive derivations, irregular forms, trick words, common ambiguities. Run it as a second eval target. Maintain it as the morphological-truth ground truth.

Cost: medium-high (the hand-curation is the work, not the code). Value: high for the next iteration — gives a metric that doesn't lie about whether changes are improvements.

### 8. Document `_next_must_be` constraint propagation

The mechanism is used in two places (POT suppletion, PROG truncation seeding) but isn't documented in CODE.md or DESIGN.md (I added a section to DESIGN.md but it's brief). A clearer write-up with examples would help future contributors not reinvent the mechanism.

Cost: low (30 min). Value: medium — pays off when the next person adds a third suppletion.

## Lower value, or higher cost

### 9. Confidence scores for parses

The current `score` is comparative (higher = better among alternatives), not interpretable as a probability. A calibrated confidence would be useful for downstream tasks (e.g. "skip tokens with confidence < 0.5"). But calibration requires either Platt scaling against held-out data or a more principled probabilistic model — both real work.

Cost: high. Value: medium — but mostly invisible until downstream tasks need it.

### 10. POSS_2SG vs pronominal-n ambiguity

`evdekinde` parses correctly, but `hayatsızlarınkinde` and similar forms sometimes go through a spurious POSS_2SG reading. A scoring tweak might help, but the underlying ambiguity is real and may require context to resolve.

Cost: medium (scoring rebalance + tests). Value: low (rare in actual text).

### 11. Reduplication

`çok çok`, `koşa koşa`, `yavaş yavaş` (intensification/adverbial). Currently UD treats these as separate tokens so it doesn't break anything; but a real Turkish NLP system might want a special relationship. Out of scope for a tokenizer per se.

Cost: medium (sentence-level pass). Value: low for tokenizer; medium for a downstream NLP system.

### 12. Learned scoring weights

The interaction between `productivity_bonus`, `rare_derivation_extra_penalty`, `voice_penalty`, etc. is delicate. A grid search or learned weights would be more robust. But it requires either a labeled training set (we'd have to make one) or a proxy objective (UD eval, which we don't fully trust).

Cost: high (research-level). Value: medium long-term — would make the parser more robust to inventory changes.

### 13. Generalization to other Turkic languages

The architecture (templates, archiphonemes, rule registry, morphotactic graph, irregulars) is general enough that adapting to Azerbaijani or Uzbek would mostly be data work (different phonology, different graph). Worth flagging for a future user but not actionable as-is.

Cost: high. Value: depends entirely on the user's interest.

## Architecture debts

These don't add user-visible features but are worth fixing for maintainability:

### 14. Public Lexicon API

`test_no_single_letter_roots` reaches into `lex._by_form`. Adding a public `Lexicon.all_forms()` API would be cleaner — single method, returns iterable of `(form, Root)` pairs.

Cost: 10 min. Value: cleanup.

### 15. Scoring config layering

Currently `ParseConfig` mixes phonological/structural penalties (compound_tense_penalty, voice_penalty) with lexical penalties (oov_penalty, derivation_penalty) and per-suffix tunings (productive_derivations, rare_derivations). A two-layer split would clarify what each dial controls.

Cost: medium. Value: cleanup; mostly pays off if you grid-search.

### 16. Documentation refresh

`CODE.md` was written for the original phase 1-5 build and doesn't reflect the productive/rare derivation classification, the constraint-propagation mechanism, the new pronoun infrastructure, the question-particle compound chain handling, etc. Worth a refresh.

Cost: a few hours. Value: cleanup; makes onboarding easier for the NEXT next Claude.

---

## My recommendation if you're picking just one

**Combination of #1 (attached-mi pre-tokenizer) + #2 (proper-noun handling) + #3 (lemma property).** Together they probably take a long afternoon and they address three of the most visible weaknesses in real text: attached question particles, proper-noun misparse, and the awkward API for downstream lemma consumers.

After that, **#7 (morphological-correctness eval)** would put the project on a much more honest footing — the eval would actually align with the design intent, and the next person making changes wouldn't be flying blind on whether they're improving or regressing.

The trick word `muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine` should continue to parse end-to-end. Test it after every parser change.
