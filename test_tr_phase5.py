"""
test_tr_phase5.py — Regression test that locks in evaluation metrics.

Runs the parser over UD_Turkish-IMST dev set with the train-only lexicon
(no leakage) and asserts that the headline metrics are at least their
current values. The thresholds are "floors": intentional improvements
should rewrite them upward; an unintended regression will fail the test.

The test is skipped if UD_Turkish-IMST/tr_imst-ud-dev.conllu is not on disk
(e.g. in CI environments without the corpus).

To update the floors after a real improvement:
    1. Run `python tr_evaluate.py UD_Turkish-IMST/tr_imst-ud-dev.conllu \\
          --lexicon lexicon_train.json --show-errors 0` and note the new
       numbers.
    2. Replace the FLOORS dict below.
    3. Commit alongside the parser change so the regression history is
       legible.
"""

import os
import unittest
from pathlib import Path

from tr_inventory     import load_inventory
from tr_lexicon       import load_lexicon
from tr_morphotactics import load_graph
from tr_parse         import Parser
from tr_evaluate      import parse_conllu, evaluate


HERE         = Path(__file__).parent
UD_DEV_PATH  = HERE / "UD_Turkish-IMST" / "tr_imst-ud-dev.conllu"
LEXICON_PATH = HERE / "lexicon_train.json"


# Floors recorded after Phase 5 + the post-Phase-5 refactor steps
# (pronominal-n rule, H-drop rule consistency, AOR allomorphy, NEG-AOR
# suppletion, OPT + IMP_3SG, a_deletable gating, bare-root bonus,
# feature-emission improvements, PASS/CAUS allomorphy, lexicon pruning,
# V→V derivational architecture, plus the trick-word infrastructure
# (VBZ_LAS, AGENT, VER, COP_EVID, CASINA, POT+NEG suppletion, double-POT
# transition, parser constraint propagation). Net of the trick-word
# additions on dev: overall root +0.6pp, VERB root +2.2pp, VERB upos
# -0.4pp. Update intentionally (and in the same commit as the parser
# change) when the parser improves.
# Floors recorded after this iteration's derivational expansion:
# - 11 new derivational suffixes added (VBZ_DA, ADJZ_MTRK/CHL, ADJ_GAN/MAZ,
#   NDER_CAGIZ/GHL, NMZ_HNTI/MACA/DHKCE, VMOD_GEL/DUR/YAZ)
# - Question particles mi/mı/mu/mü, değil, var, yok added as lexicon entries
# - Single-letter spurious noun roots filtered out at extraction
# - Productivity-bonus scoring: ADJZ_LH, ADJZ_SHZ, NDER_CH, NDER_LHK etc. now
#   win as decompositions over lexicalized headwords (heyecanlı, gazeteci...)
# - Rare-derivation penalty: NMZ_HNTI etc. don't fire speculatively
# Net effect on dev: ROOT accuracy DOWN modestly (more aggressive decomposition
# means more divergence from UD's lexicalized lemma choices), feat_F1 stable.
# Per user design intent, root accuracy is secondary to morphological correctness.
FLOORS = {
    "OVERALL": {
        "cover":      0.997,
        "root":       0.830,
        "upos":       0.890,
        "feat_exact": 0.670,
        "feat_F1":    0.870,
    },
    "ADJ": {
        "cover":      0.997,
        "root":       0.830,
        "feat_F1":    0.700,
    },
    "NOUN": {
        "cover":      0.995,
        "root":       0.810,
        "upos":       0.925,
        "feat_exact": 0.750,
        "feat_F1":    0.875,
    },
    "VERB": {
        "cover":      0.995,
        "root":       0.860,
        "upos":       0.920,
        "feat_exact": 0.530,
        "feat_F1":    0.880,
    },
}


class TestPhase5EvalFloors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not UD_DEV_PATH.exists():
            raise unittest.SkipTest(f"{UD_DEV_PATH} not present; skipping eval regression test")
        if not LEXICON_PATH.exists():
            raise unittest.SkipTest(f"{LEXICON_PATH} not present; run extract_lexicon.py first")

        inv   = load_inventory(str(HERE / "inventory.json"))
        graph = load_graph(str(HERE / "morphotactics.json"))
        lex   = load_lexicon(str(LEXICON_PATH))
        parser = Parser(lex, inv, graph)

        tokens = parse_conllu(UD_DEV_PATH)
        overall, by_upos = evaluate(parser, tokens, show_errors=0)
        # Stash a {label: Counters} dict for the test methods.
        cls.results = {"OVERALL": overall, **by_upos}

    def _assert_floor(self, label: str, metric: str, value: float, floor: float):
        self.assertGreaterEqual(
            value, floor,
            f"{label} {metric} regressed: got {value:.4f}, floor is {floor:.4f}. "
            f"If this is an intended improvement, update FLOORS in test_tr_phase5.py."
        )

    def test_floors(self):
        for label, expected in FLOORS.items():
            with self.subTest(label=label):
                self.assertIn(label, self.results, f"{label} missing from eval results")
                c = self.results[label]
                derived = {
                    "cover":      c.n_parsed       / max(c.n_tokens, 1),
                    "root":       c.n_root_correct / max(c.n_tokens, 1),
                    "upos":       c.n_upos_correct / max(c.n_tokens, 1),
                    "feat_exact": c.n_feat_exact   / max(c.n_tokens, 1),
                    "feat_F1":    c.f1()[2],
                }
                for metric, floor in expected.items():
                    self._assert_floor(label, metric, derived[metric], floor)

    def test_token_count_stable(self):
        """Catches accidental changes to the dev set or its filter logic."""
        n_overall = self.results["OVERALL"].n_tokens
        self.assertEqual(n_overall, 6363,
                         "dev-set token count changed; investigate before updating")


if __name__ == "__main__":
    unittest.main()
