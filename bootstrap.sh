#!/usr/bin/env bash
# bootstrap.sh — restore project state from /mnt/user-data/outputs/ and
# verify the parser is healthy. Run this first thing in a new session.
#
# Usage:  bash bootstrap.sh

set -e

echo "=== Step 1: Restore project files ==="
cp /mnt/user-data/outputs/*.py /mnt/user-data/outputs/*.json /mnt/user-data/outputs/*.md /mnt/user-data/outputs/*.html /home/claude/ 2>/dev/null
ls /home/claude/*.py | wc -l | xargs -I {} echo "  restored {} Python files"

echo ""
echo "=== Step 2: External data ==="
cd /home/claude
if [ ! -d UD_Turkish-IMST ]; then
    echo "  cloning UD_Turkish-IMST..."
    git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-IMST.git 2>&1 | tail -1
else
    echo "  UD_Turkish-IMST present"
fi

if [ ! -f tdk_words.json ]; then
    echo "  fetching tdk_words.json..."
    cd /tmp && [ ! -d tr-word-list ] && git clone --depth 1 https://github.com/bilalozdemir/tr-word-list.git 2>&1 | tail -1
    cp tr-word-list/files/words.json /home/claude/tdk_words.json
    cd /home/claude
else
    echo "  tdk_words.json present"
fi

echo ""
echo "=== Step 3: Sanity-check parser ==="
cd /home/claude && python <<'EOF'
from tr_inventory import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon import load_lexicon
from tr_parse import Parser

p = Parser(load_lexicon('lexicon_full.json'),
           load_inventory('inventory.json'),
           load_graph('morphotactics.json'))

cases = [
    ('heyecanlı', 'heyecan', ['ADJZ_LH']),
    ('gazeteci',  'gazete',  ['NDER_CH']),
    ('mi',        'mi',      []),
    ('musunuzdur','mu',      ['2PL_Z', 'COP_DHR']),
    ('gelmem',    'gel',     ['NEG', 'AOR', '1SG_Z']),
    ('elmalar',   'elma',    ['PLUR']),
    ('alındı',    'al',      ['PASS', 'PAST']),
    ('Bunu',      'bu',      ['ACC']),
]

ok = 0
for surface, want_root, want_suffs in cases:
    a = (p.parse(surface) or [None])[0]
    if a is None:
        print(f"  ✗ {surface}: no parse")
        continue
    got_suffs = [m.suffix_id for m in a.morphemes if m.suffix_id]
    if a.root == want_root and got_suffs == want_suffs:
        print(f"  ✓ {surface} → {a.root}+{'+'.join(got_suffs) if got_suffs else 'BARE'}")
        ok += 1
    else:
        print(f"  ✗ {surface}: got {a.root}+{'+'.join(got_suffs)}; want {want_root}+{'+'.join(want_suffs)}")

# The big test
w = 'muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine'
a = (p.parse(w) or [None])[0]
if a and not a.oov:
    print(f"  ✓ trick word: {len(a.morphemes)} morphemes")
    ok += 1
else:
    print(f"  ✗ trick word: failed")

print(f"")
print(f"  {ok}/{len(cases)+1} sanity checks passed")
EOF

echo ""
echo "=== Step 4: Run test suite ==="
cd /home/claude && python -m unittest test_tr_phonology test_tr_phase2 test_tr_phase3 test_tr_phase4 test_tr_phase5 test_tr_api 2>&1 | tail -3

echo ""
echo "Bootstrap complete. Read HANDOFF.md for context."
