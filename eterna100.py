"""
eterna100.py — Eterna100-V2 dataset loader and train/test split.

Contains 20 selected targets from the Eterna100 benchmark,
split into 15 training and 5 test targets. Targets are chosen
to cover a range of difficulties and lengths.
"""

# Each entry: (puzzle_id, name, dot-bracket structure V2)
ETERNA100_SELECTED = [
    # --- TRAIN SET (15 targets) ---
    (1,  "Simple Hairpin",
     "((((((......))))))"),
    (8,  "G-C Placement",
     "((((...))))."),
    (10, "Frog Foot",
     "..........((((....))))((((....))))((((...))))"),
    (13, "square",
     "((((((((((((((((((((((((((...))))))....)))))))....))))))....)))))))"),
    (15, "Small and Easy 6",
     "(((((.....))..((.........)))))" ),
    (23, "Shortie 4",
     "((....)).((....))" ),
    (25, "The Ministry",
     "(((.....(((.....(((.....(((.....(((........))).))).))).))).)))"),
    (26, "stickshift",
     "..((((((((.....)).)))))).." ),
    (30, "Corner bulge training",
     ".(((((((((((...)))))....))))))." ),
    (40, "Tripod5",
     "..((((((((.....))))((((.....)))))))).." ),
    (41, "Shortie 6",
     "((....)).((....)).((....)).((....))" ),
    (45, "[CloudBeta] 5 Adjacent Stack Multi-Branch Loop",
     "(((((((((....))))(((((....)))))(((((....)))))((((....)))))))))." ),
    (47, "Misfolded Aptamer",
     "((((......(((((...))).((....)).........)).....))))" ),
    (54, "7 multiloop",
     "(((((((((....)))))(((((....)))))(((((....)))))(((((....)))))(((((....)))))(((((....)))))))))" ),
    (65, "Branching Loop",
     ".(((((........)((((....))))..))))......." ),

    # --- TEST SET (5 targets) ---
    (3,  "Prion Pseudoknot",
     "((((((.((((....))))))).))).........." ),
    (11, "InfoRNA test 16",
     "((((((.((((((((....))))).)).).))))))" ),
    (20, "InfoRNA bulge test 9",
     "(((((((.(.(.(.(((((((....)))))))))))))))))" ),
    (33, "Worm 1",
     ".......(.(.(.(.(.((.((.(....).)).)).).).).).)" ),
    (59, "hard Y",
     ".....((((((.((((((((....))))))).)))((((((((((....))))))))))))))....." ),
]

TRAIN_TARGETS = ETERNA100_SELECTED[:15]
TEST_TARGETS  = ETERNA100_SELECTED[15:]

def get_train_structures():
    """Return list of (id, name, structure) for training."""
    return [(pid, name, struct) for pid, name, struct in TRAIN_TARGETS]

def get_test_structures():
    """Return list of (id, name, structure) for testing."""
    return [(pid, name, struct) for pid, name, struct in TEST_TARGETS]

if __name__ == "__main__":
    print(f"Train targets: {len(TRAIN_TARGETS)}")
    for pid, name, struct in TRAIN_TARGETS:
        print(f"  [{pid:>3}] {name:<50s} len={len(struct)}")
    print(f"\nTest targets: {len(TEST_TARGETS)}")
    for pid, name, struct in TEST_TARGETS:
        print(f"  [{pid:>3}] {name:<50s} len={len(struct)}")
