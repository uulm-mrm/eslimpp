"""
Script for generating results from Table (2).
Run script to verify results reported in publication.
Fusion type can be changed manually by uncommenting desired type.

Note: For high noise, some runs do not result in a reset.
"""

import numpy as np
import subjective_logic as sl
import tqdm

from util import generate_scenario


def run_simulation(ops: list[sl.Opinion]) -> None:

    reset_occured = {k: False for k in results.keys()}

    for idx, op in enumerate(ops):
        # Jump will happen at IDX = 25 (24 still old process, 25 new process)
        # We know detection of jump, e.g. idx = 29 -> ST = (25, 26, 27, 28, 29)
        # Thus, we expect the ST size to be idx - 25 + 1 (jump location) = 5
        # IMPORTANT: the expected_size changed between review and final submission
        # the reported percentages of early / late / correct are not affected because
        # the change was applied to both sides of the evaluation.
        # However, due to the longer ST window, the number of additional resets increased
        # but the previous trends still holds.
        expected_ST_size = idx - JUMP_LOCATION + 1

        for name, ltst in ltsts.items():
            ltst.add(op)

            if not ltst.is_last_conflicted():
                continue

            # ignore resets caused by noise before or after gt jump
            if idx < JUMP_LOCATION or idx > JUMP_LOCATION + SHORT_WINDOW_SIZE:
                continue
            if reset_occured[name]:
                results[name][3] += 1
                continue
            reset_occured[name] = True

            # increment outcome (0 = correct), (1 = early), (2 = late)
            short_size = ltst.get_short_size()
            col = 0 if expected_ST_size == short_size else (1 if expected_ST_size > short_size else 2)
            results[name][col] += 1


SEED = 42

GOOD_OP = sl.Opinion(0.8, 0.0)
BAD_OP = sl.Opinion(0.0, 0.8)
NOISE_LEVELS = {"low": 1.0, "med": 5.0, "high": 10.0}

SHORT_WINDOW_SIZE = 10
THRESHOLD = 0.2
DISCOUNT = 0.9
HANDLE_ST_CONFLICT = True

# FUSION_TYPE = sl.FusionType.AVERAGE
# FUSION_TYPE = sl.FusionType.CUMULATIVE
FUSION_TYPE = sl.FusionType.WEIGHTED

NUM_RUNS = 10000
JUMP_LOCATION = 25

ltsts = {
    "avg_dc": sl.LongShortTermMemory2d(SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, True),
    "fusion": sl.LongShortTermMemory2d(SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, HANDLE_ST_CONFLICT, False),
    "baseline": sl.LongShortTermMemory2d(SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT, FUSION_TYPE, False, False),
}

for level, variance in NOISE_LEVELS.items():
    print(f"Results for {FUSION_TYPE} and {level} noise:")
    rng = np.random.default_rng(seed=SEED)

    # (exact, early, late, additional reset)
    results = {"avg_dc": np.zeros(4, dtype=int), "fusion": np.zeros(4, dtype=int), "baseline": np.zeros(4, dtype=int)}
    scenario = [("bad", JUMP_LOCATION, variance), ("good", JUMP_LOCATION, variance)]

    for _ in tqdm.tqdm(range(NUM_RUNS)):
        for ltst in ltsts.values():
            ltst.reset()
        ops, segments = generate_scenario(scenario, GOOD_OP, BAD_OP, rng)
        run_simulation(ops)

    for name, res in results.items():
        if sum(res[:3]) != NUM_RUNS:
            print(
                f"\tSome runs did not result in a reset - can happen for high noise! Found {sum(res[:3])} valid resets."
            )
        print(
            f"\t{name}: correct: {res[0]} ({res[0] / NUM_RUNS * 100:.2f}%) "
            f"early: {res[1]} ({res[1] / NUM_RUNS * 100:.2f}%) "
            f"late: {res[2]} ({res[2] / NUM_RUNS * 100:.2f}%) "
            f"({res[3]} more resets than expected)"
        )
