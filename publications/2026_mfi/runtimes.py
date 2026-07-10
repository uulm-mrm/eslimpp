import gc
import numpy as np
import subjective_logic as sl
import time

SHORT_WINDOW_SIZE = 10
THRESHOLD = 0.2
DISCOUNT = 0.9
FUSION_TYPE = sl.FusionType.WEIGHTED

OP_GOOD = sl.Opinion(0.9, 0.0)
OP_RESET = sl.Opinion(0.0, 0.9)

NUM_WARMUP = 50
NUM_RUNS = 5000  


def report_timing(rts_us: np.ndarray) -> None:
    print(f"\tn={len(rts_us)}, "
          f"median={np.median(rts_us):.2f}us, "
          f"mean={np.mean(rts_us):.2f}us, "
          f"std={np.std(rts_us):.2f}us")
    print(f"\tp5={np.percentile(rts_us, 5):.2f}us, "
          f"p95={np.percentile(rts_us, 95):.2f}us, "
          f"p99={np.percentile(rts_us, 99):.2f}us")


def bench_reset(handle_st_conflict: bool, avg_dc: bool,
                num_warmup: int = NUM_WARMUP,
                num_runs: int = NUM_RUNS) -> np.ndarray:
    ltst = sl.LongShortTermMemory2d(
        SHORT_WINDOW_SIZE, THRESHOLD, DISCOUNT,
        FUSION_TYPE, handle_st_conflict, avg_dc
    )

    def single_trial() -> float:
        ltst.reset()
        # Fill short-term window with "good" opinions (setup cost excluded)
        for i in range(SHORT_WINDOW_SIZE):
            ltst.add(OP_GOOD)
        # Now time only the conflict-triggering additions
        for i in range(SHORT_WINDOW_SIZE):
            # suppress GC during measurement
            gc.disable()  
            start = time.perf_counter()
            ltst.add(OP_RESET)
            elapsed = time.perf_counter() - start
            gc.enable()
            if ltst.is_last_conflicted():
                return elapsed
        raise RuntimeError("Reset in buffer did not occur")

    # Warmup (discard)
    for _ in range(num_warmup):
        single_trial()

    # Measurement
    rts = np.empty(num_runs)
    for i in range(num_runs):
        rts[i] = single_trial()

    # convert to microseconds 
    return rts * 1e6 


def bench_fusion(num_warmup: int = NUM_WARMUP,
                 num_runs: int = NUM_RUNS,
                 num_time_steps: int = 100) -> np.ndarray:
    ops_list = [OP_GOOD] * num_time_steps

    # Warmup
    for _ in range(num_warmup):
        sl.Fusion.fuse_opinions(FUSION_TYPE, ops_list)

    rts = np.empty(num_runs)
    for i in range(num_runs):
        gc.disable()
        start = time.perf_counter()
        sl.Fusion.fuse_opinions(FUSION_TYPE, ops_list)
        rts[i] = time.perf_counter() - start
        gc.enable()

    return rts * 1e6


def bench_sequential(num_warmup: int = NUM_WARMUP,
                     num_runs: int = NUM_RUNS) -> np.ndarray:
    for _ in range(num_warmup):
        sl.SequentialFusion.fuse_opinions(
            FUSION_TYPE, OP_GOOD, OP_GOOD, 1.0, 1.0)

    rts = np.empty(num_runs)
    for i in range(num_runs):
        gc.disable()
        start = time.perf_counter()
        sl.SequentialFusion.fuse_opinions(
            FUSION_TYPE, OP_GOOD, OP_GOOD, 1.0, 1.0)
        rts[i] = time.perf_counter() - start
        gc.enable()

    return rts * 1e6


if __name__ == "__main__":
    configs = [
        ("ST Strategy - AVG DC",  True,  True),
        ("ST Strategy - FUSION",  True,  False),
        ("ST Strategy - BASELINE", False, False),
    ]
    for label, hst, adc in configs:
        print(label)
        rts = bench_reset(hst, adc)
        report_timing(rts)

    num_timesteps = 100
    print(f"\nFusion (MSF) ({num_timesteps} ops)")
    report_timing(bench_fusion(num_time_steps=num_timesteps))

    print("\nSequential Fusion (single operator execution)")
    report_timing(bench_sequential())