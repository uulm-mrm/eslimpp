"""
Script comparing the results of naive sequential fusion with our proposed sequential operators
and multi-source fusion. Results and errors are reported for a set of random opinions and the 
commonly used opinions from Table (1) of the paper referred to as 'JOSANG Eval Opinions'.
"""
import copy
import csv
import numpy as np
import subjective_logic as sl


def generate_sl_opinions(alphas=np.array([1, 1, 1]), num_samples: int = 1) -> list[sl.Opinion]:
    return [sl.Opinion2d(p[0], p[1]) for p in rng.dirichlet(alphas, size=num_samples)]


def single_run(ops: list[sl.Opinion], fusion_type: sl.FusionType, verbose: bool = True, save_csv: bool = False) -> None:
    # naive sequential fusion
    op_naive = copy.deepcopy(ops[0])
    for op in ops[1:]:
        if fusion_type == sl.FusionType.AVERAGE:
            op_naive = op_naive.average_fuse(op)
        elif fusion_type == sl.FusionType.WEIGHTED:
            op_naive = op_naive.wb_fuse(op)
        else:
            raise NotImplementedError

    # sequential fusion
    op_sf = copy.deepcopy(ops[0])
    weight = 1.0 if fusion_type == sl.FusionType.AVERAGE else (1 - op_sf.uncertainty())
    for op in ops[1:]:
        op_sf, weight = sl.SequentialFusion.fuse_opinions(fusion_type, op_sf, op, weight, DISCOUNT)

    # multi-source fusion
    op_msf = sl.Fusion.fuse_opinions(fusion_type, ops)

    if verbose:
        print(f"\t{'Naive:':<15} {op_naive}\n\t{'Seq (ours):':<15} {op_sf}\n\t{'Multi-Source:':<15} {op_msf}")

    err_naive = (
        abs(op_msf.belief() - op_naive.belief())
        + abs(op_msf.disbelief() - op_naive.disbelief())
        + abs(op_msf.uncertainty() - op_naive.uncertainty())
    )
    err_sf = (
        abs(op_msf.belief() - op_sf.belief())
        + abs(op_msf.disbelief() - op_sf.disbelief())
        + abs(op_msf.uncertainty() - op_sf.uncertainty())
    )

    if save_csv:
        headers = ["inp1", "inp2", "inp3", "msf", "naive", "seq"]
        beliefs = [op.belief() for op in ops] + [op_msf.belief(), op_naive.belief(), op_sf.belief()]
        disbeliefs = [op.disbelief() for op in ops] + [op_msf.disbelief(), op_naive.disbelief(), op_sf.disbelief()]
        uncs = [op.uncertainty() for op in ops] + [op_msf.uncertainty(), op_naive.uncertainty(), op_sf.uncertainty()]
        base_rates = [0.5] * len(beliefs)
        proj_prob = [b + u * a for b, u, a in zip(beliefs, uncs, base_rates)]

        with open(f"/tmp/results_{'abf' if fusion_type == sl.FusionType.AVERAGE else 'wbf'}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows([headers, beliefs, disbeliefs, uncs, base_rates, proj_prob])

    return err_naive, err_sf


def multiple_runs(fusion_type: sl.FusionType, num_runs=100, num_ops=100):
    errs = np.zeros((num_runs, 2))

    for idx in range(num_runs):
        errs[idx, :] = np.array([*single_run(generate_sl_opinions(num_samples=num_ops), fusion_type, verbose=False)])

    mean_err, std_err = np.mean(errs, axis=0), np.std(errs, axis=0)
    print(
        f"\t{'Naive error:':<20} {mean_err[0]:.4f} +/- {std_err[0]:.4f}\n"
        f"\t{'Seq (ours) error:':<20} {mean_err[1]:.4f} +/- {std_err[1]:.4f}"
    )


rng = np.random.default_rng(seed=42)

OPS_LIST = generate_sl_opinions(num_samples=100)
OPS_LIST_JOSANG = [sl.Opinion(0.1, 0.3), sl.Opinion(0.4, 0.2), sl.Opinion(0.7, 0.1)]

DISCOUNT = 1.0
NUM_RUNS = 1000

fusion_types = [sl.FusionType.AVERAGE, sl.FusionType.WEIGHTED]

for ft in fusion_types:
    print(f"Fusion Type: {ft}")
    print(f"Single run on {len(OPS_LIST)} random opinions:")
    single_run(OPS_LIST, ft)

    print(f"\nPerforming {NUM_RUNS} runs with MSF as reference:")
    multiple_runs(ft, NUM_RUNS)

    print(f"\nSingle run on JOSANG Eval Opinions (Table in Publication):")
    single_run(OPS_LIST_JOSANG, ft, save_csv=True)
    print()
