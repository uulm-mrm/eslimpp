"""
Script verifying sequential operators for average and weighted belief fusion in opinion space
by comparing the results with the evidence space implementation and the multi-source fusion.

Executing this script tests both average and weighted belief fusion and indicates whether
it was successful.
"""

import copy
import itertools
import numpy as np
import subjective_logic as sl


def abf_sequential_op(prev_op: sl.Opinion, curr_op: sl.Opinion, weight: float, discount: float) -> sl.Opinion:
    updated_weight = discount * weight + 1

    prev_belief = np.array(prev_op.belief_masses)
    prev_unc = prev_op.uncertainty()

    curr_belief = np.array(curr_op.belief_masses)
    curr_unc = curr_op.uncertainty()

    num = (updated_weight - 1) * curr_unc * prev_belief + prev_unc * curr_belief
    denom = (updated_weight - 1) * curr_unc + prev_unc
    updated_belief = num / denom
    updated_unc = (updated_weight * prev_unc * curr_unc) / denom

    return sl.Opinion([*updated_belief]), updated_weight


def wbf_sequential_op(prev_op: sl.Opinion, curr_op: sl.Opinion, weight: float, discount: float) -> sl.Opinion:
    prev_belief = np.array(prev_op.belief_masses)
    prev_unc = prev_op.uncertainty()

    curr_belief = np.array(curr_op.belief_masses)
    curr_unc = curr_op.uncertainty()

    prev_weight = weight
    weight = discount * prev_weight + (1 - curr_unc)

    num = discount * prev_weight * curr_unc * prev_belief + (1 - curr_unc) * prev_unc * curr_belief
    denom = discount * prev_weight * curr_unc + prev_unc * (1 - curr_unc)

    updated_belief = num / denom
    updated_unc = weight * prev_unc * curr_unc / denom

    return sl.Opinion([*updated_belief]), weight


def test_seq_fusion(ftype: sl.FusionType, discount: float) -> None:
    op_ev, op_op = copy.deepcopy(OPS_LIST[0]), copy.deepcopy(OPS_LIST[0])
    weight_ev = 1.0 if ftype == sl.FusionType.AVERAGE else (1 - op_ev.uncertainty())
    weight_op = weight_ev

    fuse_func = abf_sequential_op if ftype == sl.FusionType.AVERAGE else wbf_sequential_op

    for idx, op in enumerate(OPS_LIST[1:]):
        op_ev, weight_ev = sl.SequentialFusion.fuse_opinions(ftype, op_ev, op, weight_ev, discount)
        op_op, weight_op = fuse_func(op_op, op, weight_op, discount)

        assert op_op == op_ev
        assert np.isclose(weight_ev, weight_op)
        if discount == 1.0:
            op_msf = sl.Fusion.fuse_opinions(ftype, OPS_LIST[: idx + 2])
            assert op_op == op_msf


rng = np.random.default_rng()

ftypes = [sl.FusionType.AVERAGE, sl.FusionType.WEIGHTED]
apply_discount = True
dims = range(3, 12)

for dim, ftype in itertools.product(dims, ftypes):
    discount = rng.random() if apply_discount else 1.0

    alphas = [1.0] * dim
    OPS_LIST = [sl.Opinion(*p[: len(alphas) - 1]) for p in rng.dirichlet(alphas, size=25)]
    test_seq_fusion(ftype, discount)
    print(f"{ftype} successful on opinions of dim {dim-1} over {len(OPS_LIST)} opinions with discount {discount}")
