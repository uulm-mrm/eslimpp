#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt
import math
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{bm}')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

default_trust = sl.Opinion2d([0.8,0.],[1.0,0.0])


cell_opinion = sl.OpinionNoBase2d([0.,0.])
trusted_cell_op = sl.TrustedOpinionNoBase2d(default_trust, cell_opinion)

uncertainty_limit_start = 0.00
uncertainty_limit_end = 0.15
n_uncert_steps = 3
uncertainty_step = (uncertainty_limit_end - uncertainty_limit_start) / n_uncert_steps

n_steps = 50
probs = np.zeros([n_uncert_steps, n_steps])
uncerts = np.zeros([n_uncert_steps, n_steps])

probs_normal = np.zeros([n_uncert_steps, n_steps])

gt = np.zeros([n_steps])
gt[n_steps // 2:] = 1
for n in range(n_uncert_steps):

    uncertainty_limit = uncertainty_limit_start + n * uncertainty_step

    cell_estimate = sl.OpinionNoBase2d(0.0,0.0)
    trusted_cell_op = sl.TrustedOpinionNoBase2d(default_trust, cell_estimate)
    cell_opinion = sl.OpinionNoBase2d(0.0,0.0)

    probs[n,0] = cell_estimate.getBinomialProjection(0.5)
    uncerts[n,0] = cell_estimate.uncertainty()
    probs_normal[n,0] = cell_opinion.getBinomialProjection(0.5)

    for i in range(1,n_steps):

        meas_1 = sl.OpinionNoBase2d(0.,0.7)
        meas_2 = sl.OpinionNoBase2d(0.,0.7)
        meas_3 = sl.OpinionNoBase2d(0.,0.7)

        if i == 10 or i == 10:
            meas_1 = sl.OpinionNoBase(0.7,0.0)

        if i == 10 : #or i == 51:
            meas_2 = sl.OpinionNoBase(0.7,0.0)



        if i >= n_steps // 2:
            meas_1 = sl.OpinionNoBase(0.7,0.0)
            meas_2 = sl.OpinionNoBase(0.7,0.0)
            meas_3 = sl.OpinionNoBase(0.7,0.0)

        if i % 2 == 0:
            if i >= n_steps // 2:
                meas_1 = sl.OpinionNoBase(0.0,0.7)
            else:
                meas_1 = sl.OpinionNoBase(0.7,0.0)

        # if i == n_steps//2 or i == n_steps // 2 + 1:
        #     meas_3 = sl.OpinionNoBase(0.0,0.7)

        t_meas_1 = sl.TrustedOpinionNoBase2d(default_trust, meas_1)
        t_meas_2 = sl.TrustedOpinionNoBase2d(default_trust, meas_2)
        t_meas_3 = sl.TrustedOpinionNoBase2d(default_trust, meas_3)

        cell_opinion.cum_fuse_(meas_1).cum_fuse_(meas_2).cum_fuse_(meas_3)
        # cell_opinion.cum_fuse_(meas_1).cum_fuse_(meas_2)
        # cell_opinion.cum_fuse_(meas_2)

        t_vec = [trusted_cell_op,
                 t_meas_1,
                 t_meas_2,
                 t_meas_3
        ]

        if n==0:
            weighted_types = [
                # [sl.TrustRevision.TrustRevisionType.NORMAL, sl.Conflict.ConflictType.ACCUMULATE, 0.],
                (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 3.),
            ]
        else:
            weighted_types = [
                (sl.TrustRevision.TrustRevisionType.NORMAL, sl.Conflict.ConflictType.AVERAGE, 3.),
                # [sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 0.],
            ]

        cell_estimate = sl.TrustedFusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, weighted_types, t_vec)

        # if  cell_estimate.uncertainty() < uncertainty_limit:
        #     cell_estimate.trust_discount_( (1. - uncertainty_limit) / (1. - cell_estimate.uncertainty()))
        trusted_cell_op = sl.TrustedOpinionNoBase2d(default_trust, cell_estimate)

        if  cell_opinion.uncertainty() < uncertainty_limit:
            cell_opinion.trust_discount_( (1. - uncertainty_limit) / (1. - cell_opinion.uncertainty()))

        probs[n, i] = cell_estimate.getBinomialProjection(0.5)
        uncerts[n,i] = cell_estimate.uncertainty()
        probs_normal[n, i] = cell_opinion.getBinomialProjection(0.5)


blues = plt.get_cmap('Blues')
greens = plt.get_cmap('Greens')

plt.plot(gt, color="gray", linestyle='--', label='ground truth')
# for n in range(n_uncert_steps):
#     plt.plot(uncerts[n, :], color = greens(1 - n / 1.3 / n_uncert_steps))
# for n in reversed(range(n_uncert_steps)):
for n in reversed(range(n_uncert_steps)):

    uncertainty_limit = uncertainty_limit_start + n * uncertainty_step
    # plt.plot(probs[n,:], color = blues(1 - n / 1.2 / n_uncert_steps))
    plt.plot(probs_normal[n,:], color = greens(1 - (n + 1) / 1.2 / n_uncert_steps), label=f'limit {uncertainty_limit:.2f}')

plt.plot(probs[0, :], color='tab:red', label='trust revision cs', zorder=99)
plt.plot(probs[1, :], color='tab:blue', label='trust revision', zorder=98)

plt.ylim([-0.01,1.01])
plt.legend(loc='upper left')
plt.xlabel('time step')
plt.ylabel(r'$P_X(\text{occupied})$')
plt.show()

