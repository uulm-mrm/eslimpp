#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt
import bezier
import math


default_trust = sl.Opinion2d([0.8,0.],[1.0,0.0])

# meas_s1 = sl.Opinion(0.8, 0.0).trust_discount_(0.85)
meas_s1 = sl.Opinion(0.0, 0.8)
# meas_s1 = sl.Opinion(0.0, 0.0)
meas_s2 = sl.Opinion(0.0, 0.6)
# meas_s2 = sl.Opinion(0.7, 0.0)
meas_s3 = sl.Opinion(0.7, 0.0)
# meas_s3 = sl.Opinion(0.0, 0.0)
meas_s4 = sl.Opinion(0.3, 0.3)
# meas_s4 = sl.Opinion(0.0, 0.0)

cell_last = sl.Opinion(0.0,0.91)
# cell_last = sl.Opinion(0.0,0.9).trust_discount_(0.7)

# t_s1 = sl.TrustedOpinion(default_trust, meas_s1)
# t_s2 = sl.TrustedOpinion(default_trust, meas_s2)
# t_s3 = sl.TrustedOpinion(default_trust, meas_s3)

t_s1 = sl.TrustedOpinion(sl.Opinion2d([0.3,0.3],[1.0,0.0]), meas_s1)
t_s2 = sl.TrustedOpinion(sl.Opinion2d([0.3,0.2],[1.0,0.0]), meas_s2)
t_s3 = sl.TrustedOpinion(sl.Opinion2d([0.5,0.0],[1.0,0.0]), meas_s3)
t_s4 = sl.TrustedOpinion(sl.Opinion2d([0.3,0.4],[1.0,0.0]), meas_s4)
# t_s4 = sl.TrustedOpinion(default_trust, meas_s4)
# t_cell = sl.TrustedOpinion(default_trust, cell_last)
t_cell = sl.TrustedOpinion(sl.Opinion2d([0.8,0.],[1.0,0.]), cell_last)

# t_vec = [t_s1, t_s2, t_s3, t_cell]
# t_vec = [t_s1, t_s2, t_cell]
t_vec = [t_s1, t_s2, t_s3]
# t_vec = [t_s1, t_s2, t_s3, t_s4]
# t_vec = [t_s1, t_s3]
# t_vec = [t_s1, t_s2]
# t_vec = [t_s2, t_cell]
trusts_prior = [t.trust() for t in t_vec]
t_discounted = [t.discounted_opinion() for t in t_vec]
opinions = sl.TrustedOpinion2d.extractOpinions(t_vec)

# t_avg_conflict = sl.Conflict.avg_conflict(t_discounted)
t_avg_conflict = sl.Conflict.conflict(sl.Conflict.ConflictType.AVERAGE, t_discounted)


# normal_fusion = sl.Fusion.cum_fuse(t_discounted)
normal_fusion = sl.Fusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, opinions)
trusted_fusion = sl.TrustedFusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, t_vec)

conflict_weights_normal = [*range(0,20,1)]
# conflict_weights_normal = [*range(0,1,1)]
conflict_weights_ours = [*range(0,20,1)]
t_fusion = []
for cwn in conflict_weights_normal:
    t_fusion_normal = []
    for cwo in conflict_weights_ours:
        weighted_types = [
            (sl.TrustRevision.TrustRevisionType.NORMAL, sl.Conflict.ConflictType.AVERAGE, 3. * cwo),
            # (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.ACCUMULATE, 1. * cwn),
            (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 1. * cwn),
            # (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 1. * cwo),
            # (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_CUMULATIVE, 1. * cwo),
        ]
        if cwn == 1 and cwo == 0:
            fusion, t_vec_updated_cs = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types, t_vec)
        elif cwn == 0 and cwo == 1:
            fusion, t_vec_updated_bc = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE, weighted_types, t_vec)
        else:
            fusion = sl.TrustedFusion.fuse_opinions(sl.Fusion.FusionType.CUMULATIVE, weighted_types, t_vec)
        t_fusion_normal.append(fusion)

    t_fusion.append(t_fusion_normal)

oranges = plt.get_cmap('Oranges')
greens = plt.get_cmap('Greens')
fig, ax = sl.create_triangle_plot()

for idx1, cwn in enumerate(conflict_weights_normal):
    for idx2, cwo in enumerate(conflict_weights_ours):
        # if idx2 != 0 and idx1 != 0:
        #     continue
        color_1 = greens(0.5 - idx2 / len(conflict_weights_normal) / 2.)
        color_2 = oranges(0.5 - idx1 / len(conflict_weights_ours) / 2.)

        div = idx1 + idx2
        if div == 0:
            color = 0.5 * np.array(color_1) + 0.5 * np.array(color_2)
        else:
            color = idx1 / div * np.array(color_1) + idx2 / div * np.array(color_2)


        sl.draw_point(t_fusion[idx1][idx2], color=color, marker='D', s=10)

s_counter = 0
for trusted_meas in t_vec:
    # skip cell when drawing sensor measurements
    if trusted_meas == t_cell:
        continue

    dis_opinion = trusted_meas.discounted_opinion()
    sl.draw_point(dis_opinion, color="tab:blue", zorder=500)
    # sl.draw_text_at_point(dis_opinion, text=fr'$\bm{{\omega_{{s{s_counter}}}}}$', ha='left', va='top', fontsize=20, offset=[0.01, 0.00])
    s_counter += 1
    # sl.draw_point(trusted_meas.trust(), color='tab:cyan', zorder=500)
cell_discounted = t_cell.discounted_opinion()
# sl.draw_point(cell_discounted, color="tab:red", zorder=500)
# sl.draw_text_at_point(cell_discounted, text=fr'$\bm{{\omega_{{cell}}}}$', ha='right', va='top', fontsize=20, offset=[-0.01, 0.00])
# sl.draw_point(t_cell.trust(), color='tab:pink', zorder=500)
sl.draw_point(normal_fusion, color="tab:red", zorder=500)
sl.draw_point(trusted_fusion, color="tab:orange", zorder=500)

sl.reset_plot()
fig2, ax2 = sl.create_triangle_plot()

for t_op in t_vec:
    trust = t_op.trust()
    sl.draw_point(trust)

for t_op in t_vec_updated_cs:
    trust = t_op.trust()
    sl.draw_point(trust, color="tab:red", edgecolors='black', linewidth=0.5)

for t_op in t_vec_updated_bc:
    trust = t_op.trust()
    sl.draw_point(trust, color="tab:green", edgecolors='black', linewidth=0.5)


plt.show()


