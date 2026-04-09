#!/usr/bin/python3
import numpy as np
import subjective_logic as sl
import matplotlib.pyplot as plt


incorrect_observer = 5
incorrect_prediction = 15
rain_start = 30
rain_stop = 40

n_steps = 55
######################################################
# create opinions
######################################################

trust_A_A = sl.Opinion2d([0.9,0.05],[1.0,0.0])
trust_A_S = sl.Opinion2d([0.5,0.0],[1.0,0.0])
trust_A_P = sl.Opinion2d([0.5,0.0],[1.0,0.0])

rainy_observation = sl.Opinion(0.8,0.0)
sunny_observation = sl.Opinion(0.0,0.8)

predict_const_rainy = sl.Opinion(0.99, 0.0)
predict_const_sunny = sl.Opinion(0.0,0.99)

predict_change_rainy = sl.Opinion(0.0,0.99)
predict_change_sunny = sl.Opinion(0.99,0.0)

last_A_tr_ours = sl.Opinion2d()
last_A_tr_josang = sl.Opinion2d()
last_A_normal = sl.Opinion2d()
last_A_no_trust = sl.Opinion2d()

tr_weighted_types_ours = [
    sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.SHARES, sl.ConflictType.AVERAGE, 1),
]
tr_weighted_types_josang = [
    sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE, 1),
]

values_tr_ours = np.zeros((n_steps,2))
values_tr_josang = np.zeros((n_steps,2))
values_normal = np.zeros((n_steps,2))
values_no_trust = np.zeros((n_steps,2))
values_gt = np.zeros((n_steps,2))
values_gt[rain_start:rain_stop,1] = 1.0
for k in range(n_steps):
    # print(k)
    # discount last opinion
    last_A_tr_ours.trust_discount_(trust_A_A)
    last_A_tr_josang.trust_discount_(trust_A_A)
    last_A_normal.trust_discount_(trust_A_A)

    # prediction
    last_A_pred_const_tr_ours = last_A_tr_ours.deduction(predict_const_rainy, predict_const_sunny)
    last_A_pred_change_tr_ours = last_A_tr_ours.deduction(predict_change_rainy, predict_change_sunny)
    last_A_pred_const_tr_josang = last_A_tr_josang.deduction(predict_const_rainy, predict_const_sunny)
    last_A_pred_change_tr_josang = last_A_tr_josang.deduction(predict_change_rainy, predict_change_sunny)
    last_A_pred_const_normal = last_A_normal.deduction(predict_const_rainy, predict_const_sunny)
    last_A_pred_change_normal = last_A_normal.deduction(predict_change_rainy, predict_change_sunny)
    last_A_pred_const_no_trust = last_A_no_trust.deduction(predict_const_rainy, predict_const_sunny)
    last_A_pred_change_no_trust = last_A_no_trust.deduction(predict_change_rainy, predict_change_sunny)

    last_A_pred_tr_ours = last_A_pred_const_tr_ours
    last_A_pred_tr_josang = last_A_pred_const_tr_josang
    last_A_pred_normal = last_A_pred_const_normal
    last_A_pred_no_trust = last_A_pred_const_no_trust
    if k == rain_start or k == incorrect_prediction:
        last_A_pred_tr_ours = last_A_pred_change_tr_ours
        last_A_pred_tr_josang = last_A_pred_change_tr_josang
        last_A_pred_normal = last_A_pred_change_normal
        last_A_pred_no_trust = last_A_pred_change_no_trust

    t_pred_tr_ours = sl.TrustedOpinion2d(trust_A_P, last_A_pred_tr_ours)
    t_pred_tr_josang = sl.TrustedOpinion2d(trust_A_P, last_A_pred_tr_josang)
    t_pred_normal = sl.TrustedOpinion2d(trust_A_P, last_A_pred_normal)

    # sources
    source_op = sunny_observation
    if rain_start <= k < rain_stop:
        source_op = rainy_observation

    t_source_vec = [
        sl.TrustedOpinion(trust_A_S, source_op),
        sl.TrustedOpinion(trust_A_S, source_op),
        sl.TrustedOpinion(trust_A_S, source_op),
    ]
    if k == incorrect_observer:
        t_source_vec[0] = sl.TrustedOpinion(trust_A_S, rainy_observation)

    t_vec_tr_ours = t_source_vec + [t_pred_tr_ours]
    t_vec_tr_josang = t_source_vec + [t_pred_tr_josang]
    t_vec_normal = t_source_vec + [t_pred_normal]
    discounted_opinions = sl.TrustedOpinion2d.extractDiscountedOpinions(t_vec_normal)
    opinions_no_trust = sl.TrustedOpinion2d.extractOpinions(t_source_vec) + [last_A_pred_no_trust]

    op_A_normal = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, discounted_opinions)
    op_A_tr_ours = sl.TrustedFusion.fuse_opinions(
        sl.FusionType.CUMULATIVE,
        tr_weighted_types_ours,
        t_vec_tr_ours,
    )
    op_A_tr_josang = sl.TrustedFusion.fuse_opinions(
        sl.FusionType.CUMULATIVE,
        tr_weighted_types_josang,
        t_vec_tr_josang,
    )
    op_A_no_trust = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, opinions_no_trust)

    values_tr_ours[k] = [k, op_A_tr_ours.getBinomialProjection()]
    values_tr_josang[k] = [k, op_A_tr_josang.getBinomialProjection()]
    values_normal[k] = [k, op_A_normal.getBinomialProjection()]
    values_no_trust[k] = [k, op_A_no_trust.getBinomialProjection()]
    values_gt[k,0] = k

    last_A_tr_ours = op_A_tr_ours
    last_A_tr_josang = op_A_tr_josang
    last_A_normal = op_A_normal
    last_A_no_trust = op_A_no_trust

plt.plot(values_no_trust[:,1], label='no trust')
# plt.plot(values_normal[:,1], label='normal')
plt.plot(values_tr_josang[:,1], label='josang')
plt.plot(values_tr_ours[:,1], label='cs')
plt.plot(values_gt[:,1], 'r', label='gt')
plt.legend()

plt.figure()
plt.plot(np.abs((values_no_trust - values_gt)[:,1]))
plt.plot(np.abs((values_tr_josang - values_gt)[:,1]))
plt.plot(np.abs((values_tr_ours - values_gt)[:,1]))

plt.show()


plot_data = {
    "no_trust": values_no_trust,
    # "trusted_fusion": values_normal,
    "trust_revision_josang": values_tr_josang,
    "trust_revision_ours": values_tr_ours,
    "ground_truth": values_gt,
}

pgfplots_add_plot_line = "\\addplot [name path={name}, color={color}, line width=1.5pt{extra}]\n\ttable[col sep=comma]{{%\n"
pgfplots_add_legend_entry = "\\addlegendentry{{{name}}}\n"
pgfplots_end = "};\n\n"


def export_values(name, data, filename):
    with open(filename, 'w') as f:
        f.write(pgfplots_add_plot_line.format(name=name, color=name + "_color", extra=""))
        np.savetxt(f, data, delimiter=', ', )  # newline='\\\\\n')
        f.write(pgfplots_end)
        # f.write(pgfplots_add_legend_entry.format(name=name.replace('_',' ')))
        f.write("\n")

        f.close()


# for name, data in plot_data.items():
#     filename_points =f"{name}_tikz_points.txt"
#     filename_diffs =f"{name}_diffs_tikz_points.txt"
#
#     export_values(name, data, filename_points)
#     if not "truth" in name:
#         diffs = data
#         diffs[:,1] -= values_gt[:,1]
#         export_values(name, np.abs(diffs), filename_diffs)

