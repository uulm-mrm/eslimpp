import copy

import subjective_logic as sl
import random
import numpy as np

import matplotlib.pyplot as plt
from alive_progress import alive_bar

num_agents = 5
num_runs = 200
num_mc_runs = 1000

# idx_switch = num_runs
switches = [
    # num_runs // 4 * 3,
    # num_runs // 4 * 2,
    # num_runs // 4,
]

colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:olive', 'tab:brown', 'tab:purple', 'tab:orange', 'tab:cyan']
draw_divider = 1
triangle_divider = 15

default_true = sl.Opinion([0.9, 0.0], [0.5, 0.5])
default_false = sl.Opinion([0.0, 0.9], [0.5, 0.5])
# default_true = sl.Opinion([1.0, 0.0], [0.5, 0.5])
# default_false = sl.Opinion([0.0, 1.0], [0.5, 0.5])

reliabilities_raw = [
    [0.1, 0.4, 0.7, 0.9],
    [0.1, 0.9, 0.9, 0.9] ,
    [0.9, 0.7, 0.5, 0.9],
]
# select reliability intervals according to number of switches
reliabilities_raw = [reliabilities_raw[idx] for idx in range(len(switches) + 1)]
# extens all reliability intervals according to the number of agents
reliabilities_raw = [[rels[min(idx, len(rels) - 1)] for idx in range(num_agents) ] for rels in reliabilities_raw ]

weighted_types_cs_avg = [
    # (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES_ALLOW_NEGATIVE, sl.Conflict.ConflictType.AVERAGE, 0.1),
    # (sl.TrustRevision.TrustRevisionType.CONFLICT_SHARES, sl.Conflict.ConflictType.AVERAGE, 0.05),
    # (sl.TrustRevision.TrustRevisionType.HARMONY_SHARES, sl.Conflict.ConflictType.AVERAGE, 0.05),
    # (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
    # (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
    (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.3),
    (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.2),
    # (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_CUMULATIVE, 0.1),
    # (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_CUMULATIVE, 0.1),
    # (sl.TrustRevision.TrustRevisionType.NORMAL, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.05),
    # (sl.TrustRevision.TrustRevisionType.HARMONY_NORMAL, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
]


# check_event = []
# check_observs = [[] for _ in range(num_agents)]
# check_results = []
# check_match = []
trusts = [[] for _ in range(num_agents)]

print(reliabilities_raw)
projections = np.zeros((num_mc_runs, num_agents, num_runs))
with alive_bar(num_mc_runs) as bar:
    for mc_run in range(num_mc_runs):
        trusted_opinions = [
            sl.TrustedOpinion(sl.Opinion([0.0, 0.0], [0.5, 0.5]), sl.Opinion(0, 0)) for _ in range(num_agents)
        ]
        for run in range(num_runs):

            reliabilities = reliabilities_raw[0]

            for idx, switch in enumerate(switches):
                if run > switch:
                    reliabilities = reliabilities_raw[idx + 1]

            uncertainty_sum = 0
            event = random.choice([True, False])
            for idx in range(num_agents):
                if mc_run == 0:
                    trusts[idx].append(trusted_opinions[idx].trust_copy())

                projections[mc_run, idx, run] = trusted_opinions[idx].trust.getBinomialProjection()

                uncertainty_sum += trusted_opinions[idx].trust.uncertainty()

                trusted_opinions[idx].opinion = default_true
                if (reliabilities[idx] > np.random.uniform()) != event:
                    trusted_opinions[idx].opinion = default_false

                # variation = np.random.normal() * 0.05
                # # variation = abs(np.random.normal()) * 0.05
                # # trusted_opinions[idx].opinion.trust_discount(1 - variation)
                # trusted_opinions[idx].opinion.belief_masses[0] += variation
                # trusted_opinions[idx].opinion.belief_masses[1] -= variation

            # fusion_result, trusted_opinions = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE,
            # weighted_types_cs_avg, trusted_opinions)
            #
            avg_uncertainty = uncertainty_sum / num_agents
            updated_weights = []
            for entry in weighted_types_cs_avg:
                updated_weights.append((
                    entry[0],
                    entry[1],
                    # entry[2],
                    entry[2] * avg_uncertainty,
                    # 0.05 * entry[2] + 0.2 * avg_uncertainty * entry[2]
                ))


            fusion_result, trusted_opinions = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE,
                                                                              updated_weights, trusted_opinions)
            # proj = fusion_result.getBinomialProjection()
            # for t_op in trusted_opinions:
            #     t_op.trust.trust_discount_(0.9999)
                # t_op.trust.trust_discount_(0.999)
        bar()

mean_projections = np.mean(projections, axis=0)
median_projections = np.median(projections, axis=0)
q25_projections = np.quantile(projections, 0.25, axis=0)
q75_projections = np.quantile(projections, 0.75, axis=0)

plt.ylim(0,1)
for color, idx in zip(colors, range(num_agents)):
    # gt = [reliabilities_start[idx] for i in range(0, idx_switch, draw_divider)] + [reliabilities_switched[idx] for i in
    #                                                                                range(0, num_runs - idx_switch,
    #                                                                                      draw_divider)]
    # plt.plot(gt, color='tab:gray')
    # plt.plot(mean_projections[idx, ::draw_divider], color=color)
    plt.plot(median_projections[idx, ::draw_divider], color=color)
    plt.fill_between(
        range(0, num_runs // draw_divider),
        q75_projections[idx, ::draw_divider],
        q25_projections[idx, ::draw_divider],
        color='gray', alpha=0.5)

# for t_op in trusted_opinions:
#     print(t_op.trust)
#     print('proj:', t_op.trust.getBinomialProjection())
#
#     print(np.mean(check_observs[idx]))
#     print()
#
# # check_results = np.array(check_results)
# # check_event = np.array(check_event)
#
# print('event:', np.mean(check_event))
# print('result:', np.mean(check_results))
#
# print('match_rate:', np.mean(check_match))
#
# for color, (idx, trust_list) in zip(colors, enumerate(trusts)):
#     trust_projection = [trust.getBinomialProjection() for trust in trust_list[::draw_divider]]
#     plt.plot(trust_projection, color=color)
#     gt = [reliabilities_start[idx] for i in range(0,idx_switch, draw_divider)] + [reliabilities_switched[idx] for i in range(0,num_runs - idx_switch, draw_divider)]
#     plt.plot(gt, color='tab:gray')
#
#
sl.create_triangle_plot()
for color, trust_list in zip(colors, trusts):
    # sl.reset_plot()
    # sl.create_triangle_plot()
    for idx, trust in enumerate(trust_list):
        if idx % triangle_divider != 0:
            continue
        sl.draw_point(trust, color=color)

plt.show()
