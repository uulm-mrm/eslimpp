import copy

import subjective_logic as sl
import random
import numpy as np

import matplotlib.pyplot as plt
from alive_progress import alive_bar

test_uncertain = 0.4

num_agents = 5
num_runs = 1000
num_mc_runs = 10

switches = [
]

colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:orange', 'tab:cyan']
draw_divider = 1
triangle_divider = 15

opinion_dimension = 4
default_opinions = []
for i in range(opinion_dimension):
    belief_masses = [0.0] * opinion_dimension
    belief_masses[i] = 0.9
    prior = [1 / opinion_dimension] * opinion_dimension
    prior[0] = prior[0] - 0.1
    prior[-1] = prior[-1] + 0.1
    default_opinions.append(sl.Opinion(belief_masses, prior))
uncertain_opinion = sl.Opinion(*[0.]*opinion_dimension)


reliabilities = [0.1, 0.4, 0.7, 0.9, 0.9]

weighted_types_cs_avg = [
    # (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
    # (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.066),
    (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
    (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.1),
]

trusts = [[] for _ in range(num_agents)]

baseline_decisions=np.zeros((num_mc_runs, num_agents, num_runs))
projections = np.zeros((num_mc_runs, num_agents, num_runs))
with alive_bar(num_mc_runs) as bar:
    for mc_run in range(num_mc_runs):
        trusted_opinions = [
            sl.TrustedOpinion(sl.Opinion([0, 0], [0.5, 0.5]), uncertain_opinion) for _ in range(num_agents)
        ]
        for run in range(num_runs):

            uncertainty_sum = 0
            possible_events = list(range(opinion_dimension))
            # ground truth event for this run
            event = random.choice(possible_events)
            other_events = copy.copy(possible_events)
            other_events.remove(event)
            for idx in range(num_agents):
                if mc_run == 0:
                    trusts[idx].append(trusted_opinions[idx].trust_copy())

                projections[mc_run, idx, run] = trusted_opinions[idx].trust.getBinomialProjection()

                uncertainty_sum += trusted_opinions[idx].trust.uncertainty()

                # invisible and unreliable events are selected randomly for each agent independently
                invisible_event = random.choice(possible_events)
                unreliable_event = random.choice(other_events)
                # important, sample to independent random numbers
                reliable = reliabilities[idx] > np.random.uniform()
                visible = np.random.uniform() > test_uncertain
                if reliable and visible:
                    trusted_opinions[idx].opinion = default_opinions[event]
                    baseline_decisions[mc_run, idx, run] = event
                elif not visible:
                    trusted_opinions[idx].opinion = uncertain_opinion
                    baseline_decisions[mc_run, idx, run] = invisible_event
                else: # unreliable
                    trusted_opinions[idx].opinion = default_opinions[unreliable_event]
                    baseline_decisions[mc_run, idx, run] = unreliable_event

            avg_uncertainty = uncertainty_sum / num_agents
            updated_weights = []
            for entry in weighted_types_cs_avg:
                updated_weights.append((
                    entry[0],
                    entry[1],
                    entry[2] * avg_uncertainty,
                    # entry[2],
                ))

            fusion_result, trusted_opinions = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE,
                                                                              updated_weights, trusted_opinions)
        bar()

baseline_reliabilities=np.zeros((num_mc_runs, num_agents, num_runs))
for mc_run in range(num_mc_runs):
    current_run = baseline_decisions[mc_run, :, :].astype(int)
    answers = np.zeros([num_agents, num_runs])
    distributions = [sl.DirichletDistribution2f([0,0]) for _ in range(num_agents)]
    for idx in range(num_runs):
        pseudo_gt = np.argmax(np.bincount(current_run[:,idx]))
        answers[:,idx] = (current_run[:,idx] == pseudo_gt).astype(int)

        for agent_idx in range(num_agents):
            distributions[agent_idx].evidences[int(answers[agent_idx, idx])] += 1
            baseline_reliabilities[mc_run, agent_idx, idx] = distributions[agent_idx].mean()[1]

        # start_idx = max(0, idx - baseline_window_size)
        # baseline_reliabilities[mc_run,:,idx] = np.average(answers[:,start_idx:idx], axis=1)

baseline_reliabilities_mc = np.median(baseline_reliabilities, axis=0)
plt.plot(baseline_reliabilities_mc[:,].T, color='tab:blue')
print('final baseline reliability estimation\n', baseline_reliabilities_mc[:,-1])
# plt.show()
#

median_projections = np.median(projections, axis=0)
print(median_projections[:,-1])
q25_projections = np.quantile(projections, 0.25, axis=0)
q75_projections = np.quantile(projections, 0.75, axis=0)

plt.ylim(0,1)
for color, idx in zip(colors, range(num_agents)):
    # plt.plot(median_projections[idx, ::draw_divider], color=color)
    plt.plot(median_projections[idx, ::draw_divider], color='tab:green')
    plt.fill_between(
        range(0, num_runs // draw_divider),
        q75_projections[idx, ::draw_divider],
        q25_projections[idx, ::draw_divider],
        color='gray', alpha=0.5)

# fig2, ax2 = sl.create_triangle_plot(('distrust','trust'))
# for color, trust_list in zip(colors, trusts):
#     # sl.reset_plot()
#     # sl.create_triangle_plot()
#     for idx, trust in enumerate(trust_list):
#         if idx % triangle_divider != 0:
#             continue
#         sl.draw_point(trust, color=color, sizes=[100])
#
plt.show()

# export_args = {
#     "format": 'pdf',
#     "dpi": 500,
#     "transparent": True,
#     "bbox_inches": 'tight',
#     "pad_inches": 0,
# }
# fig2.savefig("04-constant_parameter_triangle.pdf", **export_args)
#
# import pandas as pd
#
# data = {'time': range(median_projections.shape[1])}
# for idx in range(num_agents):
#     data[str(idx) + '_median'] = median_projections[idx,:]
#     data[str(idx) + '_upper'] = q75_projections[idx, :]
#     data[str(idx) + '_lower'] = q25_projections[idx, :]
#
# df = pd.DataFrame(data)
# df.to_csv('04-exp_constant_parameter.csv', index=False)
#
