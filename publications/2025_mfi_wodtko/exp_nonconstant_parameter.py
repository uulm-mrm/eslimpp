import copy

import subjective_logic as sl
import random
import numpy as np

import matplotlib.pyplot as plt
from alive_progress import alive_bar

test_uncertain = 0.3
num_agents = 5
num_runs = 400
num_mc_runs = 1000

switches = [
    num_runs // 2
]

colors = ['tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:red', 'tab:purple', 'tab:orange', 'tab:cyan']

draw_divider = 1

opinion_dimension = 4
default_opinions = []
for i in range(opinion_dimension):
    belief_masses = [0.0] * opinion_dimension
    belief_masses[i] = 0.9
    prior = [1 / opinion_dimension] * opinion_dimension
    prior[0] = prior[0] - 0.1
    prior[-1] = prior[-1] + 0.1
    default_opinions.append(sl.Opinion(belief_masses, prior))
uncertain_opinion = sl.Opinion(*[0.] * opinion_dimension)

reliabilities_start = [0.1, 0.95, 0.95, 0.95, 0.95]
reliabilities_end = [0.6, 0.1, 0.3, 0.95, 0.95]

weighted_types_cs_avg = [
    (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.2),
    (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.2),
]

trusts = [[] for _ in range(num_agents)]

baseline_decisions = np.zeros((num_mc_runs, num_agents, num_runs))
projections = np.zeros((num_mc_runs, num_agents, num_runs))
with alive_bar(num_mc_runs) as bar:
    for mc_run in range(num_mc_runs):
        trusted_opinions = [
            sl.TrustedOpinion(sl.Opinion([0.0, 0.0], [0.5, 0.5]), uncertain_opinion) for _ in range(num_agents)
        ]
        for run in range(num_runs):

            if run < switches[0]:
                reliabilities = reliabilities_start
            else:
                reliabilities = reliabilities_end

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
                else:  # unreliable
                    trusted_opinions[idx].opinion = default_opinions[unreliable_event]
                    baseline_decisions[mc_run, idx, run] = unreliable_event

            avg_uncertainty = uncertainty_sum / num_agents
            updated_weights = []
            for entry in weighted_types_cs_avg:
                updated_weights.append((
                    entry[0],
                    entry[1],
                    entry[2]
                    # entry[2] * avg_uncertainty,
                ))

            fusion_result, trusted_opinions = sl.TrustedFusion.fuse_opinions_(sl.Fusion.FusionType.CUMULATIVE,
                                                                              updated_weights, trusted_opinions)
            for t_op in trusted_opinions:
                t_op.trust.trust_discount_(0.999)
        bar()

gt_draw_window = 100
end_first = switches[0]
start_first = end_first - gt_draw_window
end_second = num_runs
start_second = end_second - gt_draw_window
for idx in range(num_agents):
    plt.plot([start_first, end_first], [reliabilities_start[idx]] * 2, color='black', linewidth=1.5)
    plt.plot([start_second, end_second], [reliabilities_end[idx]] * 2, color='black', linewidth=1.5)

baseline_reliabilities = np.zeros((num_mc_runs, num_agents, num_runs))

for mc_run in range(num_mc_runs):
    current_run = baseline_decisions[mc_run, :, :].astype(int)
    answers = np.zeros([num_agents, num_runs])
    distributions = [sl.DirichletDistribution2f([0, 0]) for _ in range(num_agents)]
    for idx in range(num_runs):
        pseudo_gt = np.argmax(np.bincount(current_run[:, idx]))
        answers[:, idx] = (current_run[:, idx] == pseudo_gt).astype(int)

        for agent_idx in range(num_agents):
            distributions[agent_idx].evidences[int(answers[agent_idx, idx])] += 1
            baseline_reliabilities[mc_run, agent_idx, idx] = distributions[agent_idx].mean()[1]
            distributions[agent_idx].evidences *= 0.95

baseline_reliabilities_mc = np.median(baseline_reliabilities, axis=0)
baseline_q25_projections = np.quantile(baseline_reliabilities, 0.25, axis=0)
baseline_q75_projections = np.quantile(baseline_reliabilities, 0.75, axis=0)
for color, idx in zip(colors, range(num_agents)):
    plt.plot(baseline_reliabilities_mc[idx, ::draw_divider], ':', color=color)
print('final baseline reliability estimation\n', baseline_reliabilities_mc[:, -1])

median_projections = np.median(projections, axis=0)
print(median_projections[:, -1])
q25_projections = np.quantile(projections, 0.25, axis=0)
q75_projections = np.quantile(projections, 0.75, axis=0)

plt.ylim(0, 1)
for color, idx in zip(colors, range(num_agents)):
    plt.plot(median_projections[idx, ::draw_divider], color=color)

plt.show()

import pandas as pd

data = {'time': range(median_projections.shape[1])}
for idx in range(num_agents):
    data[str(idx) + '_median'] = median_projections[idx,:]
    data[str(idx) + '_upper'] = q75_projections[idx, :]
    data[str(idx) + '_lower'] = q25_projections[idx, :]

    data['base_' + str(idx) + '_median'] = baseline_reliabilities_mc[idx,:]
    data['base_' + str(idx) + '_upper'] = baseline_q75_projections[idx, :]
    data['base_' + str(idx) + '_lower'] = baseline_q25_projections[idx, :]

df = pd.DataFrame(data)
df.to_csv(f'04-exp_nonconstant_parameter_{num_mc_runs}_{test_uncertain}.csv', index=False)
