import copy

import subjective_logic as sl
import random
import numpy as np

import matplotlib.pyplot as plt
from alive_progress import alive_bar


test_uncertain = 0.0

num_agents = 5
num_runs = 800
num_mc_runs = 1000

opinion_dimension = 4
possible_events = list(range(opinion_dimension))

# gt_distribution = [1/opinion_dimension]*opinion_dimension
gt_distribution = [0.1, 0.1, 0.4, 0.4]

num_runs_per_experiment = 8 * opinion_dimension

# colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:orange', 'tab:cyan']
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:olive', 'tab:red', 'tab:purple', 'tab:orange', 'tab:cyan']
draw_divider = 1
triangle_divider = 15

default_opinions = []
for i in range(opinion_dimension):
    belief_masses = [0.0] * opinion_dimension
    belief_masses[i] = 0.9
    prior = [1 / opinion_dimension] * opinion_dimension
    prior[0] = prior[0] - 0.1
    prior[-1] = prior[-1] + 0.1
    default_opinions.append(sl.Opinion(belief_masses, prior))
uncertain_opinion = sl.Opinion(*[0.]*opinion_dimension)


reliabilities = [0.00, 0.1, 0.7, 1.0, 1.0]

weighted_types_cs_avg = [
    (sl.TrustRevision.TrustRevisionType.REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.3),
    (sl.TrustRevision.TrustRevisionType.HARMONY_REFERENCE_FUSION, sl.Conflict.ConflictType.BELIEF_AVERAGE, 0.3),
]

trusts = [[] for _ in range(num_agents)]

def run_experiment(reliability, visibility):
    observations = sl.DirichletDistribution([1.]*opinion_dimension)
    for _ in range(num_runs_per_experiment):
        cur_event = np.random.choice(possible_events,p=gt_distribution)
        cur_other_events = copy.copy(possible_events)
        cur_other_events.remove(cur_event)
        cur_unreliable_event = random.choice(cur_other_events)

        cur_reliable = reliability > np.random.uniform()
        cur_visible = visibility > np.random.uniform()

        if cur_reliable:
            observations.evidences[cur_event] += 1
        elif cur_visible:
            observations.evidences[cur_unreliable_event] += 1
    return observations


baseline_decisions=np.zeros((num_mc_runs, num_agents, num_runs))
projections = np.zeros((num_mc_runs, num_agents, num_runs))
with alive_bar(num_mc_runs) as bar:
    for mc_run in range(num_mc_runs):
        trusted_opinions = [
            sl.TrustedOpinion(sl.Opinion([0, 0], [0.5, 0.5]), uncertain_opinion) for _ in range(num_agents)
        ]
        for run in range(num_runs):

            uncertainty_sum = 0
            # ground truth event for this run
            event = random.choice(possible_events)
            other_events = copy.copy(possible_events)
            other_events.remove(event)
            for idx in range(num_agents):
                if mc_run == 0:
                    trusts[idx].append(trusted_opinions[idx].trust_copy())

                projections[mc_run, idx, run] = trusted_opinions[idx].trust.getBinomialProjection()

                uncertainty_sum += trusted_opinions[idx].trust.uncertainty()

                trusted_opinions[idx].opinion = run_experiment(reliabilities[idx], 1.0 - test_uncertain).as_opinion()

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


gt_draw_window = 100
end_first = num_runs
start_first = end_first - gt_draw_window
for idx in range(num_agents):
    plt.plot([start_first, end_first], [reliabilities[idx]]*2, color='black', linewidth=1.5)


median_projections = np.median(projections, axis=0)
print(median_projections[:,-1])
q25_projections = np.quantile(projections, 0.25, axis=0)
q75_projections = np.quantile(projections, 0.75, axis=0)

plt.ylim(0,1)
for color, idx in zip(colors, range(num_agents)):
    plt.plot(median_projections[idx, ::draw_divider], color=color)
    plt.fill_between(
        range(0, num_runs // draw_divider),
        q75_projections[idx, ::draw_divider],
        q25_projections[idx, ::draw_divider],
        color='gray', alpha=0.3)

plt.show()

import pandas as pd

data = {'time': range(median_projections.shape[1])}
for idx in range(num_agents):
    data[str(idx) + '_median'] = median_projections[idx,:]
    data[str(idx) + '_upper'] = q75_projections[idx, :]
    data[str(idx) + '_lower'] = q25_projections[idx, :]

df = pd.DataFrame(data)
df.to_csv(f'04-exp_distribution_{num_mc_runs}_{test_uncertain}.csv', index=False)
