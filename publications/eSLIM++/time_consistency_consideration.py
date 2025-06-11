import subjective_logic as sl
import numpy as np
from termcolor import colored

import matplotlib.pyplot as plt


dimension = 2
nst = 8
trust_discount = 0.999
trust_discount_ref = 0.95
print(f'running the experiment of time-consistency consideration with a {colored('dimension','red')} of {dimension}')


op_initial = sl.Opinion2d.VacuousBeliefOpinion()
print(f'the initial vacuous opinion is given by {op_initial}')
op_st = sl.Opinion2d.VacuousBeliefOpinion()
op_st_memory = []
op_lt = sl.Opinion2d.VacuousBeliefOpinion()
counter_op_lt = 0
op_result = sl.Opinion2d.VacuousBeliefOpinion()
threshold = 0.2

op_baseline = sl.Opinion2d.VacuousBeliefOpinion()
op_baseline_2 = sl.Opinion2d.VacuousBeliefOpinion()

observations = [0,1,0,0,0,
                0,0,1,0,0,
                0,0,1,0,0,
                0,0,0,1,0,
                0,0,1,0,0,
                0,1,0,0,0,
                0,0,0,1,0,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                0,0,1,0,1,
                0,1,0,0,1,
                0,1,0,0,1,
                0,0,1,0,1,
                0,1,0,0,1,
                0,0,1,1,0]
gt = [0.8] * 35 + [0.0]*35 + [0.6]*30
evidence_vector = np.zeros((dimension,))

long_term_prior_opinions = []
long_term_post_opinions = []
short_term_opinions = []
result_opinions = []
baseline_opinions = []
baseline_opinions_2 = []

for obs in observations:
    # gathering information
    evidence_vector[obs] = 1
    print(f'after making the observation: {obs},\nthe evidence vector is: {evidence_vector}')

    obs_dirichlet = sl.DirichletDistribution2d.from_evidences(evidence_vector)
    op_obs = obs_dirichlet.as_opinion()
    print(f'the single-time-step observation opinion is given by {op_obs}')

    # simple baseline fusing all opinions over time
    op_baseline.trust_discount_(trust_discount)
    op_baseline.cum_fuse_(op_obs)
    baseline_opinions.append(op_baseline.copy())

    op_baseline_2.trust_discount_(trust_discount_ref)
    op_baseline_2.cum_fuse_(op_obs)
    baseline_opinions_2.append(op_baseline_2.copy())

    # update short-term opinion
    op_st.cum_fuse_(op_obs)
    print(f'the {colored('short-term opinion','green')} is updated to {op_st}')
    op_st_memory.append(op_obs)

    if len(op_st_memory) < nst:
        print(f'the {colored('short-term memory','light_blue')} is still collecting evidence; the collected evidence is {len(op_st_memory)}')
        op_result = op_st
        long_term_prior_opinions.append(op_lt.copy())
    else:
        print(f'the {colored('short-term memory','light_blue')} is filled with {len(op_st_memory)} evidence')
        # oldest short term opinion
        op_old = op_st_memory[0]
        # unfusion of oldest opinion from short-term memory
        op_st.cum_unfuse_(op_old)
        # fuse to long-term opinion
        op_lt.cum_fuse_(op_old)
        counter_op_lt += 1
        del op_st_memory[0]

        long_term_prior_opinions.append(op_lt.copy())

        # if counter_op_lt >= nst:
        print(f'perform {colored('long-term','yellow')} and {colored('short-term','light_blue')}, since long-term memory is filled with {counter_op_lt} evidence')
        if op_st.degree_of_conflict(op_lt) > threshold:
            # reset long-term opinion
            print(f'{colored('conflict is detected','red')}; DC = {op_st.degree_of_conflict(op_lt)}')
            op_lt = op_initial.copy()
            counter_op_lt = 0
            print(f'the {colored('long-term opinion','yellow')} is reset to {op_lt}')
        else:
            # trust discount the long-term opinion
            print(f'{colored('NO conflict is detected','light_green')}; DC = {op_st.degree_of_conflict(op_lt)}')
            op_lt.trust_discount_(trust_discount)
            print(f'the {colored('long-term opinion','yellow')} is trust discounted to {op_lt}')

        op_result = op_st.cum_fuse(op_lt)
        # else:
        #     print(f'{colored('long-term opinion','yellow')} is still collecting evidence; collected evidence: {counter_op_lt}')
        #     op_result = op_st

    long_term_post_opinions.append(op_lt.copy())
    short_term_opinions.append(op_st.copy())
    result_opinions.append(op_result.copy())


    evidence_vector = np.zeros((dimension,))
    print()
    print()

pp_lt = [op.getBinomialProjection() for op in long_term_post_opinions]
pp_st = [op.getBinomialProjection() for op in short_term_opinions]
pp_res = [op.getBinomialProjection() for op in result_opinions]
pp_baseline = [op.getBinomialProjection() for op in baseline_opinions]
pp_baseline_2 = [op.getBinomialProjection() for op in baseline_opinions_2]
uncert_lt = [op.uncertainty() for op in long_term_post_opinions]
uncert_st = [op.uncertainty() for op in short_term_opinions]
uncert_res = [op.uncertainty() for op in result_opinions]
uncert_base = [op.uncertainty() for op in baseline_opinions]
uncert_base_2 = [op.uncertainty() for op in baseline_opinions_2]

dc_st_lt = [op_st.degree_of_conflict(op_lt) for op_st, op_lt in zip(short_term_opinions, long_term_prior_opinions)]

# plt.plot(pp_lt, label='long-term')
# plt.plot(pp_st, label='short-term')
plt.plot(gt, label='ground truth', color='tab:red')
plt.plot(pp_baseline, label='baseline', color='tab:olive')
plt.plot(pp_baseline_2, label='baseline_2', color='tab:green')
plt.plot(pp_res, label='result', color='tab:blue')
plt.legend()

plt.figure()

# plt.plot(uncert_lt, label='long-term')
# plt.plot(uncert_st, label='short-term')
plt.plot(uncert_res, label='result')
plt.plot(uncert_base, label='baseline')
plt.plot(uncert_base_2, label='baseline')
plt.legend()

plt.figure()
plt.plot(dc_st_lt, label='degree-of-conflict')
plt.legend()

plt.show()

import pandas as pd

data = {'steps': range(len(pp_lt))}
data['pp_res'] = pp_res
data['pp_baseline'] = pp_baseline
data['pp_baseline_2'] = pp_baseline_2
data['pp_gt'] = gt
data['uncert_res'] = uncert_res
data['uncert_baseline'] = uncert_base
data['uncert_baseline_2'] = uncert_base_2
data['dc_st_lt'] = dc_st_lt

df = pd.DataFrame(data)
df.to_csv('04-time_consistency_consideration.csv', index=False)