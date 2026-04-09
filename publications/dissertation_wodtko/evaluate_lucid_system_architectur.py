#!/usr/bin/python3
from subjective_logic import Opinion2d as Opinion
import subjective_logic as sl

from system_state import SystemState, Mode

import matplotlib.pyplot as plt
import numpy as np
import copy

default_true = Opinion([0.9,0.0],[0.9,0.1])
default_false = Opinion([0.0,0.9],[0.1,0.9])
# vacuous = Opinion(0,0)
vacuous = Opinion([0,0],[0.1,0.9])
default_somewhat_true = Opinion([0.5,0.1],[0.7,0.3])
default_somewhat_false = Opinion([0.1,0.5],[0.3,0.7])

steps_per_state = 100
constant_steps = 50
transition_steps = (steps_per_state - constant_steps) // 2

state = SystemState()

## System state 1

state.v_1 = default_true
state.concurrent_sa = default_true

state.v_2 = default_true

state.v_3 = default_true

state.fusion = default_true
state.planning = default_true

test = copy.deepcopy(state)
test.v_1 = vacuous
test.v_2 = vacuous
test.v_3 = vacuous
test.concurrent_sa = vacuous
test.mode = Mode.STATE_OF_HEALTH
print("soh:", test.getOverall(), "proj: ", test.getOverall().getBinomialProjection())
test.mode = Mode.MINIMAL_FEASIBLE
print("sc:", test.getOverall(), "proj: ", test.getOverall().getBinomialProjection())

states = []
states.append(copy.deepcopy(state))

states = []
states.append(copy.deepcopy(state))

## System state 2
state.v_1 = vacuous
state.v_2 = vacuous
state.concurrent_sa = vacuous
states.append(copy.deepcopy(state))

## System state 3
state.v_3 = vacuous
states.append(copy.deepcopy(state))

## System state 4
state.v_1 = default_true
state.v_2 = default_true
state.v_3 = default_true

state.v_1 = default_false
state.concurrent_sa = default_false
states.append(copy.deepcopy(state))

## System state 5
state.v_2 = default_false
states.append(copy.deepcopy(state))

## System state 6
state.v_3 = default_false
states.append(copy.deepcopy(state))

## System state 7
state.v_1 = default_true
state.v_2 = default_true
state.v_3 = default_true
state.concurrent_sa = default_true
# state.fusion = vacuous
state.fusion = default_somewhat_true
# state.planning = vacuous
state.planning = default_somewhat_true
states.append(copy.deepcopy(state))

## System state 8
state.fusion = default_false
# state.fusion = default_false
# state.fusion = vacuous
# state.fusion = default_somewhat_true
state.planning = default_true
# state.planning = default_somewhat_true
states.append(copy.deepcopy(state))

num_states = len(states)

n_total_steps = num_states * steps_per_state
# overall_projections = []
state_of_health = []
# safety_critical_projections = []
safety_critical = []
interpol_facs = []
for idx in range(n_total_steps):
    idx_state = idx // steps_per_state
    idx_current_state = idx % steps_per_state

    start_state = states[idx_state]
    end_state = states[idx_state]
    interpol_fac = 0.5
    if idx_current_state < transition_steps and idx_state > 0:
        start_state = states[idx_state-1]
        interpol_fac = 0.5 + idx_current_state / transition_steps / 2
    elif idx_current_state >= steps_per_state - transition_steps and idx_state < num_states - 1:
        interpol_fac = (idx_current_state - transition_steps - constant_steps) / transition_steps / 2
        end_state = states[idx_state+1]

    interpol_facs.append(interpol_fac)
    current_state = start_state.interpolate(end_state, interpol_fac)

    current_state.mode = Mode.STATE_OF_HEALTH
    current_overall = current_state.getOverall()
    state_of_health.append(current_overall)

    current_state.mode = Mode.MINIMAL_FEASIBLE
    current_crit = current_state.getOverall()
    safety_critical.append(current_crit)

soh_projection = [op.getBinomialProjection() for op in state_of_health]
soh_uncerts = [op.uncertainty() for op in state_of_health]
sc_projection = [op.getBinomialProjection() for op in safety_critical]
sc_uncerts = [op.uncertainty() for op in safety_critical]

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title("State of Health")
ax1.plot(soh_projection, label="projection")
ax1.plot(soh_uncerts, label="uncertainty")

ax2.set_title("Safety Critical")
ax2.plot(sc_projection, label="projection")
ax2.plot(sc_uncerts, label="uncertainty")

plt.show()

# steps = (np.array(range(len(state_of_health))) + steps_per_state//2) / steps_per_state
# export = {
#     "steps": steps,
#     "soh_proj" : soh_projection,
#     "soh_uncerts": soh_uncerts,
#     "sc_proj": sc_projection,
#     "sc_uncerts": sc_uncerts,
# }
#
# import pandas as pd
# filename = "usf_results.csv"
# df = pd.DataFrame(export)
# df.to_csv(filename, index=False)
#
