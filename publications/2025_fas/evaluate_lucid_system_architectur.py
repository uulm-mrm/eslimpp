#!/usr/bin/python3
from subjective_logic import Opinion2d as Opinion
import subjective_logic as sl

from system_state import SystemState, Mode

import matplotlib.pyplot as plt
import numpy as np
import copy

default_true = Opinion([0.9,0.0],[0.9,0.1])
default_false = Opinion([0.0,0.9],[0.1,0.9])
vacuous = Opinion(0,0)
default_somewhat_true = Opinion([0.5,0.1],[0.9,0.1])
default_somewhat_false = Opinion([0.1,0.5],[0.1,0.9])

steps_per_state = 100
constant_steps = 70
transition_steps = (steps_per_state - constant_steps) // 2

state = SystemState()

## System state 1

state.sensor_1 = default_true
state.processing_1 = default_true
state.concurrent_sa = default_true

state.sensor_2 = default_true
state.processing_2 = default_true

state.sensor_3 = default_true
state.processing_3 = default_true

state.fusion = default_true
state.planning = default_true

states = []
states.append(copy.deepcopy(state))

## System state 2

state.sensor_1 = default_false
state.concurrent_sa = default_false
states.append(copy.deepcopy(state))

## System state 3

state.sensor_2 = default_false
states.append(copy.deepcopy(state))

## System state 4

state.sensor_3 = default_false
states.append(copy.deepcopy(state))

## System state 5

state.sensor_1 = default_true
state.sensor_2 = default_true
state.sensor_3 = default_true
state.concurrent_sa = default_true
# state.fusion = vacuous
state.fusion = default_somewhat_true
# state.planning = vacuous
state.planning = default_somewhat_true
states.append(copy.deepcopy(state))

## System state 6

state.fusion = default_false
# state.fusion = default_false
# state.fusion = vacuous
# state.fusion = default_somewhat_true
state.planning = default_true
# state.planning = default_somewhat_true
states.append(copy.deepcopy(state))

num_states = len(states)

n_total_steps = num_states * steps_per_state
overall_projections = []
overall = []
safety_critical_projections = []
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
    overall.append(current_overall)
    overall_projections.append(current_overall.getBinomialProjection())

    current_state.mode = Mode.MINIMAL_FEASIBLE
    current_crit = current_state.getOverall()
    safety_critical.append(current_crit)
    safety_critical_projections.append(current_crit.getBinomialProjection())


# plt.plot(interpol_facs)
plt.figure("progressions")
plt.plot(overall_projections)
plt.plot(safety_critical_projections)

overall_projections_export = np.array([[n,val] for n,val in enumerate(overall_projections)])
critical_projection_export = np.array([[n,val] for n,val in enumerate(safety_critical_projections)])
np.savetxt('overall_projections.csv', overall_projections_export, delimiter=',', fmt='%10.5f')
np.savetxt('critical_projections.csv', critical_projection_export, delimiter=',', fmt='%10.5f')



# plt.figure("opinions")
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': " ".join([
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsfonts}',
        r'\usepackage{bm}',
    ])
})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# draw_flags_opinion = {
#     'draw_hypo_texts': False,
#     'draw_axis': False,
#     'draw_axis_label': False,
#     'draw_opinion': True,
#     'draw_opinion_label': False,
#     'draw_prior': False,
#     'draw_prior_label': False,
#     'draw_projection': False,
#     'draw_projection_label': False,
#     'belief_label_position': 0.5,
#     'disbelief_label_position': 0.6,
#     'uncertainty_label_position': 0.7,
# }

point_draw_args = {
    "color" : 'tab:gray',
    "zorder": 500,
    "s": 80,
}

default_text_draw_args = {
    "ha": 'left',
    "va": 'top',
    "fontsize": 20,
    "offset": [0.01, 0.00],
}

export_args = {
    "format": 'pdf',
    "dpi": 500,
    "transparent": True,
    "bbox_inches": 'tight',
    "pad_inches": 0,
}

fig, ax = sl.create_triangle_plot(hypo_texts=('disbelief\n(unhealty)', 'belief\n(healthy)'))

state_offset = steps_per_state // 2
# for idx in range(num_states):
for idx in range(5):
    cur_step = idx * steps_per_state + state_offset
    cur_overall = overall[idx * steps_per_state + state_offset]

    point_draw_args['color'] = '#0072BD'
    sl.draw_point(cur_overall, **point_draw_args)

    draw_options = default_text_draw_args.copy()
    draw_options['offset'] = [0.05,0.01]
    draw_options['ha'] = 'left'
    draw_options['va'] = 'center'

    if idx == 5:
        draw_options['offset'] = [-0.05, 0.01]
        draw_options['ha'] = 'right'
        draw_options['va'] = 'bottom'

    sl.draw_text_at_point(cur_overall, f'$^{{{idx+1}}}\\omega_Z^A$', **draw_options)

sl.reset_plot()
fig2, ax2 = sl.create_triangle_plot(hypo_texts=('disbelief\n(unsafe)', 'belief\n(safe)'))

state_offset = steps_per_state // 2
for idx in range(num_states):
# for idx in range(5):
    cur_step = idx * steps_per_state + state_offset
    cur_crit = safety_critical[idx * steps_per_state + state_offset]

    point_draw_args['color'] = '#D95319'
    sl.draw_point(cur_crit, **point_draw_args)

    draw_options = default_text_draw_args.copy()
    draw_options['offset'] = [0.025,0.01]
    draw_options['ha'] = 'left'
    draw_options['va'] = 'center'

    opinion_text = f'$^{{{idx+1}}}\\omega_Z^A$'
    # skip all double opinions when drawing names
    if idx in [1,2,5]:
        continue
    if idx == 0:
        draw_options['offset'] = [-0.03, 0.01]
        draw_options['ha'] = 'right'
        # opinion_text = f'$^1\\omega_Z^A$'
        # opinion_text = f'$^{{\\{{1,2\\}}}}\\omega_Z^A$'
        opinion_text = f'$^{{\\{{1,2,3\\}}}}\\omega_Z^A$'
    if idx == 3:
        # opinion_text = f'$^4\\omega_Z^A$'
        opinion_text = f'$^{{\\{{4,6\\}}}}\\omega_Z^A$'

    sl.draw_text_at_point(cur_crit, opinion_text, **draw_options)

x_start = -0.2
x_end = 1.2
y_start = -0.05
y_end = 0.95

x_diff = x_end - x_start
y_diff = y_end - y_start

relative_size = x_diff / y_diff

figure_size_y = 5

for _ax, _fig in zip([ax,ax2],[fig,fig2]):
    _ax.set_xlim(x_start, x_end)
    _ax.set_ylim(y_start, y_end)
    _fig.set_size_inches(figure_size_y * relative_size,figure_size_y)

plt.show()

fig.savefig("06-overall_bary_triangle.pdf", **export_args)
fig2.savefig("06-critical_bary_triangle.pdf", **export_args)








